"""Mask2Former (COCO-panoptic weights) on the example photos, against the same reference.

    python benchmarks/mask2former_baseline.py --name kaggle-t4
    python benchmarks/mask2former_baseline.py --name dry --random-weights --device cpu   # plumbing only

Mask2Former predicts COCO-panoptic's 133 classes in one pass. Its output is
mapped to this project's ontology (COCO_TO_ONTOLOGY below); pixels of classes
outside the ontology become void. It is then scored against the
human-corrected map exactly like this pipeline's automatic pre-annotation:

* on all classes (Mask2Former cannot predict bollard, buoy, shoe, fountain,
  boardwalk: those count against it, as they count in the corrected map);
* on the classes both can predict (common.COCO_COVERED), the fair comparison.

Post-processing is Hugging Face's `post_process_panoptic_segmentation`
(stuff classes fused, default thresholds), which upsamples masks from 384 px;
this can score lower than the original Detectron2 implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    COCO_COVERED,
    automatic_map,
    corrected_map,
    environment,
    gpu_label,
    pct,
    results_dir,
    save,
    scenes,
    timed,
)

from panoptic_prelabel.compare import Evaluation, evaluate, format_report
from panoptic_prelabel.config import Config
from panoptic_prelabel.masks import bbox_xywh
from panoptic_prelabel.panoptic import Panoptic, Segment, write_panoptic
from panoptic_prelabel.pipeline import load_rgb
from panoptic_prelabel.viz import panoptic_overlay, save_jpeg

MODELS = ["facebook/mask2former-swin-tiny-coco-panoptic", "facebook/mask2former-swin-large-coco-panoptic"]

# COCO-panoptic category name -> ontology class. Names checked against panopticapi's
# panoptic_coco_categories.json; anything not listed becomes void.
COCO_TO_ONTOLOGY = {
    "person": "person",
    "backpack": "backpack",
    "handbag": "bag",
    "suitcase": "bag",
    "bird": "bird",
    "dog": "dog",
    "cell phone": "phone",
    "car": "car",
    "truck": "truck",
    "bus": "truck",
    "stop sign": "sign",
    "sky-other-merged": "sky",
    "sea": "sea",
    "water-other": "sea",
    "river": "sea",
    "sand": "sand",
    "tree-merged": "tree",
    "grass-merged": "grass",
    "rock-merged": "rock",
    "building-other-merged": "building",
    "house": "building",
    "roof": "building",
    "pavement-merged": "pavement",
    "road": "road",
    "wall-other-merged": "wall",
    "wall-brick": "wall",
    "wall-stone": "wall",
    "wall-tile": "wall",
    "wall-wood": "wall",
}

PANOPTIC_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/cocodataset/panopticapi/"
    "7bb4655548f98f3fedc07bf37e9040a992b054b0/panoptic_coco_categories.json"
)


def coco_categories() -> list[dict]:
    with urllib.request.urlopen(PANOPTIC_CATEGORIES_URL, timeout=60) as r:
        return json.loads(r.read())


def load_model(model_id: str, device: str, random_weights: bool):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    if random_weights:  # plumbing test without downloading anything but the category names
        cats = coco_categories()
        config = Mask2FormerConfig(
            id2label={i: c["name"] for i, c in enumerate(cats)},
            label2id={c["name"]: i for i, c in enumerate(cats)},
        )
        torch.manual_seed(0)
        model = Mask2FormerForUniversalSegmentation(config)
        from transformers import Mask2FormerImageProcessor

        processor = Mask2FormerImageProcessor()
    else:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
    return processor, model.to(device).eval()  # type: ignore[arg-type]


def stuff_label_ids(id2label: dict[int, str], coco_cats: list[dict] | None) -> set[int]:
    """Label ids of stuff classes, to fuse into one segment each (COCO panoptic convention)."""
    if coco_cats is not None:
        stuff = {c["name"] for c in coco_cats if not c["isthing"]}
        return {i for i, n in id2label.items() if n in stuff}
    # the 133-class COCO-panoptic head: ids 0-79 are things, 80-132 stuff
    return {i for i in id2label if i >= 80}


def fuse_stuff(seg: np.ndarray, segments: list[dict], fuse: set[int]) -> tuple[np.ndarray, list[dict]]:
    """One segment per stuff class: merge every segment whose label is in `fuse`.

    Done here rather than with post_process_panoptic_segmentation(label_ids_to_fuse=...):
    in transformers 5.17 that option can give a thing the id of an earlier segment
    (the id counter resumes from a fused stuff id), which silently relabels pixels.
    """
    seg = seg.copy()
    first: dict[int, int] = {}
    kept = []
    for s in segments:
        label = s["label_id"]
        if label in fuse and label in first:
            seg[seg == s["id"]] = first[label]
            continue
        if label in fuse:
            first[label] = s["id"]
        kept.append(s)
    return seg, kept


def predict(processor, model, rgb: np.ndarray, device: str, fuse: set[int]) -> tuple[np.ndarray, list[dict]]:
    import torch

    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # no fusing in post-processing: every query gets its own, unique id (see fuse_stuff)
    res = processor.post_process_panoptic_segmentation(outputs, label_ids_to_fuse=set(), target_sizes=[rgb.shape[:2]])[
        0
    ]
    seg = res["segmentation"]
    seg = np.zeros(rgb.shape[:2], np.int32) if seg is None else seg.cpu().numpy().astype(np.int32)
    seg[seg < 0] = 0  # -1 = no mask found
    ids = [int(s["id"]) for s in res["segments_info"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"post-processing returned duplicate segment ids {ids}")
    segments = [{"id": int(s["id"]), "label_id": int(s["label_id"])} for s in res["segments_info"]]
    return fuse_stuff(seg, segments, fuse)


def to_ontology(seg: np.ndarray, segments: list[dict], id2label: dict[int, str], cfg: Config) -> tuple[Panoptic, float]:
    """Map a COCO-panoptic prediction to the ontology; returns it and the share of pixels made void."""
    h, w = seg.shape
    id_map = np.zeros((h, w), np.int32)
    out: list[Segment] = []
    stuff_ids: dict[str, int] = {}
    unmapped = 0
    next_id = 1
    for s in segments:
        m = seg == s["id"]
        if not m.any():
            continue
        target = COCO_TO_ONTOLOGY.get(id2label[s["label_id"]])
        if target is None:
            unmapped += int(m.sum())
            continue
        cat = cfg.by_name[target]
        if not cat.isthing and target in stuff_ids:  # several COCO stuff classes -> one ontology class
            id_map[m] = stuff_ids[target]
            continue
        id_map[m] = next_id
        if not cat.isthing:
            stuff_ids[target] = next_id
        out.append(Segment(next_id, target, cat.isthing, 0, [0, 0, 0, 0], []))
        next_id += 1
    for s in out:
        m = id_map == s.id
        s.area, s.bbox = int(m.sum()), bbox_xywh(m)
    return Panoptic("image.jpg", w, h, id_map, [s for s in out if s.area > 0]), unmapped / (h * w)


def row(name: str, ev: Evaluation) -> str:
    a, c = ev.summary(), ev.summary(only=COCO_COVERED)
    return (
        f"| {name} | {pct(a['PQ'])} | {pct(a['PQ_th'])} | {pct(a['PQ_st'])} | {pct(a['mIoU'])} "
        f"| {pct(c['PQ'])} | {pct(c['PQ_th'])} | {pct(c['PQ_st'])} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--results-root")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--repeat", type=int, default=10, help="timed runs per photo")
    ap.add_argument("--random-weights", action="store_true", help="plumbing test, no download of weights")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if device == "cuda" else None
    out = results_dir(args.name, args.results_root)
    cfg = Config.load()
    env = environment()
    coco_cats = None
    try:
        coco_cats = coco_categories()
    except OSError as e:
        print(f"note: could not fetch the COCO category list ({e}); assuming ids >= 80 are stuff")

    refs = {s.name: corrected_map(s, cfg) for s in scenes()}
    images = {s.name: load_rgb(s / "image.jpg") for s in scenes()}
    with tempfile.TemporaryDirectory() as tmp:
        autos = {s.name: automatic_map(s, cfg, Path(tmp)) for s in scenes()}

    table = [
        "| predictor | PQ | PQ things | PQ stuff | mIoU "
        "| PQ (common classes) | PQ things (common) | PQ stuff (common) |",
        "|---|---|---|---|---|---|---|---|",
        row("this pipeline, automatic part (Mask R-CNN + MobileSAM + region naming)",
            evaluate([(refs[n], autos[n]) for n in refs])),
    ]  # fmt: skip
    payload: dict = {"environment": env, "device": device, "models": {}}
    details = []
    for model_id in args.models:
        label = model_id.split("/")[-1] + (" (random weights)" if args.random_weights else "")
        try:
            result = run_model(model_id, label, args, device, sync, cfg, coco_cats, images, refs, out)
        except Exception as e:  # a failed download or an out-of-memory must not lose the other model
            print(f"[{label}] error: {type(e).__name__}: {e}")
            payload["models"][model_id] = {"error": f"{type(e).__name__}: {e}"[:300]}
            table.append(f"| {label} | error: {type(e).__name__} | | | | | | |")
        else:
            ev = result.pop("evaluation")
            table.append(row(label, ev))
            details.append(f"### {label}\n\n```\n{format_report(ev, per_class=True)}\n```")
            payload["models"][model_id] = result
        if device == "cuda":
            torch.cuda.empty_cache()
        save(out, "mask2former_baseline", payload, report(payload, table, details, images, device, env))


def run_model(model_id, label, args, device, sync, cfg, coco_cats, images, refs, out) -> dict:
    processor, model = load_model(model_id, device, args.random_weights)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    missing = sorted(set(COCO_TO_ONTOLOGY) - set(id2label.values()))
    if missing:
        print(f"warning: {model_id} has no class named {missing}")
    fuse = stuff_label_ids(id2label, coco_cats)
    preds, timing, void_share = {}, {}, {}
    for name, rgb in images.items():
        seg, segments = predict(processor, model, rgb, device, fuse)
        preds[name], void_share[name] = to_ontology(seg, segments, id2label, cfg)
        timing[name] = timed(partial(predict, processor, model, rgb, device, fuse), 2, args.repeat, sync)
        timing[name].pop("last")
        scene_out = out / "mask2former" / model_id.split("/")[-1] / name
        write_panoptic(preds[name], cfg, scene_out)
        save_jpeg(panoptic_overlay(rgb, preds[name], cfg), scene_out / "preview.jpg")
        print(f"[{label}] {name}: {timing[name]['median_ms']} ms, {100 * void_share[name]:.1f} % pixels unmapped")
    ev = evaluate([(refs[n], preds[n]) for n in refs])
    return {
        "evaluation": ev,
        "random_weights": args.random_weights,
        "summary_all": ev.summary(),
        "summary_common": ev.summary(only=COCO_COVERED),
        "timing": timing,
        "unmapped_pixel_share": void_share,
        "missing_class_names": missing,
    }


def report(payload: dict, table: list[str], details: list[str], images: dict, device: str, env: dict) -> str:
    timing_lines = ["| model | " + " | ".join(images) + " |", "|---|" + "---|" * len(images)]
    for model_id, m in payload["models"].items():
        if "timing" in m:
            cells = " | ".join(f"{m['timing'][n]['median_ms']} ms" for n in images)
            timing_lines.append(f"| {model_id.split('/')[-1]} | {cells} |")
    return "\n".join(
        [
            "# Mask2Former vs this pipeline (4 example photos)",
            "",
            "Reference: the human-corrected maps. 'Common classes' = the ontology classes a COCO-panoptic "
            "model can predict (bollard, buoy, shoe, fountain and boardwalk excluded).",
            "",
            f"Device: {device} on {gpu_label(env) if env.get('gpus') else env['cpu']}",
            "",
            *table,
            "",
            "## Time per photo (forward + post-processing, median)",
            "",
            *timing_lines,
            "",
            "## Per class",
            "",
            *details,
        ]
    )


if __name__ == "__main__":
    main()
