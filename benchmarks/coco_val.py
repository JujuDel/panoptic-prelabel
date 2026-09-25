"""Mask2Former on a sample of COCO val2017, scored by this repo's `compare` AND by panopticapi.

    python benchmarks/coco_val.py --name kaggle-t4 --images 100

Two purposes:

1. Validate `panoptic_prelabel.compare` on real data at scale: the same
   predictions are scored by the official COCO panopticapi and by `compare`,
   and the script fails if PQ / SQ / RQ (overall, things, stuff) differ by more
   than 1e-9.
2. Put the Mask2Former numbers of mask2former_baseline.py in context: PQ on a
   random (seeded) sample of COCO val2017 images, with the same inference and
   post-processing code.

Downloads: the COCO panoptic annotations (~820 MB zip, only val is extracted)
and the sampled val2017 images, one by one, from images.cocodataset.org.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import environment, pct, results_dir, save, timed
from mask2former_baseline import MODELS, load_model, predict, stuff_label_ids

from panoptic_prelabel.compare import evaluate
from panoptic_prelabel.masks import bbox_xywh
from panoptic_prelabel.panoptic import id2rgb, read_panoptic_dataset

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
IMAGES_URL = "http://images.cocodataset.org/val2017/{}"


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310 (fixed COCO URLs)
            while chunk := r.read(1 << 22):
                f.write(chunk)
        tmp.replace(dest)
    return dest


def prepare_gt(cache: Path, n: int, seed: int, local: list[str] | None) -> tuple[Path, Path, dict]:
    """The GT JSON of a seeded sample of n images, and the folder of their PNGs.

    Default: download the COCO panoptic annotations and extract val2017.
    `local` = [panoptic json, png folder] of an existing copy (any COCO-panoptic dataset).
    """
    if local:
        gt_json, gt_dir = Path(local[0]), Path(local[1])
        data = json.loads(gt_json.read_text())
    else:
        outer = fetch(ANNOTATIONS_URL, cache / "panoptic_annotations_trainval2017.zip")
        gt_dir = cache / "panoptic_val2017"
        gt_json = cache / "panoptic_val2017.json"
        with zipfile.ZipFile(outer) as z:
            if not gt_json.exists():
                gt_json.write_bytes(z.read("annotations/panoptic_val2017.json"))
            inner = cache / "panoptic_val2017.zip"
            if not inner.exists():
                inner.write_bytes(z.read("annotations/panoptic_val2017.zip"))
        data = json.loads(gt_json.read_text())
    anns = sorted(data["annotations"], key=lambda a: a["image_id"])
    sample = random.Random(seed).sample(anns, min(n, len(anns)))  # noqa: S311 (sampling, not crypto)
    ids = {a["image_id"] for a in sample}
    if not local:
        gt_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(cache / "panoptic_val2017.zip") as z:
            for a in sample:
                target = gt_dir / a["file_name"]
                if not target.exists():
                    target.write_bytes(z.read(f"panoptic_val2017/{a['file_name']}"))
    subset = {
        "images": [i for i in data["images"] if i["id"] in ids],
        "annotations": sample,
        "categories": data["categories"],
    }
    cache.mkdir(parents=True, exist_ok=True)
    sub_json = cache / f"panoptic_val2017_sample{n}_seed{seed}.json"
    sub_json.write_text(json.dumps(subset))
    return sub_json, gt_dir, subset


def to_coco_prediction(
    seg: np.ndarray, segments: list[dict], id2label: dict[int, str], name_to_cat: dict[str, int]
) -> tuple[np.ndarray, list[dict]]:
    """A Hugging Face panoptic prediction as a COCO-panoptic map + segments_info.

    Segments with no pixel are dropped (panopticapi rejects them) and so are
    pixels whose id has no segment entry."""
    infos = []
    for s in segments:
        m = seg == s["id"]
        if m.any():
            infos.append({
                "id": s["id"], "category_id": name_to_cat[id2label[s["label_id"]]],
                "iscrowd": 0, "area": int(m.sum()), "bbox": bbox_xywh(m),
            })  # fmt: skip
    keep = [i["id"] for i in infos]
    return np.where(np.isin(seg, keep), seg, 0).astype(np.int32), infos


def official_pq(gt_json: Path, gt_dir: Path, pred_json: Path, pred_dir: Path) -> dict:
    from panopticapi.evaluation import pq_compute_single_core

    gt = json.loads(gt_json.read_text())
    pred = json.loads(pred_json.read_text())
    cats = {c["id"]: c for c in gt["categories"]}
    by_image = {a["image_id"]: a for a in pred["annotations"]}
    pairs = [(a, by_image[a["image_id"]]) for a in gt["annotations"]]
    stat = pq_compute_single_core(0, pairs, str(gt_dir), str(pred_dir), cats)
    out = {}
    for key, isthing in (("", None), ("_th", True), ("_st", False)):
        res, _ = stat.pq_average(cats, isthing=isthing)
        out["PQ" + key] = res["pq"]
        if isthing is None:
            out["SQ"], out["RQ"] = res["sq"], res["rq"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--results-root")
    ap.add_argument("--cache", default="coco_cache", help="download / extraction folder")
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument(
        "--local",
        nargs=3,
        metavar=("GT_JSON", "GT_PNG_DIR", "IMAGE_DIR"),
        help="use an existing COCO-panoptic copy instead of downloading",
    )
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if device == "cuda" else None
    out = results_dir(args.name, args.results_root)
    cache = Path(args.cache)
    gt_json, gt_dir, subset = prepare_gt(cache, args.images, args.seed, args.local[:2] if args.local else None)
    name_to_cat = {c["name"]: c["id"] for c in subset["categories"]}
    gt_maps = read_panoptic_dataset(gt_json, gt_dir)

    payload: dict = {"environment": environment(), "images": args.images, "seed": args.seed, "models": {}}
    lines = [
        f"# Mask2Former on {args.images} COCO val2017 images (seed {args.seed})",
        "",
        "| model | PQ | PQ things | PQ stuff | SQ | RQ | max difference compare vs panopticapi | median time / image |",
        "|---|---|---|---|---|---|---|---|",
    ]
    footer = [
        "",
        "PQ as computed by the official panopticapi; `panoptic_prelabel.compare` gives the same numbers "
        "(column 'max difference'). Hugging Face post-processing, default thresholds.",
    ]
    disagreement = False
    for model_id in args.models:
        short = model_id.split("/")[-1] + (" (random weights)" if args.random_weights else "")
        try:
            res = run_model(model_id, args, device, sync, out, cache, subset, name_to_cat, gt_json, gt_dir, gt_maps)
        except Exception as e:  # a failed download or an out-of-memory must not lose the other model
            print(f"[{short}] error: {type(e).__name__}: {e}")
            payload["models"][model_id] = {"error": f"{type(e).__name__}: {e}"[:300]}
            lines.append(f"| {short} | error: {type(e).__name__} | | | | | | |")
        else:
            official, diff = res["official"], res["max_abs_diff"]
            disagreement |= diff > 1e-9
            payload["models"][model_id] = res
            lines.append(
                f"| {short} | {pct(official['PQ'])} | {pct(official['PQ_th'])} | {pct(official['PQ_st'])} "
                f"| {pct(official['SQ'])} | {pct(official['RQ'])} | {diff:.1e} | {res['median_ms']:.0f} ms |"
            )
        if device == "cuda":
            torch.cuda.empty_cache()
        save(out, "coco_val", payload, "\n".join(lines + footer))
    if disagreement:
        raise SystemExit("compare disagrees with panopticapi by more than 1e-9: see coco_val.json")


def run_model(model_id, args, device, sync, out, cache, subset, name_to_cat, gt_json, gt_dir, gt_maps) -> dict:
    short = model_id.split("/")[-1]
    processor, model = load_model(model_id, device, args.random_weights)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    unknown = sorted(set(id2label.values()) - set(name_to_cat))
    if unknown:
        raise ValueError(f"{model_id}: labels not in COCO panoptic categories: {unknown[:10]}")
    fuse = stuff_label_ids(id2label, subset["categories"])
    pred_dir = out / "coco_val" / short
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_anns, times = [], []
    for k, img in enumerate(subset["images"]):
        if args.local:
            path = Path(args.local[2]) / img["file_name"]
        else:
            path = fetch(IMAGES_URL.format(img["file_name"]), cache / "val2017" / img["file_name"])
        rgb = np.asarray(Image.open(path).convert("RGB"))
        seg, segments = predict(processor, model, rgb, device, fuse)
        if k < 20:  # timing on the first images is enough
            times.append(timed(partial(predict, processor, model, rgb, device, fuse), 1, 3, sync)["median_ms"])
        seg, infos = to_coco_prediction(seg, segments, id2label, name_to_cat)
        png = img["file_name"].replace(".jpg", ".png")
        Image.fromarray(id2rgb(seg)).save(pred_dir / png)
        pred_anns.append({"image_id": img["id"], "file_name": png, "segments_info": infos})
        if k % 20 == 0:
            print(f"[{short}] {k + 1}/{len(subset['images'])}", flush=True)
    pred_json = out / "coco_val" / f"{short}.json"
    pred_json.write_text(
        json.dumps({"images": subset["images"], "annotations": pred_anns, "categories": subset["categories"]})
    )
    official = official_pq(gt_json, gt_dir, pred_json, pred_dir)
    pred_maps = read_panoptic_dataset(pred_json, pred_dir)
    ours = evaluate([(gt_maps[i], pred_maps[i]) for i in gt_maps]).summary()
    diff = max(abs(ours[k] - official[k]) for k in official)
    med = float(np.median(times)) if times else float("nan")
    return {"official": official, "ours": ours, "max_abs_diff": diff, "median_ms": med}


if __name__ == "__main__":
    main()
