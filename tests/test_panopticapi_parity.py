"""compare.py against the official COCO panopticapi, on real and synthetic maps.

Skipped unless panopticapi is installed (it is not on PyPI):
    pip install git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from panoptic_prelabel.coco_io import read_coco
from panoptic_prelabel.compare import evaluate
from panoptic_prelabel.config import Config
from panoptic_prelabel.masks import bbox_xywh
from panoptic_prelabel.panoptic import Panoptic, Segment, id2rgb, read_panoptic
from panoptic_prelabel.pipeline import refine
from panoptic_prelabel.resolve import resolve

pq_eval = pytest.importorskip("panopticapi.evaluation")
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def official(pairs: list[tuple[Panoptic, Panoptic]], cfg: Config, tmp: Path) -> dict[str, float]:
    """Write the pairs as COCO panoptic files and score them with panopticapi."""
    categories = {c.id: {"id": c.id, "name": c.name, "isthing": int(c.isthing)} for c in cfg.categories}
    annotation_set = []
    for side in ("gt", "pred"):
        (tmp / side).mkdir(parents=True, exist_ok=True)
    for i, (ref, pred) in enumerate(pairs):
        anns = []
        for side, pan in (("gt", ref), ("pred", pred)):
            name = f"{i}.png"
            Image.fromarray(id2rgb(pan.id_map)).save(tmp / side / name)
            anns.append(
                {
                    "image_id": i,
                    "file_name": name,
                    "segments_info": [
                        {
                            "id": s.id,
                            "category_id": cfg.by_name[s.category].id,
                            "area": s.area,
                            "iscrowd": int(s.iscrowd),
                        }
                        for s in pan.segments
                        if (pan.id_map == s.id).any() or side == "gt"
                    ],
                }
            )
        annotation_set.append(tuple(anns))
    stat = pq_eval.pq_compute_single_core(0, annotation_set, str(tmp / "gt"), str(tmp / "pred"), categories)
    out = {}
    for key, isthing in (("", None), ("_th", True), ("_st", False)):
        res, _ = stat.pq_average(categories, isthing=isthing)
        out["PQ" + key] = res["pq"]
        if isthing is None:
            out["SQ"], out["RQ"] = res["sq"], res["rq"]
    return out


def assert_same(ours: dict[str, float], ref: dict[str, float]) -> None:
    for key in ("PQ", "SQ", "RQ", "PQ_th", "PQ_st"):
        assert ours[key] == pytest.approx(ref[key], abs=1e-9), key


def example_pairs(cfg: Config, tmp: Path, which: str) -> list[tuple[Panoptic, Panoptic]]:
    pairs = []
    for scene in sorted(p for p in EXAMPLES.iterdir() if (p / "cvat_corrected.zip").exists()):
        groups = yaml.safe_load((scene / "scene.yaml").read_text())["groups"] if (scene / "scene.yaml").exists() else []
        corrected = resolve(read_coco(scene / "cvat_corrected.zip"), cfg, groups=groups)
        if which == "auto":
            work = tmp / "work" / scene.name
            shutil.copytree(scene / "prelabel", work)
            other = resolve(read_coco(refine(work, cfg, labels=scene / "regions.yaml")), cfg)
        else:
            other = read_panoptic(scene / "reference_2026-07" / "panoptic.json")
        pairs.append((corrected, other))
    return pairs


@pytest.mark.parametrize("which", ["auto", "july"])
def test_examples_match_panopticapi(tmp_path, cfg, which):
    pairs = example_pairs(cfg, tmp_path, which)
    assert_same(evaluate(pairs).summary(), official(pairs, cfg, tmp_path / "coco"))


def random_map(rng, h, w, cats, n, crowd=False, void=True) -> Panoptic:
    id_map = np.zeros((h, w), np.int32) if void else np.ones((h, w), np.int32)
    names = {1: cats[0]} if not void else {}
    for sid in range(2, n + 2):
        y, x = rng.integers(0, h - 4), rng.integers(0, w - 4)
        id_map[y : y + rng.integers(4, h // 2), x : x + rng.integers(4, w // 2)] = sid
        names[sid] = cats[int(rng.integers(len(cats)))]
    segs = []
    for sid, c in names.items():
        m = id_map == sid
        if m.any():
            thing = c in ("person", "car", "bag")
            segs.append(Segment(sid, c, thing, int(m.sum()), bbox_xywh(m), [], iscrowd=crowd and thing and sid == 2))
    return Panoptic("x", w, h, id_map, segs)


def test_random_maps_with_void_and_crowd_match_panopticapi(tmp_path, cfg):
    rng = np.random.default_rng(0)
    cats = ["person", "car", "bag", "sky", "road", "tree"]
    pairs = []
    for _ in range(20):
        ref = random_map(rng, 48, 64, cats, 6, crowd=True, void=True)
        pred = random_map(rng, 48, 64, cats, 6, crowd=False, void=bool(rng.integers(2)))
        pairs.append((ref, pred))
    pairs.append((pairs[0][0], pairs[0][0]))  # and one perfect prediction
    assert_same(evaluate(pairs).summary(), official(pairs, cfg, tmp_path / "coco"))
