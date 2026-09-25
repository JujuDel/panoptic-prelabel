"""The glue code of benchmarks/ (the GPU runs themselves happen on Kaggle)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import common  # noqa: E402
import mask2former_baseline as m2f  # noqa: E402

from panoptic_prelabel.compare import evaluate  # noqa: E402
from panoptic_prelabel.panoptic import id2rgb, read_panoptic_dataset  # noqa: E402

ID2LABEL = {
    0: "person",
    1: "car",
    80: "sky-other-merged",
    81: "wall-brick",
    82: "wall-other-merged",
    83: "fence-merged",
}


def test_timed_reports_order_statistics():
    calls = []
    t = common.timed(lambda: calls.append(1), warmup=2, repeat=5)
    assert len(calls) == 7 and t["runs"] == 5
    assert t["min_ms"] <= t["median_ms"] <= t["p90_ms"]


def test_to_ontology_maps_fuses_and_voids(cfg):
    seg = np.array(
        [
            [1, 1, 2, 2, 3, 3],
            [4, 4, 5, 5, 0, 0],
        ],
        np.int32,
    )
    segments = [
        {"id": 1, "label_id": 0},  # person
        {"id": 2, "label_id": 80},  # sky
        {"id": 3, "label_id": 81},  # wall-brick -> wall
        {"id": 4, "label_id": 82},  # wall-other-merged -> wall: merged with id 3
        {"id": 5, "label_id": 83},  # fence: not in the ontology -> void
        {"id": 9, "label_id": 1},  # listed but no pixel -> dropped
    ]
    pan, void_share = m2f.to_ontology(seg, segments, ID2LABEL, cfg)
    cats = sorted(s.category for s in pan.segments)
    assert cats == ["person", "sky", "wall"]
    wall = next(s for s in pan.segments if s.category == "wall")
    assert wall.area == 4
    assert void_share == pytest.approx(2 / 12)
    assert (pan.id_map[1, 2:4] == 0).all()


def test_stuff_label_ids_from_categories():
    cats = [{"name": "person", "isthing": 1}, {"name": "sky-other-merged", "isthing": 0}]
    assert m2f.stuff_label_ids(ID2LABEL, cats) == {80}
    assert m2f.stuff_label_ids(ID2LABEL, None) == {80, 81, 82, 83}


def test_coco_prediction_scores_like_panopticapi(tmp_path):
    pytest.importorskip("panopticapi.evaluation")
    import coco_val

    cats = [
        {"id": 1, "name": "person", "isthing": 1},
        {"id": 3, "name": "car", "isthing": 1},
        {"id": 187, "name": "sky-other-merged", "isthing": 0},
    ]
    name_to_cat = {c["name"]: c["id"] for c in cats}
    id2label = {0: "person", 1: "car", 80: "sky-other-merged"}
    rng = np.random.default_rng(3)
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    images, gt_anns, pred_anns = [], [], []
    for k in range(6):
        gt = np.full((40, 60), 1, np.int32)  # sky everywhere
        gt[5:25, 5:25] = 2  # a person
        gt[20:35, 30:55] = 3  # a car
        gt[0:3, :] = 0  # void strip
        infos = []
        for sid, cid in ((1, 187), (2, 1), (3, 3)):
            m = gt == sid
            infos.append({"id": sid, "category_id": cid, "iscrowd": 0, "area": int(m.sum()), "bbox": [0, 0, 1, 1]})
        Image.fromarray(id2rgb(gt)).save(gt_dir / f"{k}.png")
        # the prediction: shifted by a random offset, labels in the model's id space
        pred = np.roll(gt, int(rng.integers(-6, 7)), axis=1)
        segments = [
            {"id": 1, "label_id": 80},
            {"id": 2, "label_id": 0},
            {"id": 3, "label_id": 1},
            {"id": 7, "label_id": 1},
        ]
        seg, pinfos = coco_val.to_coco_prediction(pred, segments, id2label, name_to_cat)
        Image.fromarray(id2rgb(seg)).save(pred_dir / f"{k}.png")
        images.append({"id": k, "file_name": f"{k}.jpg", "width": 60, "height": 40})
        gt_anns.append({"image_id": k, "file_name": f"{k}.png", "segments_info": infos})
        pred_anns.append({"image_id": k, "file_name": f"{k}.png", "segments_info": pinfos})
    gt_json, pred_json = tmp_path / "gt.json", tmp_path / "pred.json"
    gt_json.write_text(json.dumps({"images": images, "annotations": gt_anns, "categories": cats}))
    pred_json.write_text(json.dumps({"images": images, "annotations": pred_anns, "categories": cats}))

    official = coco_val.official_pq(gt_json, gt_dir, pred_json, pred_dir)
    g, p = read_panoptic_dataset(gt_json, gt_dir), read_panoptic_dataset(pred_json, pred_dir)
    ours = evaluate([(g[i], p[i]) for i in g]).summary()
    for key in official:
        assert ours[key] == pytest.approx(official[key], abs=1e-9)
    assert 0 < official["PQ"] < 1  # a non-trivial case


def test_fuse_stuff_merges_stuff_and_keeps_things_apart():
    seg = np.array([[1, 2, 3, 4, 5]], np.int32)
    segments = [
        {"id": 1, "label_id": 80},  # sky
        {"id": 2, "label_id": 0},  # person
        {"id": 3, "label_id": 80},  # sky again -> merged into id 1
        {"id": 4, "label_id": 1},  # car
        {"id": 5, "label_id": 0},  # a second person stays separate
    ]
    fused, kept = m2f.fuse_stuff(seg, segments, fuse={80})
    assert fused.tolist() == [[1, 2, 1, 4, 5]]
    assert [s["id"] for s in kept] == [1, 2, 4, 5]
