"""refine without any model: hand-made detector output and regions."""

import json

import numpy as np
import pytest
from conftest import rect
from PIL import Image

from panoptic_prelabel.coco_io import read_coco
from panoptic_prelabel.masks import rle_encode
from panoptic_prelabel.pipeline import UnlabeledRegions, refine
from panoptic_prelabel.stuff import partition

H, W = 60, 80


def _work(tmp_path, cfg, labels):
    person = rect(H, W, 30, 20, 45, 60)
    # SAM-like proposals: sky, sea, a post, a watch-sized blob touching the person, and a big one
    masks = [
        rect(H, W, 0, 0, W, 30),  # 1 sky (painted first: largest)
        rect(H, W, 0, 30, W, H),  # 2 sea
        rect(H, W, 60, 10, 64, 50),  # post
        rect(H, W, 45, 40, 49, 44),  # watch, touches the person
    ]
    regions = partition(masks, person, cfg)
    work = tmp_path / "work"
    work.mkdir()
    (work / "things.json").write_text(
        json.dumps(
            {
                "image": {"file_name": "x.jpg", "width": W, "height": H},
                "instances": [{"category": "person", "score": 0.99, "coco_class": "person", "rle": rle_encode(person)}],
            }
        )
    )
    Image.fromarray(regions.label_map.astype(np.uint16)).save(work / "regions.png")
    (work / "regions.json").write_text(
        json.dumps({str(k): {"point": list(regions.points[k]), "area": regions.areas[k]} for k in regions.points})
    )
    lines = ["regions:"]
    for k, (x, y) in regions.points.items():
        lines.append(f"  - {{id: {k}, point: [{x}, {y}], label: {labels(x, y)}}}")
    (work / "regions.yaml").write_text("\n".join(lines) + "\n")
    return work


def _label(x, y):
    if 58 <= x <= 65:
        return "bollard"
    if 44 <= x <= 50 and 39 <= y <= 45:
        return "person"
    return "sky" if y < 30 else "sea"


def test_partition_is_flat_and_numbered_by_area(cfg):
    masks = [rect(H, W, 0, 0, W, H), rect(H, W, 10, 10, 50, 50)]
    regions = partition(masks, np.zeros((H, W), bool), cfg)
    assert sorted(regions.points) == [1, 2]
    assert regions.areas[1] >= regions.areas[2]
    assert regions.label_map[30, 30] != regions.label_map[5, 5]  # the smaller one is on top


def test_refine_builds_the_cvat_preannotation(tmp_path, cfg):
    cfg.data["stuff"]["min_region_area"] = 10
    cfg.data["refine"]["drop_islands_below"] = 5
    work = _work(tmp_path, cfg, _label)
    out = refine(work, cfg)
    anns = read_coco(out).annotations
    cats = sorted(a.category for a in anns)
    assert cats == ["bollard", "person", "sea", "sky"]
    person = next(a for a in anns if a.category == "person")
    assert person.mask[42, 47], "the watch region was merged into the detected person"


def test_refine_refuses_unlabelled_regions(tmp_path, cfg):
    cfg.data["stuff"]["min_region_area"] = 10
    work = _work(tmp_path, cfg, lambda x, y: "null")
    with pytest.raises(UnlabeledRegions, match="no label"):
        refine(work, cfg)
    refine(work, cfg, allow_unlabeled=True)  # explicit opt-out works


def test_refine_rejects_unknown_labels(tmp_path, cfg):
    cfg.data["stuff"]["min_region_area"] = 10
    work = _work(tmp_path, cfg, lambda x, y: "lava")
    with pytest.raises(ValueError, match="unknown label"):
        refine(work, cfg)


def test_ignored_region_stays_a_hole(tmp_path, cfg):
    cfg.data["stuff"]["min_region_area"] = 10
    cfg.data["refine"]["drop_islands_below"] = 5
    cfg.data["refine"]["fill_holes_below"] = 10_000  # even hole filling must not cover it

    def label(x, y):
        return "ignore" if 58 <= x <= 65 else _label(x, y)  # the post is ignored

    anns = read_coco(refine(_work(tmp_path, cfg, label), cfg)).annotations
    covered = np.logical_or.reduce([a.mask for a in anns])
    assert not covered[30, 62], "an ignored region must not be filled with a neighbouring class"
    assert "bollard" not in {a.category for a in anns}


def test_prelabel_on_a_rotated_photo_names_the_upright_copy(tmp_path, cfg):
    from panoptic_prelabel.pipeline import UPRIGHT, prelabel
    from panoptic_prelabel.stuff import Regions

    stored = np.zeros((30, 50, 3), np.uint8)  # landscape pixels...
    photo = tmp_path / "IMG_0001.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # ...displayed rotated 90 degrees: portrait 30 x 50
    Image.fromarray(stored).save(photo, exif=exif)

    def detector(rgb):
        assert rgb.shape[:2] == (50, 30)  # the models see the upright image
        return []

    def proposer(rgb, things):
        lm = np.zeros(rgb.shape[:2], np.int32)
        lm[:, :] = 1
        return Regions(lm, {1: (5, 5)}, {1: int(lm.size)})

    work = prelabel(photo, tmp_path / "work", cfg, detector=detector, proposer=proposer)
    meta = json.loads((work / "things.json").read_text())
    assert meta["image"] == {"file_name": UPRIGHT, "width": 30, "height": 50}
    assert Image.open(work / UPRIGHT).size == (30, 50)
