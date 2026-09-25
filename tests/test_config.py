"""Config overrides: typos and wrong types fail at load time."""

import pytest

from panoptic_prelabel.config import Config


def load(tmp_path, text):
    f = tmp_path / "c.yaml"
    f.write_text(text)
    return Config.load(f)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("resolve:\n  fill: neareast\n", "not one of"),
        ("stuff:\n  min_regoin_area: 5\n", "unknown key"),
        ("stuff:\n  points_per_side: 32.5\n", "expected int"),
        ("resolve:\n  split_classes: [bagg]\n", "unknown classes"),
        ("web:\n  instance_shades: [a]\n", "expected float"),
        ("things:\n  coco_to_ontology:\n    motorcycle: scooter\n", "unknown classes"),
        ("resolve:\n  merge_stuff: 1\n", "expected bool"),
    ],
)
def test_bad_overrides_are_rejected(tmp_path, text, error):
    with pytest.raises(ValueError, match=error):
        load(tmp_path, text)


def test_good_overrides(tmp_path):
    cfg = load(
        tmp_path,
        "stuff:\n  pred_iou_thresh: 1\n"  # an int where a float is expected is fine
        "things:\n  coco_to_ontology:\n    motorcycle: car\n    bus: null\n",  # open keys; null removes
    )
    assert cfg["stuff"]["pred_iou_thresh"] == 1
    assert cfg["things"]["coco_to_ontology"]["motorcycle"] == "car"
    assert cfg["things"]["coco_to_ontology"]["bus"] is None
    assert cfg["things"]["coco_to_ontology"]["person"] == "person"  # defaults kept
