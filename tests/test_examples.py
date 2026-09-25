"""The committed example scenes: resolve the human-corrected CVAT exports and
check the result against the output of the first (July 2026) version of the
pipeline, which is what the portfolio site shows."""

from pathlib import Path

import pytest
import yaml

from panoptic_prelabel.coco_io import read_coco
from panoptic_prelabel.compare import evaluate
from panoptic_prelabel.panoptic import read_panoptic
from panoptic_prelabel.resolve import resolve

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCENES = sorted(p.name for p in EXAMPLES.iterdir() if (p / "cvat_corrected.zip").exists())


def test_there_are_examples():
    assert len(SCENES) >= 3


@pytest.mark.parametrize("name", SCENES)
def test_example_matches_reference(cfg, name):
    scene = EXAMPLES / name
    groups = []
    if (scene / "scene.yaml").exists():
        groups = yaml.safe_load((scene / "scene.yaml").read_text())["groups"]
    pan = resolve(read_coco(scene / "cvat_corrected.zip"), cfg, groups=groups)
    assert (pan.id_map > 0).all(), "no void pixel left"

    summary = evaluate([(read_panoptic(scene / "reference_2026-07" / "panoptic.json"), pan)]).summary()
    assert summary["pixel_acc"] > 0.995
    assert summary["RQ"] > 0.95
