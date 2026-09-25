"""The real models, on one example photo. Opt-in: needs `.[models]` and the weights.

    panoptic-prelabel download-weights
    RUN_MODELS=1 pytest tests/test_models.py      # about 3 minutes on a 2-vCPU CPU

Checks that `prelabel` still produces the recorded outputs in
`examples/<scene>/prelabel/` (which the evaluation replays), within tolerance:
other CPUs or library versions may change a few pixels, not the result.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from panoptic_prelabel.masks import rle_decode

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "examples" / "beach"
WEIGHTS = ROOT / "weights"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODELS") != "1" or not (WEIGHTS / "mobile_sam.pt").exists(),
    reason="set RUN_MODELS=1 and run `panoptic-prelabel download-weights` first",
)


def test_prelabel_reproduces_the_recorded_outputs(tmp_path, cfg):
    from panoptic_prelabel.pipeline import prelabel

    cfg.data["things"]["model"] = str(WEIGHTS / "MaskRCNN-12.onnx")
    cfg.data["stuff"]["checkpoint"] = str(WEIGHTS / "mobile_sam.pt")
    work = prelabel(SCENE / "image.jpg", tmp_path / "work", cfg)

    new = json.loads((work / "things.json").read_text())["instances"]
    old = json.loads((SCENE / "prelabel" / "things.json").read_text())["instances"]
    assert [i["category"] for i in new] == [i["category"] for i in old]
    for a, b in zip(new, old, strict=True):
        ma, mb = rle_decode(a["rle"]), rle_decode(b["rle"])
        assert (ma & mb).sum() / (ma | mb).sum() > 0.95
        assert abs(a["score"] - b["score"]) < 0.02

    ra = np.asarray(Image.open(work / "regions.png"))
    rb = np.asarray(Image.open(SCENE / "prelabel" / "regions.png"))
    assert (ra == rb).mean() > 0.95
