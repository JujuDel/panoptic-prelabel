"""Mask R-CNN pre/post-processing, without onnxruntime or the model file."""

import sys
import types

import numpy as np
import pytest

from panoptic_prelabel.things import MAX_H, MAX_W, ThingDetector, paste_mask, preprocess


@pytest.mark.parametrize(
    ("h", "w", "expect_ratio"),
    [
        (960, 1707, 800 / 960),  # landscape: short side 800 (1422 wide, x clipped at 1280!) -> capped
        (1707, 960, 960 / 1707),  # portrait: capped by the model's y <= 959 limit
        (600, 800, 800 / 600),  # small: upscaled to short side 800
    ],
)
def test_preprocess_respects_the_exported_graph_limits(h, w, expect_ratio):
    tensor, ratio = preprocess(np.zeros((h, w, 3), np.uint8))
    expected = min(expect_ratio, MAX_W / w, MAX_H / h)
    assert ratio == pytest.approx(expected)
    assert int(h * ratio) <= MAX_H and int(w * ratio) <= MAX_W
    assert tensor.shape[0] == 3 and tensor.shape[1] % 32 == 0 and tensor.shape[2] % 32 == 0
    assert tensor.dtype == np.float32


def test_preprocess_is_bgr_minus_mean():
    rgb = np.zeros((800, 800, 3), np.uint8)
    rgb[..., 0] = 200  # red
    tensor, _ = preprocess(rgb)
    # channel 2 of the BGR tensor is red
    assert tensor[2, 10, 10] == pytest.approx(200 - 122.7717, abs=1e-3)
    assert tensor[0, 10, 10] == pytest.approx(-102.9801, abs=1e-3)


def test_paste_mask_fills_its_box():
    full = paste_mask(np.ones((28, 28), np.float32), np.array([10.0, 20.0, 49.0, 39.0]), 100, 120, 0.5)
    ys, xs = np.nonzero(full)
    # a full soft mask covers the box (the 1-px padding only blurs the border)
    assert abs(xs.min() - 10) <= 1 and abs(xs.max() - 49) <= 1
    assert abs(ys.min() - 20) <= 1 and abs(ys.max() - 39) <= 1


def test_paste_mask_clips_at_the_image_border():
    full = paste_mask(np.ones((28, 28), np.float32), np.array([-15.0, 90.0, 20.0, 130.0]), 100, 50, 0.5)
    assert full.shape == (100, 50)
    assert full[99, 0] and not full[50, 40]


def test_detector_maps_classes_and_rescales_boxes(monkeypatch, cfg):
    calls = {}

    class FakeSession:
        def __init__(self, *a, **k):
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(name="image")]

        def run(self, _, feeds):
            t = feeds["image"]
            calls["shape"] = t.shape
            boxes = np.array([[80, 80, 400, 400], [10, 10, 50, 50], [0, 0, 5, 5]], np.float32)
            labels = np.array([1, 27, 62])  # person, handbag -> bag, chair -> dropped
            scores = np.array([0.99, 0.9, 0.99], np.float32)
            masks = np.ones((3, 1, 28, 28), np.float32)
            return boxes, labels, scores, masks

    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = FakeSession
    fake.SessionOptions = lambda: types.SimpleNamespace(log_severity_level=0)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    det = ThingDetector(cfg, model_path="unused.onnx")
    out = det(np.zeros((480, 640, 3), np.uint8))  # ratio 800/480 -> boxes divided by 5/3
    assert [a.category for a in out] == ["person", "bag"]
    assert [a.attributes["coco_class"] for a in out] == ["person", "handbag"]
    _, xs = np.nonzero(out[0].mask)
    assert abs(xs.min() - 48) <= 2 and abs(xs.max() - 240) <= 2  # 80/(5/3), 400/(5/3)
