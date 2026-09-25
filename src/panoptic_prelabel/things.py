"""Thing pre-labels: Mask R-CNN R-50-FPN from the ONNX model zoo, on CPU.

Pre/post-processing follows the model card
(onnx/models, validated/vision/object_detection_segmentation/mask-rcnn):
shorter side resized to 800, BGR, mean subtraction, zero-padded to a multiple
of 32; outputs are boxes in the resized frame, labels, scores and 28x28 masks.

One quirk the model card does not mention: this export clips every box to
x <= 1279 and y <= 959, constants frozen in the graph when it was traced. A
portrait photo resized to a short side of 800 is ~1420 px tall, so everything
below y = 960 was silently cut off (a standing person lost their legs and most
of their score). The input is therefore also capped to fit in 1280 x 960.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .coco_io import Annotation
from .config import Config

COCO_CLASSES = (
    "__background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)  # fmt: skip

MEAN_BGR = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
MAX_W, MAX_H = 1280, 960  # box clipping limits baked into MaskRCNN-12.onnx


def preprocess(rgb: np.ndarray, short_side: int = 800) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    ratio = min(short_side / min(h, w), MAX_W / w, MAX_H / h)
    nh, nw = int(ratio * h), int(ratio * w)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    bgr = resized[:, :, ::-1].astype(np.float32) - MEAN_BGR
    ph, pw = math.ceil(nh / 32) * 32, math.ceil(nw / 32) * 32
    padded = np.zeros((3, ph, pw), dtype=np.float32)
    padded[:, :nh, :nw] = bgr.transpose(2, 0, 1)
    return padded, ratio


def paste_mask(mask28: np.ndarray, box: np.ndarray, height: int, width: int, threshold: float) -> np.ndarray:
    """Paste a 28x28 soft mask into the image (maskrcnn-benchmark style, 1-px padding)."""
    m = mask28.shape[0]
    scale = (m + 2.0) / m
    padded = np.zeros((m + 2, m + 2), dtype=np.float32)
    padded[1:-1, 1:-1] = mask28
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / 2 * scale, (box[3] - box[1]) / 2 * scale
    x0, y0, x1, y1 = (round(v) for v in (cx - hw, cy - hh, cx + hw, cy + hh))
    bw, bh = max(x1 - x0 + 1, 1), max(y1 - y0 + 1, 1)
    soft = cv2.resize(padded, (bw, bh), interpolation=cv2.INTER_LINEAR) > threshold
    out = np.zeros((height, width), dtype=bool)
    ix0, iy0 = max(x0, 0), max(y0, 0)
    ix1, iy1 = min(x1 + 1, width), min(y1 + 1, height)
    if ix1 > ix0 and iy1 > iy0:
        out[iy0:iy1, ix0:ix1] = soft[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0]
    return out


class ThingDetector:
    def __init__(self, cfg: Config, model_path: str | None = None):
        import onnxruntime as ort  # heavy import kept local: resolve/tests do not need it

        tcfg = cfg["things"]
        self.cfg = tcfg
        # validated by Config; a null value removes a default mapping
        self.mapping = {k: v for k, v in tcfg["coco_to_ontology"].items() if v is not None}
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_path or tcfg["model"], sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, rgb: np.ndarray) -> list[Annotation]:
        h, w = rgb.shape[:2]
        tensor, ratio = preprocess(rgb)
        boxes, labels, scores, masks = self.session.run(None, {self.input_name: tensor})
        boxes = boxes / ratio
        out = []
        for box, label, score, mask in zip(boxes, labels, scores, masks, strict=True):
            if score < self.cfg["score_threshold"]:
                continue
            coco_name = COCO_CLASSES[int(label)]
            target = self.mapping.get(coco_name)
            if target is None:
                continue
            full = paste_mask(mask[0], box, h, w, self.cfg["mask_threshold"])
            if full.sum() < self.cfg["min_area"]:
                continue
            out.append(Annotation(target, full, score=float(score), attributes={"coco_class": coco_name}))
        return out
