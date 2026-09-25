"""Region proposals: MobileSAM automatic mask generation, on CPU.

SAM-style models segment *everything* but name *nothing*. This module turns
the overlapping SAM masks into a flat partition of the image into numbered,
class-agnostic regions (``regions.png``), which a labeller then names in
``regions.yaml``: a stuff class (sky, sea, ...), a thing class the detector
does not know (bollard, buoy, ...), or ``ignore``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config


@dataclass
class Regions:
    label_map: np.ndarray  # (H, W) int32, 0 = no region
    points: dict[int, tuple[int, int]]  # region id -> an interior (x, y) pixel
    areas: dict[int, int]


def interior_point(mask: np.ndarray) -> tuple[int, int]:
    """The pixel deepest inside the mask (max of the distance transform)."""
    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y)


def partition(masks: list[np.ndarray], things: np.ndarray, cfg: Config) -> Regions:
    """Flatten overlapping SAM masks into non-overlapping regions.

    Larger masks are painted first and smaller ones on top, so a small region
    nested in a big one (a rock in the sea) survives. Masks lying mostly on a
    detected thing are dropped: the detector owns those pixels.
    """
    scfg = cfg["stuff"]
    h, w = things.shape
    canvas = np.zeros((h, w), dtype=np.int32)
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    k = 0
    for i in order:
        m = masks[i]
        area = int(m.sum())
        if area == 0 or (m & things).sum() / area > scfg["max_thing_overlap"]:
            continue
        k += 1
        canvas[m & ~things] = k

    # renumber by decreasing visible area, dropping regions that got too small
    ids, counts = np.unique(canvas[canvas > 0], return_counts=True)
    keep = [(int(c), int(i)) for i, c in zip(ids, counts, strict=True) if c >= scfg["min_region_area"]]
    keep.sort(reverse=True)
    label_map = np.zeros_like(canvas)
    points, areas = {}, {}
    for new_id, (area, old_id) in enumerate(keep, start=1):
        m = canvas == old_id
        label_map[m] = new_id
        points[new_id] = interior_point(m)
        areas[new_id] = area
    return Regions(label_map, points, areas)


class RegionProposer:
    def __init__(self, cfg: Config, checkpoint: str | None = None):
        import warnings

        import torch  # heavy imports kept local: resolve/tests do not need them

        with warnings.catch_warnings():  # timm deprecation / registry notices from MobileSAM's imports
            warnings.simplefilter("ignore")
            from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

        scfg = cfg["stuff"]
        self.cfg = cfg
        torch.manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sam = sam_model_registry[scfg["model_type"]](checkpoint=checkpoint or scfg["checkpoint"])
        self.device = scfg["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("stuff.device is 'cuda' but torch sees no CUDA device")
        sam.to(self.device).eval()
        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=scfg["points_per_side"],
            pred_iou_thresh=scfg["pred_iou_thresh"],
            stability_score_thresh=scfg["stability_score_thresh"],
            points_per_batch=scfg["points_per_batch"],
        )

    def __call__(self, rgb: np.ndarray, things: np.ndarray) -> Regions:
        import torch

        h, w = rgb.shape[:2]
        scale = min(1.0, self.cfg["stuff"]["max_side"] / max(h, w))
        small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            proposals = self.generator.generate(small)
        masks = [
            cv2.resize(p["segmentation"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            for p in proposals
        ]
        return partition(masks, things, self.cfg)
