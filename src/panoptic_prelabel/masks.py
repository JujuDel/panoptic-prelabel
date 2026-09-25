"""Binary-mask helpers: COCO polygons and RLE, clean-up, bounding boxes.

Everything here is plain NumPy/OpenCV so the resolve stage and the tests do
not need pycocotools or any deep-learning framework.
"""

from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------
# COCO encodings
# --------------------------------------------------------------------------


def rle_encode(mask: np.ndarray) -> dict:
    """Uncompressed COCO RLE (column-major run lengths, starting with zeros).

    This is the form CVAT writes for "mask" shapes in a COCO 1.0 export.
    """
    h, w = mask.shape
    flat = np.asarray(mask, dtype=bool).T.reshape(-1)
    # indices where the value changes, plus both ends
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts = [0, *counts]
    return {"counts": counts, "size": [h, w]}


def rle_decode(rle: dict) -> np.ndarray:
    """Decode an uncompressed COCO RLE into a boolean (H, W) mask."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        raise ValueError(
            "compressed RLE strings are not supported; export from CVAT as COCO 1.0 "
            "(which writes uncompressed counts) or decode with pycocotools first"
        )
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, counts)
    if flat.size != h * w:
        raise ValueError(f"RLE covers {flat.size} pixels, expected {h * w}")
    return flat.reshape(w, h).T


def polygons_decode(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    """Rasterise COCO polygons ([x0, y0, x1, y1, ...] lists) into a mask."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            continue
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas.astype(bool)


def segmentation_decode(segmentation, height: int, width: int) -> np.ndarray:
    """Decode either COCO segmentation form (polygon list or RLE dict)."""
    if isinstance(segmentation, dict):
        return rle_decode(segmentation)
    return polygons_decode(segmentation, height, width)


def mask_to_polygons(mask: np.ndarray, epsilon: float = 1.5) -> list[list[float]]:
    """Outer contours of a mask as COCO polygons (holes are not representable)."""
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for c in contours:
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, True)
        if len(c) < 3:
            continue
        polygons.append(c.reshape(-1).astype(float).tolist())
    return polygons


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def bbox_xywh(mask: np.ndarray) -> list[int]:
    """Tight [x, y, w, h] box of a mask, COCO convention (inclusive pixels)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def connected_components(mask: np.ndarray, min_area: int = 0) -> list[np.ndarray]:
    """Split a mask into its 8-connected components, largest first."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype=np.uint8), connectivity=8)
    comps = [
        (int(stats[i, cv2.CC_STAT_AREA]), labels == i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    comps.sort(key=lambda t: -t[0])
    return [m for _, m in comps]


def largest_hole(mask: np.ndarray) -> int:
    """Area of the biggest hole (background region not touching the border)."""
    inv = ~np.asarray(mask, dtype=bool)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=4)
    h, w = mask.shape
    best = 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches:
            best = max(best, int(area))
    return best


def clean_mask(mask: np.ndarray, fill_holes_below: int, drop_islands_below: int) -> np.ndarray:
    """Fill small holes and drop small disconnected fragments."""
    m = np.asarray(mask, dtype=bool).copy()
    if drop_islands_below > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < drop_islands_below:
                m[labels == i] = False
    if fill_holes_below > 0:
        inv = (~m).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        h, w = m.shape
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            touches = x == 0 or y == 0 or x + bw == w or y + bh == h
            if not touches and area < fill_holes_below:
                m[labels == i] = True
    return m
