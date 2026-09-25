"""Colors and preview images (JPEGs for humans, not data)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import Config, hex_to_rgb

if TYPE_CHECKING:
    from .coco_io import Annotation
    from .panoptic import Panoptic
    from .stuff import Regions

DEFAULT_SHADES = (1.0, 1.35, 0.7, 1.6, 0.5, 1.85)


def shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    r, g, b = (int(min(255, c * factor)) for c in rgb)
    return r, g, b


def segment_colors(pan: Panoptic, cfg: Config) -> dict[int, tuple[int, int, int]]:
    """Base class color for the 1st instance of a class, shades for the next ones.

    Instances are numbered in segment-id order, which is paint order
    (front-most first), so colors are stable for a given resolve output.
    """
    shades = cfg.data.get("web", {}).get("instance_shades", DEFAULT_SHADES)
    seen: dict[str, int] = {}
    colors = {}
    for s in sorted(pan.segments, key=lambda s: s.id):
        k = seen.get(s.category, 0)
        seen[s.category] = k + 1
        colors[s.id] = shade(cfg.by_name[s.category].rgb, shades[k % len(shades)])
    return colors


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSansMono-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_jpeg(rgb: np.ndarray, path: str | Path, quality: int = 88) -> None:
    Image.fromarray(rgb).save(path, quality=quality, optimize=True)


def boundaries(id_map: np.ndarray) -> np.ndarray:
    """Pixels whose right or bottom neighbour belongs to another segment."""
    edge = np.zeros(id_map.shape, dtype=bool)
    edge[:, :-1] |= id_map[:, :-1] != id_map[:, 1:]
    edge[:-1, :] |= id_map[:-1, :] != id_map[1:, :]
    return edge


def panoptic_overlay(rgb: np.ndarray, pan: Panoptic, cfg: Config, alpha: float = 0.6) -> np.ndarray:
    colors = segment_colors(pan, cfg)
    lut = np.zeros((pan.id_map.max() + 1, 3), dtype=np.float32)
    for sid, c in colors.items():
        lut[sid] = c
    color = lut[pan.id_map]
    valid = (pan.id_map > 0)[..., None]
    out = np.where(valid, rgb * (1 - alpha) + color * alpha, rgb)
    out[boundaries(pan.id_map)] = 255
    return out.astype(np.uint8)


def draw_regions(rgb: np.ndarray, regions: Regions, things: list[Annotation], cfg: Config) -> np.ndarray:
    """Numbered region overlay: what the labeller looks at to fill regions.yaml."""
    rng = np.random.default_rng(0)
    lm = regions.label_map
    palette = rng.integers(40, 256, size=(lm.max() + 1, 3)).astype(np.float32)
    out = rgb.astype(np.float32)
    has = (lm > 0)[..., None]
    out = np.where(has, out * 0.45 + palette[lm] * 0.55, out * 0.6)
    for t in things:  # detector instances in their class color, unnumbered
        c = np.array(cfg.by_name[t.category].rgb, np.float32)
        out[t.mask] = out[t.mask] * 0.4 + c * 0.6
    out[boundaries(lm)] = 255
    img = Image.fromarray(out.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    font = load_font(max(14, max(rgb.shape[:2]) // 60))
    for rid, (x, y) in regions.points.items():
        text = str(rid)
        left, top, right, bottom = draw.textbbox((x, y), text, font=font, anchor="mm")
        draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(255, 255, 255), anchor="mm")
    return np.asarray(img)


def draw_boxes(rgb: np.ndarray, boxes: list[dict], width: int, height: int) -> np.ndarray:
    """Boxes given in percent of the image (the web export format)."""
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    h, w = rgb.shape[:2]
    font = load_font(max(11, max(h, w) // 70))
    lw = max(2, max(h, w) // 400)
    for b in boxes:
        x0, y0 = b["x"] / 100 * w, b["y"] / 100 * h
        x1, y1 = x0 + b["w"] / 100 * w, y0 + b["h"] / 100 * h
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255), width=lw)
        left, top, right, bottom = draw.textbbox((0, 0), b["cls"], font=font)
        th = bottom - top
        ty = y0 - th - 6 if y0 - th - 6 > 0 else y0 + lw
        draw.rectangle((x0, ty, x0 + (right - left) + 8, ty + th + 6), fill=hex_to_rgb(b["color"]))
        draw.text((x0 + 4, ty + 3 - top), b["cls"], font=font, fill=(11, 13, 18))
    return np.asarray(img)


def resize_long(rgb: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
