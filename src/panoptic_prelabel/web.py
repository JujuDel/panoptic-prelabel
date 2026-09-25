"""export-web: the files the portfolio carousel consumes.

For a scene called NAME it writes, next to each other:

    NAME_raw.jpg     the photo, with its metadata stripped (losslessly)
    NAME_pan.png     PANOPTIC view: every segment in its color, white boundaries
    NAME_inst.png    INSTANCES view: things in their instance color, stuff dimmed
    NAME_boxes.json  instance boxes in percent of the image, plus a JS snippet

Both PNGs are transparent overlays with the photo's aspect ratio, scaled so
their long side is ``web.long_side``; the page stretches them over the photo.
Boxes are in percent so they do not depend on the displayed size.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import Config, hex_to_rgb
from .jpeg import copy_without_metadata
from .panoptic import Panoptic
from .viz import boundaries, rgb_hex, segment_colors


def _scaled_ids(pan: Panoptic, long_side: int) -> np.ndarray:
    s = long_side / max(pan.width, pan.height)
    size = (int(pan.width * s), int(pan.height * s))  # truncate, as the site assets do
    return cv2.resize(pan.id_map.astype(np.int32), size, interpolation=cv2.INTER_NEAREST)


def _thick(edge: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return edge
    return cv2.dilate(edge.astype(np.uint8), np.ones((width, width), np.uint8)).astype(bool)


def overlays(pan: Panoptic, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(pan RGBA, inst RGBA) at the export resolution."""
    wcfg = cfg["web"]
    ids = _scaled_ids(pan, wcfg["long_side"])
    n = int(max(ids.max(), max((s.id for s in pan.segments), default=0))) + 1
    colors = segment_colors(pan, cfg)
    is_thing = np.zeros(n, dtype=bool)
    lut = np.zeros((n, 3), dtype=np.uint8)
    for s in pan.segments:
        lut[s.id] = colors[s.id]
        is_thing[s.id] = s.isthing
    edge_rgb = np.array(hex_to_rgb(wcfg["edge_color"]), np.uint8)
    edges = _thick(boundaries(ids), wcfg["edge_width"])
    valid = ids > 0

    pan_rgba = np.zeros((*ids.shape, 4), dtype=np.uint8)
    pan_rgba[..., :3] = lut[ids]
    pan_rgba[..., 3] = np.where(valid, wcfg["pan_alpha"], 0)
    pan_rgba[edges & valid] = [*edge_rgb, wcfg["edge_alpha"]]

    thing_px = is_thing[ids]
    inst = np.zeros_like(pan_rgba)
    inst[..., :3] = np.where(thing_px[..., None], lut[ids], np.array(hex_to_rgb(wcfg["dim_color"]), np.uint8))
    inst[..., 3] = np.where(thing_px, wcfg["inst_alpha"], np.where(valid, wcfg["dim_alpha"], 0))
    # only outline things: stuff/stuff boundaries stay quiet in this view
    thing_edges = _thick(boundaries(np.where(thing_px, ids, 0)), wcfg["edge_width"])
    inst[thing_edges & valid] = [*edge_rgb, wcfg["edge_alpha"]]
    return pan_rgba, inst


def boxes(pan: Panoptic, cfg: Config) -> list[dict]:
    """Instance boxes of the visible thing segments, in percent of the image."""
    wcfg = cfg["web"]
    colors = segment_colors(pan, cfg)
    out = []
    for s in sorted(pan.segments, key=lambda s: s.id):
        if not s.isthing or s.category in wcfg["box_exclude"] or s.area < wcfg["box_min_area"]:
            continue
        x, y, w, h = s.bbox
        out.append(
            {
                "cls": s.category.upper(),
                "color": rgb_hex(colors[s.id]),
                "x": round(100 * x / pan.width, 2),
                "y": round(100 * y / pan.height, 2),
                "w": round(100 * w / pan.width, 2),
                "h": round(100 * h / pan.height, 2),
            }
        )
    return out


NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def scene_snippet(name: str, pan: Panoptic, bxs: list[dict], asset_dir: str = "assets/scenes") -> str:
    """The entry to paste into the site's SCENES array."""
    lines = [
        f'{{ name:"{name.upper()}", w:{pan.width}, h:{pan.height},',
        f'  photo:"{asset_dir}/{name}_raw.jpg", pan:"{asset_dir}/{name}_pan.png", inst:"{asset_dir}/{name}_inst.png",',
    ]
    items = [f'{{cls:"{b["cls"]}",color:"{b["color"]}",x:{b["x"]},y:{b["y"]},w:{b["w"]},h:{b["h"]}}}' for b in bxs]
    lines.append("  boxes:[" + ",\n         ".join(items) + "]}")
    return "\n".join(lines)


def export_web(pan: Panoptic, image_path: str | Path, name: str, out_dir: str | Path, cfg: Config) -> dict:
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"scene name {name!r} must match {NAME_RE.pattern} (it becomes file names and JS)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{name}_raw.jpg"
    with Image.open(image_path) as im:
        rotated = im.getexif().get(0x0112, 1) != 1  # EXIF Orientation
        upright = ImageOps.exif_transpose(im) if rotated else im
        if upright.size != (pan.width, pan.height):
            raise ValueError(f"image is {upright.size}, panoptic map is {(pan.width, pan.height)}")
        if im.format == "JPEG" and not rotated:
            copy_without_metadata(image_path, raw)  # lossless: same pixels, no EXIF/GPS
        else:
            # rotated by its EXIF tag, or not a JPEG: re-encode the upright pixels
            # (a stripped file would lose the Orientation tag and show sideways)
            upright.convert("RGB").save(raw, quality=95, optimize=True, icc_profile=im.info.get("icc_profile"))
    pan_rgba, inst_rgba = overlays(pan, cfg)
    Image.fromarray(pan_rgba, "RGBA").save(out_dir / f"{name}_pan.png", optimize=True)
    Image.fromarray(inst_rgba, "RGBA").save(out_dir / f"{name}_inst.png", optimize=True)
    bxs = boxes(pan, cfg)
    payload = {"name": name.upper(), "w": pan.width, "h": pan.height, "boxes": bxs}
    (out_dir / f"{name}_boxes.json").write_text(json.dumps(payload, indent=1), "utf-8")
    (out_dir / f"{name}_scene.js").write_text(scene_snippet(name, pan, bxs) + "\n", "utf-8")
    return payload
