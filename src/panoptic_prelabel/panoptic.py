"""The panoptic result type and its COCO-panoptic serialisation (PNG + JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config


@dataclass
class Segment:
    id: int
    category: str
    isthing: bool
    area: int
    bbox: list[int]  # [x, y, w, h], full-resolution pixels
    source_ids: list[int]


@dataclass
class Panoptic:
    file_name: str
    width: int
    height: int
    id_map: np.ndarray  # (H, W) int32, 0 = void
    segments: list[Segment]

    def segment(self, seg_id: int) -> Segment:
        return next(s for s in self.segments if s.id == seg_id)


def id2rgb(id_map: np.ndarray) -> np.ndarray:
    """COCO panoptic PNG encoding: id = R + 256 * G + 256**2 * B."""
    ids = id_map.astype(np.uint32)
    return np.stack([ids & 255, (ids >> 8) & 255, (ids >> 16) & 255], axis=-1).astype(np.uint8)


def rgb2id(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int32)
    return rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]


def panoptic_json(pan: Panoptic, cfg: Config, png_name: str) -> dict:
    return {
        "info": {"description": "panoptic-prelabel output (COCO panoptic format)"},
        "images": [{"id": 1, "file_name": pan.file_name, "width": pan.width, "height": pan.height}],
        "annotations": [
            {
                "image_id": 1,
                "file_name": png_name,
                "segments_info": [
                    {
                        "id": s.id,
                        "category_id": cfg.by_name[s.category].id,
                        "iscrowd": 0,
                        "area": s.area,
                        "bbox": s.bbox,
                    }
                    for s in pan.segments
                ],
            }
        ],
        "categories": [
            {"id": c.id, "name": c.name, "isthing": int(c.isthing), "color": list(c.rgb)} for c in cfg.categories
        ],
    }


def write_panoptic(pan: Panoptic, cfg: Config, out_dir: str | Path, stem: str = "panoptic") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    js = out_dir / f"{stem}.json"
    Image.fromarray(id2rgb(pan.id_map)).save(png, optimize=True)
    js.write_text(json.dumps(panoptic_json(pan, cfg, png.name), indent=1), "utf-8")
    return png, js


def read_panoptic(json_path: str | Path) -> Panoptic:
    """Read a COCO panoptic JSON (single image) and its PNG next to it."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text("utf-8"))
    img = data["images"][0]
    ann = data["annotations"][0]
    cats = {c["id"]: c for c in data["categories"]}
    id_map = rgb2id(np.asarray(Image.open(json_path.parent / ann["file_name"]).convert("RGB")))
    segments = [
        Segment(
            id=s["id"],
            category=cats[s["category_id"]]["name"],
            isthing=bool(cats[s["category_id"]]["isthing"]),
            area=int(s["area"]),
            bbox=[int(v) for v in s["bbox"]],
            source_ids=[],
        )
        for s in ann["segments_info"]
    ]
    return Panoptic(img["file_name"], int(img["width"]), int(img["height"]), id_map, segments)
