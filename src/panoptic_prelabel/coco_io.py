"""Reading and writing annotations in the COCO 1.0 layout CVAT imports/exports.

A CVAT "COCO 1.0" archive is a zip holding ``annotations/instances_default.json``
(one image per archive in this project). Polygons come back as polygon lists;
shapes drawn with the brush tool come back as uncompressed RLE with
``iscrowd: 1``. Both are handled.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .masks import bbox_xywh, largest_hole, mask_to_polygons, rle_encode, segmentation_decode

CVAT_JSON = "annotations/instances_default.json"


@dataclass
class Annotation:
    """One shape: a class name and a full-resolution boolean mask."""

    category: str
    mask: np.ndarray
    source_id: int | None = None  # id in the file it was read from
    score: float | None = None  # detector confidence, when there is one
    attributes: dict = field(default_factory=dict)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class ImageAnnotations:
    file_name: str
    width: int
    height: int
    annotations: list[Annotation]


def load_coco_json(data: dict) -> ImageAnnotations:
    if len(data["images"]) != 1:
        raise ValueError(f"expected exactly one image, got {len(data['images'])}")
    img = data["images"][0]
    h, w = int(img["height"]), int(img["width"])
    names = {c["id"]: c["name"] for c in data["categories"]}
    anns = []
    for a in data["annotations"]:
        mask = segmentation_decode(a["segmentation"], h, w)
        anns.append(
            Annotation(
                category=names[a["category_id"]],
                mask=mask,
                source_id=a.get("id"),
                score=a.get("score"),
                attributes=dict(a.get("attributes") or {}),
            )
        )
    return ImageAnnotations(img["file_name"], w, h, anns)


def read_coco(path: str | Path) -> ImageAnnotations:
    """Read a CVAT COCO 1.0 zip, or a bare instances JSON file."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = CVAT_JSON if CVAT_JSON in zf.namelist() else _find_json(zf)
            data = json.loads(zf.read(name))
    else:
        data = json.loads(path.read_text("utf-8"))
    return load_coco_json(data)


def _find_json(zf: zipfile.ZipFile) -> str:
    candidates = [n for n in zf.namelist() if n.endswith(".json")]
    if len(candidates) != 1:
        raise ValueError(f"cannot find the annotation JSON in the archive: {candidates}")
    return candidates[0]


def to_coco_json(image: ImageAnnotations, cfg: Config) -> dict:
    """Serialise annotations; masks with large holes become RLE, others polygons."""
    rcfg = cfg["refine"]
    out_anns = []
    for i, a in enumerate(image.annotations, start=1):
        cat = cfg.by_name[a.category]
        seg: dict | list[list[float]]
        if largest_hole(a.mask) >= rcfg["rle_if_holes_above"]:
            seg, iscrowd = rle_encode(a.mask), 1
        else:
            seg, iscrowd = mask_to_polygons(a.mask, rcfg["polygon_epsilon"]), 0
            if not seg:
                continue
        entry: dict[str, Any] = {
            "id": i,
            "image_id": 1,
            "category_id": cat.id,
            "segmentation": seg,
            "area": a.area,
            "bbox": bbox_xywh(a.mask),
            "iscrowd": iscrowd,
        }
        if a.score is not None:
            entry["score"] = round(float(a.score), 4)
        out_anns.append(entry)
    return {
        "info": {"description": "panoptic-prelabel pre-annotation"},
        "licenses": [],
        "images": [{"id": 1, "file_name": image.file_name, "width": image.width, "height": image.height}],
        "categories": [
            {"id": c.id, "name": c.name, "supercategory": "thing" if c.isthing else "stuff"} for c in cfg.categories
        ],
        "annotations": out_anns,
    }


def write_cvat_zip(image: ImageAnnotations, cfg: Config, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_coco_json(image, cfg))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(CVAT_JSON, payload)
    return path
