"""resolve: overlapping annotations -> one panoptic map + instance boxes.

Annotations coming back from CVAT overlap freely (a person polygon drawn on
top of the pavement, a backpack on top of the person). A panoptic map needs
every pixel to belong to exactly one segment, so this stage decides who wins:

1. Things always win over stuff.
2. Between two classes, the one listed first in ``resolve.priority`` wins.
3. Within one class, the smaller annotation wins.
4. Classes in ``resolve.split_classes`` are split into connected components
   (CVAT's COCO export merges same-label shapes into one multi-polygon).
   Explicit ``groups`` do the opposite: they merge several annotations that
   are parts of one object split by an occluder.
5. Stuff annotations of one class are merged into one segment.
6. Pixels no annotation covers go to the nearest segment (``fill: nearest``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .coco_io import Annotation, ImageAnnotations
from .config import Config
from .masks import bbox_xywh, connected_components
from .panoptic import Panoptic, Segment


@dataclass
class _Item:
    category: str
    mask: np.ndarray
    source_ids: list[int]


def _prepare_items(image: ImageAnnotations, cfg: Config, groups: list[list[int]]) -> list[_Item]:
    rcfg = cfg["resolve"]
    for a in image.annotations:
        if a.category not in cfg.by_name:
            raise ValueError(f"annotation {a.source_id} has unknown class {a.category!r}")

    ids = [a.source_id for a in image.annotations if a.source_id is not None]
    if len(ids) != len(set(ids)):
        raise ValueError("annotation ids must be unique within an image")
    by_id: dict[int, Annotation] = {a.source_id: a for a in image.annotations if a.source_id is not None}
    grouped: set[int] = set()
    items: list[_Item] = []

    # explicit groups first: several shapes, one instance
    seen: set[int] = set()
    for group in groups:
        if seen & set(group):
            raise ValueError(f"annotation ids {sorted(seen & set(group))} appear in more than one group")
        seen.update(group)
        members = [by_id[i] for i in group if i in by_id]
        if len(members) != len(group):
            missing = sorted(set(group) - set(by_id))
            raise ValueError(f"group {group} references unknown annotation ids {missing}")
        cats = {m.category for m in members}
        if len(cats) != 1:
            raise ValueError(f"group {group} mixes classes {sorted(cats)}")
        if not cfg.by_name[members[0].category].isthing:
            raise ValueError(
                f"group {group} is made of stuff ({members[0].category}); stuff is merged per class anyway"
            )
        mask = np.logical_or.reduce([m.mask for m in members])
        items.append(_Item(members[0].category, mask, list(group)))
        grouped.update(group)

    stuff_union: dict[str, _Item] = {}
    for a in image.annotations:
        if a.source_id in grouped:
            continue
        sid = [a.source_id] if a.source_id is not None else []
        cat = cfg.by_name[a.category]
        if not cat.isthing and rcfg["merge_stuff"]:
            if a.category in stuff_union:
                prev = stuff_union[a.category]
                prev.mask = prev.mask | a.mask
                prev.source_ids += sid
            else:
                stuff_union[a.category] = _Item(a.category, a.mask.copy(), list(sid))
            continue
        if a.category in rcfg["split_classes"]:
            for comp in connected_components(a.mask, rcfg["split_min_area"]):
                items.append(_Item(a.category, comp, list(sid)))
            continue
        items.append(_Item(a.category, a.mask, list(sid)))
    items.extend(stuff_union.values())
    return items


def _raster_key(mask: np.ndarray) -> bytes:
    """Orders masks by their pixels in raster order (top-most, then left-most set pixel first)."""
    return (~np.packbits(mask.reshape(-1))).tobytes()


def paint_order(items: list[_Item], cfg: Config) -> list[_Item]:
    """Front-to-back order: things, then class priority, then smaller first.

    Ties (same class, same area) are broken by the masks' pixels in raster
    order: at the first pixel where two masks differ, the one covering it
    comes first. The result never depends on the order of the annotations
    in the file.
    """

    def key(it: _Item):
        cat = cfg.by_name[it.category]
        # the packed mask itself is the last key: two different masks never tie
        return (0 if cat.isthing else 1, cfg.priority_rank(it.category), int(it.mask.sum()), _raster_key(it.mask))

    return sorted(items, key=key)


def resolve(image: ImageAnnotations, cfg: Config, groups: list[list[int]] | None = None) -> Panoptic:
    h, w = image.height, image.width
    items = paint_order(_prepare_items(image, cfg, groups or []), cfg)

    id_map = np.zeros((h, w), dtype=np.int32)
    kept: list[_Item] = []
    for it in items:
        free = it.mask & (id_map == 0)
        if not free.any():
            continue  # fully hidden behind higher-priority segments
        kept.append(it)
        id_map[free] = len(kept)

    if cfg["resolve"]["fill"] == "nearest" and kept and (id_map == 0).any():
        void = id_map == 0
        _, (iy, ix) = ndimage.distance_transform_edt(void, return_indices=True)
        id_map = id_map[iy, ix]

    segments = []
    for i, it in enumerate(kept, start=1):
        m = id_map == i
        segments.append(
            Segment(
                id=i,
                category=it.category,
                isthing=cfg.by_name[it.category].isthing,
                area=int(m.sum()),
                bbox=bbox_xywh(m),
                source_ids=it.source_ids,
            )
        )
    return Panoptic(image.file_name, w, h, id_map, segments)
