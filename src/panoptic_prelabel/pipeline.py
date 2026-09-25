"""Stage orchestration and the on-disk layout of a work directory.

<work>/
  things.json      prelabel  Mask R-CNN instances (RLE masks, scores)
  regions.png      prelabel  MobileSAM regions, 16-bit label map (0 = none)
  regions.json     prelabel  per-region interior point and area
  regions.jpg      prelabel  numbered overlay, what the labeller looks at
  regions.yaml     labeller  the class of each region (template written by prelabel)
  upright.jpg      prelabel  only for photos with an EXIF rotation: the image to upload to CVAT
  coco_for_cvat.zip   refine  pre-annotation to import into CVAT (COCO 1.0)
  refine_preview.jpg  refine  what the pre-annotation resolves to
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps
from scipy import ndimage

from .coco_io import Annotation, ImageAnnotations, write_cvat_zip
from .config import Config
from .masks import clean_mask, connected_components, rle_decode, rle_encode
from .stuff import Regions
from .viz import draw_regions, panoptic_overlay, save_jpeg

THINGS = "things.json"
REGIONS_PNG = "regions.png"
REGIONS_JSON = "regions.json"
REGIONS_JPG = "regions.jpg"
LABELS = "regions.yaml"
CVAT_ZIP = "coco_for_cvat.zip"
UPRIGHT = "upright.jpg"
IGNORE = "ignore"


class UnlabeledRegions(RuntimeError):
    pass


EXIF_ORIENTATION = 0x0112


def exif_orientation(path: str | Path) -> int:
    """EXIF Orientation tag (1 = upright, 6 = rotated 90° clockwise, ...)."""
    with Image.open(path) as im:
        return int(im.getexif().get(EXIF_ORIENTATION, 1))


def load_rgb(path: str | Path) -> np.ndarray:
    """The image as displayed: EXIF orientation applied, RGB, uint8 (H, W, 3).

    Phones store portrait photos as landscape pixels plus an Orientation tag;
    every stage works on the upright image so masks, CVAT and the website agree.
    """
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


# --------------------------------------------------------------------------
# prelabel
# --------------------------------------------------------------------------


def prelabel(image_path: str | Path, work: str | Path, cfg: Config, detector=None, proposer=None) -> Path:
    """Run both models and write things.json, regions.* and a labels template."""
    from .stuff import RegionProposer
    from .things import ThingDetector

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(image_path)
    h, w = rgb.shape[:2]
    file_name = Path(image_path).name
    if exif_orientation(image_path) != 1:
        # CVAT must show the same upright pixels the masks were computed on, and the
        # COCO file must name that image or CVAT will not attach the annotations to it
        with Image.open(image_path) as im:
            Image.fromarray(rgb).save(work / UPRIGHT, quality=95, icc_profile=im.info.get("icc_profile"))
        file_name = UPRIGHT
        print(f"note: {image_path} has an EXIF rotation; upload {work / UPRIGHT} to CVAT, not the original")

    detector = detector or ThingDetector(cfg)
    things = detector(rgb)
    thing_union = np.logical_or.reduce([t.mask for t in things]) if things else np.zeros((h, w), bool)

    proposer = proposer or RegionProposer(cfg)
    regions = proposer(rgb, thing_union)

    (work / THINGS).write_text(
        json.dumps(
            {
                "image": {"file_name": file_name, "width": w, "height": h},
                "instances": [
                    {
                        "category": t.category,
                        "score": round(float(t.score), 4),
                        "coco_class": t.attributes.get("coco_class"),
                        "rle": rle_encode(t.mask),
                    }
                    for t in things
                ],
            }
        ),
        "utf-8",
    )
    if regions.label_map.max() > np.iinfo(np.uint16).max:
        raise ValueError("more than 65535 regions: raise stuff.min_region_area")
    Image.fromarray(regions.label_map.astype(np.uint16)).save(work / REGIONS_PNG)
    (work / REGIONS_JSON).write_text(
        json.dumps({str(k): {"point": list(regions.points[k]), "area": regions.areas[k]} for k in regions.points}),
        "utf-8",
    )
    save_jpeg(draw_regions(rgb, regions, things, cfg), work / REGIONS_JPG)
    if not (work / LABELS).exists():
        write_labels_template(regions, work / LABELS)
    return work


def write_labels_template(regions: Regions, path: Path) -> None:
    lines = [
        "# Class of each numbered region in regions.jpg.",
        "# label: a class name from the config (stuff such as sky, or a thing the",
        "# detector does not know such as bollard), or `ignore`: no annotation is made",
        "# for the region and it stays a hole in the pre-annotation, to draw in CVAT.",
        "# Regions are matched by `point` (x, y), so this file survives a re-run.",
        "regions:",
    ]
    for rid in sorted(regions.points):
        x, y = regions.points[rid]
        lines.append(f"  - {{id: {rid}, point: [{x}, {y}], label: null}}")
    path.write_text("\n".join(lines) + "\n", "utf-8")


def read_regions(work: Path) -> Regions:
    label_map = np.asarray(Image.open(work / REGIONS_PNG)).astype(np.int32)
    meta = json.loads((work / REGIONS_JSON).read_text("utf-8"))
    return Regions(
        label_map,
        {int(k): tuple(v["point"]) for k, v in meta.items()},
        {int(k): int(v["area"]) for k, v in meta.items()},
    )


def assign_labels(regions: Regions, labels_path: Path, cfg: Config, allow_unlabeled: bool) -> dict[int, str]:
    """Map region ids to class names using the points in the labels file."""
    data = yaml.safe_load(labels_path.read_text("utf-8")) or {}
    assigned: dict[int, str] = {}
    h, w = regions.label_map.shape
    for entry in data.get("regions", []):
        label = entry.get("label")
        if label is None:
            continue
        if label != IGNORE and label not in cfg.by_name:
            raise ValueError(f"{labels_path}: unknown label {label!r} (region {entry.get('id')})")
        x, y = (int(v) for v in entry["point"])
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"{labels_path}: point {x, y} is outside the image")
        rid = int(regions.label_map[y, x])
        if rid == 0:
            continue  # the point no longer falls in a region (models or thresholds changed)
        if assigned.get(rid, label) != label:
            raise ValueError(f"{labels_path}: region {rid} labelled both {assigned[rid]!r} and {label!r}")
        assigned[rid] = label
    missing = sorted(set(regions.points) - set(assigned))
    if missing and not allow_unlabeled:
        raise UnlabeledRegions(
            f"{len(missing)} region(s) have no label in {labels_path}: {missing}. "
            f"Open {labels_path.parent / REGIONS_JPG}, fill in the labels, and run refine again "
            "(or pass --allow-unlabeled to drop them)."
        )
    return assigned


# --------------------------------------------------------------------------
# refine
# --------------------------------------------------------------------------


def refine(
    work: str | Path,
    cfg: Config,
    labels: str | Path | None = None,
    allow_unlabeled: bool = False,
    image_path: str | Path | None = None,
) -> Path:
    """Detector instances + labelled regions -> clean COCO pre-annotation for CVAT."""
    work = Path(work)
    rcfg = cfg["refine"]
    meta = json.loads((work / THINGS).read_text("utf-8"))
    info = meta["image"]
    h, w = info["height"], info["width"]
    regions = read_regions(work)
    assigned = assign_labels(regions, Path(labels) if labels else work / LABELS, cfg, allow_unlabeled)

    anns: list[Annotation] = []
    for inst in meta["instances"]:
        m = clean_mask(rle_decode(inst["rle"]), rcfg["fill_holes_below"], rcfg["drop_islands_below"])
        if m.any():
            anns.append(Annotation(inst["category"], m, score=inst["score"]))
    thing_union = np.logical_or.reduce([a.mask for a in anns]) if anns else np.zeros((h, w), bool)

    # Stuff: the thin gaps SAM leaves between regions (pixels in no region and
    # on no detected thing) go to the nearest stuff region. Regions that were
    # ignored, or left unlabelled with --allow-unlabeled, are NOT filled: they
    # stay holes in the pre-annotation, for the annotator to draw in CVAT.
    stuff_ids = [rid for rid, lab in assigned.items() if lab != IGNORE and not cfg.by_name[lab].isthing]
    stuff_map = np.where(np.isin(regions.label_map, stuff_ids), regions.label_map, 0)
    if stuff_ids:
        hole = (regions.label_map == 0) & ~thing_union
        if hole.any():
            _, (iy, ix) = ndimage.distance_transform_edt(stuff_map == 0, return_indices=True)
            stuff_map = np.where(hole, stuff_map[iy, ix], stuff_map)

    by_label: dict[str, np.ndarray] = {}
    for rid, lab in assigned.items():
        if lab == IGNORE:
            continue
        src = stuff_map if not cfg.by_name[lab].isthing else regions.label_map
        m = src == rid
        by_label[lab] = by_label.get(lab, np.zeros((h, w), bool)) | m
    # regions nobody named: never covered by the hole filling of clean_mask either
    unnamed = (regions.label_map > 0) & ~np.isin(regions.label_map, [r for r, lab in assigned.items() if lab != IGNORE])
    for lab, m in sorted(by_label.items(), key=lambda kv: cfg.priority_rank(kv[0])):
        m = clean_mask(m, rcfg["fill_holes_below"], rcfg["drop_islands_below"]) & ~unnamed
        if cfg.by_name[lab].isthing:
            # thing classes named by the labeller: one instance per connected blob,
            # unless the blob touches a detector instance of the same class, in which
            # case it completes that instance (a wrist watch SAM cut out of a person)
            detected = [a for a in anns if a.category == lab and a.score is not None]
            kernel = np.ones((5, 5), np.uint8)
            for c in connected_components(m, rcfg["drop_islands_below"]):
                grown = cv2.dilate(c.astype(np.uint8), kernel).astype(bool)
                host = next((a for a in detected if (a.mask & grown).any()), None)
                if host is not None:
                    host.mask = host.mask | c
                else:
                    anns.append(Annotation(lab, c))
        elif m.any():
            anns.append(Annotation(lab, m))

    image = ImageAnnotations(info["file_name"], w, h, anns)
    out = write_cvat_zip(image, cfg, work / CVAT_ZIP)

    img = image_path or (work.parent / info["file_name"])
    if Path(img).exists():
        from .resolve import resolve

        save_jpeg(panoptic_overlay(load_rgb(img), resolve(image, cfg), cfg), work / "refine_preview.jpg")
    return out
