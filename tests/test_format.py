"""Output formats: COCO panoptic PNG/JSON, CVAT COCO 1.0 zip, web export."""

import json
import zipfile

import numpy as np
import pytest
from conftest import rect, scene
from PIL import Image

from panoptic_prelabel.coco_io import CVAT_JSON, read_coco, write_cvat_zip
from panoptic_prelabel.config import Config
from panoptic_prelabel.masks import rle_decode, rle_encode
from panoptic_prelabel.panoptic import id2rgb, read_panoptic, rgb2id, write_panoptic
from panoptic_prelabel.resolve import resolve
from panoptic_prelabel.web import boxes, export_web, overlays

H, W = 40, 60


def _pan(cfg):
    return resolve(
        scene(
            [
                ("sky", rect(H, W, 0, 0, W, 20)),
                ("pavement", rect(H, W, 0, 20, W, H)),
                ("person", rect(H, W, 10, 5, 22, 38)),
                ("person", rect(H, W, 30, 8, 40, 38)),
                ("bollard", rect(H, W, 50, 10, 53, 38)),
            ]
        ),
        cfg,
    )


def test_id_rgb_roundtrip():
    ids = np.array([[0, 1, 255], [256, 65535, 70000]], dtype=np.int32)
    assert np.array_equal(rgb2id(id2rgb(ids)), ids)


def test_panoptic_files_are_valid_coco_panoptic(cfg, tmp_path):
    pan = _pan(cfg)
    png, js = write_panoptic(pan, cfg, tmp_path)
    data = json.loads(js.read_text())
    assert set(data) >= {"images", "annotations", "categories"}
    img, ann = data["images"][0], data["annotations"][0]
    assert (img["width"], img["height"]) == (W, H)
    assert ann["file_name"] == png.name
    cats = {c["id"]: c for c in data["categories"]}
    assert all({"id", "name", "isthing", "color"} <= set(c) for c in cats.values())

    ids = rgb2id(np.asarray(Image.open(png).convert("RGB")))
    seg_ids = [s["id"] for s in ann["segments_info"]]
    assert len(seg_ids) == len(set(seg_ids)), "segment ids must be unique"
    assert set(np.unique(ids)) <= set(seg_ids) | {0}
    for s in ann["segments_info"]:
        m = ids == s["id"]
        assert s["area"] == int(m.sum())
        ys, xs = np.nonzero(m)
        assert s["bbox"] == [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1]
        assert s["category_id"] in cats
    assert sum(s["area"] for s in ann["segments_info"]) == H * W  # no void left


def test_read_panoptic_roundtrip(cfg, tmp_path):
    pan = _pan(cfg)
    _, js = write_panoptic(pan, cfg, tmp_path)
    back = read_panoptic(js)
    assert np.array_equal(back.id_map, pan.id_map)
    assert [(s.id, s.category, s.area, s.bbox) for s in back.segments] == [
        (s.id, s.category, s.area, s.bbox) for s in pan.segments
    ]


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for first in (False, True):
        m = rng.random((17, 23)) > 0.5
        m[0, 0] = first
        rle = rle_encode(m)
        assert sum(rle["counts"]) == m.size
        assert np.array_equal(rle_decode(rle), m)


def test_cvat_zip_roundtrip(tmp_path):
    cfg = Config.load()
    cfg.data["refine"]["rle_if_holes_above"] = 100
    ring = rect(H, W, 5, 5, 45, 35) & ~rect(H, W, 15, 12, 35, 28)  # a 320-px hole -> exported as RLE
    img = scene([("sky", ring), ("person", rect(H, W, 50, 5, 58, 35))])
    path = write_cvat_zip(img, cfg, tmp_path / "coco.zip")
    with zipfile.ZipFile(path) as zf:
        data = json.loads(zf.read(CVAT_JSON))
    kinds = {a["category_id"]: a["iscrowd"] for a in data["annotations"]}
    assert kinds[cfg.by_name["sky"].id] == 1 and kinds[cfg.by_name["person"].id] == 0
    back = read_coco(path)
    assert np.array_equal(back.annotations[0].mask, ring)  # RLE is exact
    person = back.annotations[1].mask
    assert abs(int(person.sum()) - 8 * 30) <= 40  # polygons are close


def test_web_boxes_are_percent_and_skip_excluded_classes(cfg):
    pan = _pan(cfg)
    bxs = boxes(pan, cfg)
    assert [b["cls"] for b in bxs] == ["PERSON", "PERSON"]  # bollards are excluded by default
    first = bxs[0]
    assert first["x"] == round(100 * 30 / W, 2) and first["w"] == round(100 * 10 / W, 2)
    assert first["color"] == cfg.by_name["person"].color  # 1st instance: base color
    assert bxs[1]["color"] != first["color"]  # 2nd instance: a shade of it


def test_web_overlays(cfg):
    pan = _pan(cfg)
    pan_rgba, inst_rgba = overlays(pan, cfg)
    assert pan_rgba.shape[2] == 4 and max(pan_rgba.shape[:2]) == cfg["web"]["long_side"]
    assert pan_rgba.shape[:2] == inst_rgba.shape[:2]
    assert set(np.unique(pan_rgba[..., 3])) <= {cfg["web"]["pan_alpha"], cfg["web"]["edge_alpha"]}
    # stuff is dimmed in the instances view
    dim = np.array([8, 10, 18])
    assert (inst_rgba[5, 5, :3] == dim).all()


def test_export_web_strips_metadata(cfg, tmp_path):
    pan = _pan(cfg)
    photo = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[0x010F] = "SomeCamera"  # Make
    Image.new("RGB", (W, H), (120, 60, 30)).save(photo, exif=exif)
    assert Image.open(photo).getexif()
    export_web(pan, photo, "demo", tmp_path / "web", cfg)
    raw = Image.open(tmp_path / "web" / "demo_raw.jpg")
    assert not raw.getexif()
    assert np.array_equal(np.asarray(raw), np.asarray(Image.open(photo)))  # lossless
    payload = json.loads((tmp_path / "web" / "demo_boxes.json").read_text())
    assert payload["name"] == "DEMO" and (payload["w"], payload["h"]) == (W, H)


def test_jpeg_strip_drops_trailing_images_and_keeps_pixels(tmp_path):
    from panoptic_prelabel.jpeg import strip_metadata

    photo = tmp_path / "p.jpg"
    exif = Image.Exif()
    exif[0x010F] = "SomeCamera"
    for progressive in (False, True):
        Image.new("RGB", (W, H), (10, 200, 90)).save(photo, exif=exif, progressive=progressive)
        data = photo.read_bytes() + b"\xff\xd8 appended secondary image with XMP \xff\xd9"
        clean = strip_metadata(data)
        assert clean.endswith(b"\xff\xd9") and b"appended" not in clean and b"SomeCamera" not in clean
        (tmp_path / "c.jpg").write_bytes(clean)
        assert np.array_equal(np.asarray(Image.open(tmp_path / "c.jpg")), np.asarray(Image.open(photo)))


def test_jpeg_strip_keeps_the_icc_profile(tmp_path):
    from panoptic_prelabel.jpeg import strip_metadata

    icc = b"fake-icc-profile-bytes" * 4
    photo = tmp_path / "icc.jpg"
    exif = Image.Exif()
    exif[0x010F] = "SomeCamera"
    Image.new("RGB", (W, H), (1, 2, 3)).save(photo, exif=exif, icc_profile=icc)
    clean = tmp_path / "clean.jpg"
    clean.write_bytes(strip_metadata(photo.read_bytes()))
    with Image.open(clean) as im:
        assert im.info.get("icc_profile") == icc
        assert not im.getexif()


def test_export_web_applies_exif_rotation(cfg, tmp_path):
    # pixels stored landscape (W x H) with Orientation=6: displayed as portrait (H x W)
    stored = np.zeros((H, W, 3), np.uint8)
    stored[:, : W // 2] = 255
    photo = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(stored).save(photo, exif=exif, quality=100)
    portrait = resolve(scene([("sky", rect(W, H, 0, 0, H, W))], h=W, w=H), cfg)  # a map of the upright size
    export_web(portrait, photo, "rot", tmp_path / "web", cfg)
    with Image.open(tmp_path / "web" / "rot_raw.jpg") as raw:
        assert raw.size == (H, W)
        assert raw.getexif().get(0x0112, 1) == 1


def test_export_web_rejects_unsafe_names(cfg, tmp_path):
    pan = _pan(cfg)
    photo = tmp_path / "p.jpg"
    Image.new("RGB", (W, H)).save(photo)
    with pytest.raises(ValueError, match="scene name"):
        export_web(pan, photo, 'x"};alert(1)//', tmp_path / "web", cfg)
