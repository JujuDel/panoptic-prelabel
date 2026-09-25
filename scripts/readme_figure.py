"""Build docs/<scene>.jpg for the README: photo | automatic pre-labels | final.

python scripts/readme_figure.py SCENE WORK FINAL_JSON
  SCENE       examples/<name>             (the photo)
  WORK        output of `prelabel` + `refine` for that photo
  FINAL_JSON  panoptic.json resolved from the CVAT-corrected export
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from panoptic_prelabel.coco_io import read_coco
from panoptic_prelabel.config import Config
from panoptic_prelabel.panoptic import read_panoptic
from panoptic_prelabel.pipeline import load_rgb
from panoptic_prelabel.resolve import resolve
from panoptic_prelabel.viz import draw_boxes, load_font, panoptic_overlay, resize_long
from panoptic_prelabel.web import boxes


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    font = load_font(max(14, im.width // 26))
    left, top, right, bottom = d.textbbox((0, 0), text, font=font)
    d.rectangle((10, 10, 10 + right - left + 16, 10 + bottom - top + 12), fill=(11, 13, 18))
    d.text((18, 16 - top), text, font=font, fill=(255, 255, 255))
    return np.asarray(im)


def main(scene: str, work: str, final_json: str, height: int = 820) -> None:
    cfg = Config.load()
    rgb = load_rgb(Path(scene) / "image.jpg")
    auto = resolve(read_coco(Path(work) / "coco_for_cvat.zip"), cfg)
    final = read_panoptic(final_json)
    panels = [
        label(rgb, "INPUT"),
        label(panoptic_overlay(rgb, auto, cfg), "PRE-LABELS (AUTO)"),
        label(draw_boxes(panoptic_overlay(rgb, final, cfg), boxes(final, cfg), final.width, final.height), "FINAL"),
    ]
    panels = [resize_long(p, height) for p in panels]
    h = min(p.shape[0] for p in panels)
    gap = 8
    canvas = Image.new("RGB", (sum(p.shape[1] for p in panels) + gap * 2, h), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(Image.fromarray(p[:h]), (x, 0))
        x += p.shape[1] + gap
    out = Path("docs") / f"{Path(scene).name}.jpg"
    out.parent.mkdir(exist_ok=True)
    canvas.save(out, quality=84, optimize=True)
    print(f"wrote {out} {canvas.size}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
