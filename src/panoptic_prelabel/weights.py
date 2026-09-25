"""Download the two model files. Nothing is committed to the repository."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Weight:
    filename: str
    url: str
    sha256: str
    license: str


MOBILESAM_COMMIT = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
ONNX_MODELS_COMMIT = "4f43949841cb55a0b98dc8fcd045431ccafd9f96"  # onnx/models main, 2026-09-25

WEIGHTS = (
    Weight(
        "MaskRCNN-12.onnx",
        f"https://media.githubusercontent.com/media/onnx/models/{ONNX_MODELS_COMMIT}/validated/vision/"
        "object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx",
        "5bf940c124d9117f6fae350c4e0d46df180cbb01981832b35fdbed148078b946",
        "MIT (model card, onnx/models); converted from facebookresearch/maskrcnn-benchmark (MIT)",
    ),
    Weight(
        "mobile_sam.pt",
        f"https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/{MOBILESAM_COMMIT}/weights/mobile_sam.pt",
        "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
        "Apache-2.0 (ChaoningZhang/MobileSAM)",
    ),
)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: str | Path = "weights", force: bool = False) -> list[Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for w in WEIGHTS:
        path = dest / w.filename
        if path.exists() and not force and sha256sum(path) == w.sha256:
            print(f"ok       {path}")
            paths.append(path)
            continue
        print(f"fetching {w.url}")
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(w.url, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310 (fixed https URLs)
                shutil.copyfileobj(r, f)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        digest = sha256sum(tmp)
        if digest != w.sha256:
            tmp.unlink()
            raise RuntimeError(f"{w.filename}: sha256 {digest} does not match the expected {w.sha256}")
        tmp.replace(path)
        print(f"ok       {path}  ({w.license})")
        paths.append(path)
    return paths
