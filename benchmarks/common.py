"""Shared helpers for the benchmarks: environment report, timing, result files, examples.

Benchmarks are scripts, not part of the package: they need extra libraries
(onnxruntime-gpu, TensorRT, transformers) and write to benchmarks/results/.
"""

from __future__ import annotations

import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from panoptic_prelabel.coco_io import read_coco
from panoptic_prelabel.config import Config
from panoptic_prelabel.panoptic import Panoptic
from panoptic_prelabel.pipeline import refine
from panoptic_prelabel.resolve import resolve

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# Ontology classes a COCO-panoptic model can predict at all (see mask2former_baseline.COCO_TO_ONTOLOGY).
# The rest (bollard, buoy, shoe, fountain, boardwalk) exist only through region naming or correction.
COCO_COVERED = {
    "person", "backpack", "bag", "bird", "dog", "phone", "car", "truck", "sign",
    "sky", "sea", "sand", "tree", "grass", "rock", "building", "pavement", "road", "wall",
}  # fmt: skip


def scenes() -> list[Path]:
    found = sorted(p for p in EXAMPLES.iterdir() if (p / "cvat_corrected.zip").is_file())
    if not found:
        raise SystemExit(f"no example scene in {EXAMPLES} (was examples/*/cvat_corrected.zip unpacked by an upload?)")
    return found


def corrected_map(scene: Path, cfg: Config) -> Panoptic:
    f = scene / "scene.yaml"
    groups = yaml.safe_load(f.read_text())["groups"] if f.exists() else []
    return resolve(read_coco(scene / "cvat_corrected.zip"), cfg, groups=groups)


def automatic_map(scene: Path, cfg: Config, tmp: Path) -> Panoptic:
    work = tmp / scene.name
    shutil.copytree(scene / "prelabel", work, dirs_exist_ok=True)
    return resolve(read_coco(refine(work, cfg, labels=scene / "regions.yaml")), cfg)


# --------------------------------------------------------------------------- timing


def timed(fn: Callable[[], Any], warmup: int, repeat: int, sync: Callable[[], None] | None = None) -> dict:
    """Median / p90 / min wall time in ms of `fn`, after `warmup` untimed calls.

    `sync` (e.g. torch.cuda.synchronize) is called before reading the clock, so
    asynchronous GPU work is included in the measurement. The value returned by
    the last call is under the key "last" (pop it before saving the stats).
    """
    last = None
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        last = fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {
        "median_ms": round(statistics.median(times), 2),
        "p90_ms": round(times[min(len(times) - 1, int(0.9 * len(times)))], 2),
        "min_ms": round(times[0], 2),
        "runs": repeat,
        "warmup": warmup,
        "last": last,
    }


# --------------------------------------------------------------------------- environment


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment() -> dict:
    """What the numbers were measured on. Saved next to every result."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": _cpu_count(),
        "packages": {
            d: _version(d)
            for d in (
                "panoptic-prelabel",
                "numpy",
                "opencv-python-headless",
                "onnxruntime",
                "onnxruntime-gpu",
                "torch",
                "torchvision",
                "timm",
                "mobile_sam",
                "transformers",
                "tensorrt",
                "tensorrt-cu12",
                "tensorrt-cu13",
                "tensorrt_cu12_libs",
                "tensorrt_cu13_libs",
                "polygraphy",
            )
        },
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=20,
        )
        env["gpus"] = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        env["gpus"] = []
    try:
        import torch

        env["torch_cuda"] = torch.version.cuda
        env["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["torch_device"] = torch.cuda.get_device_name(0)
            env["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        env["onnxruntime_available_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    return env


def _cpu_count() -> int:
    import os

    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 0)


# --------------------------------------------------------------------------- results


def results_dir(name: str, root: str | Path | None = None) -> Path:
    out = Path(root) if root else ROOT / "benchmarks" / "results"
    out = out / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save(out: Path, stem: str, payload: dict, markdown: str) -> None:
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=1, default=str), "utf-8")
    (out / f"{stem}.md").write_text(markdown.rstrip() + "\n", "utf-8")
    print(markdown)
    print(f"\nwrote {out / stem}.json and .md")


def pct(x: float) -> str:
    return f"{100 * x:.1f}"
