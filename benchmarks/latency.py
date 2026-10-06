"""Latency of the two pre-labelling models, CPU vs GPU, on the example photos.

    python benchmarks/latency.py --name kaggle-t4             # everything available
    python benchmarks/latency.py --name local --cpu-only      # no GPU

Mask R-CNN (ONNX) runs through onnxruntime with each execution provider that
loads: CPU, CUDA, and TensorRT (FP32 and FP16). For every provider the
detections are compared with the CPU ones (same classes, mask IoU, score
delta), so a faster provider that changes the result is visible.

MobileSAM's automatic mask generation runs with PyTorch on CPU and CUDA
(at two prompt batch sizes). Its regions are compared with the recorded CPU
output in examples/*/prelabel/regions.png.

A TensorRT-only build of the ONNX graph (no onnxruntime) is attempted with
`polygraphy inspect capability`, which reports which parts of the graph
TensorRT can take; the output is saved as is.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, environment, gpu_label, results_dir, save, scenes, timed

from panoptic_prelabel.config import Config
from panoptic_prelabel.masks import rle_decode
from panoptic_prelabel.pipeline import load_rgb
from panoptic_prelabel.things import ThingDetector, preprocess

WEIGHTS = ROOT / "weights"


def load_tensorrt_libs() -> str:
    """Make the pip-installed TensorRT libraries visible to onnxruntime's TensorRT provider."""
    try:
        import tensorrt_libs  # installed by `pip install tensorrt-cu12`
    except ImportError:
        return "tensorrt_libs not installed"
    libdir = Path(tensorrt_libs.__file__).parent
    loaded = []
    for pattern in ("libnvinfer.so.*", "libnvinfer_plugin.so.*", "libnvonnxparser.so.*"):
        libs = sorted(libdir.glob(pattern))
        if not libs:
            return f"{pattern} not found in {libdir}"
        for lib in libs:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                loaded.append(lib.name)
            except OSError as e:
                return f"failed to load {lib.name}: {e}"
    return "loaded " + ", ".join(loaded)


def peak_rss_gb() -> float | None:
    """Peak resident memory of this process so far (Linux), to see which step grows it."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


# onnx-tensorrt imports RoiAlign as the ROIAlign_TRT plugin and then looks up
# libnvinfer_vc_plugin, which the pip wheels do not ship: every subgraph holding a
# RoiAlign fails to parse, and onnxruntime keeps re-splitting the graph until the
# machine runs out of memory. Excluded, RoiAlign runs on CUDA.
TRT_EXCLUDED_OPS = "RoiAlign"


def shape_inferred_model(cache_dir: Path) -> Path:
    """MaskRCNN-12.onnx with the shapes of its intermediate tensors filled in.

    The TensorRT provider rejects the original ("TensorRT input: ... has no shape
    specified"); its documented fix is onnxruntime's symbolic shape inference. Only
    shape annotations are added: the computation is the same.
    """
    out = cache_dir / "MaskRCNN-12.shapes.onnx"
    if not out.exists():
        import onnx
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        model = SymbolicShapeInference.infer_shapes(onnx.load(str(WEIGHTS / "MaskRCNN-12.onnx")), auto_merge=True)
        tmp = out.with_suffix(".tmp")
        onnx.save(model, str(tmp))
        tmp.replace(out)
    return out


def provider_configs(cache_dir: Path) -> dict[str, list]:
    trt = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_dir),
        "trt_max_workspace_size": 4 << 30,
        "trt_op_types_to_exclude": TRT_EXCLUDED_OPS,
    }
    return {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt_fp32": [("TensorrtExecutionProvider", trt), "CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt_fp16": [
            ("TensorrtExecutionProvider", {**trt, "trt_fp16_enable": True}),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    }


def match_detections(ref: list, new: list) -> dict:
    """Match a backend's detections to the CPU ones, greedily by mask IoU within each class.

    Robust to order changes and to a detection crossing the score threshold:
    those show up as unmatched counts instead of hiding every other difference.
    """

    def iou(a, b) -> float:
        return float((a.mask & b.mask).sum() / max((a.mask | b.mask).sum(), 1))

    pairs = sorted(
        ((iou(a, b), i, j) for i, a in enumerate(ref) for j, b in enumerate(new) if a.category == b.category),
        reverse=True,
    )
    used_r, used_n, matched = set(), set(), []
    for v, i, j in pairs:
        if v > 0.5 and i not in used_r and j not in used_n:
            used_r.add(i)
            used_n.add(j)
            matched.append((v, abs(ref[i].score - new[j].score)))
    return {
        "matched": len(matched),
        "only_cpu": len(ref) - len(matched),
        "only_this_backend": len(new) - len(matched),
        "min_mask_iou": round(min(v for v, _ in matched), 4) if matched else None,
        "max_score_delta": round(max(d for _, d in matched), 4) if matched else None,
    }


def region_agreement(new: np.ndarray, recorded: np.ndarray) -> dict:
    """How close a region map is to the recorded one.

    `exact`: same region id per pixel. Regions are numbered by area, so one region
    appearing or vanishing renumbers the smaller ones; `matched` maps each new
    region to the recorded region it overlaps most, which is insensitive to that.
    """
    keys, counts = np.unique(new.astype(np.int64) * (1 << 20) + recorded, return_counts=True)
    best: dict[int, tuple[int, int]] = {}
    for k, c in zip(keys, counts, strict=True):
        n, r = divmod(int(k), 1 << 20)
        if c > best.get(n, (0, -1))[0]:
            best[n] = (int(c), r)
    matched = sum(c for n, (c, r) in best.items() if n != 0 and r != 0) + int(((new == 0) & (recorded == 0)).sum())
    return {"exact": round(float((new == recorded).mean()), 4), "matched": round(matched / new.size, 4)}


def _error_row(model: str, backend: str, e: BaseException, **extra) -> dict:
    msg = f"{type(e).__name__}: {e}".replace("\n", " ")[:300]
    print(f"[{backend}] error: {msg}")
    return {"model": model, "backend": backend, **extra, "error": msg}


def init_onnxruntime() -> None:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):  # onnxruntime >= 1.21: CUDA / cuDNN from the nvidia pip packages
        try:
            ort.preload_dlls()
        except Exception as e:
            print(f"note: preload_dlls failed: {e}")


def bench_detector(cfg: Config, images: dict[str, np.ndarray], name: str, args, reference: dict) -> list[dict]:
    """One backend; `reference` holds the CPU detections per scene (filled by the "cpu" run).

    One call per backend, so each session is freed before the next one is created.
    """
    import onnxruntime as ort

    cache = ROOT / ".cache" / "trt"  # engines are large: kept out of the results folder
    cache.mkdir(parents=True, exist_ok=True)
    providers = provider_configs(cache)[name]
    model = "Mask R-CNN (ONNX)"
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    onnx_path = WEIGHTS / "MaskRCNN-12.onnx"
    if name.startswith("tensorrt"):
        try:
            onnx_path = shape_inferred_model(cache)
        except Exception as e:
            return [_error_row(model, name, e)]
    print(f"[{name}] creating the session ...", flush=True)
    try:
        session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    except Exception as e:
        return [_error_row(model, name, e)]
    wanted = providers[0] if isinstance(providers[0], str) else providers[0][0]
    if session.get_providers()[0] != wanted:
        return [_error_row(model, name, RuntimeError(f"fell back to {session.get_providers()}"))]
    # TensorRT builds its engine at the first run; if that fails, onnxruntime would
    # silently continue on CUDA and this row would time CUDA under a TensorRT label
    session.disable_fallback()
    det = ThingDetector(cfg, session=session)
    rows = []
    for scene, rgb in images.items():
        if name.startswith("tensorrt") and scene not in args.trt_scenes:
            continue  # every new input size means a new engine build (minutes each)
        tensor, _ = preprocess(rgb)
        feed = {det.input_name: tensor}
        n_warm = 3 if name != "cpu" else 1
        n_rep = args.repeat_gpu if name != "cpu" else args.repeat_cpu
        if name.startswith("tensorrt"):
            print(f"[{name}] {scene}: building the TensorRT engine (minutes, cached afterwards) ...", flush=True)
        try:
            t_net = timed(partial(session.run, None, feed), warmup=n_warm, repeat=n_rep)
            t_net.pop("last")
            t_all = timed(partial(det, rgb), warmup=1, repeat=max(1, n_rep // 2))
            dets = t_all.pop("last")
        except Exception as e:
            rows.append(_error_row(model, name, e, scene=scene))
            continue
        if name == "cpu":
            reference[scene] = dets
        row = {
            "model": model,
            "backend": name,
            "providers": session.get_providers(),  # after the runs: what really ran
            "onnx": onnx_path.name,
            "scene": scene,
            "input": list(tensor.shape),
            "network": t_net,
            "with_pre_post": t_all,
            "detections": len(dets),
            "peak_rss_gb": peak_rss_gb(),
        }
        if scene in reference and name != "cpu":
            row["vs_cpu"] = match_detections(reference[scene], dets)
        rows.append(row)
        print(
            f"[{name}] {scene}: network {t_net['median_ms']} ms, total {t_all['median_ms']} ms, "
            f"peak RSS {row['peak_rss_gb']} GB",
            flush=True,
        )
    return rows


def bench_sam(cfg: Config, images: dict[str, np.ndarray], args) -> list[dict]:
    import json

    import torch

    from panoptic_prelabel.pipeline import THINGS
    from panoptic_prelabel.stuff import RegionProposer

    runs = [("cpu", 64)]
    if not args.cpu_only and torch.cuda.is_available():
        runs += [("cuda", 64), ("cuda", 256)]
    model = "MobileSAM automatic masks"
    rows = []
    for device, ppb in runs:
        backend = f"torch {device}"
        cfg.data["stuff"]["device"] = device
        cfg.data["stuff"]["points_per_batch"] = ppb
        try:
            proposer = RegionProposer(cfg, checkpoint=str(WEIGHTS / "mobile_sam.pt"))
        except Exception as e:
            rows.append(_error_row(model, backend, e, points_per_batch=ppb))
            continue
        sync = torch.cuda.synchronize if device == "cuda" else None
        todo = list(images.items())[: args.sam_cpu_images] if device == "cpu" else list(images.items())
        for scene, rgb in todo:
            meta = json.loads((ROOT / "examples" / scene / "prelabel" / THINGS).read_text())
            things = np.zeros(rgb.shape[:2], bool)
            for inst in meta["instances"]:
                things |= rle_decode(inst["rle"])
            rep = 1 if device == "cpu" else args.repeat_sam_gpu
            try:
                t = timed(partial(proposer, rgb, things), warmup=0 if device == "cpu" else 1, repeat=rep, sync=sync)
            except Exception as e:  # e.g. CUDA out of memory at a large prompt batch
                rows.append(_error_row(model, backend, e, points_per_batch=ppb, scene=scene))
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            regions = t.pop("last")
            recorded = np.asarray(Image.open(ROOT / "examples" / scene / "prelabel" / "regions.png")).astype(np.int32)
            agree = region_agreement(regions.label_map, recorded)
            rows.append(
                {
                    "model": model,
                    "backend": backend,
                    "points_per_batch": ppb,
                    "scene": scene,
                    "time": t,
                    "regions": len(regions.points),
                    "vs_recorded_cpu": agree,
                }
            )
            print(f"[sam {device} ppb={ppb}] {scene}: {t['median_ms'] / 1000:.1f} s, regions vs recorded: {agree}")
        del proposer
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def trt_capability(out: Path) -> dict:
    """Ask TensorRT (through polygraphy) which parts of the ONNX graph it supports."""
    target = out / "trt_capability"
    target.mkdir(exist_ok=True)
    cmd = ["polygraphy", "inspect", "capability", str(WEIGHTS / "MaskRCNN-12.onnx"), "-o", str(target)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # noqa: S603 (fixed command)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": " ".join(cmd), "error": str(e)}
    (target / "stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-25:]
    return {"command": " ".join(cmd), "returncode": proc.returncode, "last_lines": tail}


def markdown(env: dict, det_rows: list[dict], sam_rows: list[dict], cap: dict | None) -> str:
    lines = [
        "# Latency",
        "",
        f"GPU: {gpu_label(env)}  ",
        f"CPU: {env['cpu']} ({env['cpu_count']} threads)",
        "",
        "## Mask R-CNN (ONNX), per image",
        "",
        "| backend | scene | input | network (median / p90) | with pre/post (median) | vs CPU detections |",
        "|---|---|---|---|---|---|",
    ]
    for r in det_rows:
        if "error" in r:
            lines.append(f"| {r['backend']} | {r.get('scene', '-')} | - | - | - | error: {r['error'][:120]} |")
            continue
        a = r.get("vs_cpu")
        if r["backend"] == "cpu":
            agree = "reference"
        elif a:
            agree = (
                f"{a['matched']} matched (min IoU {a['min_mask_iou']}, max score delta {a['max_score_delta']}), "
                f"{a['only_cpu']} only on CPU, {a['only_this_backend']} only here"
            )
        else:
            agree = "-"
        lines.append(
            f"| {r['backend']} | {r['scene']} | {r['input'][1]}x{r['input'][2]} "
            f"| {r['network']['median_ms']} / {r['network']['p90_ms']} ms | {r['with_pre_post']['median_ms']} ms "
            f"| {agree} |"
        )
    if any(r["backend"].startswith("tensorrt") for r in det_rows):
        lines += [
            "",
            "TensorRT rows: the model after onnxruntime's symbolic shape inference (same computation, "
            f"shapes annotated). {TRT_EXCLUDED_OPS} excluded (its TensorRT plugin needs libnvinfer_vc_plugin, "
            "which the pip wheels do not ship); it and the nodes TensorRT's parser rejects (the UINT8 casts and "
            "the nodes reading them, one NonMaxSuppression) run on CUDA.",
        ]
    lines += [
        "",
        "## MobileSAM automatic masks, per image",
        "",
        "| backend | prompts / batch | scene | time (median) | regions | vs recorded CPU regions (exact / matched) |",
        "|---|---|---|---|---|---|",
    ]
    for r in sam_rows:
        if "error" in r:
            lines.append(
                f"| {r['backend']} | {r.get('points_per_batch', '-')} | {r.get('scene', '-')} | - | - "
                f"| error: {r['error'][:120]} |"
            )
            continue
        a = r["vs_recorded_cpu"]
        lines.append(
            f"| {r['backend']} | {r['points_per_batch']} | {r['scene']} | {r['time']['median_ms'] / 1000:.2f} s "
            f"| {r['regions']} | {100 * a['exact']:.1f} % / {100 * a['matched']:.1f} % |"
        )
    if cap is not None:
        tail = cap.get("last_lines") or [cap.get("error", "")]
        start = [i for i, line in enumerate(tail) if "===== Summary =====" in line]
        lines += [
            "",
            "## TensorRT capability of MaskRCNN-12.onnx (polygraphy)",
            "",
            "Summary only; the full output is in `trt_capability/stdout.txt`.",
            "",
            "```",
            *(tail[start[-1] :] if start else tail),
            "```",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="results/<name>/")
    ap.add_argument("--results-root")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--skip-sam", action="store_true")
    ap.add_argument("--skip-tensorrt", action="store_true", help="no TensorRT rows: CPU and CUDA only")
    ap.add_argument("--skip-capability", action="store_true")
    ap.add_argument("--repeat-cpu", type=int, default=5)
    ap.add_argument("--repeat-gpu", type=int, default=30)
    ap.add_argument("--repeat-sam-gpu", type=int, default=3)
    ap.add_argument("--sam-cpu-images", type=int, default=1, help="MobileSAM on CPU takes minutes per photo")
    ap.add_argument("--scenes", nargs="*", help="default: all example scenes")
    ap.add_argument(
        "--trt-scenes", nargs="*", default=["bay", "wharf"], help="TensorRT runs (one landscape, one portrait photo)"
    )
    args = ap.parse_args()

    out = results_dir(args.name, args.results_root)
    cfg = Config.load()
    env = environment()
    env["tensorrt_libs"] = load_tensorrt_libs()
    print(env)
    names = args.scenes or [s.name for s in scenes()]
    images = {n: load_rgb(ROOT / "examples" / n / "image.jpg") for n in names}

    payload: dict = {"environment": env, "detector": [], "sam": [], "trt_capability": None}

    def checkpoint() -> None:  # save after every step: a later crash keeps what was measured
        save(out, "latency", payload, markdown(env, payload["detector"], payload["sam"], payload["trt_capability"]))

    init_onnxruntime()
    reference: dict = {}

    def detector(backends: list[str]) -> None:
        for name in backends:
            payload["detector"] += bench_detector(cfg, images, name, args, reference)
            checkpoint()

    # TensorRT last: its engine builds are the step most likely to exhaust the memory
    detector(["cpu"] if args.cpu_only else ["cpu", "cuda"])
    if not args.skip_sam:
        payload["sam"] = bench_sam(cfg, images, args)
        checkpoint()
    if not (args.cpu_only or args.skip_tensorrt):
        detector(["tensorrt_fp32", "tensorrt_fp16"])
    if not (args.cpu_only or args.skip_capability):
        payload["trt_capability"] = trt_capability(out)
        checkpoint()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    main()
