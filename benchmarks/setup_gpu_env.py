"""Install the GPU libraries that match the machine's PyTorch/CUDA, then report what loads.

    python benchmarks/setup_gpu_env.py            # install + check
    python benchmarks/setup_gpu_env.py --check    # check only

Meant for a fresh Kaggle (or any Linux + NVIDIA) notebook where PyTorch with
CUDA is preinstalled; the preinstalled torch is kept, never replaced.

onnxruntime-gpu is built against one CUDA major version and one TensorRT major
version (checked by reading the NEEDED entries of its provider libraries):

    onnxruntime-gpu 1.25.0 -> CUDA 12, cuDNN 9, TensorRT 10
    onnxruntime-gpu 1.30.0 -> CUDA 13, cuDNN 9, TensorRT 10

so the pair is chosen from torch's CUDA version, with the latest TensorRT 10.
TensorRT is installed last and is optional: without it, only the TensorRT
rows of the latency benchmark are missing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

ORT = {"12": "onnxruntime-gpu==1.25.0", "13": "onnxruntime-gpu==1.30.0"}
# PyPI only has a stub for these that fetches the real wheels from NVIDIA's index
TRT = {"12": "tensorrt-cu12==10.16.1.11", "13": "tensorrt-cu13==10.16.1.11"}
COMMON = ["polygraphy==0.53.6", "transformers==5.17.0", "timm==1.0.30"]
GIT = [
    "mobile_sam @ git+https://github.com/ChaoningZhang/MobileSAM.git@f706ad9c4eb7f219c00d9050e46328518ffb65d2",
    "panopticapi @ git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0",
]


def pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)  # noqa: S603 (fixed pip command)


def check() -> bool:
    ok = True
    try:
        import torch

        print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device: {torch.cuda.get_device_name(0)}")
        else:
            ok = False
            print("  no GPU visible to torch: in Kaggle, Settings > Accelerator > GPU T4 x2")
    except ImportError:
        print("torch is not installed")
        return False
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        print(f"onnxruntime {ort.__version__}, providers: {ort.get_available_providers()}")
    except Exception as e:
        ok = False
        print(f"onnxruntime: {e}")
    try:
        import tensorrt

        print(f"tensorrt {tensorrt.__version__}")
    except Exception as e:
        print(f"tensorrt: {e} (the TensorRT rows of the latency table will be skipped)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report, install nothing")
    args = ap.parse_args()
    if not args.check:
        import torch

        major = (torch.version.cuda or "").split(".")[0]
        if major not in ORT:
            raise SystemExit(f"torch reports CUDA {torch.version.cuda!r}; expected a CUDA 12 or 13 build")
        # the CPU build and the GPU build both provide the `onnxruntime` module: keep only one
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], check=False)
        pip("install", "-q", ORT[major], *COMMON, *GIT)
        # TensorRT last and optional: if it fails, CUDA and every other benchmark still run
        trt = [sys.executable, "-m", "pip", "install", "-q", "--extra-index-url", "https://pypi.nvidia.com", TRT[major]]
        print("+", " ".join(trt), flush=True)
        if subprocess.run(trt, check=False).returncode != 0:  # noqa: S603 (fixed pip command)
            print("TensorRT could not be installed: the TensorRT rows of the latency table will show an error")
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
