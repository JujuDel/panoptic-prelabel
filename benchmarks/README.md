# GPU benchmarks

Three measurements that need an NVIDIA GPU and internet access to model hubs. They were run once on Kaggle (Tesla T4); the results are in [`results/kaggle/`](results/kaggle/), and section 5 of the [main README](../README.md) sums them up.

| Script | Question it answers | Output |
|---|---|---|
| `latency.py` | How fast are the two pre-labelling models on CPU, CUDA and TensorRT (FP32 / FP16), and does the GPU give the same result as the CPU? Which parts of `MaskRCNN-12.onnx` can TensorRT itself take? | `results/<name>/latency.{md,json}` |
| `mask2former_baseline.py` | Would a single COCO-panoptic model (Mask2Former, Swin-T and Swin-L) have done better than this pipeline's automatic part, on the same 4 photos and against the same human-corrected reference? | `results/<name>/mask2former_baseline.{md,json}`, predictions and previews |
| `coco_val.py` | The same Mask2Former models on 100 COCO val2017 images, scored by the official panopticapi **and** by `panoptic_prelabel.compare`. The script fails if the two disagree by more than 1e-9. | `results/<name>/coco_val.{md,json}` |

`setup_gpu_env.py` installs the matching `onnxruntime-gpu` + TensorRT pair for the machine's PyTorch/CUDA version (the preinstalled PyTorch is kept) and reports which execution providers load. `kaggle_gpu.ipynb` runs everything in order.

## Run it on Kaggle (free T4 GPU)

You need a Kaggle account with a **verified phone number**: without it, Kaggle gives neither GPU nor internet.

**1. Zip the repository.** In PowerShell, from the repository folder:

```powershell
git archive --format=zip -o "$env:USERPROFILE\Desktop\panoptic-prelabel.zip" HEAD
```

This packs the committed files only: no weights, nothing private.

**2. Upload it as a private dataset.**
- On kaggle.com: *Create* → *New Dataset* → drop `panoptic-prelabel.zip`.
- Title: `panoptic-prelabel-src`, visibility *Private*, then *Create*.
- Kaggle unzips it automatically, including the four nested `examples/*/cvat_corrected.zip`, which become folders. The notebook's first cell zips them back.

**3. Import the notebook.**
- *Create* → *New Notebook*, then *File* → *Import Notebook*.
- Upload `benchmarks/kaggle_gpu.ipynb` from this repository.

**4. Set the notebook up** (right-hand panel):
- *Session options* → *Accelerator*: **GPU T4 x2**. The benchmarks use one GPU. Do not pick P100: current TensorRT versions do not support it.
- *Session options* → *Internet*: **On**.
- *Input* → *Add Input* → *Your Work* / *Datasets* → `panoptic-prelabel-src`.

**5. Run it.** Choose one of two ways:
- **In the background (recommended).** Click *Save Version* → *Save & Run All (Commit)*. You can close the browser, and the run keeps going for up to 12 h. When it is done, the version's *Output* tab holds `results_kaggle.zip`.
- **Interactively.** *Run All*, then download `results_kaggle.zip` from the *Output* panel (`/kaggle/working/`) before the session stops.

Expected time is about 1 to 1.5 h in total:
- the first TensorRT engine builds (a few minutes per photo and per precision);
- the COCO annotation download (~820 MB);
- one CPU run of MobileSAM (a few minutes).

**6. Bring the results back.** Unzip `results_kaggle.zip` into `benchmarks/results/`, so the files end up as `benchmarks/results/kaggle/latency.md` and so on, and commit them apart from any code change. The predictions on the COCO images are git-ignored; the summaries, and the predictions on the 4 example photos, are kept.

### If something fails

Each benchmark records a failure as an error row instead of crashing: a TensorRT provider that does not load, a model that cannot be downloaded, a CUDA out-of-memory. The results are saved after every section, so a later crash keeps what was already measured. The usual causes of failure:

| Symptom | Cause |
|---|---|
| `no GPU visible to torch` | The accelerator is not set to GPU (step 4). |
| `pip` cannot download anything | Internet is off, or the phone number is not verified. |
| TensorRT rows show an error | TensorRT could not be installed, or its version does not match the image's CUDA. It is installed last and is optional, so the CUDA rows and the other benchmarks are still valid. The row shows the first 120 characters of the error. |
| "Your notebook tried to allocate more memory than is available" during `latency.py` | The TensorRT runs come last, and the results are saved after each backend, so `latency.md` already holds the CPU, CUDA and MobileSAM rows. Look at the last `peak RSS` line printed. To get the rest without TensorRT, re-run with `--skip-tensorrt`. |
| `ModelImporter ... legalUINT8: TensorRT does not support UINT8 types` | Expected: `MaskRCNN-12.onnx` casts to UINT8 inside the graph, which TensorRT does not take. onnxruntime gives that part to CUDA and TensorRT the rest. |
| `Unable to open library: libnvinfer_vc_plugin.so.10`, repeated | TensorRT tried to take a `RoiAlign` node: its plugin needs a library the pip wheels do not ship. `latency.py` excludes `RoiAlign` from TensorRT (`trt_op_types_to_exclude`); if the message is back, that option was dropped. |
| `TensorRT input: ... has no shape specified` | The TensorRT rows run on `.cache/trt/MaskRCNN-12.shapes.onnx`, the model after onnxruntime's symbolic shape inference. Delete that file if it comes from an older run. |
| `coco_val.py` is slow or times out on the download | Re-run it with `--images 50`. |

## Run the plumbing without a GPU

Every script has a CPU path that checks the code before spending GPU time:

```bash
python benchmarks/latency.py --name local --cpu-only --skip-sam --repeat-cpu 2
python benchmarks/mask2former_baseline.py --name dry --random-weights --device cpu --repeat 1 \
    --models facebook/mask2former-swin-tiny-coco-panoptic
```

`--random-weights` builds Mask2Former without downloading weights, so its scores are meaningless: it only exercises the code. `tests/test_benchmarks.py` covers the conversion and scoring glue.

## Reading the numbers

- **Timing.** Median and p90 over repeated runs, after warm-up runs. GPU timings wait for the GPU to finish (`torch.cuda.synchronize`). The environment is saved with every result: GPU, driver, CUDA and library versions.
- **Scores.** "Common classes" restricts the averages to the classes a COCO-panoptic model can predict at all. Bollard, buoy, shoe, fountain and boardwalk exist only through region naming or manual correction.
- **Pre- and post-processing.** Mask2Former goes through Hugging Face's image processor, which resizes every image to 384 × 384, and its post-processing with default thresholds. The original evaluation uses a short side of 800 px, so the scores can be below the published ones; `coco_val.py` measures by how much on the sample.
