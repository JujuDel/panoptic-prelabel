# Latency

GPU: 2 x Tesla T4, 580.178.04, 15360 MiB (the benchmarks use the first)  
CPU: Intel(R) Xeon(R) CPU @ 2.00GHz (4 threads)

## Mask R-CNN (ONNX), per image

| backend | scene | input | network (median / p90) | with pre/post (median) | vs CPU detections |
|---|---|---|---|---|---|
| cpu | bay | 736x1280 | 2220.41 / 2368.1 ms | 2217.68 ms | reference |
| cpu | beach | 800x1088 | 2159.7 / 2249.02 ms | 2155.89 ms | reference |
| cpu | street | 960x736 | 2010.7 / 2063.91 ms | 2033.96 ms | reference |
| cpu | wharf | 960x544 | 1666.3 / 1702.57 ms | 1656.85 ms | reference |
| cuda | bay | 736x1280 | 107.21 / 109.48 ms | 127.19 ms | 1 matched (min IoU 1.0, max score delta 0.0), 0 only on CPU, 0 only here |
| cuda | beach | 800x1088 | 109.33 / 110.8 ms | 127.14 ms | 2 matched (min IoU 1.0, max score delta 0.0), 0 only on CPU, 0 only here |
| cuda | street | 960x736 | 96.84 / 98.77 ms | 120.67 ms | 7 matched (min IoU 1.0, max score delta 0.0), 0 only on CPU, 0 only here |
| cuda | wharf | 960x544 | 80.67 / 82.04 ms | 93.09 ms | 3 matched (min IoU 1.0, max score delta 0.0), 0 only on CPU, 0 only here |
| tensorrt_fp32 | bay | 736x1280 | 105.53 / 108.12 ms | 124.71 ms | 1 matched (min IoU 1.0, max score delta 0.0), 0 only on CPU, 0 only here |
| tensorrt_fp32 | wharf | 960x544 | 80.89 / 82.95 ms | 94.73 ms | 3 matched (min IoU 1.0, max score delta 0.0), 0 only on CPU, 0 only here |
| tensorrt_fp16 | bay | 736x1280 | 45.71 / 49.27 ms | 70.22 ms | 1 matched (min IoU 0.9152, max score delta 0.0011), 0 only on CPU, 0 only here |
| tensorrt_fp16 | wharf | 960x544 | 40.92 / 42.01 ms | 57.79 ms | 3 matched (min IoU 0.8256, max score delta 0.0379), 0 only on CPU, 0 only here |

TensorRT rows: the model after onnxruntime's symbolic shape inference (same computation, shapes annotated). RoiAlign excluded (its TensorRT plugin needs libnvinfer_vc_plugin, which the pip wheels do not ship); it and the nodes TensorRT's parser rejects (the UINT8 casts and the nodes reading them, one NonMaxSuppression) run on CUDA.

## MobileSAM automatic masks, per image

| backend | prompts / batch | scene | time (median) | regions | vs recorded CPU regions (exact / matched) |
|---|---|---|---|---|---|
| torch cpu | 64 | bay | 169.36 s | 65 | 100.0 % / 100.0 % |
| torch cuda | 64 | bay | 6.89 s | 65 | 100.0 % / 100.0 % |
| torch cuda | 64 | beach | 7.92 s | 22 | 100.0 % / 100.0 % |
| torch cuda | 64 | street | 7.69 s | 70 | 100.0 % / 100.0 % |
| torch cuda | 64 | wharf | 6.64 s | 30 | 100.0 % / 100.0 % |
| torch cuda | 256 | bay | 7.10 s | 65 | 100.0 % / 100.0 % |
| torch cuda | 256 | beach | 8.21 s | 22 | 100.0 % / 100.0 % |
| torch cuda | 256 | street | 7.97 s | 70 | 100.0 % / 100.0 % |
| torch cuda | 256 | wharf | 6.81 s | 30 | 100.0 % / 100.0 % |

## TensorRT capability of MaskRCNN-12.onnx (polygraphy)

Summary only; the full output is in `trt_capability/stdout.txt`.

```
[I] ===== Summary =====
    Stack trace  | Operator  | Node  | Reason                                                                                                                                                                            
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                 | RoiAlign  | 2569  | In node 1481 with name: 2569 and operator: RoiAlign (importRoiAlign): UNSUPPORTED_NODE: Assertion failed: plugin != nullptr: ROIAlign plugin was not found in the plugin registry!
[I] Saving results to /kaggle/working/panoptic-prelabel/benchmarks/results/kaggle/trt_capability/results.txt
```
