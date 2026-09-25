# Third-party notices

This repository contains no third-party source code and no model weights. The
models are downloaded by `panoptic-prelabel download-weights` from the URLs
below and verified against the SHA-256 digests in
`src/panoptic_prelabel/weights.py`. Python dependencies are installed by pip
from their own distributions.

Licenses were checked on 2026-09-24 in the files named below.

| Component | Used for | Source | License (where it was read) |
|---|---|---|---|
| Mask R-CNN R-50-FPN, `MaskRCNN-12.onnx` | thing pre-labels | [onnx/models](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/mask-rcnn), file `validated/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx`, sha256 `5bf940c1…8b946` | **MIT**: `SPDX-License-Identifier: MIT` in the model's `README.md`. The onnx/models repository itself is under **Apache-2.0** (`LICENSE`). |
| maskrcnn-benchmark (origin of the weights above) | – | [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) | **MIT**: `LICENSE`, "Copyright (c) 2018 Facebook" |
| COCO 2017 (training data of the weights above) | – | [cocodataset.org](https://cocodataset.org) | see the COCO terms of use on cocodataset.org (not re-read for this notice) |
| MobileSAM, code and `weights/mobile_sam.pt` | region proposals | [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM) @ `f706ad9c4eb7f219c00d9050e46328518ffb65d2`, sha256 of the weights `6dbb9052…6c2f` | **Apache-2.0**: `LICENSE` at that commit. The weights file is distributed in the same repository, and no separate license is stated for it. |
| Segment Anything (MobileSAM builds on it) | – | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | **Apache-2.0**: `LICENSE` |
| TinyViT (MobileSAM's image encoder) | – | `mobile_sam/modeling/tiny_vit_sam.py` | **MIT**: file header, "Copyright (c) 2022 Microsoft" |
| onnxruntime | Mask R-CNN inference | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | **MIT**: `LICENSE` |
| PyTorch, torchvision | MobileSAM inference | [pytorch/pytorch](https://github.com/pytorch/pytorch), [pytorch/vision](https://github.com/pytorch/vision) | **BSD-3-Clause**: `LICENSE` of both repositories |
| timm | MobileSAM dependency | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | **Apache-2.0**: `LICENSE` |

Not used, although often associated with Mask R-CNN: **Detectron2** (Apache-2.0)
and **torchvision's `maskrcnn_resnet50_fpn`**. torchvision is installed only
because MobileSAM imports it.

Example photos (`examples/*/image.jpg`) are not covered by this repository's
Apache-2.0 license. All rights reserved.
