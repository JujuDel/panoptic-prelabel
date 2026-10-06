# Mask2Former on 100 COCO val2017 images (seed 0)

| model | PQ | PQ things | PQ stuff | SQ | RQ | max difference compare vs panopticapi | median time / image |
|---|---|---|---|---|---|---|---|
| mask2former-swin-tiny-coco-panoptic | 43.4 | 48.6 | 36.1 | 67.0 | 51.7 | 3.3e-16 | 68 ms |
| mask2former-swin-large-coco-panoptic | 46.6 | 51.1 | 40.3 | 69.1 | 55.5 | 2.2e-16 | 134 ms |

PQ as computed by the official panopticapi; `panoptic_prelabel.compare` gives the same numbers (column 'max difference'). Hugging Face post-processing, default thresholds.
