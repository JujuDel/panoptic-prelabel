# Example scenes

The four scenes of the [juliendelclos.com](https://juliendelclos.com) hero carousel.

| file | what it is |
|---|---|
| `image.jpg` | the photo, metadata removed (no EXIF / GPS) |
| `regions.yaml` | the class of each MobileSAM region, written by Claude (Anthropic's model) from `regions.jpg` |
| `cvat_corrected.zip` | the annotation after manual correction in CVAT, exported as COCO 1.0 |
| `scene.yaml` | (wharf only) annotation ids that are parts of one object |
| `prelabel/` | the recorded outputs of `panoptic-prelabel prelabel` for this photo (Mask R-CNN instances, MobileSAM regions), so `refine` and `scripts/evaluate.py` replay the automatic part without the models |
| `reference_2026-07/` | the panoptic map produced in July 2026 by the first version of the pipeline, used on the website |

```bash
panoptic-prelabel run examples/bay --out out/bay                  # all stages (needs the models)
panoptic-prelabel run examples/bay --out out/bay --skip-prelabel  # no model: replays prelabel/, then refine, resolve, export-web
```

The photos are not covered by the repository's Apache-2.0 license. All rights reserved.
