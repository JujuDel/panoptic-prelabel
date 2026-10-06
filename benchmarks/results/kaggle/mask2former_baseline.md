# Mask2Former vs this pipeline (4 example photos)

Reference: the human-corrected maps. 'Common classes' = the ontology classes a COCO-panoptic model can predict (bollard, buoy, shoe, fountain and boardwalk excluded).

Device: cuda on 2 x Tesla T4, 580.178.04, 15360 MiB (the benchmarks use the first)

| predictor | PQ | PQ things | PQ stuff | mIoU | PQ (common classes) | PQ things (common) | PQ stuff (common) |
|---|---|---|---|---|---|---|---|
| this pipeline, automatic part (Mask R-CNN + MobileSAM + region naming) | 45.0 | 36.1 | 54.8 | 55.7 | 46.8 | 36.8 | 54.8 |
| mask2former-swin-tiny-coco-panoptic | 49.4 | 30.7 | 68.0 | 52.6 | 54.8 | 38.4 | 68.0 |
| mask2former-swin-large-coco-panoptic | 45.5 | 22.1 | 71.2 | 52.3 | 50.3 | 27.1 | 71.2 |

## Time per photo (forward + post-processing, median)

| model | bay | beach | street | wharf |
|---|---|---|---|---|
| mask2former-swin-tiny-coco-panoptic | 78.95 ms | 75.01 ms | 96.87 ms | 86.23 ms |
| mask2former-swin-large-coco-panoptic | 146.51 ms | 141.09 ms | 157.58 ms | 148.51 ms |

## Per class

### mask2former-swin-tiny-coco-panoptic

```
images           4
PQ / SQ / RQ     49.4 / 60.1 / 61.6
PQ things/stuff  30.7 / 68.0
mIoU             52.6
pixel accuracy   88.78 %
class       kind     PQ    SQ    RQ   IoU  TP FP FN
backpack    thing  35.9  89.7  40.0  37.6   1  1  2
bag         thing  22.9  80.2  28.6  13.9   1  2  3
bird        thing  35.1  87.9  40.0  84.6   1  0  3
bollard     thing   0.0   0.0   0.0   0.0   0  0  8
buoy        thing   0.0   0.0   0.0   0.0   0  0  1
car         thing  72.4  72.4 100.0  72.4   1  0  0
person      thing  73.8  82.0  90.0  84.6   9  1  1
phone       thing   0.0   0.0   0.0   0.0   0  0  1
sign        thing   0.0   0.0   0.0   0.0   0  0  1
truck       thing  67.4  67.4 100.0  67.4   1  0  0
building    stuff  21.5  64.6  33.3  32.6   1  2  2
grass       stuff   0.0   0.0   0.0   0.0   0  0  1
pavement    stuff  86.9  86.9 100.0  87.2   2  0  0
road        stuff  73.1  73.1 100.0  73.1   1  0  0
rock        stuff  51.7  51.7 100.0  51.7   1  0  0
sand        stuff  97.5  97.5 100.0  97.5   1  0  0
sea         stuff  83.8  83.8 100.0  84.2   2  0  0
sky         stuff  95.0  95.0 100.0  96.0   3  0  0
tree        stuff  80.2  80.2 100.0  78.9   4  0  0
wall        stuff  90.1  90.1 100.0  90.1   1  0  0
```
### mask2former-swin-large-coco-panoptic

```
images           4
PQ / SQ / RQ     45.5 / 54.7 / 54.1
PQ things/stuff  22.1 / 71.2
mIoU             52.3
pixel accuracy   90.97 %
class       kind     PQ    SQ    RQ   IoU  TP FP FN
backpack    thing  61.5  92.3  66.7  88.2   2  1  1
bag         thing  22.8  57.1  40.0   2.5   1  0  3
bird        thing  35.5  88.6  40.0  85.4   1  0  3
bollard     thing   0.0   0.0   0.0   0.0   0  0  8
buoy        thing   0.0   0.0   0.0   0.0   0  0  1
car         thing  46.9  70.4  66.7  60.9   1  1  0
dog         thing   0.0   0.0   0.0   0.0   0  1  0
person      thing  76.9  85.4  90.0  90.7   9  1  1
phone       thing   0.0   0.0   0.0   0.0   0  0  1
sign        thing   0.0   0.0   0.0   0.0   0  0  1
truck       thing   0.0   0.0   0.0   0.0   0  0  1
building    stuff  20.8  62.5  33.3  34.9   1  2  2
grass       stuff  90.1  90.1 100.0  90.1   1  0  0
pavement    stuff  91.1  91.1 100.0  90.7   2  0  0
road        stuff  72.4  72.4 100.0  72.4   1  0  0
rock        stuff   0.0   0.0   0.0  43.2   0  1  1
sand        stuff  96.1  96.1 100.0  96.1   1  0  0
sea         stuff  80.9  80.9 100.0  81.2   2  0  0
sky         stuff  92.0  92.0 100.0  93.6   3  0  0
tree        stuff  80.7  80.7 100.0  79.4   4  0  0
wall        stuff  88.4  88.4 100.0  88.4   1  0  0
```
