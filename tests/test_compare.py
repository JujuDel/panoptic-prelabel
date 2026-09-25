"""compare: hand-computed PQ / mIoU on tiny maps."""

import numpy as np
import pytest

from panoptic_prelabel.compare import evaluate
from panoptic_prelabel.masks import bbox_xywh
from panoptic_prelabel.panoptic import Panoptic, Segment


def pan(id_map, cats):
    id_map = np.asarray(id_map, dtype=np.int32)
    segs = [
        Segment(i, c, c in {"person", "car"}, int((id_map == i).sum()), bbox_xywh(id_map == i), [])
        for i, c in cats.items()
    ]
    return Panoptic("x", id_map.shape[1], id_map.shape[0], id_map, segs)


def test_identical_maps_score_100():
    m = [[1, 1, 2, 2], [1, 1, 2, 2]]
    ref = pan(m, {1: "sky", 2: "person"})
    s = evaluate([(ref, ref)]).summary()
    assert s["PQ"] == s["mIoU"] == s["pixel_acc"] == 1.0


def test_known_tp_fp_fn():
    # one row, sky=1 person=2 car=3
    #   ref : s p p p p s c c s s
    #   pred: s s p p p p s s c c
    # person: inter 3, union 5 -> IoU 0.6, a match; car: no overlap -> FP + FN;
    # sky: inter 1, union 7 -> no match -> FP + FN
    ref = pan([[1, 2, 2, 2, 2, 1, 3, 3, 1, 1]], {1: "sky", 2: "person", 3: "car"})
    pred = pan([[1, 1, 2, 2, 2, 2, 1, 1, 3, 3]], {1: "sky", 2: "person", 3: "car"})
    ev = evaluate([(ref, pred)])
    counts = {n: (c.tp, c.fp, c.fn) for n, c in ev.classes.items()}
    assert counts == {"person": (1, 0, 0), "car": (0, 1, 1), "sky": (0, 1, 1)}
    s = ev.summary()
    assert ev.classes["person"].pq == pytest.approx(0.6)
    assert s["PQ_th"] == pytest.approx(0.3)  # mean of person 0.6 and car 0
    assert s["PQ_st"] == 0.0
    assert s["PQ"] == pytest.approx(0.2)
    assert s["mIoU"] == pytest.approx((0.6 + 0.0 + 1 / 7) / 3)
    assert s["pixel_acc"] == pytest.approx(0.4)


def test_pooling_over_images_is_not_averaging_per_image():
    # image 1: person matched perfectly; image 2: person missed entirely
    ref1 = pan([[1, 2], [1, 2]], {1: "sky", 2: "person"})
    ref2 = pan([[1, 2], [1, 2]], {1: "sky", 2: "person"})
    pred2 = pan([[1, 1], [1, 1]], {1: "sky"})
    ev = evaluate([(ref1, ref1), (ref2, pred2)])
    person = ev.classes["person"]
    assert (person.tp, person.fp, person.fn) == (1, 0, 1)
    assert person.pq == pytest.approx(1.0 / 1.5)


def test_prediction_on_reference_void_is_not_a_false_positive():
    ref = pan([[1, 1, 0, 0], [1, 1, 0, 0]], {1: "sky"})
    pred = pan([[1, 1, 2, 2], [1, 1, 2, 2]], {1: "sky", 2: "car"})
    car = evaluate([(ref, pred)]).classes["car"]
    assert car.fp == 0


def test_large_coco_ids_are_supported():
    # COCO panoptic ids use 24 bits; a map compared with itself must score 100
    big = 3 * 256**2 + 7
    ref = pan([[big, big, 5, 5]], {big: "sky", 5: "person"})
    assert evaluate([(ref, ref)]).summary()["PQ"] == 1.0


def test_segment_listed_but_absent_from_the_png():
    ref = pan([[1, 1]], {1: "sky"})
    ref.segments.append(Segment(9, "sea", False, 0, [0, 0, 0, 0], []))  # hidden entirely
    assert evaluate([(ref, ref)]).classes["sea"].fn == 1
