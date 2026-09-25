"""resolve on small synthetic cases: one rule per test."""

import numpy as np
import pytest
from conftest import rect, scene

from panoptic_prelabel.config import Config
from panoptic_prelabel.resolve import resolve

H, W = 40, 60


def cls_at(pan, y, x):
    return pan.segment(int(pan.id_map[y, x])).category


def test_things_win_over_stuff(cfg):
    pan = resolve(scene([("sky", rect(H, W, 0, 0, W, H)), ("person", rect(H, W, 10, 10, 20, 30))]), cfg)
    assert cls_at(pan, 15, 15) == "person"
    assert cls_at(pan, 2, 2) == "sky"


def test_class_priority_between_things(cfg):
    # a backpack drawn over a person: backpack is listed before person
    person = rect(H, W, 10, 5, 40, 35)
    backpack = rect(H, W, 25, 10, 45, 25)
    pan = resolve(scene([("person", person), ("backpack", backpack)]), cfg)
    assert cls_at(pan, 15, 30) == "backpack"
    # ...and the order of the annotations in the file does not matter
    pan2 = resolve(scene([("backpack", backpack), ("person", person)]), cfg)
    assert np.array_equal(class_map(pan), class_map(pan2))
    assert cls_at(pan2, 15, 30) == "backpack"


def class_map(pan):
    names = {s.id: s.category for s in pan.segments}
    return np.vectorize(lambda i: names.get(int(i), ""))(pan.id_map)


def test_equal_area_tie_does_not_depend_on_file_order(cfg):
    # two people of exactly the same area overlapping: the winner must be stable
    a = rect(H, W, 10, 5, 30, 25)
    b = rect(H, W, 20, 10, 40, 30)
    one = resolve(scene([("person", a), ("person", b)]), cfg)
    two = resolve(scene([("person", b), ("person", a)]), cfg)
    winner_one = one.segment(int(one.id_map[15, 25]))
    winner_two = two.segment(int(two.id_map[15, 25]))
    # `a` (top-most first pixel) wins in both orders: it is id 1 in `one`, id 2 in `two`
    assert winner_one.source_ids == [1] and winner_two.source_ids == [2]


def test_tie_with_the_same_first_pixel_is_still_order_independent(cfg):
    # same area (400 px), same top-left pixel, different shapes
    a = rect(H, W, 5, 5, 25, 25)  # 20 x 20
    b = rect(H, W, 5, 5, 45, 15)  # 40 x 10
    one = resolve(scene([("person", a), ("person", b)]), cfg)
    two = resolve(scene([("person", b), ("person", a)]), cfg)
    owner_one = one.segment(int(one.id_map[10, 10])).source_ids  # a pixel both masks cover
    owner_two = two.segment(int(two.id_map[10, 10])).source_ids
    # the same mask wins whatever its position in the file: id 1 in `one` <=> id 2 in `two`
    assert (owner_one == [1]) == (owner_two == [2])


def test_an_id_in_two_groups_is_rejected(cfg):
    ann = scene(
        [("bag", rect(H, W, 0, 0, 5, 5)), ("bag", rect(H, W, 10, 10, 15, 15)), ("bag", rect(H, W, 20, 20, 25, 25))]
    )
    with pytest.raises(ValueError, match="more than one group"):
        resolve(ann, cfg, groups=[[1, 2], [2, 3]])


def test_groups_reject_stuff(cfg):
    ann = scene([("sky", rect(H, W, 0, 0, 20, 10)), ("sky", rect(H, W, 30, 0, 50, 10))])
    with pytest.raises(ValueError, match="stuff"):
        resolve(ann, cfg, groups=[[1, 2]])


def test_duplicate_ids_are_rejected(cfg):
    ann = scene([("person", rect(H, W, 0, 0, 5, 5)), ("person", rect(H, W, 10, 10, 15, 15))])
    ann.annotations[1].source_id = 1
    with pytest.raises(ValueError, match="unique"):
        resolve(ann, cfg)


def test_bollard_in_front_of_person(cfg):
    pan = resolve(scene([("person", rect(H, W, 10, 5, 40, 35)), ("bollard", rect(H, W, 20, 0, 24, H))]), cfg)
    assert cls_at(pan, 20, 22) == "bollard"


def test_stuff_priority(cfg):
    pan = resolve(scene([("sky", rect(H, W, 0, 0, W, 25)), ("tree", rect(H, W, 20, 5, 50, 40))]), cfg)
    assert cls_at(pan, 10, 30) == "tree"


def test_same_class_smaller_in_front(cfg):
    big = rect(H, W, 0, 0, 40, 40)
    small = rect(H, W, 30, 10, 50, 30)
    pan = resolve(scene([("person", big), ("person", small)]), cfg)
    front = pan.id_map[20, 35]
    assert pan.segment(int(front)).area < pan.segment(int(pan.id_map[5, 5])).area
    assert pan.segment(int(front)).source_ids == [2]


def test_split_classes_become_instances(cfg):
    two_bags = rect(H, W, 2, 2, 10, 10) | rect(H, W, 40, 20, 50, 30)
    pan = resolve(scene([("pavement", rect(H, W, 0, 0, W, H)), ("bag", two_bags)]), cfg)
    bags = [s for s in pan.segments if s.category == "bag"]
    assert len(bags) == 2
    assert {b.area for b in bags} == {64, 100}


def test_groups_merge_parts_of_one_object(cfg):
    a = rect(H, W, 10, 10, 15, 20)
    b = rect(H, W, 25, 10, 30, 20)  # the same backpack, cut in two by an arm
    ann = scene([("sky", rect(H, W, 0, 0, W, H)), ("backpack", a), ("backpack", b)])
    assert len([s for s in resolve(ann, cfg).segments if s.category == "backpack"]) == 2
    grouped = resolve(ann, cfg, groups=[[2, 3]])
    packs = [s for s in grouped.segments if s.category == "backpack"]
    assert len(packs) == 1
    assert packs[0].bbox == [10, 10, 20, 10]


def test_groups_reject_mixed_classes(cfg):
    ann = scene([("person", rect(H, W, 0, 0, 5, 5)), ("bag", rect(H, W, 10, 10, 15, 15))])
    with pytest.raises(ValueError, match="mixes classes"):
        resolve(ann, cfg, groups=[[1, 2]])


def test_stuff_of_one_class_is_one_segment(cfg):
    pan = resolve(
        scene([("sky", rect(H, W, 0, 0, 20, 10)), ("sky", rect(H, W, 40, 0, W, 10)), ("sea", rect(H, W, 0, 10, W, H))]),
        cfg,
    )
    assert [s.category for s in pan.segments].count("sky") == 1


def test_gaps_filled_with_nearest_segment(cfg):
    # a 2-px vertical gap between sky (left) and sea (right)
    pan = resolve(scene([("sky", rect(H, W, 0, 0, 29, H)), ("sea", rect(H, W, 31, 0, W, H))]), cfg)
    assert (pan.id_map > 0).all()
    assert cls_at(pan, 5, 29) == "sky"
    assert cls_at(pan, 5, 30) == "sea"


def test_gaps_can_stay_void():
    cfg = Config.load()
    cfg.data["resolve"]["fill"] = "void"
    pan = resolve(scene([("sky", rect(H, W, 0, 0, 29, H)), ("sea", rect(H, W, 31, 0, W, H))]), cfg)
    assert (pan.id_map[:, 29:31] == 0).all()


def test_fully_hidden_annotation_is_dropped(cfg):
    pan = resolve(scene([("person", rect(H, W, 0, 0, 30, 30)), ("sky", rect(H, W, 5, 5, 10, 10))]), cfg)
    assert [s.category for s in pan.segments] == ["person"]


def test_unknown_class_is_an_error(cfg):
    with pytest.raises(ValueError, match="unknown class"):
        resolve(scene([("unicorn", rect(H, W, 0, 0, 5, 5))]), cfg)


def test_priority_must_cover_every_class():
    cfg = Config.load()
    data = dict(cfg.data)
    data["resolve"] = dict(data["resolve"], priority=data["resolve"]["priority"][:-1])
    with pytest.raises(ValueError, match="missing"):
        Config(data)
