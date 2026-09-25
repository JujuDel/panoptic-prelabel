"""Panoptic quality and semantic IoU between a reference and a prediction.

The matching follows panopticapi (Kirillov et al., "Panoptic Segmentation",
CVPR 2019): a reference segment and a predicted segment of the same class
match when IoU > 0.5, which makes matches unique. Pixels that are void in the
reference are left out of the union, and a predicted segment lying mostly
(> 50 %) on reference void is not counted as a false positive.

Statistics are accumulated per class over *all* the image pairs and averaged
over classes at the end (pooled, as in COCO panoptic), then split into
things (PQ_th) and stuff (PQ_st). mIoU is the semantic view: per-class pixel
IoU pooled over the images, averaged over classes.

Used here to measure agreement between two versions of a label (automatic vs
human-corrected, this code vs the first version); when the reference is not
independent ground truth, read the numbers as agreement, not accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .panoptic import Panoptic

OFFSET = 256**3  # > any COCO panoptic id (24-bit RGB): packs (ref id, pred id) into one int64


@dataclass
class ClassStat:
    isthing: bool
    iou: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    inter_px: int = 0  # semantic: pixels both maps give this class (reference void excluded)
    union_px: int = 0

    @property
    def pq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou / d if d else 0.0

    @property
    def sq(self) -> float:
        return self.iou / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d else 0.0


@dataclass
class Evaluation:
    classes: dict[str, ClassStat] = field(default_factory=dict)
    pixels_same: int = 0
    pixels_total: int = 0
    images: int = 0

    def _stat(self, name: str, isthing: bool) -> ClassStat:
        if name not in self.classes:
            self.classes[name] = ClassStat(isthing)
        return self.classes[name]

    def add(self, ref: Panoptic, pred: Panoptic) -> None:
        """Accumulate one image pair (reference first)."""
        if ref.id_map.shape != pred.id_map.shape:
            raise ValueError(f"size mismatch: {ref.id_map.shape} vs {pred.id_map.shape}")
        self.images += 1
        r_ids, p_ids = ref.id_map.astype(np.int64), pred.id_map.astype(np.int64)
        cat_r = {s.id: s.category for s in ref.segments}
        cat_p = {s.id: s.category for s in pred.segments}
        for s in [*ref.segments, *pred.segments]:
            self._stat(s.category, s.isthing)

        # pixel-level (semantic) agreement, reference void excluded
        valid = r_ids > 0
        index = {n: i for i, n in enumerate(sorted(self.classes))}
        lut_r = np.full(max(int(r_ids.max()), *cat_r, 0) + 1, -1)
        for sid, c in cat_r.items():
            lut_r[sid] = index[c]
        lut_p = np.full(max(int(p_ids.max()), *cat_p, 0) + 1, -1)
        for sid, c in cat_p.items():
            lut_p[sid] = index[c]
        cr, cp = lut_r[r_ids][valid], lut_p[p_ids][valid]
        self.pixels_same += int((cr == cp).sum())
        self.pixels_total += int(valid.sum())
        for name, i in index.items():
            in_r, in_p = cr == i, cp == i
            self.classes[name].inter_px += int((in_r & in_p).sum())
            self.classes[name].union_px += int((in_r | in_p).sum())

        # segment-level (panoptic) matching
        keys, counts = np.unique(r_ids * OFFSET + p_ids, return_counts=True)
        inter = {(int(k) // OFFSET, int(k) % OFFSET): int(n) for k, n in zip(keys, counts, strict=True)}
        area_r = {s.id: s.area for s in ref.segments}
        area_p = {s.id: s.area for s in pred.segments}
        on_void = {pid: n for (rid, pid), n in inter.items() if rid == 0}
        matched_r, matched_p = set(), set()
        for (rid, pid), n in inter.items():
            if rid == 0 or pid == 0 or cat_r.get(rid) != cat_p.get(pid):
                continue
            iou = n / (area_r[rid] + area_p[pid] - n - on_void.get(pid, 0))
            if iou > 0.5:
                matched_r.add(rid)
                matched_p.add(pid)
                st = self.classes[cat_r[rid]]
                st.iou += iou
                st.tp += 1
        for s in ref.segments:
            if s.id not in matched_r:
                self.classes[s.category].fn += 1
        for s in pred.segments:
            if s.id not in matched_p and on_void.get(s.id, 0) / max(s.area, 1) <= 0.5:
                self.classes[s.category].fp += 1

    # ---------------------------------------------------------------- summary

    def _mean(self, attr: str, isthing: bool | None = None) -> float:
        vals = [
            getattr(c, attr)
            for c in self.classes.values()
            if (isthing is None or c.isthing == isthing) and (c.tp + c.fp + c.fn) > 0
        ]
        return float(np.mean(vals)) if vals else 0.0

    def summary(self) -> dict[str, float]:
        ious = [c.inter_px / c.union_px for c in self.classes.values() if c.union_px]
        return {
            "PQ": self._mean("pq"),
            "SQ": self._mean("sq"),
            "RQ": self._mean("rq"),
            "PQ_th": self._mean("pq", True),
            "PQ_st": self._mean("pq", False),
            "mIoU": float(np.mean(ious)) if ious else 0.0,
            "pixel_acc": self.pixels_same / self.pixels_total if self.pixels_total else 0.0,
        }


def evaluate(pairs: list[tuple[Panoptic, Panoptic]]) -> Evaluation:
    ev = Evaluation()
    for ref, pred in pairs:
        ev.add(ref, pred)
    return ev


def format_report(ev: Evaluation, per_class: bool = False) -> str:
    s = {k: 100 * v for k, v in ev.summary().items()}
    lines = [
        f"images           {ev.images}",
        f"PQ / SQ / RQ     {s['PQ']:.1f} / {s['SQ']:.1f} / {s['RQ']:.1f}",
        f"PQ things/stuff  {s['PQ_th']:.1f} / {s['PQ_st']:.1f}",
        f"mIoU             {s['mIoU']:.1f}",
        f"pixel accuracy   {s['pixel_acc']:.2f} %",
    ]
    if per_class:
        lines.append(f"{'class':<11} {'kind':<5} {'PQ':>5} {'SQ':>5} {'RQ':>5} {'IoU':>5}  TP FP FN")
        for name in sorted(ev.classes, key=lambda n: (not ev.classes[n].isthing, n)):
            c = ev.classes[name]
            iou = c.inter_px / c.union_px if c.union_px else 0.0
            lines.append(
                f"{name:<11} {'thing' if c.isthing else 'stuff':<5} {100 * c.pq:5.1f} {100 * c.sq:5.1f} "
                f"{100 * c.rq:5.1f} {100 * iou:5.1f} {c.tp:3d}{c.fp:3d}{c.fn:3d}"
            )
    return "\n".join(lines)
