import numpy as np
import pytest

from panoptic_prelabel.coco_io import Annotation, ImageAnnotations
from panoptic_prelabel.config import Config


@pytest.fixture
def cfg() -> Config:
    return Config.load()


def rect(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def scene(anns, h=40, w=60) -> ImageAnnotations:
    """Annotations given as (class, mask) pairs; source ids are 1..n."""
    return ImageAnnotations(
        "synthetic.jpg", w, h, [Annotation(c, m, source_id=i) for i, (c, m) in enumerate(anns, start=1)]
    )
