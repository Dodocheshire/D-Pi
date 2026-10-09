import numpy as np

from .natural_backgrounds import empty_rectangle


def test_natural_patch_contains_no_foreground_and_has_usable_geometry():
    mask=np.zeros((160,200),dtype=bool);mask[40:120,60:140]=True
    box=empty_rectangle(mask,minimum=32)
    assert box is not None
    x1,y1,x2,y2=box
    assert not mask[y1:y2,x1:x2].any()
    assert min(x2-x1,y2-y1)>=32
    assert max((x2-x1)/(y2-y1),(y2-y1)/(x2-x1))<=2


def test_no_patch_available_when_every_pixel_is_foreground():
    assert empty_rectangle(np.ones((160,200),dtype=bool)) is None
