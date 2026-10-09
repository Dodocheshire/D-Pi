"""Guard the upstream center-versus-corner crop error in detector proposals."""

import torch
from transformers.image_transforms import center_to_corners_format

from .gdino import crop_bounds


def test_detector_center_boxes_become_native_corner_crops():
    normalized=torch.tensor([0.5,0.5,0.4,0.2])
    corners=(center_to_corners_format(normalized)*torch.tensor([100,100,100,100])).round().tolist()
    assert crop_bounds(corners,(100,100))==(30,40,70,60)
    assert crop_bounds([-3,-4,105,110],(100,100))==(0,0,100,100)
    assert crop_bounds([90,90,80,80],(100,100)) is None
