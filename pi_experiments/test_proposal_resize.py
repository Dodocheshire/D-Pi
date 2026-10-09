import cv2
import numpy as np
import pytest
from PIL import Image

from .prepare import resize_sam_image
from .backbone_prepare import proposal_image


def test_long_side_resize_matches_official_helper_pixels_and_floor_geometry():
    pixels = np.arange(23*37*3,dtype=np.uint8).reshape(23,37,3)
    image = Image.fromarray(pixels)
    resized = resize_sam_image(image,16)
    expected = cv2.resize(pixels,(16,int(23*16/37)))
    assert resized.size==(16,9)
    assert np.array_equal(np.asarray(resized),expected)
    assert resize_sam_image(image,0) is image
    assert resize_sam_image(image,1024) is image


def test_backbone_crop_reconstruction_uses_recorded_source_and_proposal_frames():
    image = Image.fromarray(np.arange(23*37*3,dtype=np.uint8).reshape(23,37,3))
    record = {'source_image_size':[37,23],'image_size':[16,9]}
    reconstructed = proposal_image(image,record,{'max_long_side':16})
    assert np.array_equal(np.asarray(reconstructed),np.asarray(resize_sam_image(image,16)))
    with pytest.raises(AssertionError,match='Original source'):
        proposal_image(Image.new('RGB',(23,37)),record,{'max_long_side':16})
    with pytest.raises(AssertionError,match='different image coordinates'):
        proposal_image(image,{'image_size':[16,9]}, {})
