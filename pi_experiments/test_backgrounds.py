import numpy as np
import torch
from PIL import Image

from .backgrounds import frozen_attribute_metrics, remove_foreground, replace_background


def test_replacement_preserves_foreground_bytes():
    pixels = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    mask = np.zeros((8, 9), dtype=bool)
    mask[2:6, 3:7] = True
    changed = np.asarray(replace_background(Image.fromarray(pixels), mask, Image.new('RGB', (4, 4), (255, 0, 0))))
    assert np.array_equal(changed[mask], pixels[mask])
    assert np.all(changed[~mask] == np.array([255, 0, 0]))


def test_donor_removal_uses_only_background_pixels():
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True
    pixels = np.zeros((9, 9, 3), dtype=np.uint8)
    pixels[mask] = [255, 0, 0]
    pixels[~mask] = [0, 0, 255]
    changed = np.asarray(remove_foreground(Image.fromarray(pixels), mask))
    assert np.all(changed == np.array([0, 0, 255]))


def test_background_attribute_threshold_is_frozen():
    reference = {'details': [{'attribute': 0, 'name': 'color', 'prototypes': [0], 'threshold': 10.}]}
    annotations = {'test': {'attributes': torch.tensor([[0], [0], [1], [1]]),
                            'certainty': torch.full((4, 1), 3)}}
    concepts = torch.tensor([[0.], [1.], [2.], [3.]])
    report = frozen_attribute_metrics(concepts, reference, annotations)
    assert report['details'][0]['threshold'] == 10.
    assert report['attribute_macro_balanced_accuracy'] == .5
    assert report['attribute_macro_auroc'] == 1.
