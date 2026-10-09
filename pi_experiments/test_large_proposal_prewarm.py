"""Check rejection of source/frame corruption before large-cache reuse."""

import copy

import pytest
import torch

from .large_proposal_prewarm import verify_record


def sample():
    return {'source_index': 7, 'label': 2, 'source_image_size': [2001, 999],
            'image_size': [1024, 511], 'features': torch.ones(1, 1024),
            'boxes': torch.tensor([[0, 0, 1024, 511]])}


def test_stale_source_or_label_is_rejected():
    for field, value in [('source_index', 8), ('label', 3)]:
        record = sample()
        record[field] = value
        with pytest.raises(AssertionError):
            verify_record(record, 7, 2, (2001, 999))


def test_original_frame_boxes_cannot_be_used_as_capped_frame_boxes():
    record = sample()
    assert verify_record(record, 7, 2, (2001, 999)) == 1
    record['boxes'] = torch.tensor([[0, 0, 2001, 999]])
    with pytest.raises(AssertionError, match='exceeds proposal frame'):
        verify_record(record, 7, 2, (2001, 999))
    record = sample()
    record['image_size'] = [1024, 512]
    with pytest.raises(AssertionError, match='capped proposal frame'):
        verify_record(record, 7, 2, (2001, 999))


def test_empty_proposals_are_explicit_and_nonfinite_embeddings_rejected():
    record = sample()
    empty = copy.deepcopy(record)
    empty.update(features=torch.empty(0, 1024), boxes=torch.empty(0, 4))
    assert verify_record(empty, 7, 2, (2001, 999)) == 0
    record['features'][0, 0] = float('nan')
    with pytest.raises(AssertionError):
        verify_record(record, 7, 2, (2001, 999))
