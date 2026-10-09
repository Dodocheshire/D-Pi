"""A named part in the proposal can be absent from the encoded CLIP region."""

import pytest
import torch

from .visible_semantics import filtered_assignments


def test_mapping_support_must_survive_clip_crop_and_use_training_source():
    pack={'metadata':{'concept_sources':[(7,[0,0,20,40]),(7,[0,0,20,40])]}}
    parts=torch.zeros(1,15,3)
    parts[0,8]=torch.tensor([10,5,1])  # In the rectangle, outside the encoded square.
    parts[0,13]=torch.tensor([10,20,1])
    annotations={'attribute_names':['has_wing_color::black','has_tail_color::black'],
                 'train':{'parts':parts,'attributes':torch.ones(1,2),'certainty':torch.full((1,2),3)}}
    original={0:[0],1:[1]}
    assigned,removed=filtered_assignments(pack,annotations,{'train_indices':[7]},original)
    assert assigned=={1:[1]} and [r['concept'] for r in removed]==[0]
    assert original=={0:[0],1:[1]}  # The original mapping is retained.
    with pytest.raises(KeyError):
        filtered_assignments(pack,annotations,{'train_indices':[8]},original)
