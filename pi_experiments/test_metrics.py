"""Check calibration separation and undefined semantic evaluation handling."""

import json

import numpy as np
import torch

from .cub_metrics import attribute_metrics, attribute_scores


def test_corruption_randomness_is_paired_and_does_not_change_global_rng():
    from PIL import Image
    from .corruptions import corrupt

    image=Image.fromarray(np.full((32,32,3),128,dtype=np.uint8))
    np.random.seed(14)
    expected=np.random.random()
    np.random.seed(14)
    first=np.asarray(corrupt(image,'gaussian_noise',0.05,index=21))
    assert np.random.random()==expected
    assert np.array_equal(first,np.asarray(corrupt(image,'gaussian_noise',0.05,index=21)))
    assert not np.array_equal(first,np.asarray(corrupt(image,'gaussian_noise',0.05,index=22)))
    assert first.shape==(32,32,3)


def annotations(test_targets):
    return {'attribute_names':['has_wing_color::black'],
            'val':{'attributes':torch.tensor([[0],[1]]*5),
                   'certainty':torch.full((10,1),3)},
            'test':{'attributes':torch.tensor(test_targets).reshape(-1,1),
                    'certainty':torch.full((len(test_targets),1),3)}}


def test_attribute_threshold_uses_validation_only():
    val=np.array([[0.1],[0.8]]*5,dtype=np.float32)
    test=np.array([[0.2],[0.9],[0.3],[0.7]],dtype=np.float32)
    first=attribute_metrics(val,test,annotations([0,1,0,1]),{0:[0]})
    second=attribute_metrics(val,test,annotations([1,0,1,0]),{0:[0]})
    assert first['details'][0]['threshold']==second['details'][0]['threshold']
    assert first['attribute_macro_auroc']==1
    assert second['attribute_macro_auroc']==0


def test_constant_scores_produce_finite_all_negative_threshold():
    scores=np.zeros((10,1),dtype=np.float32)
    result=attribute_metrics(scores,scores,annotations([0,1]*5),{0:[0]})
    assert result['attribute_macro_balanced_accuracy']==0.5
    threshold=result['details'][0]['threshold']
    assert np.isfinite(threshold) and not (scores>=threshold).any()
    json.dumps(result,allow_nan=False)


def test_no_mapped_attribute_is_reported_as_unavailable():
    scores=attribute_scores(torch.zeros(10,4),{})
    result=attribute_metrics(scores,scores,annotations([0,1]*5),{})
    assert scores.shape==(10,0)
    assert result['evaluated_attributes']==0
    assert result['attribute_macro_balanced_accuracy'] is None
    assert result['attribute_macro_auroc'] is None
    json.dumps(result,allow_nan=False)


def test_guessed_attributes_are_excluded_from_primary_evaluation():
    scores=np.array([[0],[1]]*5,dtype=np.float32)
    data=annotations([0,1]*5)
    data['val']['certainty'].fill_(2);data['test']['certainty'].fill_(2)
    primary=attribute_metrics(scores,scores,data,{0:[0]})
    sensitivity=attribute_metrics(scores,scores,data,{0:[0]},min_certainty=2)
    assert primary['evaluated_attributes']==0 and primary['attribute_macro_auroc'] is None
    assert sensitivity['evaluated_attributes']==1 and sensitivity['attribute_macro_auroc']==1
def test_crop_part_uses_exclusive_pil_right_and_bottom():
    from .cub_metrics import crop_contains_part
    assert crop_contains_part([10,20,1],[10,20,30,40])
    assert not crop_contains_part([30,20,1],[10,20,30,40])
    assert not crop_contains_part([10,40,1],[10,20,30,40])
    assert not crop_contains_part([11,21,0],[10,20,30,40])
