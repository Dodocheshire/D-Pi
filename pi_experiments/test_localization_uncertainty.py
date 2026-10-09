"""Tests for eligible-concept weighting and common image pairing."""

import numpy as np

from .localization_uncertainty import paired_ratio_bootstrap


def test_ratio_weights_concepts_instead_of_averaging_image_rates():
    # Image rates [1,0] average to 1/2, but one hit over ten concepts is 1/10.
    denominator=np.array([[1.],[9.]])
    control=np.zeros((2,2,1));adapted=np.array([[[1.],[0.]],[[1.],[0.]]])
    result=paired_ratio_bootstrap(control,adapted,denominator,draws=50)
    assert abs(result['metrics'][0]['delta']-.1)<1e-12
    assert result['eligible_concepts_per_metric']==[10.]


def test_identical_paired_models_have_zero_interval():
    denominator=np.ones((4,1))
    control=np.array([[[1.],[0.],[1.],[0.]],[[0.],[1.],[0.],[1.]]])
    result=paired_ratio_bootstrap(control,control.copy(),denominator,draws=75)
    assert result['metrics'][0]['delta']==0
    assert result['metrics'][0]['ci95']==[0.,0.]
    assert result['metrics'][0]['valid_draws']==75


def test_common_images_preserve_opposite_metric_effects():
    denominator=np.ones((3,2))
    control=np.array([[[0.,1.],[0.,1.],[1.,0.]]])
    adapted=1-control
    result=paired_ratio_bootstrap(control,adapted,denominator,draws=175)
    left,right=result['metrics']
    assert abs(left['delta']+right['delta'])<1e-12
    np.testing.assert_allclose(left['ci95'],[-right['ci95'][1],-right['ci95'][0]],rtol=0,atol=1e-12)


def test_zero_eligibility_replicas_are_disclosed():
    denominator=np.array([[0.],[0.],[1.]])
    control=np.zeros((1,3,1));adapted=np.array([[[0.],[0.],[1.]]])
    result=paired_ratio_bootstrap(control,adapted,denominator,draws=200)
    metric=result['metrics'][0]
    assert metric['delta']==1 and metric['ci95']==[1.,1.]
    assert 0<metric['undefined_draws']<200
    assert metric['undefined_draws']+metric['valid_draws']==200
