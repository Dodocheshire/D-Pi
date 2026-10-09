"""Check balanced class weighting, unknown labels and shared-image uncertainty."""

import numpy as np

from .semantic_uncertainty import paired_attribute_bootstrap


def test_balanced_accuracy_does_not_reduce_to_unbalanced_task_accuracy():
    positive = np.array([1,1,0,0,0,0,0,0,0,0,1,1],dtype=bool)[:,None]
    known = np.ones_like(positive);known[-2:]=False
    clean = np.zeros((5,12,1),dtype=bool)
    adapted = np.broadcast_to(positive,clean.shape).copy()
    # Unknown attribute values must not contribute even when their predictions differ.
    result = paired_attribute_bootstrap(clean,adapted,positive,known,draws=300)
    assert result['balanced_accuracy_delta'] == .5  # Ordinary accuracy improves by .2.
    np.testing.assert_allclose(result['ci95'],[.5,.5])
    assert result['undefined_draws'] > 0 and result['valid_draws']+result['undefined_draws']==300
    adapted[:,-2:] = ~adapted[:,-2:]
    assert paired_attribute_bootstrap(clean,adapted,positive,known,draws=300) == result


def test_common_image_resample_preserves_opposite_attribute_effects():
    labels = np.array([1]*4+[0]*8,dtype=bool)
    positive = np.stack([labels,labels],axis=-1);known=np.ones_like(positive)
    clean = np.zeros((5,12,2),dtype=bool);clean[:,:,1]=True
    adapted = clean.copy();adapted[:,[0,1,4],0]=True;adapted[:,:,1]=~adapted[:,:,0]
    # Each attribute changes on different classes; their paired macro effects cancel.
    result = paired_attribute_bootstrap(clean,adapted,positive,known,draws=300)
    assert result['balanced_accuracy_delta'] == 0
    np.testing.assert_allclose(result['ci95'],[0,0],atol=1e-15)
