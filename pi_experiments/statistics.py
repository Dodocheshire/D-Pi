"""Paired uncertainty across adaptation seeds and shared test images."""

import numpy as np


def hierarchical_paired_bootstrap(controls,adapted,labels,seed=731,draws=2000):
    labels=np.asarray(labels)
    controls,adapted=np.asarray(controls),np.asarray(adapted)
    assert controls.shape==adapted.shape and controls.ndim==3
    assert controls.shape[1]==len(labels)
    differences=(adapted.argmax(-1)==labels).astype(float)-(controls.argmax(-1)==labels).astype(float)
    seeds,images=differences.shape
    rng=np.random.default_rng(seed)
    values=np.empty(draws)
    for start in range(0,draws,50):
        batch=min(50,draws-start)
        seed_ids=rng.integers(0,seeds,size=(batch,seeds))
        per_image=differences[seed_ids].mean(1)
        # One image resample per replicate, shared by all selected seeds.
        image_ids=rng.integers(0,images,size=(batch,images))
        values[start:start+batch]=np.take_along_axis(per_image,image_ids,axis=1).mean(1)
    return {'accuracy_delta':float(differences.mean()),'ci95':np.quantile(values,[0.025,0.975]).tolist(),
            'adaptation_seeds':seeds,'test_images':images,'draws':draws,'seed':seed,
            'protocol':'resample paired adaptation seeds and a common image sample; fixed prototypes and initial classifier head'}
