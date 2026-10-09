"""Validate localization accounting against known heatmap geometry."""

import numpy as np
import torch

from .localization import foreground_metrics, grid_plan, part_points, quadrant_metrics


def test_grid_distractors_and_position_balance():
    labels=np.repeat(np.arange(5),9)
    plan=grid_plan(labels)
    assert plan==grid_plan(labels)
    counts=np.bincount([p['target_quadrant'] for p in plan],minlength=4)
    assert counts.max()-counts.min()<=1
    for row,item in enumerate(plan):
        assert item['indices'][item['target_quadrant']]==row
        assert len(set(labels[item['indices']]))==4


def test_localized_cam_and_flat_ties_have_expected_credit():
    cam=torch.zeros(8,8)
    cam[4:,4:]=1
    result=quadrant_metrics(cam,3)
    assert result['relative_attribution']==1
    assert result['max_point_accuracy']==1
    assert result['quadrant_gini_normalized']==1
    for quadrant in range(4):
        flat=quadrant_metrics(torch.ones(8,8),quadrant)
        assert flat['max_point_accuracy']==0.25
        assert flat['relative_attribution']==0.25
        assert flat['quadrant_gini']==0
    assert quadrant_metrics(torch.zeros(8,8),0) is None


def test_part_coordinates_respect_center_crop_visibility():
    parts=torch.zeros(15,3)
    parts[1]=torch.tensor([200,100,1])
    assert part_points(parts,'has_bill_shape::curved',(400,200))==[[112,112]]
    parts[1]=torch.tensor([1,100,1])
    assert part_points(parts,'has_bill_shape::curved',(400,200))==[]


def test_foreground_mass_and_tied_maxima():
    mask=torch.zeros(8,8,dtype=torch.bool)
    mask[4:,4:]=True
    flat=foreground_metrics(torch.ones(8,8),mask)
    assert flat['foreground_attribution']==.25
    assert flat['foreground_max_point']==.25
    localized=foreground_metrics(mask.float(),mask)
    assert localized['foreground_attribution']==1
    assert localized['foreground_max_point']==1
    assert foreground_metrics(torch.zeros(8,8),mask) is None
    assert foreground_metrics(torch.ones(8,8),torch.zeros_like(mask)) is None
