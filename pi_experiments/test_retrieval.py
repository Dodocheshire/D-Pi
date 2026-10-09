import torch

from .region_retrieval import precision_at_five,regional_truth
from .retrieval_regions import visible_window


def test_effective_region_window_matches_clip_center_crop():
    assert visible_window([0,0,20,40])==[0.,10.,20.,30.]
    assert visible_window([0,0,40,20])==[10.,0.,30.,20.]


def test_retrieval_truth_requires_certainty_and_part_in_effective_crop():
    parts=torch.zeros(3,15,3);parts[:,1]=torch.tensor([10.,10.,1.])
    annotations={'test':{'certainty':torch.tensor([[3],[2],[4]]),'attributes':torch.tensor([[1],[1],[1]]),'parts':parts}}
    regions={'image_positions':torch.arange(3),'visible_windows':torch.tensor([[0.,0.,20.,20.],[0.,0.,20.,20.],[10.,10.,10.,20.]])}
    known,truth=regional_truth(0,'has_bill_shape::curved',annotations,regions)
    assert known.tolist()==[True,False,True]
    assert truth.tolist()==[True,True,False]


def test_retrieval_skips_unknown_and_reports_distinct_images():
    scores=torch.tensor([10.,9.,8.,7.,6.,5.,4.]);known=torch.tensor([False,True,True,True,True,True,True])
    truth=torch.tensor([True,True,False,True,False,True,True]);images=torch.tensor([0,0,0,1,2,3,4])
    result=precision_at_five(scores,known,truth,images)
    assert result['top_region_indices']==[1,2,3,4,5]
    assert abs(result['precision_at_5']-.6)<1e-6
    assert abs(result['precision_at_5_distinct_images']-.8)<1e-6
    assert result['top_distinct_image_region_indices']==[1,3,4,5,6]
