import pytest
import torch

from .matched_nec import common_targets,truncate_exact


def test_common_nec_targets_exclude_unsupported_counts_in_any_class():
    a=torch.ones(2,12)
    b=torch.ones(2,12);b[0,7:]=0;b[1,9:]=1e-6
    assert common_targets([a,b])==[1,5,7]
    for k in common_targets([a,b]):
        for weight in [a,b]:
            actual=truncate_exact(weight,k)
            assert (actual.ne(0).sum(-1)==k).all()
            assert ((actual.abs()>1e-5).sum(-1)==k).all()
    with pytest.raises(AssertionError):truncate_exact(b,8)
