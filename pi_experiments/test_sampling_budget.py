import torch

from .model import DPi, DProjection
from .sampling_budget import BUDGETS, collect_budgets


def test_sampling_budgets_use_probability_prefixes_and_restore_rng():
    torch.manual_seed(51)
    projection = DProjection(torch.randn(4,3),torch.zeros(4),torch.ones(4))
    model = DPi(projection,torch.randn(2,4),torch.zeros(2),torch.ones(3),initial_scale=.4)
    split = {'features':torch.randn(7,3),'labels':torch.zeros(7,dtype=torch.long)}
    rng = torch.get_rng_state().clone()
    probabilities, diagnostics = collect_budgets(model,split,seed=9,batch_size=7)
    assert torch.equal(rng,torch.get_rng_state())
    torch.manual_seed(9)
    draws = torch.stack([model(split['features'])[0].softmax(-1) for _ in range(16)])
    for n in BUDGETS:
        torch.testing.assert_close(probabilities[str(n)],draws[:n].mean(0))
    assert diagnostics['1']['sampling_top5_stability']==1
    assert diagnostics['1']['concept_sampling_variance']==0
