"""Validate inference caching for all core corruption comparison modes."""

import pytest
import torch

from .analyze import collect
from .corruptions import cache_cpu_generator
from .model import DPi, DProjection


@pytest.mark.parametrize('mode', ['none', 'fixed', 'mean_only', 'adapter', 'mean_scale'])
def test_cpu_cache_preserves_every_core_output_and_random_state(mode):
    torch.manual_seed(19)
    projection = DProjection(torch.randn(8, 5), torch.full((8,), .125), torch.full((8,), .07))
    model = DPi(projection, torch.randn(3, 8), torch.randn(3), torch.ones(5), mode=mode).eval()
    if model.generator is not None:
        with torch.no_grad():
            model.generator.mean.weight.normal_(0, .1)
            model.generator.log_scale.weight.normal_(0, .03)
    split = {'features': torch.randn(13, 5)}
    before = torch.random.get_rng_state().clone()
    expected = collect(model, split, samples=8, seed=4091, batch_size=5)
    enabled = cache_cpu_generator(model, 'cpu')
    assert enabled == (mode in ['mean_only', 'adapter', 'mean_scale'])
    actual = collect(model, split, samples=8, seed=4091, batch_size=5)
    assert set(actual) == set(expected)
    for key in actual:
        assert torch.equal(actual[key], expected[key]), (mode, key)
    assert torch.equal(before, torch.random.get_rng_state())
