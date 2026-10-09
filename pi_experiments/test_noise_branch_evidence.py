"""Verify that cached branch diagnostics preserve outputs and coupled randomness."""

import torch

from .analyze import collect
from .model import DPi, DProjection
from .noise_branch_evidence import CachedParameters, SCENARIOS
from .noise_diagnostics import NoiseOverride


def model():
    torch.manual_seed(27)
    projection = DProjection(torch.randn(7, 5), torch.full((7,), .1), torch.full((7,), .08))
    result = DPi(projection, torch.randn(3, 7), torch.randn(3), torch.ones(5)).eval()
    with torch.no_grad():
        result.generator.mean.weight.normal_(0, .2)
        result.generator.log_scale.weight.normal_(0, .03)
    return result


def test_cached_scenarios_preserve_all_outputs_and_rng():
    m = model();base = m.generator
    split = {'features': torch.randn(11, 5)}
    before = torch.random.get_rng_state().clone()
    reference = {}
    for name, alpha, beta in SCENARIOS:
        m.generator = NoiseOverride(base, alpha, beta).eval()
        reference[name] = collect(m, split, samples=4, seed=3010, batch_size=4)
    cached = CachedParameters(base).eval()
    for name, alpha, beta in SCENARIOS:
        m.generator = NoiseOverride(cached, alpha, beta).eval()
        actual = collect(m, split, samples=4, seed=3010, batch_size=4)
        for key, value in actual.items():
            assert torch.equal(value, reference[name][key]), (name, key)
    assert len(cached.cache) == 3
    assert torch.equal(before, torch.random.get_rng_state())


def test_mutated_input_does_not_reuse_stale_parameters():
    m = model();cached = CachedParameters(m.generator).eval()
    features = torch.randn(4, 5)
    with torch.no_grad():
        old, _ = cached.parameters_for(features)
        features.add_(torch.randn_like(features))
        actual = cached.parameters_for(features)
        expected = m.generator.parameters_for(features)
    assert not torch.equal(old, actual[0])
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_deterministic_repeated_mean_uses_a_float_error_bound():
    m = model();base = m.generator
    split = {'features': torch.randn(100, 5)}
    single = collect(m, split, samples=1, mean=True)['probabilities']
    m.generator = NoiseOverride(CachedParameters(base).eval(), 1, 0).eval()
    repeated = collect(m, split, samples=16, seed=3010)['probabilities']
    torch.testing.assert_close(single, repeated, rtol=1e-6, atol=1e-7)
    assert torch.equal(single.argmax(-1), repeated.argmax(-1))
