import torch

from .model import DPi, DProjection
from .random_bank_train import MatchedBankDPi


def make(cls, conditioning):
    centers = torch.arange(1, 25).float().reshape(6, 4)
    projection = DProjection(centers, torch.zeros(6), torch.ones(6))
    return cls(projection, torch.ones(3, 6), torch.zeros(3), torch.ones(4),
               conditioning=conditioning, hidden=8)


def test_random_bank_matches_parameter_initialization_and_rng():
    torch.manual_seed(3)
    real = make(DPi, 'concept')
    original_rng = torch.random.get_rng_state()
    torch.manual_seed(3)
    random = make(MatchedBankDPi, 'random_concept')
    assert torch.equal(original_rng, torch.random.get_rng_state())
    for (name, value), (other_name, other) in zip(real.named_parameters(), random.named_parameters()):
        assert name == other_name and torch.equal(value, other)
    assert torch.equal(real.projection.centers, random.projection.centers)
    assert not torch.equal(real.generator.prototypes, random.generator.prototypes)
    torch.testing.assert_close(random.generator.prototypes.norm(dim=-1), torch.ones(6))


def test_matched_bank_keeps_frozen_projection_and_head():
    torch.manual_seed(2)
    model = make(MatchedBankDPi, 'random_concept')
    logits, _, _, energy = model(torch.randn(5, 4))
    (logits.square().mean() + energy).backward()
    assert model.generator.mean.weight.grad is not None
    assert model.head.weight.grad is None
    assert model.projection.centers.grad is None
