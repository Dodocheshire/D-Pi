import torch

from .model import DPi,DProjection
from .rotated_bank_train import RotatedBankDPi


def test_rotation_preserves_gram_initial_parameters_rng_and_projection():
    torch.manual_seed(12)
    projection = DProjection(torch.randn(7,11),torch.zeros(7),torch.ones(7))
    weight,bias = torch.randn(3,7),torch.randn(3)
    for seed in range(5):
        torch.manual_seed(seed);real = DPi(projection,weight,bias,torch.ones(11));state = torch.get_rng_state().clone()
        torch.manual_seed(seed);rotated = RotatedBankDPi(projection,weight,bias,torch.ones(11),conditioning='random_concept')
        assert torch.equal(state,torch.get_rng_state())
        for (na,a),(nb,b) in zip(real.named_parameters(),rotated.named_parameters()):assert na==nb and torch.equal(a,b)
        x,y = real.generator.prototypes,rotated.generator.prototypes
        assert not torch.equal(x,y)
        torch.testing.assert_close(x@x.T,y@y.T)
        assert torch.equal(real.projection.centers,rotated.projection.centers)
