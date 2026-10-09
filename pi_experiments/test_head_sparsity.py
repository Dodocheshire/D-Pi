import torch
from torch.nn import functional as F

from .head_sparsity import grid_loss


def test_head_grid_gradients_match_independent_heads():
    torch.manual_seed(12)
    x = torch.randn(7, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    weights = torch.randn(5, 3, 4, requires_grad=True)
    biases = torch.randn(5, 3, requires_grad=True)
    lambdas = torch.tensor([1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    logits = torch.einsum('nk,lck->lnc', x, weights) + biases[:, None, :]
    grid_loss(logits, labels, weights, lambdas).backward()
    for i in range(5):
        w = weights[i].detach().clone().requires_grad_()
        b = biases[i].detach().clone().requires_grad_()
        (F.cross_entropy(x @ w.T + b, labels) + lambdas[i] * w.abs().sum()).backward()
        torch.testing.assert_close(weights.grad[i], w.grad)
        torch.testing.assert_close(biases.grad[i], b.grad)
