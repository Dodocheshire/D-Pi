import torch

from .proximal_sparsity import soft_threshold
from . import proximal_sparsity


def test_proximal_l1_solution_and_kkt():
    # Exact minimizer of 0.5||w-v||² + lambda||w||₁, including zeros.
    v = torch.tensor([-3., -0.5, 0., 0.5, 3.])
    penalty = 1.
    w = soft_threshold(v, penalty)
    torch.testing.assert_close(w, torch.tensor([-2., 0., 0., 0., 2.]))
    active = w != 0
    torch.testing.assert_close((w-v)[active] + penalty*w[active].sign(), torch.zeros(2))
    assert ((w-v)[~active].abs() <= penalty).all()
    torch.testing.assert_close(soft_threshold(v, 0), v)


def test_full_solver_matches_binary_regularized_optimum(monkeypatch):
    monkeypatch.setattr(proximal_sparsity, 'LAMBDAS', [0.1])
    train = torch.cat([torch.ones(10, 1), -torch.ones(10, 1)])
    labels = torch.cat([torch.ones(10), torch.zeros(10)]).long()
    # Validation frequencies match the regularized solution's 0.9 confidence.
    val_labels = torch.tensor([1]*9+[0]+[0]*9+[1])
    weight, bias, nll, _, _ = proximal_sparsity.fit_grid(
        train, labels, train.unsqueeze(0), val_labels, torch.zeros(2, 1), torch.zeros(2), 200, 0)
    expected = torch.log(torch.tensor(9.)) / 2
    torch.testing.assert_close(weight[0, :, 0], torch.stack([-expected, expected]), atol=0.003, rtol=0)
    # A common intercept shift leaves softmax probabilities unchanged.
    torch.testing.assert_close(bias[0] - bias[0].mean(), torch.zeros(2), atol=1e-5, rtol=0)
    assert nll[0] < 0.326
