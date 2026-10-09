import torch

from .rendition_eval import restricted_probabilities


def test_restrict_each_draw_before_averaging():
    logits = torch.tensor([[2., 0., 20.], [0., 2., -20.]])
    allowed = torch.tensor([0, 1])
    restricted = restricted_probabilities(logits, allowed).mean(0)
    torch.testing.assert_close(restricted, torch.tensor([.5, .5]))
    # Averaging full thousand-class probabilities and then restricting weights
    # draws by their unrelated excluded-class mass, changing the result.
    wrong = logits.softmax(-1).mean(0)[allowed]
    wrong /= wrong.sum()
    assert not torch.allclose(restricted, wrong)
