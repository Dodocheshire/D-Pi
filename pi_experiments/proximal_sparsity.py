"""Fit genuinely sparse heads with full-gradient proximal optimization."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from . import head_sparsity as base
from .resources import gpu_lease
from .suite import Runner


LAMBDAS = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def soft_threshold(values, threshold):
    return values.sign() * (values.abs() - threshold).clamp_min(0)


def fit_grid(train, labels, validation, validation_labels, weight, bias, epochs, seed):
    count = len(LAMBDAS)
    weights = weight.expand(count, -1, -1).clone()
    biases = bias.expand(count, -1).clone()
    lambdas = weights.new_tensor(LAMBDAS)[:, None, None]
    augmented = torch.cat([train, train.new_ones(len(train), 1)], -1)
    # Softmax covariance has spectral norm <=1/2. Including the intercept
    # in this Gram matrix bounds the joint weight/bias Hessian for every head.
    gram = augmented.T @ augmented / len(train)
    step = 1 / (0.5 * torch.linalg.eigvalsh(gram.double())[-1].item())
    best_nll = weights.new_full((count,), float('inf'))
    best_weight, best_bias = weights.clone(), biases.clone()
    best_epoch = torch.zeros(count, dtype=torch.long, device=train.device)
    current_w, current_b = weights.clone(), biases.clone()
    acceleration = 1.0; history = []
    started = time.monotonic()
    for iteration in range(epochs + 1):
        if iteration % 25 == 0 or iteration == epochs:
            with torch.no_grad():
                logits = torch.einsum('dnk,lck->ldnc', validation, weights) + biases[:, None, None, :]
                probabilities = logits.softmax(-1).mean(1)
                nll = -probabilities[:, torch.arange(len(validation_labels), device=train.device),
                                     validation_labels].clamp_min(1e-12).log().mean(-1)
                improved = nll < best_nll
                best_nll[improved] = nll[improved]
                best_weight[improved] = weights[improved]
                best_bias[improved] = biases[improved]
                best_epoch[improved] = iteration
                history.append({'iteration': iteration, 'val_nll': nll.tolist(),
                                'nonzero_per_class': weights.ne(0).float().sum(-1).mean(-1).tolist()})
            if iteration % 250 == 0:
                print('proximal iterations', iteration, 'best NLL', best_nll.tolist(),
                      'elapsed', round(time.monotonic() - started, 1), flush=True)
        if iteration == epochs:
            break
        w = current_w.detach().requires_grad_(); b = current_b.detach().requires_grad_()
        logits = torch.einsum('nk,lck->lnc', train, w) + b[:, None, :]
        loss = F.cross_entropy(logits.reshape(-1, weight.shape[0]), labels.repeat(count), reduction='sum') / len(train)
        grad_w, grad_b = torch.autograd.grad(loss, (w, b))
        with torch.no_grad():
            next_w = soft_threshold(w - step * grad_w, step * lambdas)
            next_b = b - step * grad_b
            next_acceleration = (1 + math.sqrt(1 + 4 * acceleration ** 2)) / 2
            momentum = (acceleration - 1) / next_acceleration
            current_w = next_w + momentum * (next_w - weights)
            current_b = next_b + momentum * (next_b - biases)
            weights, biases = next_w, next_b
            acceleration = next_acceleration
    return best_weight, best_bias, best_nll, best_epoch, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=2000, help='Full-gradient proximal iterations; validation every 25.')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); args.dataset = 'cifar10'
    torch.set_num_threads(8)
    base.LAMBDAS = LAMBDAS
    base.fit_grid = fit_grid
    runner = Runner('artifacts/logs/proximal_sparsity_v1')
    pack = torch.load('artifacts/datasets/cifar10/baseline_k256_seed0.pt', weights_only=True, map_location='cpu')
    root = Path('outputs/proximal_sparsity_v1/cifar10')
    for variant in ['clean', 'pi']:
        for seed in range(5):
            run = root / f'{variant}_seed{seed}'
            if (run / 'head_grid.json').exists():
                continue
            runner.state.update(phase='waiting_for_gpu', next_case=run.name); runner.save()
            with gpu_lease(f'Proximal L1 head {run.name}', args.device):
                runner.state.update(phase=run.name); runner.save()
                base.run_case(pack, run, variant, seed, args)
                path = run / 'head_grid.json'; result = json.loads(path.read_text())
                result['solver'] = {'name': 'Full-gradient FISTA; exact soft threshold; bias unpenalized',
                                    'max_iterations': args.epochs, 'validation_interval': 25,
                                    'lipschitz_bound': 'Half the largest eigenvalue of augmented training Gram',
                                    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
                heads = torch.load(run / 'heads.pt', weights_only=True, map_location='cpu')
                for candidate, weights in zip(result['candidates'], heads['weights']):
                    candidate['nec_exact_nonzero'] = weights.ne(0).float().sum(-1).mean().item()
                result['selected'] = result['candidates'][result['selection']['selected_index']]
                path.write_text(json.dumps(result, indent=2, allow_nan=False))
            runner.state['completed_commands'].append({'name': run.name, 'returncode': 0, 'finished_at': time.time()})
            runner.save()
    base.summarize(root, Path('reports/proximal_sparsity_v1/cifar10'))
    runner.state.update(phase='complete', finished_at=time.time()); runner.save()


if __name__ == '__main__':
    main()
