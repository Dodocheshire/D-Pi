"""Validation-select L1 after fixed-checkpoint head refits and compare equal sparsity."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from .analyze import load_model, task_metrics
from .prepare import atomic_save
from .resources import gpu_lease
from .statistics import hierarchical_paired_bootstrap
from .suite import Runner


LAMBDAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]


def grid_loss(logits, labels, weights, lambdas):
    heads, batch, classes = logits.shape
    return (F.cross_entropy(logits.reshape(-1, classes), labels.repeat(heads), reduction='sum') / batch
            + (weights.abs().sum((1, 2)) * lambdas).sum())


def fit_grid(train, labels, validation, validation_labels, weight, bias, epochs, seed):
    count = len(LAMBDAS)
    weights = torch.nn.Parameter(weight.expand(count, -1, -1).clone())
    biases = torch.nn.Parameter(bias.expand(count, -1).clone())
    lambdas = weights.new_tensor(LAMBDAS)
    optimizer = torch.optim.Adam([weights, biases], lr=1e-4)
    best_nll = weights.new_full((count,), float('inf'))
    best_weight, best_bias = weights.detach().clone(), biases.detach().clone()
    best_epoch = torch.zeros(count, dtype=torch.long, device=weights.device)
    rng = torch.Generator(device=weights.device).manual_seed(seed + 4000)
    history = []
    started = time.monotonic()
    for epoch in range(epochs):
        for ids in torch.randperm(len(train), device=train.device, generator=rng).split(32):
            optimizer.zero_grad()
            logits = torch.einsum('nk,lck->lnc', train[ids], weights) + biases[:, None, :]
            grid_loss(logits, labels[ids], weights, lambdas).backward()
            optimizer.step()
        with torch.no_grad():
            logits = torch.einsum('dnk,lck->ldnc', validation, weights) + biases[:, None, None, :]
            probabilities = logits.softmax(-1).mean(1)
            nll = -probabilities[:, torch.arange(len(validation_labels), device=train.device),
                                 validation_labels].clamp_min(1e-12).log().mean(-1)
            improved = nll < best_nll
            best_nll[improved] = nll[improved]
            best_weight[improved] = weights[improved]
            best_bias[improved] = biases[improved]
            best_epoch[improved] = epoch + 1
        history.append({'epoch': epoch + 1, 'val_nll': nll.tolist()})
        if (epoch + 1) % 20 == 0:
            print('head L1 grid epoch', epoch + 1, 'best NLL', best_nll.tolist(),
                  'elapsed', round(time.monotonic() - started, 1), flush=True)
    return best_weight, best_bias, best_nll, best_epoch, history


@torch.no_grad()
def concept_draws(model, split, samples, seed, mean=False):
    device = model.feature_scale.device
    batches = []
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        for features in split['features'].split(512):
            features = features.to(device)
            batches.append(torch.stack([model(features, sample=not mean)[1] for _ in range(samples)]))
    return torch.cat(batches, dim=1)


@torch.no_grad()
def probabilities(concepts, weight, bias):
    return (concepts @ weight.T + bias).softmax(-1).mean(0).cpu()


def run_case(pack, root, variant, seed, args):
    source = Path(f'outputs/study_v1/{args.dataset}/{variant}_seed{seed}/model.pt')
    model, _ = load_model(pack, source, args.device)
    model.eval().requires_grad_(False)
    train = concept_draws(model, pack['train'], 1, seed + 3000)[0]
    val = concept_draws(model, pack['val'], 8, seed + 1000)
    weights, biases, nll, epochs, history = fit_grid(
        train, pack['train']['labels'].to(args.device), val, pack['val']['labels'].to(args.device),
        pack['weight'].to(args.device), pack['bias'].to(args.device), args.epochs, seed)
    chosen = int(nll.argmin())
    selection = {'criterion': 'Validation MC8 NLL only, before any test concept evaluation',
                 'lambdas': LAMBDAS, 'selected_index': chosen, 'selected_l1': LAMBDAS[chosen],
                 'val_nll': nll.tolist(), 'best_epoch': epochs.tolist(),
                 'source_checkpoint': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                 'seed': seed, 'variant': variant, 'config': vars(args)}
    root.mkdir(parents=True, exist_ok=True)
    (root / 'selection.json').write_text(json.dumps(selection, indent=2))
    # Only after every L1/epoch selection has finished do we read test values.
    draws = concept_draws(model, pack['test'], 8, seed + 2000)
    means = concept_draws(model, pack['test'], 1, seed + 2000, mean=True)
    labels = pack['test']['labels']
    reports = []
    for index, l1 in enumerate(LAMBDAS):
        mc8 = probabilities(draws, weights[index], biases[index])
        mean = probabilities(means, weights[index], biases[index])
        reports.append({'l1': l1, 'val_nll': nll[index].item(), 'best_epoch': epochs[index].item(),
                        'nec_threshold_1e-5': (weights[index].abs() > 1e-5).float().sum(-1).mean().item(),
                        'mc8': task_metrics(mc8, labels), 'mean': task_metrics(mean, labels)})
        if index == chosen:
            atomic_save({'labels': labels, 'mc8': mc8, 'mean': mean}, root / 'predictions.pt')
    retained = sorted({min(k, weights.shape[-1]) for k in [1, 5, 10, 20, 40, 80, 160, weights.shape[-1]]})
    pruning = []
    order = weights[chosen].abs().argsort(-1, descending=True)
    for k in retained:
        mask = torch.zeros_like(weights[chosen]).scatter_(1, order[:, :k], 1)
        pruning.append({'retained_per_class': k,
                        **task_metrics(probabilities(draws, weights[chosen] * mask, biases[chosen]), labels)})
    atomic_save({'weights': weights.cpu(), 'biases': biases.cpu(), 'selection': selection}, root / 'heads.pt')
    result = {'evaluation_split': 'test', 'selection': selection, 'candidates': reports,
              'selected': reports[chosen], 'pruning': pruning, 'history': history,
              'protocol': 'Frozen existing Pi checkpoint; single sampled train concept per image; '
                          'all L1 heads share starting weights, bias, minibatch order and validation draws.'}
    temporary = root / 'head_grid.partial.json'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(root / 'head_grid.json')


def summarize(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    groups = {}
    for variant in ['clean', 'pi']:
        paths = [root / f'{variant}_seed{s}' for s in range(5)]
        if not all((p / 'head_grid.json').exists() for p in paths):
            return
        groups[variant] = [json.loads((p / 'head_grid.json').read_text()) for p in paths]
    a = [torch.load(root / f'clean_seed{s}/predictions.pt', weights_only=True) for s in range(5)]
    b = [torch.load(root / f'pi_seed{s}/predictions.pt', weights_only=True) for s in range(5)]
    for x, y in zip(a, b):
        torch.testing.assert_close(x['labels'], y['labels'])
    result = {'groups': groups, 'pi_minus_clean': hierarchical_paired_bootstrap(
        np.stack([r['mc8'].numpy() for r in a]), np.stack([r['mc8'].numpy() for r in b]), a[0]['labels'].numpy())}
    (destination / 'results.json').write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for variant, rows in groups.items():
        ks = [r['retained_per_class'] for r in rows[0]['pruning']]
        values = np.array([[r['accuracy'] * 100 for r in row['pruning']] for row in rows])
        ax.errorbar(ks, values.mean(0), yerr=values.std(0, ddof=1), fmt='-o', capsize=3, label=f'{variant} (n=5)')
    ax.set_xscale('log')
    ax.set_xlabel('Retained concepts per class after validation-selected L1 refit')
    ax.set_ylabel('Test accuracy (%)')
    ax.grid(alpha=.2)
    ax.legend()
    for extension in ['svg', 'pdf', 'png']:
        fig.savefig(destination / f'matched_sparsity.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path('outputs/head_sparsity_v1') / args.dataset
    runner = Runner('artifacts/logs/head_sparsity_v1')
    pack = torch.load(f'artifacts/datasets/{args.dataset}/baseline_k256_seed0.pt', weights_only=True)
    for variant in ['clean', 'pi']:
        for seed in range(5):
            run = root / f'{variant}_seed{seed}'
            if (run / 'head_grid.json').exists():
                continue
            runner.state.update(phase='waiting_for_gpu', next_case=run.name)
            runner.save()
            with gpu_lease(f'L1 head grid {run.name}', args.device):
                runner.state.update(phase=run.name)
                runner.save()
                run_case(pack, run, variant, seed, args)
            runner.state['completed_commands'].append({'name': run.name, 'finished_at': time.time(), 'returncode': 0})
            runner.save()
    summarize(root, Path('reports/head_sparsity_v1') / args.dataset)
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
