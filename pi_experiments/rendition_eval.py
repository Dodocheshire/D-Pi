"""Evaluate fixed ImageNet models on matched 200-class ID and ImageNet-R data."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .analyze import load_model, task_metrics
from .prepare import atomic_save
from .resources import gpu_lease
from .statistics import hierarchical_paired_bootstrap
from .suite import Runner, pid_identity


def restricted_probabilities(logits, allowed):
    return logits.index_select(-1, allowed).softmax(-1)


@torch.no_grad()
def evaluate(model, features, allowed, seed, samples=8):
    device = model.feature_scale.device
    probabilities, means = [], []
    drift_sum = energy_sum = 0.0
    allowed = allowed.to(device)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        for values in features.split(512):
            values = values.to(device)
            draws, concepts = [], []
            for _ in range(samples):
                logits, current, clean, energy = model(values)
                draws.append(restricted_probabilities(logits, allowed))
                concepts.append(current)
                energy_sum += energy.item() * len(values) / samples
            probabilities.append(torch.stack(draws).mean(0).cpu())
            drift_sum += (torch.stack(concepts).mean(0) - clean).square().mean().item() * len(values)
            logits = model(values, sample=False)[0]
            means.append(restricted_probabilities(logits, allowed).cpu())
    return {'mc8': torch.cat(probabilities), 'mean': torch.cat(means)}, {
        'mean_concept_drift_mse': drift_sum / len(features), 'expected_noise_energy': energy_sum / len(features)}


def summarize(root, destination):
    groups = {}
    for variant in ['clean', 'fixed', 'mean_only', 'adapter_matched', 'pi']:
        rows = []
        for seed in range(5):
            path = root / f'{variant}_seed{seed}/rendition.json'
            if path.exists():
                rows.append(json.loads(path.read_text()))
        if rows:
            groups[variant] = {'seeds': [row['seed'] for row in rows], 'metrics': {}}
            for split in ['imagenet_200', 'imagenet_r']:
                groups[variant]['metrics'][split] = {
                    key: {'mean': float(np.mean([r[split]['mc8'][key] for r in rows])),
                          'std': float(np.std([r[split]['mc8'][key] for r in rows], ddof=1)) if len(rows) > 1 else None}
                    for key in ['accuracy', 'nll', 'ece_15_bins', 'macro_f1']}
    comparisons = []
    pi_seeds = groups.get('pi', {}).get('seeds', [])
    for variant in ['clean', 'fixed', 'mean_only', 'adapter_matched', 'same_checkpoint_mean']:
        seeds = pi_seeds if variant == 'same_checkpoint_mean' else sorted(set(pi_seeds) & set(groups.get(variant, {}).get('seeds', [])))
        if not seeds:
            continue
        b = [torch.load(root / f'pi_seed{s}/rendition_predictions.pt', weights_only=True) for s in seeds]
        a = b if variant == 'same_checkpoint_mean' else [torch.load(root / f'{variant}_seed{s}/rendition_predictions.pt', weights_only=True) for s in seeds]
        for split in ['imagenet_200', 'imagenet_r']:
            for x, y in zip(a, b):
                torch.testing.assert_close(x[split]['labels'], y[split]['labels'])
            comparisons.append({'split': split, 'control': variant, 'seeds': seeds,
                                **hierarchical_paired_bootstrap(
                                    np.stack([r[split]['mean' if variant == 'same_checkpoint_mean' else 'mc8'].numpy() for r in a]),
                                    np.stack([r[split]['mc8'].numpy() for r in b]), a[0][split]['labels'].numpy())})
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'results.json').write_text(json.dumps({'groups': groups, 'pi_comparisons': comparisons}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--upstream-status', default='artifacts/logs/large_v1/status.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--directory', default='artifacts/datasets/imagenet_r')
    parser.add_argument('--pack', default='artifacts/datasets/imagenet/baseline_k4096_seed0.pt')
    parser.add_argument('--runs', default='outputs/study_v1/imagenet')
    parser.add_argument('--output', default='outputs/rendition_eval_v1/imagenet')
    parser.add_argument('--reports', default='reports/rendition_eval_v1')
    parser.add_argument('--logs', default='artifacts/logs/rendition_eval_v1')
    args = parser.parse_args()
    runner = Runner(args.logs)
    identity = pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid) == identity:
        runner.state.update(phase='waiting_for_imagenet_models', waiting_pid=args.wait_pid, checked_at=time.time())
        runner.save()
        time.sleep(10)
    upstream = json.loads(Path(args.upstream_status).read_text())
    if upstream['manager_pid'] != args.wait_pid or upstream['phase'] != 'complete':
        runner.state.update(phase='upstream_failed', upstream_state=upstream)
        runner.save()
        raise RuntimeError('Expected ImageNet model study did not complete')
    torch.set_num_threads(8)
    directory = Path(args.directory)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rendition = torch.load(directory / 'test_features.pt', weights_only=True)
    assert rendition['manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert torch.equal(rendition['labels'], rendition['restricted_class_indices'][rendition['restricted_labels']])
    pack = torch.load(args.pack, weights_only=True)
    assert pack['metadata']['classes'] == manifest['classes_1000']
    assert pack['metadata']['backbone'] == manifest['backbone']
    allowed = rendition['restricted_class_indices']
    mapping = torch.full((1000,), -1, dtype=torch.long)
    mapping[allowed] = torch.arange(len(allowed))
    id_labels = mapping[pack['test']['labels']]
    mask = id_labels >= 0
    features = {'imagenet_200': pack['test']['features'][mask], 'imagenet_r': rendition['features']}
    labels = {'imagenet_200': id_labels[mask], 'imagenet_r': rendition['restricted_labels']}
    assert len(features['imagenet_200']) == 10000
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for variant in ['clean', 'fixed', 'mean_only', 'adapter_matched', 'pi']:
        for seed in range(5):
            output = root / f'{variant}_seed{seed}'
            if (output / 'rendition.json').exists():
                continue
            checkpoint = Path(args.runs) / f'{variant}_seed{seed}/model.pt'
            runner.state.update(phase='waiting_for_gpu', next_case=output.name)
            runner.save()
            with gpu_lease(f'ImageNet-R {output.name}', args.device):
                runner.state.update(phase=output.name)
                runner.save()
                model, _ = load_model(pack, checkpoint, args.device)
                record = {'seed': seed, 'variant': variant,
                          'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          'manifest_sha256': rendition['manifest_sha256'],
                          'class_policy': 'Restrict logits to 200 synsets before every draw softmax; average probabilities.'}
                predictions = {}
                for split in features:
                    arrays, diagnostics = evaluate(model, features[split], allowed, seed + 2110)
                    predictions[split] = {**arrays, 'labels': labels[split]}
                    record[split] = {'images': len(labels[split]), 'diagnostics': diagnostics,
                                     **{name: task_metrics(p, labels[split]) for name, p in arrays.items()}}
                output.mkdir(parents=True, exist_ok=True)
                atomic_save(predictions, output / 'rendition_predictions.pt')
                temporary = output / 'rendition.partial.json'
                temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
                temporary.replace(output / 'rendition.json')
                del model
            runner.state['completed_commands'].append({'name': output.name, 'returncode': 0, 'finished_at': time.time()})
            runner.save()
        summarize(root, Path(args.reports))
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
