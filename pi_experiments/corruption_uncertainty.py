"""Audit complete CIFAR stress tests and retain pairing across corruption cases."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .analyze import task_metrics
from .corruptions import SETTINGS
from .large_proposal_prewarm import file_sha256
from .suite import Runner, pid_identity
from .visible_semantics import write_json


VARIANTS = ['clean', 'fixed', 'mean_only', 'adapter_matched', 'pi']
MODES = dict(zip(VARIANTS, ['none', 'fixed', 'mean_only', 'adapter', 'mean_scale']))
CASES = [f'{kind}_{parameter}' for kind, parameters in SETTINGS.items() for parameter in parameters]


def paired_interval(differences, draws=2000, seed=731):
    """Bootstrap paired seeds and source images; fixed conditions stay coupled."""
    differences = np.asarray(differences, dtype=np.float64)
    assert differences.ndim == 2 and np.isfinite(differences).all()
    seeds, images = differences.shape
    assert seeds > 0 and images > 0 and draws > 0
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for start in range(0, draws, 50):
        count = min(50, draws - start)
        selected_seeds = rng.integers(0, seeds, size=(count, seeds))
        per_image = differences[selected_seeds].mean(1)
        selected_images = rng.integers(0, images, size=(count, images))
        values[start:start + count] = np.take_along_axis(per_image, selected_images, axis=1).mean(1)
    return {'accuracy_delta': float(differences.mean()),
            'ci95': np.quantile(values, [.025, .975]).tolist(),
            'adaptation_seeds': seeds, 'source_test_images': images, 'draws': draws, 'seed': seed}


def ready(dataset):
    verification = Path('artifacts/datasets') / dataset / 'corruptions/prewarm_verification.json'
    if not verification.exists():
        return False
    root = Path('outputs/study_v1') / dataset
    return all((root / f'{variant}_seed{seed}' / name).exists()
               for variant in VARIANTS for seed in range(5)
               for name in ['model.pt', 'corruptions.json', 'corruption_predictions.pt'])


def audit(dataset):
    root = Path('outputs/study_v1') / dataset
    directory = Path('artifacts/datasets') / dataset
    verification = json.loads((directory / 'corruptions/prewarm_verification.json').read_text())
    assert set(verification['cases']) == set(CASES)
    pack_path = directory / ('baseline_k256_seed0.pt' if dataset == 'cifar10' else 'baseline_k2048_seed0.pt')
    pack = torch.load(pack_path, weights_only=True, map_location='cpu')
    labels = pack['test']['labels']
    groups, correct, sources = {}, {}, {}
    for variant in VARIANTS:
        rows, correctness = [], []
        for seed in range(5):
            run = root / f'{variant}_seed{seed}'
            state = torch.load(run / 'model.pt', weights_only=True, map_location='cpu')
            assert state['config']['seed'] == seed and state['config']['mode'] == MODES[variant]
            assert not state['config'].get('validation_only') and not state['config'].get('refit_epochs')
            for tensor, original in [('projection.centers', 'centers'), ('projection.mean', 'mean'),
                                     ('projection.scale', 'scale'), ('head.weight', 'weight'), ('head.bias', 'bias')]:
                assert torch.equal(state['state_dict'][tensor], pack[original]), (run, tensor)
            del state
            report = json.loads((run / 'corruptions.json').read_text())
            prediction = torch.load(run / 'corruption_predictions.pt', weights_only=True, map_location='cpu')
            assert report['evaluation_split'] == 'test' and report['mc_samples'] == 8
            assert torch.equal(prediction['labels'], labels)
            assert set(prediction) == set(CASES) | {'labels'}
            by_case = {row['case']: row for row in report['cases']}
            assert len(report['cases']) == len(by_case) == len(CASES)
            metrics, flags = {}, []
            for name in CASES:
                p = prediction[name]
                assert p.shape == (len(labels), len(pack['metadata']['classes']))
                assert torch.isfinite(p).all() and (p >= 0).all()
                assert torch.allclose(p.sum(-1), torch.ones(len(labels)), atol=1e-5)
                actual = task_metrics(p, labels)
                for key, value in actual.items():
                    assert abs(value - by_case[name][key]) < 1e-7, (run, name, key)
                cache_path = directory / 'corruptions' / f'{name}.pt'
                # Hash every source cache once, then confirm every report's metadata.
                if variant == 'clean' and seed == 0:
                    assert file_sha256(cache_path) == verification['cases'][name]['sha256']
                    cache = torch.load(cache_path, weights_only=True, map_location='cpu')
                    assert torch.equal(cache['indices'], pack['test']['indices'])
                    assert torch.equal(cache['labels'], labels)
                    sources[name] = {'sha256': verification['cases'][name]['sha256'], 'metadata': cache['metadata']}
                    del cache
                assert by_case[name]['metadata'] == sources[name]['metadata']
                metrics[name] = actual
                flags.append((p.argmax(-1) == labels).numpy())
            rows.append(metrics)
            correctness.append(np.stack(flags))
            sources[str(run)] = {name: file_sha256(run / name)
                                 for name in ['model.pt', 'corruptions.json', 'corruption_predictions.pt']}
            print('AUDITED', dataset, run.name, flush=True)
        correct[variant] = np.stack(correctness).astype(np.float64)
        case_metrics = {name: {key: {'mean': float(np.mean([row[name][key] for row in rows])),
                                   'std': float(np.std([row[name][key] for row in rows], ddof=1))}
                              for key in rows[0][name]} for name in CASES}
        mean_accuracy = correct[variant].mean((1, 2))
        groups[variant] = {'cases': case_metrics, 'seeds': list(range(5)),
                           'equal_weight_case_accuracy': {'mean': float(mean_accuracy.mean()),
                                                         'std': float(mean_accuracy.std(ddof=1))},
                           'per_kind_accuracy': {kind: {
                               'mean': float(correct[variant][:, [i for i, name in enumerate(CASES)
                                                                  if name.startswith(kind + '_')]].mean()),
                               'std': float(correct[variant][:, [i for i, name in enumerate(CASES)
                                                                 if name.startswith(kind + '_')]].mean((1, 2)).std(ddof=1))}
                               for kind in SETTINGS}}
    comparisons = []
    for control in VARIANTS[:-1]:
        differences = correct['pi'] - correct[control]
        for i, name in enumerate(CASES):
            comparisons.append({'control': control, 'case': name, **paired_interval(differences[:, i])})
        comparisons.append({'control': control, 'case': 'all_15_equal_weight',
                            **paired_interval(differences.mean(1))})
    result = {'dataset': dataset, 'groups': groups, 'comparisons': comparisons,
              'case_order': CASES, 'source_sha256': sources,
              'cache_verification_sha256': file_sha256(directory / 'corruptions/prewarm_verification.json'),
              'baseline_pack_sha256': file_sha256(pack_path), 'code_sha256': file_sha256(__file__),
              'verified_frozen_checkpoints': 25, 'reproduced_prediction_cases': 25 * len(CASES),
              'checked_at': time.time(),
              'limits': 'Custom native-resolution stress tests, not CIFAR-C. All 15 preset conditions retained. '
                        'For the aggregate, average conditions within each seed/source image before resampling; '
                        'conditions are fixed, not independent replicates. Unadjusted descriptive intervals condition '
                        'on the fixed bank, head, selected generators, one fixed corruption field and MC8 draws. '
                        'Differences against separately trained controls do not isolate an inference-only sigma effect.'}
    write_json(Path('reports/corruption_uncertainty_v1') / f'{dataset}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', required=True, type=int)
    args = parser.parse_args()
    torch.set_num_threads(8)
    runner = Runner('artifacts/logs/corruption_uncertainty_v1')
    identity = pid_identity(args.wait_pid)
    assert identity is not None, 'Expected live postprocess manager'
    done = set()
    while len(done) < 2:
        for dataset in ['cifar10', 'cifar100']:
            if dataset in done or not ready(dataset):
                continue
            runner.state.update(phase=dataset + '_audit')
            runner.save()
            audit(dataset)
            done.add(dataset)
            runner.state['completed_commands'].append({'name': dataset, 'returncode': 0, 'finished_at': time.time()})
            runner.save()
        if len(done) == 2:
            break
        assert pid_identity(args.wait_pid) == identity, 'Postprocess ended before both complete stress tests arrived'
        runner.state.update(phase='waiting_for_complete_corruption_predictions', waiting_pid=args.wait_pid,
                            waiting_pid_start_ticks=identity, checked_at=time.time())
        runner.save()
        time.sleep(30)
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
