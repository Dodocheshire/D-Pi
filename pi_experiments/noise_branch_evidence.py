"""Preserve original inference branch diagnostics with per-image paired evidence."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .analyze import collect, load_model, task_metrics
from .large_proposal_prewarm import file_sha256
from .noise_diagnostics import NoiseOverride
from .prepare import atomic_save
from .statistics import hierarchical_paired_bootstrap
from .suite import CLUSTERS, Runner, pid_identity
from .visible_semantics import write_json


SCENARIOS = [('no_noise', 0, 0), ('scale_only', 0, 1), ('mean_only', 1, 0)]
SCENARIOS += [(f'mean_plus_scale_x{factor}', 1, factor) for factor in [.25, .5, 1, 2, 4]]


class CachedParameters(nn.Module):
    """Cache deterministic outputs only during inference on unchanged parameters."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.parameter_versions = tuple(p._version for p in base.parameters())
        self.cache = {}

    @property
    def query(self):
        return self.base.query

    @property
    def prototypes(self):
        return self.base.prototypes

    def parameters_for(self, features):
        assert not torch.is_grad_enabled() and not self.training
        assert self.parameter_versions == tuple(p._version for p in self.base.parameters())
        key = (features.data_ptr(), tuple(features.shape), tuple(features.stride()),
               features.device, features.dtype, features._version)
        if key not in self.cache:
            # Retain the input to prevent pointer reuse for a different tensor.
            self.cache[key] = (features, self.base.parameters_for(features))
        return self.cache[key][1]


def run_seed(pack, run, destination):
    source_digest = file_sha256(run / 'model.pt')
    model, config = load_model(pack, run / 'model.pt', 'cpu')
    assert config['mode'] == 'mean_scale' and not config.get('validation_only')
    assert all(torch.equal(model.state_dict()[name], pack[original])
               for name, original in [('projection.centers', 'centers'), ('projection.mean', 'mean'),
                                      ('projection.scale', 'scale')])
    if not config.get('refit_epochs'):
        assert torch.equal(model.head.weight, pack['weight']) and torch.equal(model.head.bias, pack['bias'])
    cached = CachedParameters(model.generator).eval()
    model.generator = cached
    split = pack['test']
    mean = collect(model, split, samples=1, mean=True)
    labels = split['labels']
    mean_metrics = task_metrics(mean['probabilities'], labels)
    probabilities = {'mean_reference': mean['probabilities']}
    rows = []
    for name, alpha, beta in SCENARIOS:
        model.generator = NoiseOverride(cached, alpha, beta).eval()
        result = collect(model, split, samples=16, seed=3010)
        mu_energy = result['mu_rms'].square().mean().item()
        sigma_energy = result['sigma_rms'].square().mean().item()
        row = {'scenario': name, 'mean_multiplier': alpha, 'scale_multiplier': beta,
               **task_metrics(result['probabilities'], labels),
               'mean_energy': mu_energy, 'scale_energy': sigma_energy,
               'scale_energy_fraction': sigma_energy / (mu_energy + sigma_energy) if mu_energy + sigma_energy else 0.,
               'standardized_concept_mse': (result['concepts'] - result['clean_concepts']).square().mean().item(),
               'concept_sampling_variance': result['concept_sampling_variance'].mean().item()}
        probabilities[name] = result['probabilities']
        row['nll_delta_vs_mean'] = row['nll'] - mean_metrics['nll']
        rows.append(row)
        print('SCENARIO', run, name, row['accuracy'], flush=True)
        del result
    # Averaging sixteen identical float32 probabilities may round differently
    # from returning one draw. Preserve the original reduction and audit it.
    torch.testing.assert_close(probabilities['mean_reference'], probabilities['mean_only'], rtol=1e-6, atol=1e-7)
    mean_reduction_error = (probabilities['mean_reference'] - probabilities['mean_only']).abs().max().item()
    mean_class_changes = (probabilities['mean_reference'].argmax(-1) != probabilities['mean_only'].argmax(-1)).sum().item()
    legacy_path = run / 'noise_diagnostics_test.json'
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text())
        assert legacy['evaluation_split'] == 'test' and legacy['mc_samples'] == 16
        assert legacy['sampling_seed'] == 3010
        for row, previous in zip(rows, legacy['scenarios']):
            assert row['scenario'] == previous['scenario']
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    assert abs(value - previous[key]) < 1e-7, (run, row['scenario'], key, value, previous[key])
    assert file_sha256(run / 'model.pt') == source_digest
    atomic_save({'labels': labels, 'indices': split['indices'], **probabilities}, destination / 'predictions.pt')
    write_json(destination / 'metrics.json', {
        'source_checkpoint': str(run / 'model.pt'), 'source_sha256': source_digest,
        'config': config, 'evaluation_split': 'test', 'mc_samples': 16, 'sampling_seed': 3010,
        'scenarios': rows, 'mean_reference': mean_metrics,
        'legacy_scalar_metrics_reproduced': legacy_path.exists(),
        'deterministic_average_max_probability_error': mean_reduction_error,
        'deterministic_average_class_changes': mean_class_changes,
        'legacy_report_sha256': file_sha256(legacy_path) if legacy_path.exists() else None,
        'code_sha256': file_sha256(__file__), 'predictions_sha256': file_sha256(destination / 'predictions.pt'),
        'protocol': 'Original collect/forward and fixed seed3010, MC16 and eight original scenarios. '
                    'Only deterministic generator parameters for identical input batches are cached. No retraining.',
    })


def summarize(dataset, variant, root):
    runs = [root / f'{variant}_seed{s}' for s in range(5)]
    records = [json.loads((run / 'metrics.json').read_text()) for run in runs]
    packs = [torch.load(run / 'predictions.pt', weights_only=True, map_location='cpu') for run in runs]
    labels, indices = packs[0]['labels'], packs[0]['indices']
    refitted_head = variant == 'pi_refit'
    for seed, (run, record, p) in enumerate(zip(runs, records, packs)):
        assert record['config']['seed'] == seed and record['sampling_seed'] == 3010
        assert bool(record['config'].get('refit_epochs')) == refitted_head
        assert record['source_sha256'] == file_sha256(record['source_checkpoint'])
        assert record['predictions_sha256'] == file_sha256(run / 'predictions.pt')
        assert torch.equal(p['labels'], labels) and torch.equal(p['indices'], indices)
        for probability in [p['mean_reference'], *[p[name] for name, _, _ in SCENARIOS]]:
            assert torch.isfinite(probability).all() and (probability >= 0).all()
            assert torch.allclose(probability.sum(-1), torch.ones(len(labels)), atol=1e-5)
    groups = {}
    comparisons = []
    for name, _, _ in SCENARIOS:
        rows = [next(row for row in record['scenarios'] if row['scenario'] == name) for record in records]
        groups[name] = {key: {'mean': float(np.mean([r[key] for r in rows])),
                             'std': float(np.std([r[key] for r in rows], ddof=1))}
                        for key in rows[0] if isinstance(rows[0][key], (int, float))}
        for p, row in zip(packs, rows):
            for key, value in task_metrics(p[name], labels).items():
                assert abs(value - row[key]) < 1e-7
        if name == 'mean_only':
            continue
        interval = hierarchical_paired_bootstrap(np.stack([p['mean_reference'].numpy() for p in packs]),
                                                 np.stack([p[name].numpy() for p in packs]), labels.numpy())
        head = 'validation-selected refitted head' if refitted_head else 'original initial head'
        interval['protocol'] = ('Resample paired adaptation seeds and common test images; fixed prototype bank, '
                                f'selected source-checkpoint generator and its {head} within every inference pair.')
        comparisons.append({'scenario': name, 'reference': 'mean_reference', **interval})
    write_json(Path('reports/noise_branch_evidence_v1') / dataset / f'{variant}.json', {
        'dataset': dataset, 'variant': variant, 'seeds': list(range(5)), 'groups': groups,
        'comparisons': comparisons, 'source_checkpoint_sha256': [r['source_sha256'] for r in records],
        'predictions_sha256': [r['predictions_sha256'] for r in records],
        'run_code_sha256': [r['code_sha256'] for r in records],
        'head_protocol': 'Each source checkpoint head remains fixed within paired branch interventions; '
                         + ('head selected by the original refit validation rule' if refitted_head else 'original initial head'),
        'code_sha256': file_sha256(__file__), 'checked_at': time.time(),
        'limits': 'Inference interventions on each unchanged selected checkpoint; no multiplier or budget selection. '
                  'Paired seeds/common images, unadjusted descriptive intervals conditional on fixed banks, heads, '
                  'selected generators and one shared MC16 realization. Do not infer effects of retraining without a branch. '
                  'Energy fractions are before feature-RMS scaling; response drift is not true concept accuracy.',
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    runner = Runner('artifacts/logs/noise_branch_evidence_v1')
    identity = pid_identity(args.wait_pid)
    assert identity is not None
    done = set()
    while len(done) < 6:
        for dataset in ['cifar10', 'cifar100', 'cub']:
            for variant in ['pi', 'pi_refit']:
                if (dataset, variant) in done:
                    continue
                source = Path('outputs/study_v1') / dataset
                if not all((source / f'{variant}_seed{s}/analysis.json').exists() for s in range(5)):
                    continue
                pack = torch.load(Path('artifacts/datasets') / dataset / f'baseline_k{CLUSTERS[dataset]}_seed0.pt',
                                  weights_only=True, map_location='cpu')
                root = Path('outputs/noise_branch_evidence_v1') / dataset
                for seed in range(5):
                    run = source / f'{variant}_seed{seed}'
                    destination = root / run.name
                    if (destination / 'metrics.json').exists():
                        continue
                    runner.state.update(phase=f'{dataset}_{run.name}_cpu');runner.save()
                    run_seed(pack, run, destination)
                    runner.state['completed_commands'].append({'name': f'{dataset}_{run.name}',
                                                               'returncode': 0, 'finished_at': time.time()})
                    runner.save()
                summarize(dataset, variant, root)
                done.add((dataset, variant))
                del pack
        if len(done) == 6:
            break
        assert pid_identity(args.wait_pid) == identity, 'Main study ended before all required checkpoints arrived'
        runner.state.update(phase='waiting_for_complete_models', waiting_pid=args.wait_pid,
                            waiting_pid_start_ticks=identity, checked_at=time.time());runner.save()
        time.sleep(30)
    runner.state.update(phase='complete', finished_at=time.time());runner.save()


if __name__ == '__main__':
    main()
