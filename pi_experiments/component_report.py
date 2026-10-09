"""Report completed component controls with paired uncertainty and concept drift."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .statistics import hierarchical_paired_bootstrap


def main():
    root = Path('outputs/study_v1/cifar10')
    destination = Path('reports/component_diagnostics')
    destination.mkdir(parents=True, exist_ok=True)
    variants = ['pi', 'pi_mlp', 'pi_random_concepts', 'pi_concept_scores',
                'pi_no_preservation', 'pi_no_energy']
    groups = []
    for variant in variants:
        runs = [root / f'{variant}_seed{seed}' for seed in range(5)]
        if not all((run / 'analysis.json').exists() for run in runs):
            raise RuntimeError(f'Incomplete five-seed comparison: {variant}')
        reports = [json.loads((run / 'analysis.json').read_text())['inference']['mc8'] for run in runs]
        groups.append({'variant': variant, 'seeds': list(range(5)), 'metrics': {
            key: {'mean': float(np.mean([r[key] for r in reports])),
                  'std': float(np.std([r[key] for r in reports], ddof=1))}
            for key in ['accuracy', 'nll', 'standardized_concept_mse', 'top5_clean_agreement', 'sigma_rms']}})
    reference = [torch.load(root / f'pi_seed{s}/predictions.pt', weights_only=True) for s in range(5)]
    comparisons = []
    for variant in variants[1:]:
        predictions = [torch.load(root / f'{variant}_seed{s}/predictions.pt', weights_only=True) for s in range(5)]
        for a, b in zip(reference, predictions):
            torch.testing.assert_close(a['labels'], b['labels'])
        comparisons.append({'variant': variant, 'reference': 'pi',
                            **hierarchical_paired_bootstrap(np.stack([r['mc8'].numpy() for r in reference]),
                                                          np.stack([r['mc8'].numpy() for r in predictions]),
                                                          reference[0]['labels'].numpy())})
    result = {'dataset': 'cifar10', 'groups': groups, 'comparisons': comparisons,
              'parameter_counts': json.loads(Path('reports/CONDITIONING_PARAMETER_COUNTS.json').read_text()),
              'limits': ['Fixed prototype bank, initial head, and common regularization selected for Pi.',
                         'MLP has fewer parameters; random-bank initialization advances the RNG before parameter initialization.',
                         'Concept response drift and top-5 agreement do not measure true concept accuracy.']}
    (destination / 'cifar10.json').write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = ['Pi', 'MLP context', 'Random bank', 'Score injection', 'No preservation', 'No energy penalty']
    for axis, metric, factor, ylabel in [(axes[0], 'accuracy', 100, 'Test accuracy (%)'),
                                        (axes[1], 'standardized_concept_mse', 1, 'Concept response MSE')]:
        means = [g['metrics'][metric]['mean'] * factor for g in groups]
        errors = [g['metrics'][metric]['std'] * factor for g in groups]
        axis.errorbar(range(len(groups)), means, yerr=errors, fmt='o', capsize=4)
        axis.set_xticks(range(len(groups)), labels, rotation=32, ha='right')
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=.25)
    axes[1].set_yscale('log')
    fig.suptitle('CIFAR-10 components: five adaptation seeds, mean ± SD')
    for extension in ['svg', 'pdf', 'png']:
        fig.savefig(destination / f'cifar10_components.{extension}', dpi=180)
    plt.close(fig)
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
