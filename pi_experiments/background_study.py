"""Prepare shared CUB context changes, then evaluate successful model studies."""

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .backgrounds import CASES
from .statistics import hierarchical_paired_bootstrap
from .suite import Runner, pid_identity


def summarize(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    groups = []
    comparisons = []
    primary = ['clean', 'fixed', 'mean_only', 'adapter_matched', 'pi']
    for variant in primary:
        paths = [root / f'{variant}_seed{seed}' for seed in range(5)]
        rows = [json.loads((p / 'backgrounds.json').read_text()) for p in paths if (p / 'backgrounds.json').exists()]
        if not rows:
            continue
        group = {'variant': variant, 'completed_seeds': [int(p.name.rsplit('seed', 1)[1]) for p in paths if (p / 'backgrounds.json').exists()], 'cases': {}}
        for index, case in enumerate(CASES):
            metrics = [r['cases'][index] for r in rows]
            assert all(r['case'] == case for r in metrics)
            group['cases'][case] = {}
            for key in ['accuracy', 'nll', 'ece_15_bins']:
                values = [r['mc8'][key] for r in metrics]
                group['cases'][case][key] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)) if len(values) > 1 else None}
            for key in ['attribute_macro_balanced_accuracy', 'attribute_macro_auroc']:
                values = [r['semantic'][key] for r in metrics if r['semantic'][key] is not None]
                group['cases'][case][key] = {'mean': float(np.mean(values)) if values else None,
                                           'std': float(np.std(values, ddof=1)) if len(values) > 1 else None}
        groups.append(group)
    for control in primary[:-1] + ['same_checkpoint_mean']:
        for case in CASES:
            a, b = [], []
            seeds = []
            for seed in range(5):
                target = root / f'pi_seed{seed}/background_predictions.pt'
                source = target if control == 'same_checkpoint_mean' else root / f'{control}_seed{seed}/background_predictions.pt'
                if target.exists() and source.exists():
                    av, bv = torch.load(source, weights_only=True), torch.load(target, weights_only=True)
                    torch.testing.assert_close(av['labels'], bv['labels'])
                    a.append(av[case]['mean' if control == 'same_checkpoint_mean' else 'mc8'].numpy())
                    b.append(bv[case]['mc8'].numpy())
                    seeds.append(seed)
            if seeds:
                comparisons.append({'case': case, 'control': control, 'seeds': seeds,
                                    **hierarchical_paired_bootstrap(np.stack(a), np.stack(b), av['labels'].numpy())})
    (destination / 'results.json').write_text(json.dumps({'groups': groups, 'pi_comparisons': comparisons}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for group in groups:
        for ax, key, ylabel in [(axes[0], 'accuracy', 'Task accuracy (%)'),
                                (axes[1], 'attribute_macro_balanced_accuracy', 'Attribute balanced accuracy (%)')]:
            values = [group['cases'][case][key]['mean'] for case in CASES]
            ax.plot(range(len(CASES)), [x * 100 if x is not None else np.nan for x in values], '-o',
                    label=f"{group['variant']} (n={len(group['completed_seeds'])})")
            ax.set_xticks(range(len(CASES)), CASES, rotation=30, ha='right')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
    if groups:
        axes[0].legend(fontsize=8)
    fig.suptitle('CUB majority foreground preserved; original validation thresholds frozen')
    for extension in ['svg', 'pdf', 'png']:
        fig.savefig(destination / f'background_task_semantics.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--upstream-status', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--model-directory', required=True)
    parser.add_argument('--logs', required=True)
    parser.add_argument('--reports', required=True)
    parser.add_argument('--prepare-after-wait', action='store_true')
    args = parser.parse_args()
    runner = Runner(args.logs)
    directory = Path('artifacts/datasets/cub')
    if not args.prepare_after_wait and not all((directory / 'backgrounds' / f'{case}.pt').exists() for case in CASES):
        runner.command('cub_background_features', ['pi_experiments.backgrounds', '--stage', 'prepare'])
    identity = pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid) == identity:
        runner.state.update(phase='waiting_for_cub_models', waiting_pid=args.wait_pid, checked_at=time.time())
        runner.save()
        time.sleep(10)
    upstream = json.loads(Path(args.upstream_status).read_text())
    if upstream['manager_pid'] != args.wait_pid or upstream['phase'] != 'complete':
        runner.state.update(phase='upstream_failed', upstream_state=upstream)
        runner.save()
        raise RuntimeError('Expected CUB model study did not complete successfully')
    if not all((directory / 'backgrounds' / f'{case}.pt').exists() for case in CASES):
        runner.command('cub_background_features', ['pi_experiments.backgrounds', '--stage', 'prepare'])
    pack = Path(args.model_directory) / 'baseline_k2048_seed0.pt'
    root = Path(args.runs) / 'cub'
    for variant in ['clean', 'fixed', 'mean_only', 'adapter_matched', 'pi']:
        for seed in range(5):
            run = root / f'{variant}_seed{seed}'
            if not (run / 'backgrounds.json').exists():
                runner.command(run.name + '_backgrounds', ['pi_experiments.backgrounds', '--stage', 'evaluate',
                                                          '--pack', str(pack), '--run', str(run)])
        summarize(root, Path(args.reports))
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
