"""Select and test paired core controls for two ViT backbones on three datasets."""

import argparse
import json
from pathlib import Path
import time

from .concept_study import StudyRunner
from .suite import CLUSTERS, ROOTS, pid_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-weights-pid', type=int, required=True)
    parser.add_argument('--proposal-pid', type=int, required=True)
    parser.add_argument('--datasets',nargs='+',choices=list(ROOTS),default=['cifar10','cifar100','cub'])
    parser.add_argument('--logs',default='artifacts/logs/backbone_v1')
    args = parser.parse_args()
    runner = StudyRunner(args.logs)
    identity = pid_identity(args.wait_weights_pid)
    while identity is not None and pid_identity(args.wait_weights_pid) == identity:
        runner.state.update(phase='waiting_for_official_weights', waiting_pid=args.wait_weights_pid, checked_at=time.time())
        runner.save()
        time.sleep(10)
    for dataset in args.datasets:
        identity=pid_identity(args.proposal_pid)
        while not (Path('artifacts/datasets')/dataset/'manifest.json').exists():
            if identity is None or pid_identity(args.proposal_pid)!=identity:
                runner.state.update(phase='upstream_failed',missing_manifest=dataset);runner.save()
                raise RuntimeError('Parent study ended before the required dataset manifest exists')
            runner.state.update(phase='waiting_for_dataset_manifest',dataset=dataset,waiting_pid=args.proposal_pid,checked_at=time.time())
            runner.save();time.sleep(10)
        for backbone in ['ViT-B/16', 'ViT-L/14']:
            slug = backbone.replace('/', '')
            directory = Path('artifacts/backbone_ablation_v1') / slug / dataset
            output = Path('outputs/backbone_v1') / slug / dataset
            output.mkdir(parents=True, exist_ok=True)
            prep = ['pi_experiments.backbone_prepare', '--dataset', dataset, '--backbone', backbone,
                    '--output', str(directory), '--wait-pid', str(args.proposal_pid), '--stage']
            runner.command(f'{slug}_{dataset}_features', prep + ['features'])
            runner.command(f'{slug}_{dataset}_regions', prep + ['crops'])
            pack = directory / f'baseline_k{CLUSTERS[dataset]}_seed0.pt'
            if not pack.exists():
                baseline_args=['pi_experiments.prepare', '--dataset', dataset,
                               '--root', ROOTS[dataset], '--output', str(directory), '--backbone', backbone,
                               '--stage', 'baseline', '--clusters', str(CLUSTERS[dataset])]
                if dataset in ['places365','imagenet']:baseline_args+=['--head-batch-size','512']
                runner.command(f'{slug}_{dataset}_baseline',baseline_args)
            selected_path = output / 'selected_hyperparameters.json'
            if not selected_path.exists():
                candidates = []
                grid = sorted({(c, e) for c in [0, .01, .1, 1] for e in [0, .001, .01, .1]} |
                              {(c, e) for c in [1, 3, 10] for e in [.1, .3, 1]})
                for c, e in grid:
                    run = output / 'validation_search' / f'c{c:g}_e{e:g}'
                    if not (run / 'metrics.json').exists():
                        runner.command(f'{slug}_{dataset}_search_c{c}_e{e}', ['pi_experiments.train', '--pack', str(pack),
                                       '--output', str(run), '--device', 'cuda', '--validation-only',
                                       '--concept-weight', str(c), '--energy-weight', str(e)]+
                                       (['--batch-size','512'] if dataset in ['places365','imagenet'] else []))
                    report = json.loads((run / 'metrics.json').read_text())
                    assert report['evaluation_split'] == 'val'
                    candidates.append({'concept_weight': c, 'energy_weight': e, 'val_nll': report['metrics']['pi']['nll']})
                selected_path.write_text(json.dumps({'criterion': 'Validation NLL only', 'candidates': candidates,
                                                     'selected': min(candidates, key=lambda r: r['val_nll'])}, indent=2))
            selected = json.loads(selected_path.read_text())['selected']
            for variant, mode in [('clean', 'none'), ('fixed', 'fixed'), ('mean_only', 'mean_only'),
                                  ('adapter_matched', 'adapter'), ('pi', 'mean_scale')]:
                for seed in range(5):
                    run = output / f'{variant}_seed{seed}'
                    if not (run / 'metrics.json').exists():
                        runner.command(f'{slug}_{dataset}_{run.name}', ['pi_experiments.train', '--pack', str(pack),
                                       '--output', str(run), '--device', 'cuda', '--mode', mode, '--seed', str(seed),
                                       '--concept-weight', str(selected['concept_weight']), '--energy-weight', str(selected['energy_weight'])]+
                                       (['--batch-size','512'] if dataset in ['places365','imagenet'] else []))
                    if not (run / 'analysis.json').exists():
                        runner.command(f'{slug}_{dataset}_{run.name}_analysis', ['pi_experiments.analyze', '--pack', str(pack), '--run', str(run)])
                    if dataset == 'cub' and not (run / 'cub_concepts.json').exists():
                        from .resources import gpu_lease
                        with gpu_lease(f'{slug} CUB attribute evaluation'):
                            runner.command(f'{slug}_{run.name}_concepts', ['pi_experiments.cub_metrics', '--pack', str(pack),
                                           '--run', str(run), '--directory', str(directory)])
                runner.command(f'{slug}_{dataset}_{variant}_summary', ['pi_experiments.summarize', '--root', str(output.parent),
                               '--output', str(Path('reports/backbone_v1') / slug)])
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
