"""Run paired controls for nested source/K cases at fixed reference regularization."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .resources import gpu_lease
from .suite import Runner


PRIMARY = [('clean', 'none'), ('fixed', 'fixed'), ('mean_only', 'mean_only'),
           ('adapter_matched', 'adapter'), ('pi', 'mean_scale')]


class StudyRunner(Runner):
    def command(self, name, args):
        if args[0] == 'pi_experiments.train':
            self.state.update(phase='waiting_for_gpu', next_command=name)
            self.save()
            with gpu_lease(name):
                return super().command(name, args)
        return super().command(name, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--cases-root', default='artifacts/concept_ablation_v1')
    parser.add_argument('--output', default='outputs/concept_study_v1')
    parser.add_argument('--report', default='reports/concept_study_v1')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    cases = json.loads((Path(args.cases_root) / f'{args.dataset}_cases.json').read_text())['cases']
    reference = Path('outputs/study_v1') / args.dataset
    selection_path = reference / 'selected_hyperparameters.json'
    selection = json.loads(selection_path.read_text())['selected']
    for case in cases:
        pack = Path(case['directory']) / f'baseline_k{case["clusters"]}_seed0.pt'
        if not pack.exists():
            raise RuntimeError(f'Complete all case baselines first: {pack}')
    protocol = {'dataset': args.dataset, 'cases': cases, 'seeds': list(range(5)),
                'controls': PRIMARY, 'regularization': selection,
                'selection_sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                'interpretation': 'Fixed reference validation-selected regularization; no per-case retuning. '
                                  'Effects are conditional on each case prototype bank and initial head.'}
    if args.plan:
        print(json.dumps(protocol, indent=2))
        return
    root = Path(args.output) / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Existing protocol differs'
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    runner = StudyRunner(f'artifacts/logs/concept_study_v1/{args.dataset}')
    for case in cases:
        pack = Path(case['directory']) / f'baseline_k{case["clusters"]}_seed0.pt'
        output = root / case['name']
        output.mkdir(exist_ok=True)
        for variant, mode in PRIMARY:
            for seed in range(5):
                run = output / f'{variant}_seed{seed}'
                existing = reference / run.name
                # The 50-image/K256 case is exactly the original bank, not a new seed.
                original_pack = Path(f'artifacts/datasets/{args.dataset}/baseline_k256_seed0.pt')
                if (case['images_per_class'] == 50 and case['clusters'] == 256
                        and pack.resolve() == original_pack.resolve()
                        and all((existing / f).exists() for f in ['model.pt', 'analysis.json', 'predictions.pt'])):
                    config = json.loads((existing / 'metrics.json').read_text())['config']
                    assert config['mode'] == mode and config['seed'] == seed
                    if mode not in {'none', 'fixed'}:
                        assert config['concept_weight'] == selection['concept_weight']
                        assert config['energy_weight'] == selection['energy_weight']
                    if not run.exists():
                        run.symlink_to(os.path.relpath(existing.resolve(), output.resolve()), target_is_directory=True)
                    else:
                        assert run.resolve() == existing.resolve()
                if not (run / 'metrics.json').exists():
                    runner.command(f'{case["name"]}_{run.name}',
                                   ['pi_experiments.train', '--pack', str(pack), '--output', str(run),
                                    '--device', 'cuda', '--mode', mode, '--seed', str(seed),
                                    '--concept-weight', str(selection['concept_weight']),
                                    '--energy-weight', str(selection['energy_weight'])])
                if not (run / 'analysis.json').exists():
                    runner.command(f'{case["name"]}_{run.name}_analysis',
                                   ['pi_experiments.analyze', '--pack', str(pack), '--run', str(run)])
        runner.command(f'{case["name"]}_summary', ['pi_experiments.summarize', '--root', str(root),
                                                '--output', str(Path(args.report) / args.dataset)])
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
