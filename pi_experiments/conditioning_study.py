"""Five seeds for the initialization-matched random conditioning control."""

import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .concept_study import StudyRunner
from .statistics import hierarchical_paired_bootstrap


def main():
    reference = Path('outputs/study_v1/cifar10')
    root = Path('outputs/conditioning_v1/cifar10')
    root.mkdir(parents=True, exist_ok=True)
    selected = json.loads((reference / 'selected_hyperparameters.json').read_text())['selected']
    runner = StudyRunner('artifacts/logs/conditioning_v1')
    pack = 'artifacts/datasets/cifar10/baseline_k256_seed0.pt'
    for seed in range(5):
        original = reference / f'pi_seed{seed}'
        assert (original / 'analysis.json').exists()
        target = root / original.name
        if not target.exists():
            target.symlink_to(os.path.relpath(original.resolve(), root.resolve()), target_is_directory=True)
        run = root / f'pi_random_matched_seed{seed}'
        if not (run / 'metrics.json').exists():
            # StudyRunner guards the regular trainer. Explicitly hold the lease
            # here because this separate entry point delegates to that trainer.
            from .resources import gpu_lease
            runner.state.update(phase='waiting_for_gpu', next_command=run.name)
            runner.save()
            with gpu_lease(run.name):
                runner.command(run.name, ['pi_experiments.random_bank_train', '--pack', pack,
                                          '--output', str(run), '--conditioning', 'random_concept',
                                          '--seed', str(seed), '--device', 'cuda',
                                          '--concept-weight', str(selected['concept_weight']),
                                          '--energy-weight', str(selected['energy_weight'])])
        if not (run / 'analysis.json').exists():
            runner.command(run.name + '_analysis', ['pi_experiments.analyze', '--pack', pack, '--run', str(run)])
    runner.command('conditioning_summary', ['pi_experiments.summarize', '--root', 'outputs/conditioning_v1',
                                            '--output', 'reports/conditioning_v1'])
    original = [torch.load(root / f'pi_seed{s}/predictions.pt', weights_only=True) for s in range(5)]
    control = [torch.load(root / f'pi_random_matched_seed{s}/predictions.pt', weights_only=True) for s in range(5)]
    for a, b in zip(original, control):
        torch.testing.assert_close(a['labels'], b['labels'])
    comparison = {'comparison': 'Initialization-matched random bank minus real semantic bank',
                  **hierarchical_paired_bootstrap(np.stack([r['mc8'].numpy() for r in original]),
                                                np.stack([r['mc8'].numpy() for r in control]),
                                                original[0]['labels'].numpy())}
    Path('reports/conditioning_v1/conditioning_comparison.json').write_text(json.dumps(comparison, indent=2))
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
