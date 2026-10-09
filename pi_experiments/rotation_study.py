"""Pair geometry-preserving conditioning rotations with the completed real bank."""

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .resources import gpu_lease
from .statistics import hierarchical_paired_bootstrap
from .suite import Runner


def main():
    root = Path('outputs/rotation_conditioning_v1/cifar10');root.mkdir(parents=True,exist_ok=True)
    source = Path('outputs/study_v1/cifar10')
    selection_path = source/'selected_hyperparameters.json'
    selected = json.loads(selection_path.read_text())['selected']
    protocol = {'reference_selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                'selected':selected,'seeds':list(range(5)),
                'control':'Signed dimension permutation of normalized real conditioning prototypes; independent RNG seed+20000.'}
    path = root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2))
    runner = Runner('artifacts/logs/rotation_conditioning_v1')
    pack = 'artifacts/datasets/cifar10/baseline_k256_seed0.pt'
    for seed in range(5):
        real = root/f'pi_seed{seed}'
        if not real.exists():real.symlink_to(os.path.relpath((source/real.name).resolve(),root.resolve()),target_is_directory=True)
        assert real.resolve()==(source/real.name).resolve()
        run = root/f'pi_rotated_matched_seed{seed}'
        for phase,module,additional in [
            ('train','pi_experiments.rotated_bank_train',['--output',str(run),'--conditioning','random_concept',
             '--mode','mean_scale','--seed',str(seed),'--concept-weight',str(selected['concept_weight']),
             '--energy-weight',str(selected['energy_weight'])]),
            ('analysis','pi_experiments.analyze',['--run',str(run)])]:
            if (run/('metrics.json' if phase=='train' else 'analysis.json')).exists():continue
            runner.state.update(phase='waiting_for_gpu',next_case=run.name);runner.save()
            with gpu_lease(f'Rotated conditioning {run.name} {phase}'):
                runner.command(f'{run.name}_{phase}',[module,'--pack',pack,'--device','cuda',*additional])
    runner.command('summary',['pi_experiments.summarize','--root','outputs/rotation_conditioning_v1',
                             '--output','reports/rotation_conditioning_v1'])
    a = [torch.load(root/f'pi_seed{s}/predictions.pt',weights_only=True,map_location='cpu') for s in range(5)]
    b = [torch.load(root/f'pi_rotated_matched_seed{s}/predictions.pt',weights_only=True,map_location='cpu') for s in range(5)]
    for x,y in zip(a,b):assert torch.equal(x['labels'],y['labels'])
    comparison = {'comparison':'Geometry-preserving conditioning rotation minus aligned real bank',
                  **hierarchical_paired_bootstrap(np.stack([r['mc8'].numpy() for r in a]),
                                                 np.stack([r['mc8'].numpy() for r in b]),a[0]['labels'].numpy())}
    Path('reports/rotation_conditioning_v1/conditioning_comparison.json').write_text(json.dumps(comparison,indent=2))
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
