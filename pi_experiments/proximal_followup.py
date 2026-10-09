"""Extend a binding validation-selected iteration cap without replacing the first run."""

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

from . import head_sparsity as base
from .proximal_sparsity import LAMBDAS, fit_grid
from .resources import gpu_lease
from .suite import Runner, pid_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    parser.add_argument('--epochs',type=int,default=20000)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args();args.dataset='cifar10';torch.set_num_threads(8)
    runner = Runner('artifacts/logs/proximal_sparsity_v2')
    identity = pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid)==identity:
        runner.state.update(phase='waiting_for_initial_sparse_study',waiting_pid=args.wait_pid,checked_at=time.time())
        runner.save();time.sleep(10)
    upstream = json.loads(Path('artifacts/logs/proximal_sparsity_v1/status.json').read_text())
    assert upstream['manager_pid']==args.wait_pid and upstream['phase']=='complete'
    # These separate files contain validation selection only, rather than the
    # test candidate metrics. A binding selected cap motivates this extension.
    selections = [json.loads((Path('outputs/proximal_sparsity_v1/cifar10')/f'{v}_seed{s}/selection.json').read_text())
                  for v in ['clean','pi'] for s in range(5)]
    binding = [row for row in selections if row['best_epoch'][row['selected_index']]>=row['config']['epochs']-25]
    root = Path('outputs/proximal_sparsity_v2/cifar10');root.mkdir(parents=True,exist_ok=True)
    trigger = {'criterion':'Selected validation iteration at the prior cap; no test metrics used.',
               'binding_cases':[f"{r['variant']}_seed{r['seed']}" for r in binding],
               'prior_cap':2000,'extended_cap':args.epochs,'validation_interval':25,'lambdas':LAMBDAS}
    (root/'extension_protocol.json').write_text(json.dumps(trigger,indent=2))
    assert binding, 'No binding validation cap; this extension is unnecessary'
    base.LAMBDAS=LAMBDAS;base.fit_grid=fit_grid
    pack = torch.load('artifacts/datasets/cifar10/baseline_k256_seed0.pt',weights_only=True,map_location='cpu')
    for variant in ['clean','pi']:
        for seed in range(5):
            run = root/f'{variant}_seed{seed}'
            if (run/'head_grid.json').exists():continue
            runner.state.update(phase='waiting_for_gpu',next_case=run.name);runner.save()
            with gpu_lease(f'Extended proximal head {run.name}',args.device):
                runner.state.update(phase=run.name);runner.save()
                base.run_case(pack,run,variant,seed,args)
                path = run/'head_grid.json';result=json.loads(path.read_text())
                result['solver']={'name':'Full-gradient FISTA; exact soft threshold; bias unpenalized',
                                  'max_iterations':args.epochs,'validation_interval':25,
                                  'fit_code_sha256':hashlib.sha256(Path('pi_experiments/proximal_sparsity.py').read_bytes()).hexdigest(),
                                  'entry_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  'limits':'Validation may select iteration zero, which retains the original head. This is not a convergence certificate.'}
                heads = torch.load(run/'heads.pt',weights_only=True,map_location='cpu')
                for candidate,weights in zip(result['candidates'],heads['weights']):
                    candidate['nec_exact_nonzero']=weights.ne(0).float().sum(-1).mean().item()
                result['selected']=result['candidates'][result['selection']['selected_index']]
                path.write_text(json.dumps(result,indent=2,allow_nan=False))
            runner.state['completed_commands'].append({'name':run.name,'returncode':0,'finished_at':time.time()});runner.save()
    destination = Path('reports/proximal_sparsity_v2/cifar10')
    base.summarize(root,destination)
    fig,axis=plt.subplots(figsize=(7,5),constrained_layout=True)
    for variant in ['clean','pi']:
        rows=[json.loads((root/f'{variant}_seed{s}/head_grid.json').read_text()) for s in range(5)]
        nec=np.array([[c['nec_exact_nonzero'] for c in r['candidates']] for r in rows])
        accuracy=np.array([[c['mc8']['accuracy']*100 for c in r['candidates']] for r in rows])
        axis.errorbar(nec.mean(0),accuracy.mean(0),xerr=nec.std(0,ddof=1),yerr=accuracy.std(0,ddof=1),
                      fmt='o',capsize=3,label=variant)
    axis.set_xlabel('Mean strictly nonzero coefficients per class')
    axis.set_ylabel('Test accuracy (%)');axis.set_title('Predefined L1 candidates, validation-selected iterations')
    axis.legend();axis.grid(alpha=.2)
    for extension in ['svg','pdf','png']:fig.savefig(destination/f'actual_nec.{extension}',dpi=160)
    plt.close(fig)
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
