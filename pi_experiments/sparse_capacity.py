"""Separate deterministic adaptation/capacity from Pi at exactly matched NEC."""

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import head_sparsity as base
from .analyze import load_model, task_metrics
from .matched_nec import common_targets, truncate_exact
from .prepare import atomic_save
from .proximal_sparsity import LAMBDAS, fit_grid
from .resources import gpu_lease
from .statistics import hierarchical_paired_bootstrap
from .suite import Runner, pid_identity
from .visible_semantics import link, sha, write_json


VARIANTS = ['clean','mean_only','adapter_matched','pi']


def summarize(root, destination, targets):
    groups = {variant:[json.loads((root/f'{variant}_seed{s}/head_grid.json').read_text()) for s in range(5)]
              for variant in VARIANTS}
    predictions = {variant:[torch.load(root/f'{variant}_seed{s}/exact_predictions.pt',weights_only=True,
                                       map_location='cpu') for s in range(5)] for variant in VARIANTS}
    labels = predictions['pi'][0]['labels']
    comparisons = []
    for control in VARIANTS[:-1]:
        for k in targets:
            a,b = predictions[control],predictions['pi']
            assert all(torch.equal(row['labels'],labels) for row in a+b)
            interval = hierarchical_paired_bootstrap(np.stack([r['mc8'][str(k)].numpy() for r in a]),
                         np.stack([r['mc8'][str(k)].numpy() for r in b]),labels.numpy())
            interval['protocol'] = 'Paired adaptation seeds and common images; fixed prototype bank, selected generators and validation-selected refitted heads.'
            comparisons.append({'control':control,'nec_per_class':k,**interval})
    stochastic = []
    for k in targets:
        a = predictions['pi']
        interval = hierarchical_paired_bootstrap(np.stack([r['mean'][str(k)].numpy() for r in a]),
                                                np.stack([r['mc8'][str(k)].numpy() for r in a]),labels.numpy())
        interval['protocol'] = 'Paired adaptation seeds and common images; same selected Pi generator and same truncated head for both inference modes.'
        stochastic.append({'nec_per_class':k,**interval})
    write_json(destination/'results.json',{
        'groups':groups,'targets':targets,'pi_minus_control':comparisons,'pi_mc8_minus_mean':stochastic,
        'limits':'Validation-selected unpruned heads are truncated at common feasible counts; '
                 'heads are not separately optimized for each NEC. Generator checkpoints, regularization '
                 'grid, initialization and iteration cap are fixed. No test-based NEC or lambda selection.'})
    fig,axis = plt.subplots(figsize=(8,5),constrained_layout=True)
    for variant in VARIANTS:
        values = np.array([[json.loads((root/f'{variant}_seed{s}/exact_metrics.json').read_text())
                            ['points'][i]['mc8']['accuracy']*100 for i in range(len(targets))] for s in range(5)])
        axis.errorbar(targets,values.mean(0),yerr=values.std(0,ddof=1),fmt='-o',capsize=3,label=variant)
    axis.set_xlabel('Exactly nonzero coefficients in every class')
    axis.set_ylabel('Test accuracy (%)');axis.legend();axis.grid(alpha=.2)
    for extension in ['svg','pdf','png']:fig.savefig(destination/f'exact_nec_capacity.{extension}',dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    parser.add_argument('--epochs',type=int,default=20000)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args();args.dataset='cifar10';torch.set_num_threads(8)
    runner = Runner('artifacts/logs/sparse_capacity_v1')
    upstream = Path('artifacts/logs/matched_nec_v2/status.json')
    assert json.loads(upstream.read_text())['manager_pid']==args.wait_pid
    identity = pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid)==identity:
        runner.state.update(phase='waiting_for_initial_exact_nec_study',checked_at=time.time());runner.save();time.sleep(10)
    state = json.loads(upstream.read_text())
    assert state['manager_pid']==args.wait_pid and state['phase']=='complete'
    root = Path('outputs/sparse_capacity_v1/cifar10');root.mkdir(parents=True,exist_ok=True)
    reused = {}
    for variant in ['clean','pi']:
        for seed in range(5):
            source = Path('outputs/proximal_sparsity_v2/cifar10')/f'{variant}_seed{seed}'
            selection = json.loads((source/'selection.json').read_text())
            assert selection['lambdas']==LAMBDAS and selection['config']['epochs']==args.epochs
            reused[source.name] = {file:sha(source/file) for file in ['selection.json','heads.pt','head_grid.json','predictions.pt']}
            run = root/source.name
            for file in reused[source.name]:link(source/file,run/file)
    protocol = {'variants':VARIANTS,'adaptation_seeds':list(range(5)),'max_iterations':args.epochs,
                'lambdas':LAMBDAS,'device':args.device,'validation_interval':25,'reused_heads':reused,
                'reason':'Clean/Pi fixed-NEC gains alone cannot isolate deterministic adaptation, capacity and random distributions.',
                'selection':'Same original heads, frozen selected generators, one fixed training draw (seed+3000), '
                            'MC8 validation (seed+1000); validation NLL alone selects lambda and iteration.',
                'code_sha256':sha(__file__)}
    path = root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:write_json(path,protocol)
    base.LAMBDAS=LAMBDAS;base.fit_grid=fit_grid
    pack = torch.load('artifacts/datasets/cifar10/baseline_k256_seed0.pt',weights_only=True,map_location='cpu')
    for variant in ['mean_only','adapter_matched']:
        for seed in range(5):
            run = root/f'{variant}_seed{seed}'
            if (run/'head_grid.json').exists():continue
            runner.state.update(phase='waiting_for_gpu',next_case=run.name);runner.save()
            with gpu_lease(f'Sparse deterministic control {run.name}',args.device):
                runner.state.update(phase=run.name);runner.save()
                base.run_case(pack,run,variant,seed,args)
                report = json.loads((run/'head_grid.json').read_text())
                heads = torch.load(run/'heads.pt',weights_only=True,map_location='cpu')
                for row,weight in zip(report['candidates'],heads['weights']):
                    row['nec_exact_nonzero']=weight.ne(0).float().sum(-1).mean().item()
                report['selected']=report['candidates'][report['selection']['selected_index']]
                report['solver']={'name':'Full-gradient FISTA; exact soft threshold; bias unpenalized',
                                  'max_iterations':args.epochs,'validation_interval':25,
                                  'fit_code_sha256':sha('pi_experiments/proximal_sparsity.py')}
                report['protocol']='Frozen selected generator; fixed training concepts; all L1 heads share the original head initialization and validation draws; full-gradient proximal optimization.'
                write_json(run/'head_grid.json',report)
            runner.state['completed_commands'].append({'name':run.name,'returncode':0,'finished_at':time.time()});runner.save()
    selected = {}
    for variant in VARIANTS:
        for seed in range(5):
            run = root/f'{variant}_seed{seed}'
            heads = torch.load(run/'heads.pt',weights_only=True,map_location='cpu')
            index = heads['selection']['selected_index']
            selected[run.name] = (heads['weights'][index],heads['biases'][index])
    targets = common_targets([w for w,_ in selected.values()])
    exact_protocol = {'targets':targets,'nec':'Every class has exactly k strict nonzeros, also above 1e-5.',
                      'selection':'Common feasible count from validation-selected weights only; test labels not read.',
                      'sampling':'Shared MC8 concept draws across all k (seed+2000); mean inference on the same head.',
                      'head_sha256':{name:sha(root/name/'heads.pt') for name in selected},'device':args.device}
    write_json(root/'exact_protocol.json',exact_protocol)
    for variant in VARIANTS:
        for seed in range(5):
            run = root/f'{variant}_seed{seed}'
            if (run/'exact_metrics.json').exists():continue
            runner.state.update(phase='waiting_for_gpu',next_case=run.name+'_exact_nec');runner.save()
            with gpu_lease(f'Sparse capacity exact NEC {run.name}',args.device):
                runner.state.update(phase=run.name+'_exact_nec');runner.save()
                model,_ = load_model(pack,Path('outputs/study_v1/cifar10')/run.name/'model.pt',args.device)
                draws = base.concept_draws(model,pack['test'],8,seed+2000)
                means = base.concept_draws(model,pack['test'],1,seed+2000,mean=True)
                weight,bias = selected[run.name];bias=bias.to(args.device)
                arrays = {kind:{str(k):base.probabilities(values,truncate_exact(weight,k).to(args.device),bias)
                                for k in targets} for kind,values in [('mc8',draws),('mean',means)]}
                del model,draws,means
                if str(args.device).startswith('cuda'):torch.cuda.empty_cache()
            arrays['labels']=pack['test']['labels'];atomic_save(arrays,run/'exact_predictions.pt')
            write_json(run/'exact_metrics.json',{'variant':variant,'seed':seed,
                      'points':[{'nec_per_class':k,**{kind:task_metrics(arrays[kind][str(k)],arrays['labels'])
                                                    for kind in ['mc8','mean']}} for k in targets]})
            runner.state['completed_commands'].append({'name':run.name+'_exact_nec','returncode':0,'finished_at':time.time()});runner.save()
    summarize(root,Path('reports/sparse_capacity_v1/cifar10'),targets)
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
