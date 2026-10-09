"""Follow completed core Pi models across datasets/backbones with budget curves."""

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .sampling_budget import BUDGETS
from .suite import CLUSTERS, Runner, pid_identity


def specifications():
    cases = [(f'RN50/{d}', Path('outputs/study_v1')/d, Path('artifacts/datasets')/d) for d in CLUSTERS]
    cases.append(('RN50/cub_gdino', Path('outputs/gdino_v1/cub'), Path('artifacts/datasets/cub_gdino')))
    cases += [(f'{b}/{d}', Path('outputs/backbone_v1')/b/d, Path('artifacts/backbone_ablation_v1')/b/d)
              for b in ['ViT-B16', 'ViT-L14'] for d in CLUSTERS]
    return cases


def summarize(cases):
    groups = []
    for name, root, _ in cases:
        records = [json.loads(p.read_text()) for s in range(5)
                   if (p := root/f'pi_seed{s}/sampling_budget.json').exists()]
        if not records: continue
        groups.append({'case': name, 'seeds': [s for s in range(5) if (root/f'pi_seed{s}/sampling_budget.json').exists()],
                       'mean': {key: {'mean': float(np.mean([r['mean'][key] for r in records])),
                                      'std': float(np.std([r['mean'][key] for r in records], ddof=1)) if len(records)>1 else None}
                                for key in ['accuracy','nll','ece_15_bins']},
                       'budgets': {str(n): {key: {'mean': float(np.mean([r['inference'][str(n)][key] for r in records])),
                                                'std': float(np.std([r['inference'][str(n)][key] for r in records], ddof=1)) if len(records)>1 else None}
                                           for key in ['accuracy','nll','ece_15_bins','standardized_concept_mean_mse','sampling_top5_stability']}
                                   for n in BUDGETS}})
    destination = Path('reports/sampling_budget_v1'); destination.mkdir(parents=True, exist_ok=True)
    (destination/'results.json').write_text(json.dumps({'groups':groups,'protocol':'Fixed selected models; no retraining or budget selection.'},indent=2))
    if not groups: return
    fig, axes = plt.subplots(1, 2, figsize=(12,4), constrained_layout=True)
    for group in groups:
        for axis, metric, factor in zip(axes,['accuracy','nll'],[100,1]):
            axis.plot(BUDGETS,[group['budgets'][str(n)][metric]['mean']*factor for n in BUDGETS],'-o',label=group['case'])
            axis.set_xscale('log',base=2); axis.set_xticks(BUDGETS,BUDGETS);axis.set_xlabel('Monte Carlo draws')
            axis.set_ylabel('Accuracy (%)' if metric=='accuracy' else 'NLL');axis.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    for extension in ['svg','pdf','png']: fig.savefig(destination/f'budget_curves.{extension}',dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pids', nargs='+', type=int, required=True)
    parser.add_argument('--upstream-statuses', nargs='+', required=True)
    args = parser.parse_args(); assert len(args.wait_pids)==len(args.upstream_statuses)
    runner = Runner('artifacts/logs/sampling_budget_v1'); cases = specifications()
    identities = [pid_identity(pid) for pid in args.wait_pids]
    while True:
        for name, root, directory in cases:
            dataset = 'cub' if name.endswith('cub_gdino') else name.split('/')[-1]
            pack = directory/f'baseline_k{CLUSTERS[dataset]}_seed0.pt'
            if not pack.exists(): continue
            for seed in range(5):
                run = root/f'pi_seed{seed}'
                if (run/'sampling_budget.json').exists(): continue
                if not all((run/f).exists() for f in ['model.pt','analysis.json','predictions.pt']): continue
                if time.time()-(run/'predictions.pt').stat().st_mtime<10: continue
                runner.command(f'{name.replace("/","_")}_seed{seed}',
                               ['pi_experiments.sampling_budget','--pack',str(pack),'--run',str(run)])
                summarize(cases)
        alive = [identity is not None and pid_identity(pid)==identity for pid,identity in zip(args.wait_pids,identities)]
        if not any(alive): break
        runner.state.update(phase='waiting_for_models', checked_at=time.time());runner.save();time.sleep(10)
    upstream = [json.loads(Path(path).read_text()) for path in args.upstream_statuses]
    if any(state['manager_pid']!=pid or state['phase']!='complete' for pid,state in zip(args.wait_pids,upstream)):
        runner.state.update(phase='upstream_failed');runner.save();raise RuntimeError('Expected model studies did not all complete')
    missing = [str(root/f'pi_seed{s}') for _,root,_ in cases for s in range(5)
               if not (root/f'pi_seed{s}/sampling_budget.json').exists()]
    assert not missing, missing
    summarize(cases);runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__': main()
