"""Compare exact per-class NEC where all selected heads support the same count."""

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

from .analyze import load_model, task_metrics
from .head_sparsity import concept_draws, probabilities
from .prepare import atomic_save
from .resources import gpu_lease
from .statistics import hierarchical_paired_bootstrap
from .suite import Runner, pid_identity


THRESHOLD = 1e-5


def common_targets(weights):
    admissible = min(int((weight.abs()>THRESHOLD).sum(-1).min()) for weight in weights)
    return sorted({k for k in [1,5,10,20,40,80,160] if k<=admissible} | {admissible})


def truncate_exact(weight, count):
    assert count>=0 and ((weight.abs()>THRESHOLD).sum(-1)>=count).all()
    order = weight.abs().argsort(-1,descending=True)
    mask = torch.zeros_like(weight).scatter_(1,order[:,:count],1)
    result = weight*mask
    assert (result.ne(0).sum(-1)==count).all()
    assert ((result.abs()>THRESHOLD).sum(-1)==count).all()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    parser.add_argument('--heads',default='outputs/proximal_sparsity_v2/cifar10')
    parser.add_argument('--upstream-status',default='artifacts/logs/proximal_sparsity_v2/status.json')
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args();torch.set_num_threads(8)
    runner = Runner('artifacts/logs/matched_nec_v2')
    identity = pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid)==identity:
        runner.state.update(phase='waiting_for_selected_heads',waiting_pid=args.wait_pid,checked_at=time.time())
        runner.save();time.sleep(10)
    state = json.loads(Path(args.upstream_status).read_text())
    assert state['manager_pid']==args.wait_pid and state['phase']=='complete'
    selected={}; source_hashes={}
    for variant in ['clean','pi']:
        for seed in range(5):
            key=f'{variant}_seed{seed}';path=Path(args.heads)/key/'heads.pt'
            record=torch.load(path,weights_only=True,map_location='cpu')
            index=record['selection']['selected_index']
            selected[key]=(record['weights'][index],record['biases'][index])
            source_hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    targets=common_targets([weight for weight,_ in selected.values()])
    root=Path('outputs/matched_nec_v2/cifar10');root.mkdir(parents=True,exist_ok=True)
    protocol={'targets':targets,'threshold':THRESHOLD,'head_sha256':source_hashes,
              'selection':'Common admissible count determined only from validation-selected head weights, not test labels.',
              'nec':'Exactly k nonzero coefficients, also exceeding 1e-5, in every class of every compared head.',
              'inference':'Frozen original models, eight draws; average probabilities after each truncated head.'}
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2))
    pack=torch.load('artifacts/datasets/cifar10/baseline_k256_seed0.pt',weights_only=True,map_location='cpu')
    labels=pack['test']['labels']
    for variant in ['clean','pi']:
        for seed in range(5):
            key=f'{variant}_seed{seed}';run=root/key
            if (run/'metrics.json').exists():continue
            runner.state.update(phase='waiting_for_gpu',next_case=key);runner.save()
            with gpu_lease(f'Exact NEC {key}',args.device):
                runner.state.update(phase=key);runner.save()
                model,_=load_model(pack,Path('outputs/study_v1/cifar10')/key/'model.pt',args.device)
                draws=concept_draws(model,pack['test'],8,seed+2000)
                weight,bias=selected[key];bias=bias.to(args.device)
                arrays={str(k):probabilities(draws,truncate_exact(weight,k).to(args.device),bias) for k in targets}
                del draws,model
                if str(args.device).startswith('cuda'):torch.cuda.empty_cache()
            atomic_save({'labels':labels,**arrays},run/'predictions.pt')
            report={'seed':seed,'variant':variant,'head_sha256':source_hashes[key],
                    'points':[{'nec_per_class':k,**task_metrics(arrays[str(k)],labels)} for k in targets]}
            temp=run/'metrics.partial.json';temp.write_text(json.dumps(report,indent=2));temp.replace(run/'metrics.json')
            runner.state['completed_commands'].append({'name':key,'returncode':0,'finished_at':time.time()});runner.save()
    groups={v:[json.loads((root/f'{v}_seed{s}/metrics.json').read_text()) for s in range(5)] for v in ['clean','pi']}
    a=[torch.load(root/f'clean_seed{s}/predictions.pt',weights_only=True,map_location='cpu') for s in range(5)]
    b=[torch.load(root/f'pi_seed{s}/predictions.pt',weights_only=True,map_location='cpu') for s in range(5)]
    for x,y in zip(a,b):assert torch.equal(x['labels'],y['labels'])
    comparisons=[{'nec_per_class':k,**hierarchical_paired_bootstrap(
                 np.stack([r[str(k)].numpy() for r in a]),np.stack([r[str(k)].numpy() for r in b]),labels.numpy())} for k in targets]
    destination=Path('reports/matched_nec_v2/cifar10');destination.mkdir(parents=True,exist_ok=True)
    (destination/'results.json').write_text(json.dumps({'protocol':protocol,'groups':groups,'pi_minus_clean':comparisons},indent=2))
    fig,axis=plt.subplots(figsize=(7,5),constrained_layout=True)
    for variant,rows in groups.items():
        values=np.array([[r['accuracy']*100 for r in row['points']] for row in rows])
        axis.errorbar(targets,values.mean(0),yerr=values.std(0,ddof=1),fmt='-o',capsize=3,label=variant)
    axis.set_xlabel('Exactly retained nonzero coefficients in every class')
    axis.set_ylabel('Test accuracy (%)');axis.legend();axis.grid(alpha=.2)
    for extension in ['svg','pdf','png']:fig.savefig(destination/f'matched_nec.{extension}',dpi=160)
    plt.close(fig)
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
