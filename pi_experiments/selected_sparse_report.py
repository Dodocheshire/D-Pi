"""Compare complete five-seed validation-selected heads before NEC truncation."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from .analyze import task_metrics
from .statistics import hierarchical_paired_bootstrap
from .visible_semantics import sha, write_json


def main():
    torch.set_num_threads(8)
    root=Path('outputs/sparse_capacity_v1/cifar10');groups={};predictions={};sources={}
    for variant in ['clean','mean_only','adapter_matched','pi']:
        runs=[root/f'{variant}_seed{s}' for s in range(5)]
        if not all(all((r/f).exists() for f in ['head_grid.json','heads.pt','predictions.pt']) for r in runs):continue
        rows=[];packs=[];hashes=[]
        for seed,run in enumerate(runs):
            r=json.loads((run/'head_grid.json').read_text());h=torch.load(run/'heads.pt',weights_only=True,map_location='cpu')
            p=torch.load(run/'predictions.pt',weights_only=True,map_location='cpu');selected=h['selection']['selected_index']
            assert h['selection']['seed']==seed and h['selection']['variant']==variant
            assert h['selection']['source_sha256']==sha(h['selection']['source_checkpoint'])
            assert r['selection']==h['selection'] and r['selected']==r['candidates'][selected]
            actual=task_metrics(p['mc8'],p['labels'])
            for key,value in actual.items():assert abs(value-r['selected']['mc8'][key])<1e-7
            for kind in ['mc8','mean']:
                assert torch.isfinite(p[kind]).all() and (p[kind]>=0).all()
                assert torch.allclose(p[kind].sum(-1),torch.ones(len(p['labels'])),atol=1e-5)
            rows.append({'seed':seed,'accuracy':actual['accuracy'],
                         'selected_l1':h['selection']['selected_l1'],'best_iteration':r['selected']['best_epoch'],
                         'strict_nec':float(h['weights'][selected].ne(0).sum(-1).float().mean()),
                         'threshold_nec':float((h['weights'][selected].abs()>1e-5).sum(-1).float().mean())})
            packs.append(p);hashes.append({f:sha(run/f) for f in ['heads.pt','predictions.pt','head_grid.json']})
        groups[variant]={'seeds':list(range(5)),'rows':rows,
                         'accuracy_mean':float(np.mean([r['accuracy'] for r in rows])),
                         'accuracy_std':float(np.std([r['accuracy'] for r in rows],ddof=1)),
                         'strict_nec_mean':float(np.mean([r['strict_nec'] for r in rows]))}
        predictions[variant]=packs;sources[variant]=hashes
    contrasts=[]
    if 'pi' in predictions:
        labels=predictions['pi'][0]['labels']
        for control in ['clean','mean_only','adapter_matched']:
            if control not in predictions:continue
            a,b=predictions[control],predictions['pi']
            assert all(torch.equal(r['labels'],labels) for r in a+b)
            interval=hierarchical_paired_bootstrap(np.stack([r['mc8'].numpy() for r in a]),
                                                  np.stack([r['mc8'].numpy() for r in b]),labels.numpy())
            interval['protocol']='Paired adaptation seeds and common test images; frozen selected generators and validation-selected proximal refit heads. Each model retains its own selected NEC.'
            contrasts.append({'control':control,'adapted':'pi',**interval})
    result={'groups':groups,'contrasts':contrasts,'source_sha256':sources,'checked_at':time.time(),
            'code_sha256':sha(__file__),'limits':'Only complete five-seed groups. Unpruned validation-selected heads may have different NEC; '
                'these are not exact-NEC contrasts or sigma-only causal effects. The exact-NEC four-control study remains separate. '
                'Descriptive unadjusted intervals condition on the fixed bank, selected generators, heads and inference draws.'}
    write_json('reports/sparse_capacity_v1/cifar10/selected_head_comparisons.json',result)
    print(json.dumps({'completed_groups':list(groups),'comparisons':len(contrasts)},indent=2))


if __name__=='__main__':main()
