"""Paired adaptation-seed/image uncertainty for fixed CUB attribute thresholds."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import torch

from .analyze import collect, load_model
from .cub_metrics import attribute_scores
from .prepare import atomic_save
from .suite import Runner, pid_identity
from .visible_semantics import BANKS, ready, sha, write_json


def paired_attribute_bootstrap(controls, adapted, positive, known, seed=731, draws=2000):
    """Resample common images and paired seeds, preserving the attribute cohort."""
    controls, adapted = np.asarray(controls,dtype=bool), np.asarray(adapted,dtype=bool)
    positive, known = np.asarray(positive,dtype=bool), np.asarray(known,dtype=bool)
    assert controls.shape == adapted.shape and controls.ndim == 3
    assert controls.shape[1:] == positive.shape == known.shape
    seeds, images, attributes = controls.shape
    assert seeds > 0 and images > 0 and attributes > 0 and draws > 0
    pos, neg = positive & known, ~positive & known
    p, n = pos.sum(0), neg.sum(0)
    assert (p>0).all() and (n>0).all()
    difference = adapted.astype(np.int8)-controls.astype(np.int8)
    dp = (difference*pos).astype(np.float32)
    dn = (-difference*neg).astype(np.float32)
    point = .5*(dp.mean(0,dtype=np.float64).sum(0)/p + dn.mean(0,dtype=np.float64).sum(0)/n).mean()
    rng = np.random.default_rng(seed)
    values = []
    with threadpool_limits(limits=8,user_api='blas'):
        for start in range(0,draws,50):
            batch = min(50,draws-start)
            seed_ids = rng.integers(seeds,size=(batch,seeds))
            seed_weights = np.stack([np.bincount(row,minlength=seeds) for row in seed_ids])/seeds
            image_ids = rng.integers(images,size=(batch,images))
            counts = np.stack([np.bincount(row,minlength=images) for row in image_ids]).astype(np.float32)
            bp, bn = counts@pos.astype(np.float32), counts@neg.astype(np.float32)
            valid = ((bp>0)&(bn>0)).all(-1)
            tp, tn = np.zeros((batch,attributes)), np.zeros((batch,attributes))
            for index in range(seeds):
                tp += seed_weights[:,index,None]*(counts@dp[index])
                tn += seed_weights[:,index,None]*(counts@dn[index])
            values.extend((.5*(tp[valid]/bp[valid]+tn[valid]/bn[valid])).mean(-1).tolist())
    assert values, 'All replicates lost an attribute class; interval unavailable for this cohort'
    return {'balanced_accuracy_delta':float(point),'ci95':np.quantile(values,[.025,.975]).tolist(),
            'adaptation_seeds':seeds,'test_images':images,'attributes':attributes,
            'draws':draws,'valid_draws':len(values),'undefined_draws':draws-len(values),'seed':seed,
            'minimum_positive_samples':int(p.min()),'minimum_negative_samples':int(n.min()),
            'protocol':'Paired adaptation seeds and one common image resample; fixed validation thresholds, '
                       'semantic mapping and attribute cohort. Balanced accuracy normalizes the positive and negative classes separately.',
            'undefined_policy':'Discard a replicate if either class is absent for any fixed evaluated attribute; '
                               'disclose the count rather than changing its attribute cohort.'}


def cache_scores(run, directory):
    report = json.loads((run/'cub_concepts.json').read_text())
    pack = torch.load(directory/'baseline_k2048_seed0.pt',weights_only=True,map_location='cpu')
    annotations = torch.load(directory/'attributes.pt',weights_only=True,map_location='cpu')
    mapping_path = Path(report['mapping_file'])
    assert report['mapping_sha256'] == sha(mapping_path) and report['device'] == 'cpu'
    assert report['source_checkpoint_sha256'] == sha(run/'model.pt')
    assignments = {int(a):v for a,v in json.loads(mapping_path.read_text())['assignments'].items()}
    model, config = load_model(pack,run/'model.pt','cpu')
    assert not config.get('validation_only')
    outputs = collect(model,pack['test'],samples=8,seed=2110)
    scores = attribute_scores(outputs['concepts'],assignments)
    lookup = {a:i for i,a in enumerate(sorted(assignments))}
    details = sorted(report['details'],key=lambda r:r['attribute'])
    ids = [r['attribute'] for r in details]
    scores = torch.tensor(scores[:,[lookup[a] for a in ids]])
    thresholds = torch.tensor([r['threshold'] for r in details],dtype=scores.dtype)
    positive = annotations['test']['attributes'][:,ids]>0
    known = annotations['test']['certainty'][:,ids]>=3
    predicted = scores>=thresholds
    assert len(positive) == len(pack['test']['labels'])
    pos, neg = known & positive, known & ~positive
    if ids:
        actual = .5*((predicted&pos).sum(0).double()/pos.sum(0) + (~predicted&neg).sum(0).double()/neg.sum(0)).mean().item()
        auc = float(np.mean([roc_auc_score(positive[:,i][known[:,i]].numpy(),scores[:,i][known[:,i]].numpy())
                             for i in range(len(ids))]))
        assert abs(actual-report['attribute_macro_balanced_accuracy']) < 1e-10
        assert abs(auc-report['attribute_macro_auroc']) < 1e-10
    else:
        actual, auc = None, None
    metadata = {'mapping_sha256':sha(mapping_path),'checkpoint_sha256':sha(run/'model.pt'),
                'attribute_report_sha256':sha(run/'cub_concepts.json'),'code_sha256':sha(__file__),
                'samples':8,'sampling_seed':2110,'device':'cpu','min_certainty':3,
                'evaluation_split':'test','balanced_accuracy':actual,'auroc':auc,
                'limits':'Reproduce the unchanged CPU visible-support evaluation. Thresholds are read from its original validation calibration.'}
    atomic_save({'scores':scores,'predicted':predicted,'attribute_ids':torch.tensor(ids),
                 'positive':positive,'known':known,'indices':pack['test']['indices'],
                 'thresholds':thresholds,'metadata':metadata},run/'semantic_predictions.pt')
    print('Cached attribute predictions',run,len(ids),flush=True)


def summarize():
    comparisons = []
    for bank,_,_,variants in BANKS:
        root = Path('outputs/visible_semantics_v1')/bank
        targets = [root/f'pi_seed{s}/semantic_predictions.pt' for s in range(5)]
        if not all(p.exists() for p in targets):continue
        b = [torch.load(p,weights_only=True,map_location='cpu') for p in targets]
        for variant in variants:
            if variant == 'pi':continue
            sources = [root/f'{variant}_seed{s}/semantic_predictions.pt' for s in range(5)]
            if not all(p.exists() for p in sources):continue
            destination = Path('reports/semantic_uncertainty_v1')/bank/f'{variant}_vs_pi.json'
            provenance = {'control_predictions_sha256':[sha(p) for p in sources],
                          'pi_predictions_sha256':[sha(p) for p in targets]}
            if destination.exists():
                result = json.loads(destination.read_text())
                assert result['provenance'] == provenance
            else:
                a = [torch.load(p,weights_only=True,map_location='cpu') for p in sources]
                for left,right in zip(a,b):
                    for field in ['attribute_ids','positive','known','indices']:
                        assert torch.equal(left[field],right[field]), (bank,variant,field)
                    assert left['metadata']['mapping_sha256'] == right['metadata']['mapping_sha256']
                if len(b[0]['attribute_ids']):
                    result = paired_attribute_bootstrap(
                        np.stack([r['predicted'].numpy() for r in a]),np.stack([r['predicted'].numpy() for r in b]),
                        b[0]['positive'].numpy(),b[0]['known'].numpy())
                else:
                    result = {'balanced_accuracy_delta':None,'ci95':None,'attributes':0,'reason':'No evaluable mapped attributes'}
                result.update(bank=bank,control=variant,adapted='pi',provenance=provenance,
                              attribute_ids=b[0]['attribute_ids'].tolist(),code_sha256=sha(__file__))
                write_json(destination,result)
            comparisons.append(result)
    write_json('reports/semantic_uncertainty_v1/results.json',{'comparisons':comparisons,
               'limits':'Only complete five-seed comparisons. Conditional on fixed prototypes, mapping, '
                        'validation thresholds and one inference realization. AUROC means and seed SD remain '
                        'in visible_semantics_v1; these intervals concern macro balanced accuracy. '
                        'Intervals are descriptive and unadjusted for multiple comparisons.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    args = parser.parse_args();torch.set_num_threads(8)
    upstream_path = Path('artifacts/logs/visible_semantics_v1/status.json')
    assert json.loads(upstream_path.read_text())['manager_pid'] == args.wait_pid
    identity = pid_identity(args.wait_pid)
    runner = Runner('artifacts/logs/semantic_uncertainty_v1')
    while True:
        for bank,_,directory,variants in BANKS:
            for variant in variants:
                for seed in range(5):
                    run = Path('outputs/visible_semantics_v1')/bank/f'{variant}_seed{seed}'
                    if not ready(run/'cub_concepts.json') or (run/'semantic_predictions.pt').exists():continue
                    runner.state.update(phase=f'{bank}_{run.name}_scores');runner.save()
                    cache_scores(run,Path(directory))
                    runner.state['completed_commands'].append({'name':f'{bank}_{run.name}','returncode':0,'finished_at':time.time()})
                    runner.save();summarize()
        if identity is None or pid_identity(args.wait_pid) != identity:break
        runner.state.update(phase='waiting_for_visible_semantic_reports',checked_at=time.time());runner.save();time.sleep(10)
    upstream = json.loads(upstream_path.read_text())
    assert upstream['manager_pid'] == args.wait_pid and upstream['phase'] == 'complete'
    missing = [str(path) for bank,_,_,variants in BANKS for variant in variants for seed in range(5)
               if not (path:=Path('outputs/visible_semantics_v1')/bank/f'{variant}_seed{seed}/semantic_predictions.pt').exists()]
    assert not missing, missing
    summarize();runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
