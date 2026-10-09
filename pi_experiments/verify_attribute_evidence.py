"""Reproduce full CUB attribute evidence from native annotations and saved scores."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from .large_proposal_prewarm import file_sha256
from .semantic_uncertainty import paired_attribute_bootstrap
from .visible_semantics import write_json


VARIANTS = ['clean','mean_only','adapter_matched','pi']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',choices=['RN50_SAM2','RN50_GDINO'],default='RN50_SAM2')
    args=parser.parse_args();torch.set_num_threads(8)
    directory=Path('artifacts/datasets')/('cub' if args.bank=='RN50_SAM2' else 'cub_gdino')
    root=Path('outputs/visible_semantics_v1')/args.bank
    hashes={};snapshots={}

    def read(path):
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        hashes[str(path)]=digest;snapshots[(path.stem,digest)]=raw
        return json.loads(raw)

    def load(path):
        hashes[str(path)]=file_sha256(path)
        return torch.load(path,weights_only=True,map_location='cpu')

    pack=load(directory/'baseline_k2048_seed0.pt')
    annotations=load(directory/'attributes.pt')
    assert len(pack['test']['labels'])==5794
    groups={};predictions={};rows=[];reference=None
    for variant in VARIANTS:
        values=[];predictions[variant]=[]
        for seed in range(5):
            run=root/f'{variant}_seed{seed}'
            report=read(run/'cub_concepts.json')
            p=load(run/'semantic_predictions.pt')
            checkpoint=load(run/'model.pt')
            metadata=p['metadata']
            assert metadata['checkpoint_sha256']==report['source_checkpoint_sha256']==hashes[str(run/'model.pt')]
            assert metadata['attribute_report_sha256']==hashes[str(run/'cub_concepts.json')]
            assert metadata['evaluation_split']==report['evaluation_split']=='test'
            assert metadata['samples']==8 and metadata['sampling_seed']==2110
            assert metadata['min_certainty']==report['min_certainty']==3
            assert metadata['device']=='cpu' and not checkpoint['config'].get('validation_only')
            assert checkpoint['config']['seed']==seed and not checkpoint['config'].get('refit_epochs')
            for key,original in [('projection.centers','centers'),('projection.mean','mean'),
                                 ('projection.scale','scale'),('head.weight','weight'),('head.bias','bias')]:
                assert torch.equal(checkpoint['state_dict'][key],pack[original]),(run,key)
            mapping_path=Path(report['mapping_file']);mapping=read(mapping_path)
            assert hashes[str(mapping_path)]==report['mapping_sha256']==metadata['mapping_sha256']
            ids=p['attribute_ids'];count=len(ids)
            assert count==report['evaluated_attributes'] and torch.equal(ids,ids.unique(sorted=True))
            assert ((ids>=0)&(ids<312)).all() and count>0
            details=sorted(report['details'],key=lambda r:r['attribute'])
            assert ids.tolist()==[r['attribute'] for r in details]
            assert all(str(int(i)) in mapping['assignments'] for i in ids)
            assert torch.equal(p['indices'],pack['test']['indices'])
            assert torch.equal(p['positive'],annotations['test']['attributes'][:,ids]>0)
            assert torch.equal(p['known'],annotations['test']['certainty'][:,ids]>=3)
            assert p['scores'].shape==(5794,count) and torch.isfinite(p['scores']).all()
            assert torch.equal(p['thresholds'],p['scores'].new_tensor([r['threshold'] for r in details]))
            assert torch.equal(p['predicted'],p['scores']>=p['thresholds'])
            fields=['attribute_ids','indices','positive','known']
            if reference is None:reference={key:p[key] for key in fields}
            else:assert all(torch.equal(p[key],reference[key]) for key in fields)
            pos,neg=p['positive']&p['known'],~p['positive']&p['known']
            assert (pos.sum(0)>0).all() and (neg.sum(0)>0).all()
            balanced=.5*((p['predicted']&pos).sum(0).double()/pos.sum(0)
                          +(~p['predicted']&neg).sum(0).double()/neg.sum(0))
            aurocs=[roc_auc_score(p['positive'][:,i][p['known'][:,i]].numpy(),
                                  p['scores'][:,i][p['known'][:,i]].numpy()) for i in range(count)]
            for i,detail in enumerate(details):
                assert abs(balanced[i].item()-detail['balanced_accuracy'])<1e-10
                assert abs(aurocs[i]-detail['auroc'])<1e-10
            metrics={'attribute_macro_balanced_accuracy':balanced.mean().item(),
                     'attribute_macro_auroc':float(np.mean(aurocs))}
            for key,value in metrics.items():assert abs(value-report[key])<1e-10
            values.append(metrics);predictions[variant].append(p['predicted'].numpy())
            rows.append({'variant':variant,'seed':seed,'native_truth_exact':True,
                         'source_model_frozen_invariants_exact':True,'thresholds_from_fixed_report_exact':True,
                         'per_attribute_metrics_reproduced':True,'evaluated_attributes':count})
        groups[variant]={key:{'mean':float(np.mean([r[key] for r in values])),
                              'std':float(np.std([r[key] for r in values],ddof=1))} for key in values[0]}
        print('VERIFIED',args.bank,variant,5,'full-test attribute caches',flush=True)
    comparisons=[]
    for control in VARIANTS[:-1]:
        path=Path('reports/semantic_uncertainty_v1')/args.bank/f'{control}_vs_pi.json'
        row=read(path)
        assert row['adaptation_seeds']==5 and row['test_images']==5794
        assert row['attributes']==len(reference['attribute_ids'])
        assert row['attribute_ids']==reference['attribute_ids'].tolist()
        assert row['provenance']['control_predictions_sha256']==[
            hashes[str(root/f'{control}_seed{s}'/'semantic_predictions.pt')] for s in range(5)]
        assert row['provenance']['pi_predictions_sha256']==[
            hashes[str(root/f'pi_seed{s}'/'semantic_predictions.pt')] for s in range(5)]
        actual=paired_attribute_bootstrap(np.stack(predictions[control]),np.stack(predictions['pi']),
                                         reference['positive'].numpy(),reference['known'].numpy(),
                                         seed=row['seed'],draws=row['draws'])
        for key in ['balanced_accuracy_delta','minimum_positive_samples','minimum_negative_samples',
                    'valid_draws','undefined_draws']:
            assert abs(actual[key]-row[key])<1e-12,(control,key)
        assert np.allclose(actual['ci95'],row['ci95'],rtol=0,atol=1e-12)
        comparisons.append(row)
        print('REPRODUCED',args.bank,control,'paired interval',flush=True)
    snapshot=Path('artifacts/report_assets/attribute_evidence_source_snapshots');snapshot.mkdir(exist_ok=True)
    for (stem,digest),raw in snapshots.items():(snapshot/f'{stem}_{digest}.json').write_bytes(raw)
    audit={'bank':args.bank,'groups':groups,'comparisons':comparisons,'verified_checkpoints':rows,
           'test_images':5794,'evaluated_attributes':len(reference['attribute_ids']),
           'mapped_attributes':report['mapped_attributes'],'attribute_ids':reference['attribute_ids'].tolist(),
           'source_sha256':hashes,'source_snapshots_directory':str(snapshot.resolve()),
           'code_sha256':file_sha256(__file__),'checked_at':time.time(),
           'limits':'Native held-out annotations, fixed visible-support mapping and saved validation thresholds. '
                    'Reproduces each attribute BA/AUROC and all three main paired intervals. No re-training or '
                    're-calibration; intervals condition on one fixed bank and sampling realization and are '
                    'not multiplicity-adjusted. Different banks have different attribute cohorts.'}
    write_json(Path('reports/attribute_evidence_verification_v1')/f'{args.bank}.json',audit)
    print(json.dumps({'bank':args.bank,'verified_checkpoints':len(rows),'attributes':audit['evaluated_attributes'],
                      'paired_intervals_reproduced':len(comparisons)},indent=2))


if __name__=='__main__':main()
