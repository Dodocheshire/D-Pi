"""Paired image/seed intervals for full-test, fixed-concept localization."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .suite import Runner, pid_identity
from .visible_semantics import ready, write_json


METRICS = ['grid_attribution','grid_max_point','foreground_attribution','foreground_max_point','part_hit']
VARIANTS = ['clean','fixed','mean_only','adapter_matched','pi']
BANKS = ['RN50_GDINO','RN50_SAM2']
COMPLETED_CACHE = {}


def paired_ratio_bootstrap(control, adapted, denominator, seed=731, draws=2000):
    """Resample images together; weight eligible concepts, not image proportions."""
    a,b,d = map(lambda x:np.asarray(x,dtype=np.float64),(control,adapted,denominator))
    assert a.shape==b.shape and a.ndim==3 and d.shape==a.shape[1:]
    assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(d).all()
    assert (d>=0).all() and (d.sum(0)>0).all() and draws>0
    assert (a>=0).all() and (b>=0).all() and (a<=d[None]+1e-10).all() and (b<=d[None]+1e-10).all()
    seeds,images,metrics=a.shape
    delta=b-a
    point=delta.mean(0).sum(0)/d.sum(0)
    rng=np.random.default_rng(seed);values=[]
    for start in range(0,draws,25):
        batch=min(25,draws-start)
        sampled_seeds=rng.integers(seeds,size=(batch,seeds))
        per_image=delta[sampled_seeds].mean(1)
        sampled_images=rng.integers(images,size=(batch,images))
        numer=np.take_along_axis(per_image,sampled_images[:,:,None],axis=1).sum(1)
        denom=d[sampled_images].sum(1)
        estimates=np.full_like(numer,np.nan)
        np.divide(numer,denom,out=estimates,where=denom>0)
        values.append(estimates)
    values=np.concatenate(values)
    result=[]
    for i in range(metrics):
        valid=np.isfinite(values[:,i]);assert valid.any()
        result.append({'delta':float(point[i]),'ci95':np.quantile(values[valid,i],[.025,.975]).tolist(),
                       'valid_draws':int(valid.sum()),'undefined_draws':int((~valid).sum())})
    return {'metrics':result,'adaptation_seeds':seeds,'test_images':images,'draws':draws,'seed':seed,
            'eligible_concepts_per_metric':d.sum(0).tolist(),
            'protocol':'Paired adaptation seeds and one common image resample. Aggregate eligible-concept '
                       'numerators over denominators within each resample; zero heatmaps score zero. '
                       'Fixed clean top-5 targets, grids, part eligibility and visible-support mapping.'}


def arrays(report):
    assert not report['debug_limit'] and report['evaluated_images']==report['test_images']
    count=report['test_images'];assert len(report['rows'])==count
    numer=np.zeros((count,len(METRICS)));denom=np.zeros_like(numer);identities=[]
    for i,row in enumerate(report['rows']):
        assert row['row']==i
        assert [c['concept'] for c in row['grid']]==row['concepts']
        assert [c['concept'] for c in row['foreground']]==row['concepts']
        identities.append([row['source_index'],row['target_quadrant'],row['concepts']])
        denom[i,:2]=len(row['grid'])
        denom[i,2:4]=row['foreground_candidates']
        denom[i,4]=row['part_localizable']
        for c in row['grid']:
            if c['metrics'] is not None:
                numer[i,0]+=c['metrics']['relative_attribution'];numer[i,1]+=c['metrics']['max_point_accuracy']
        for c in row['foreground']:
            if c['metrics'] is not None:
                numer[i,2]+=c['metrics']['foreground_attribution'];numer[i,3]+=c['metrics']['foreground_max_point']
        numer[i,4]=sum(p['hit_radius_0_1_diagonal'] for p in row['parts'])
    expected=[report['grid_all_concepts_mean']['relative_attribution'],report['grid_all_concepts_mean']['max_point_accuracy'],
              report['foreground_all_candidate_mean']['foreground_attribution'],report['foreground_all_candidate_mean']['foreground_max_point'],
              report['part_hit_rate_all_localizable']]
    actual=numer.sum(0)/denom.sum(0)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    identity=hashlib.sha256(json.dumps(identities,separators=(',',':')).encode()).hexdigest()
    return numer,denom,identity


def summarize():
    completed=[]
    for bank in BANKS:
        root=Path('outputs/visible_semantics_v1')/bank
        for control in VARIANTS[:-1]:
            paths={v:[root/f'{v}_seed{s}/localization.json' for s in range(5)] for v in [control,'pi']}
            if not all(ready(p) for group in paths.values() for p in group):continue
            signature=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for group in paths.values() for p in group)
            if (bank,control) in COMPLETED_CACHE and COMPLETED_CACHE[bank,control][0]==signature:
                completed.append(COMPLETED_CACHE[bank,control][1]);continue
            hashes={v:[hashlib.sha256(p.read_bytes()).hexdigest() for p in ps] for v,ps in paths.items()}
            destination=Path('reports/localization_uncertainty_v1')/bank/f'{control}_vs_pi.json'
            if destination.exists():
                result=json.loads(destination.read_text());assert result['source_sha256']==hashes
            else:
                predictions={};common_den=None;common_identity=None;grid=None;mapping=None
                for v,ps in paths.items():
                    items=[]
                    for path in ps:
                        report=json.loads(path.read_text());a,d,identity=arrays(report)
                        if common_den is None:
                            common_den=d;common_identity=identity;grid=report['grid_plan_sha256'];mapping=report['visible_mapping_sha256']
                        assert np.array_equal(d,common_den) and identity==common_identity,(path,'Eligibility or targets differ')
                        assert grid==report['grid_plan_sha256'] and mapping==report['visible_mapping_sha256']
                        assert report['test_images']==5794
                        items.append(a)
                    predictions[v]=np.stack(items)
                result=paired_ratio_bootstrap(predictions[control],predictions['pi'],common_den)
                result['metrics']={key:value for key,value in zip(METRICS,result['metrics'])}
                result.update(bank=bank,control=control,adapted='pi',source_sha256=hashes,
                              fixed_targets_sha256=common_identity,grid_plan_sha256=grid,mapping_sha256=mapping,
                              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                write_json(destination,result)
                print('Computed full-test localization intervals',bank,control,flush=True)
            completed.append(result)
            COMPLETED_CACHE[bank,control]=(signature,result)
    write_json('reports/localization_uncertainty_v1/results.json',{'comparisons':completed,
               'limits':'Complete five-seed comparisons only. Descriptive 95% intervals without multiplicity '
                        'correction, conditional on fixed bank, mapping, concept targets, grid and one MC8 CAM '
                        'realization. Grid distractor images are shared/reused; image bootstrap conditions on '
                        'this fixed grid plan and does not resample the source pool or model initialization.'})
    return len(completed)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    if args.wait_pid is None:summarize();return
    state_path=Path('artifacts/logs/visible_semantics_v1/status.json')
    assert json.loads(state_path.read_text())['manager_pid']==args.wait_pid
    identity=pid_identity(args.wait_pid);runner=Runner('artifacts/logs/localization_uncertainty_v1')
    while True:
        count=summarize()
        runner.state.update(phase='waiting_for_full_test_localization',completed_comparisons=count,checked_at=time.time());runner.save()
        if identity is None or pid_identity(args.wait_pid)!=identity:break
        time.sleep(30)
    state=json.loads(state_path.read_text());assert state['manager_pid']==args.wait_pid and state['phase']=='complete'
    assert summarize()==8
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
