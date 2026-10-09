"""Wait for successful CUB models and the shared pool, then compare regional P@5."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .suite import Runner,pid_identity


def summarize(root,destination):
    destination.mkdir(parents=True,exist_ok=True)
    groups=[];reports={}
    for variant in ['clean','fixed','mean_only','adapter_matched','pi']:
        rows={}
        for seed in range(5):
            path=root/f'{variant}_seed{seed}/region_retrieval.json'
            if path.exists():rows[seed]=json.loads(path.read_text())
        if not rows:continue
        reports[variant]=rows
        group={'variant':variant,'seeds':sorted(rows),'metrics':{}}
        for key in ['precision_at_5_macro','precision_at_5_distinct_images_macro','excess_precision_over_chance','evaluated_queries']:
            values=[r[key] for r in rows.values() if r[key] is not None]
            group['metrics'][key]={'mean':float(np.mean(values)) if values else None,
                                   'std':float(np.std(values,ddof=1)) if len(values)>1 else None}
        groups.append(group)
    comparisons=[]
    for variant in ['clean','fixed','mean_only','adapter_matched']:
        seeds=sorted(set(reports.get('pi',{}))&set(reports.get(variant,{})))
        if not seeds:continue
        differences=[]
        for seed in seeds:
            a,b=reports[variant][seed],reports['pi'][seed]
            assert a['pool_protocol_sha256']==b['pool_protocol_sha256'] and a['mapping_sha256']==b['mapping_sha256']
            assert [q['concept'] for q in a['queries']]==[q['concept'] for q in b['queries']]
            differences.append(b['precision_at_5_macro']-a['precision_at_5_macro'])
        rng=np.random.default_rng(919)
        boot=np.array(differences)[rng.integers(len(seeds),size=(2000,len(seeds)))].mean(-1)
        comparisons.append({'control':variant,'seeds':seeds,'precision_delta':float(np.mean(differences)),
                            'ci95':np.quantile(boot,[.025,.975]).tolist(),
                            'uncertainty':'Paired adaptation seeds only; fixed region pool and prototype queries. '
                                          'Image, region and prototype-building uncertainty not represented.'})
    (destination/'results.json').write_text(json.dumps({'groups':groups,'pi_paired_comparisons':comparisons},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    parser.add_argument('--upstream-status',required=True)
    parser.add_argument('--region-pid',type=int,required=True)
    parser.add_argument('--runs',required=True)
    parser.add_argument('--model-directory',required=True)
    parser.add_argument('--logs',required=True)
    parser.add_argument('--reports',required=True)
    args=parser.parse_args()
    runner=Runner(args.logs)
    identity=pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid)==identity:
        runner.state.update(phase='waiting_for_cub_models',waiting_pid=args.wait_pid,checked_at=time.time())
        runner.save();time.sleep(10)
    upstream=json.loads(Path(args.upstream_status).read_text())
    if upstream['manager_pid']!=args.wait_pid or upstream['phase']!='complete':
        runner.state.update(phase='upstream_failed',upstream_state=upstream);runner.save()
        raise RuntimeError('Expected CUB model study did not complete successfully')
    identity=pid_identity(args.region_pid)
    while identity is not None and pid_identity(args.region_pid)==identity:
        runner.state.update(phase='waiting_for_region_pool',waiting_pid=args.region_pid,checked_at=time.time())
        runner.save();time.sleep(10)
    if not Path('artifacts/datasets/cub_retrieval/regions.pt').exists():
        runner.state.update(phase='upstream_failed',upstream='retrieval region pool');runner.save()
        raise RuntimeError('Regional pool extraction ended without the complete pool')
    pack=Path(args.model_directory)/'baseline_k2048_seed0.pt'
    root=Path(args.runs)/'cub'
    for variant in ['clean','fixed','mean_only','adapter_matched','pi']:
        for seed in range(5):
            run=root/f'{variant}_seed{seed}'
            if not (run/'region_retrieval.json').exists():
                runner.command(run.name+'_retrieval',['pi_experiments.region_retrieval','--pack',str(pack),
                               '--run',str(run),'--model-directory',args.model_directory])
        summarize(root,Path(args.reports))
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
