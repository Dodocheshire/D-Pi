"""Select internal Pi on validation data, then run five paired seeds and controls."""

import argparse
import json
from pathlib import Path
import time

from .suite import CLUSTERS, Runner, pid_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=['cifar10','cifar100','cub'],default=['cifar10'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--upstream-status')
    parser.add_argument('--output-root',default='outputs/internal_v1')
    parser.add_argument('--logs',default='artifacts/logs/internal_v1')
    parser.add_argument('--grid',choices=['initial','expanded'],default='initial')
    args=parser.parse_args()
    runner=Runner(args.logs)
    if args.wait_pid:
        identity=pid_identity(args.wait_pid)
        while identity is not None and pid_identity(args.wait_pid)==identity:
            runner.state.update(phase='waiting_for_main_study',waiting_pid=args.wait_pid,checked_at=time.time())
            runner.save();time.sleep(10)
        if args.upstream_status:
            upstream=json.loads(Path(args.upstream_status).read_text())
            if upstream['manager_pid']!=args.wait_pid or upstream['phase']!='complete':
                runner.state.update(phase='upstream_failed',upstream_state=upstream);runner.save()
                raise RuntimeError('Expected initial internal study did not complete')
    for dataset in args.datasets:
        directory=Path('artifacts/datasets')/dataset
        pack=directory/f'baseline_k{CLUSTERS[dataset]}_seed0.pt'
        if not pack.exists():raise RuntimeError(f'Complete global baseline first: {pack}')
        if not (directory/'internal/channel_scale.pt').exists():
            runner.command(f'{dataset}_cache',['pi_experiments.internal_cache','--directory',str(directory)])
        output=Path(args.output_root)/dataset;output.mkdir(parents=True,exist_ok=True)
        selected_path=output/'selected_hyperparameters.json'
        if not selected_path.exists():
            candidates=[]
            grid=sorted({(c,e) for c in [1,3,10] for e in [0.1,0.3,1]} |
                        ({(c,e) for c in [0,0.01,0.1,1] for e in [0,0.001,0.01,0.1]}
                         if args.grid=='expanded' else set()))
            for c,e in grid:
                    run=output/'validation_search'/f'c{c:g}_e{e:g}'
                    if not (run/'metrics.json').exists():
                        runner.command(f'{dataset}_search_c{c}_e{e}',
                                       ['pi_experiments.internal_train','--directory',str(directory),'--pack',str(pack),
                                        '--output',str(run),'--validation-only','--resident-maps',
                                        '--concept-weight',str(c),'--energy-weight',str(e)])
                    report=json.loads((run/'metrics.json').read_text())
                    assert report['evaluation_split']=='val'
                    candidates.append({'concept_weight':c,'energy_weight':e,'val_nll':report['inference']['mc8']['nll'],'run':str(run)})
            selected=min(candidates,key=lambda r:r['val_nll'])
            selected_path.write_text(json.dumps({'criterion':'validation NLL only','grid':args.grid,
                                                'selected':selected,'candidates':candidates},indent=2))
        selected=json.loads(selected_path.read_text())['selected']
        for name,mode in [('clean','none'),('fixed','fixed'),('mean_only','mean_only'),('adapter_matched','adapter'),('pi','mean_scale')]:
            for seed in args.seeds:
                run=output/f'{name}_seed{seed}'
                if not (run/'metrics.json').exists():
                    runner.command(f'{dataset}_{name}_seed{seed}',
                                   ['pi_experiments.internal_train','--directory',str(directory),'--pack',str(pack),'--output',str(run),
                                    '--resident-maps','--mode',mode,'--seed',str(seed),
                                    '--concept-weight',str(selected['concept_weight']),'--energy-weight',str(selected['energy_weight'])])
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
