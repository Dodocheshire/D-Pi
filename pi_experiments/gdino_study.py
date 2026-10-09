"""Complete the CUB GDINO comparison using the same split, encoder and Pi controls."""

import argparse
import json
from pathlib import Path
import time

from .resources import gpu_lease
from .suite import Runner, pid_identity


class GuardedRunner(Runner):
    def command(self,name,args):
        if args[0] in {'pi_experiments.train','pi_experiments.analyze','pi_experiments.cub_metrics'}:
            with gpu_lease(name):return super().command(name,args)
        return super().command(name,args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--directory',default='artifacts/datasets/cub_gdino')
    args=parser.parse_args()
    runner=GuardedRunner('artifacts/logs/gdino_v1')
    if args.wait_pid:
        identity=pid_identity(args.wait_pid)
        while identity is not None and pid_identity(args.wait_pid)==identity:
            runner.state.update(phase='waiting_for_existing_proposals',waiting_pid=args.wait_pid,checked_at=time.time())
            runner.save();time.sleep(10)
    directory=Path(args.directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    shards=directory/'proposals'
    if not all((shards/f'{i:08d}.pt').exists() for i in manifest['concept_source_indices']):
        runner.command('cub_gdino_proposals',['pi_experiments.gdino','--output',str(directory)])
    pack=directory/'baseline_k2048_seed0.pt'
    if not pack.exists():
        runner.command('cub_gdino_baseline',['pi_experiments.prepare','--dataset','cub','--root',manifest['root'],
                                           '--output',str(directory),'--stage','baseline','--clusters','2048'])
    output=Path('outputs/gdino_v1/cub');output.mkdir(parents=True,exist_ok=True)
    selected_path=output/'selected_hyperparameters.json'
    if not selected_path.exists():
        grid=sorted({(c,e) for c in [0,0.01,0.1,1] for e in [0,0.001,0.01,0.1]}|
                    {(c,e) for c in [1,3,10] for e in [0.1,0.3,1]})
        candidates=[]
        for c,e in grid:
            destination=output/'validation_search'/f'c{c:g}_e{e:g}'
            if not (destination/'metrics.json').exists():
                runner.command(f'cub_search_c{c}_e{e}',
                               ['pi_experiments.train','--pack',str(pack),'--output',str(destination),'--device','cuda',
                                '--validation-only','--concept-weight',str(c),'--energy-weight',str(e)])
            report=json.loads((destination/'metrics.json').read_text());assert report['evaluation_split']=='val'
            candidates.append({'concept_weight':c,'energy_weight':e,'val_nll':report['metrics']['pi']['nll']})
        selected_path.write_text(json.dumps({'criterion':'validation NLL only','selected':min(candidates,key=lambda r:r['val_nll']),
                                            'candidates':candidates},indent=2))
    selected=json.loads(selected_path.read_text())['selected']
    for name,mode in [('clean','none'),('fixed','fixed'),('mean_only','mean_only'),('adapter_matched','adapter'),('pi','mean_scale')]:
        for seed in range(5):
            destination=output/f'{name}_seed{seed}'
            if not (destination/'metrics.json').exists():
                runner.command(f'cub_{name}_seed{seed}',
                               ['pi_experiments.train','--pack',str(pack),'--output',str(destination),'--device','cuda','--mode',mode,
                                '--seed',str(seed),'--concept-weight',str(selected['concept_weight']),'--energy-weight',str(selected['energy_weight'])])
            if not (destination/'analysis.json').exists():
                runner.command(f'cub_{name}_seed{seed}_analysis',['pi_experiments.analyze','--pack',str(pack),'--run',str(destination)])
            if not (destination/'cub_concepts.json').exists():
                runner.command(f'cub_{name}_seed{seed}_concepts',
                               ['pi_experiments.cub_metrics','--pack',str(pack),'--run',str(destination),'--directory',str(directory)])
        runner.command(f'cub_{name}_summary',['pi_experiments.summarize','--root','outputs/gdino_v1','--output','reports/gdino_v1'])
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
