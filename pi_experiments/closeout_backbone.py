"""Use the final GPU window for the original B/16 CUB protocol without shortening it."""

import argparse
import json
from pathlib import Path
import time

from .concept_study import StudyRunner
from .resources import gpu_lease
from .suite import ROOTS
from .visible_semantics import write_json


class WindowEnded(Exception):
    pass


class DeadlineRunner(StudyRunner):
    def command(self,name,args):
        if time.time()>=self.deadline:
            raise WindowEnded(name)
        return super().command(name,args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-deadline',type=float,required=True)
    args=parser.parse_args()
    runner=DeadlineRunner('artifacts/logs/closeout_backbone_v1');runner.deadline=args.launch_deadline
    directory=Path('artifacts/backbone_ablation_v1/ViT-B16/cub')
    output=Path('outputs/backbone_v1/ViT-B16/cub');output.mkdir(parents=True,exist_ok=True)
    try:
        for stage in ['features','crops']:
            runner.command(f'ViT-B16_cub_{stage}',['pi_experiments.backbone_prepare','--dataset','cub',
                           '--backbone','ViT-B/16','--output',str(directory),'--stage',stage])
        pack=directory/'baseline_k2048_seed0.pt'
        if not pack.exists():
            runner.command('ViT-B16_cub_baseline',['pi_experiments.prepare','--dataset','cub',
                           '--root',ROOTS['cub'],'--output',str(directory),'--backbone','ViT-B/16',
                           '--stage','baseline','--clusters','2048'])
        selection=output/'selected_hyperparameters.json'
        if not selection.exists():
            candidates=[]
            grid=sorted({(c,e) for c in [0,.01,.1,1] for e in [0,.001,.01,.1]}|
                        {(c,e) for c in [1,3,10] for e in [.1,.3,1]})
            for c,e in grid:
                run=output/'validation_search'/f'c{c:g}_e{e:g}'
                if not (run/'metrics.json').exists():
                    runner.command(f'ViT-B16_cub_search_c{c:g}_e{e:g}',['pi_experiments.train','--pack',str(pack),
                                   '--output',str(run),'--device','cuda','--validation-only',
                                   '--concept-weight',str(c),'--energy-weight',str(e)])
                report=json.loads((run/'metrics.json').read_text());assert report['evaluation_split']=='val'
                candidates.append({'concept_weight':c,'energy_weight':e,'val_nll':report['metrics']['pi']['nll']})
            assert len(candidates)==24
            write_json(selection,{'criterion':'Validation NLL only','candidates':candidates,
                                  'selected':min(candidates,key=lambda r:r['val_nll'])})
        selected=json.loads(selection.read_text())['selected']
        for variant,mode in [('clean','none'),('fixed','fixed'),('mean_only','mean_only'),
                             ('adapter_matched','adapter'),('pi','mean_scale')]:
            for seed in range(5):
                run=output/f'{variant}_seed{seed}'
                if not (run/'metrics.json').exists():
                    runner.command(f'ViT-B16_cub_{run.name}',['pi_experiments.train','--pack',str(pack),
                                   '--output',str(run),'--device','cuda','--mode',mode,'--seed',str(seed),
                                   '--concept-weight',str(selected['concept_weight']),
                                   '--energy-weight',str(selected['energy_weight'])])
                if not (run/'analysis.json').exists():
                    runner.command(f'ViT-B16_cub_{run.name}_analysis',['pi_experiments.analyze',
                                   '--pack',str(pack),'--run',str(run)])
                if not (run/'cub_concepts.json').exists():
                    with gpu_lease('ViT-B16 CUB attribute evaluation'):
                        runner.command(f'ViT-B16_cub_{run.name}_concepts',['pi_experiments.cub_metrics',
                                       '--pack',str(pack),'--run',str(run),'--directory',str(directory)])
            runner.command(f'ViT-B16_cub_{variant}_summary',['pi_experiments.summarize','--root',str(output.parent),
                           '--output','reports/backbone_v1/ViT-B16'])
        runner.state.update(phase='complete',finished_at=time.time());runner.save()
    except WindowEnded as error:
        runner.state.update(phase='deferred_deadline',next_command=str(error),finished_at=time.time());runner.save()


if __name__=='__main__':main()
