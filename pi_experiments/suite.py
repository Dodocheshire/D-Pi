"""Run the preregistered real-data D-Pi study with per-command logs and live PID records."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOTS={
    'cifar10':'/home/pengcheng/VLG-CBM/datasets',
    'cifar100':'/home/pengcheng/VLG-CBM/datasets',
    'cub':'/home/pengcheng/VLG-CBM/datasets/CUB',
    'places365':'/home/pengcheng/VLG-CBM/datasets/places365_torch',
    'imagenet':'/home/pengcheng/VLG-CBM/datasets/imagenet/ILSVRC/Data/CLS-LOC/ImageNet_train_200_10_clean',
}
CLUSTERS={'cifar10':256,'cifar100':2048,'cub':2048,'places365':2048,'imagenet':4096}


def pid_identity(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():
        return None
    data=path.read_text().split(') ',1)[1].split()
    return data[19] if data[0]!='Z' else None


class Runner:
    def __init__(self,root):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.state={'manager_pid':os.getpid(),'started_at':time.time(),'completed_commands':[], 'phase':'initializing'}
        self.save()

    def save(self):
        temp=self.root/'status.partial.json'
        temp.write_text(json.dumps(self.state,indent=2))
        temp.replace(self.root/'status.json')

    def command(self,name,args):
        command=[sys.executable,'-u','-m',*args]
        log=self.root/f'{name}.log'
        with log.open('a') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            self.state.update({'phase':name,'child_pid':child.pid,'command':command,'log':str(log)})
            self.save()
            print('START',name,'pid',child.pid,flush=True)
            result=child.wait()
        self.state['child_pid']=None
        self.state['completed_commands'].append({'name':name,'returncode':result,'log':str(log),'finished_at':time.time()})
        self.save()
        if result:
            self.state['phase']='failed';self.save()
            raise RuntimeError(f'{name} failed with {result}; inspect {log}')
        print('DONE',name,flush=True)


def study_dataset(runner,dataset,args):
    directory=Path('artifacts/datasets')/dataset
    prep=['pi_experiments.prepare','--dataset',dataset,'--root',ROOTS[dataset],
          '--output',str(directory),'--stage']
    extra=['--test-root','/home/pengcheng/VLG-CBM/datasets/imagenet/ILSVRC/Data/CLS-LOC/ImageNet_val'] if dataset=='imagenet' else []
    runner.command(f'{dataset}_features',prep+['features']+extra)
    if dataset=='cub' and not (directory/'attributes.pt').exists():
        runner.command('cub_annotations',['pi_experiments.cub_annotations'])
    runner.command(f'{dataset}_proposals',prep+['proposals']+extra)
    pack=directory/f'baseline_k{CLUSTERS[dataset]}_seed0.pt'
    if not pack.exists():
        runner.command(f'{dataset}_baseline',prep+['baseline','--clusters',str(CLUSTERS[dataset])]+extra)
    output=Path('outputs/study_v1')/dataset
    selected_file=output/'selected_hyperparameters.json'
    output.mkdir(parents=True,exist_ok=True)
    grid=[(c,e) for c in [0.0,0.01,0.1,1.0] for e in [0.0,0.001,0.01,0.1]]
    # CIFAR-10's initial optimum hit both upper bounds. Add stronger
    # regularization before committing to the final paired comparisons.
    grid=sorted(set(grid+[(c,e) for c in [1.0,3.0,10.0] for e in [0.1,0.3,1.0]]))
    previous=json.loads(selected_file.read_text()) if selected_file.exists() else None
    completed_grid={(r['concept_weight'],r['energy_weight']) for r in previous['candidates']} if previous else set()
    if set(grid)!=completed_grid:
        candidates=[]
        for concept_weight,energy_weight in grid:
            destination=output/'validation_search'/f'c{concept_weight}_e{energy_weight}'
            if not (destination/'metrics.json').exists():
                runner.command(f'{dataset}_search_c{concept_weight}_e{energy_weight}',
                               ['pi_experiments.train','--pack',str(pack),'--output',str(destination),
                                '--mode','mean_scale','--device','cuda','--validation-only',
                                '--concept-weight',str(concept_weight),'--energy-weight',str(energy_weight)])
            result=json.loads((destination/'metrics.json').read_text())
            assert result['evaluation_split']=='val'
            candidates.append({'concept_weight':concept_weight,'energy_weight':energy_weight,
                               'val_nll':result['metrics']['pi']['nll'],'run':str(destination)})
        selected=min(candidates,key=lambda x:x['val_nll'])
        if previous:
            initial_file=output/'selected_hyperparameters.initial_grid.json'
            if not initial_file.exists():initial_file.write_text(json.dumps(previous,indent=2))
        selected_file.write_text(json.dumps({'criterion':'validation NLL only','search_version':2,
                                            'adjustment':'extend original grid because CIFAR-10 selected both upper bounds',
                                            'selected':selected,'candidates':candidates},indent=2))
    selected=json.loads(selected_file.read_text())['selected']
    variants=[('clean','none','concept','feature',0,{}),
              ('fixed','fixed','concept','feature',0,{}),
              ('mean_only','mean_only','concept','feature',0,{}),
              ('adapter_matched','adapter','concept','feature',0,{}),
              ('pi','mean_scale','concept','feature',0,{}),
              ('pi_mlp','mean_scale','mlp','feature',0,{}),
              ('pi_random_concepts','mean_scale','random_concept','feature',0,{}),
              ('pi_concept_scores','mean_scale','concept','concept',0,{}),
              ('pi_no_preservation','mean_scale','concept','feature',0,{'concept_weight':0}),
              ('pi_no_energy','mean_scale','concept','feature',0,{'energy_weight':0}),
              ('clean_refit','none','concept','feature',200,{}),
              ('pi_refit','mean_scale','concept','feature',200,{})]
    for name,mode,conditioning,site,refit,overrides in variants:
        for seed in args.seeds:
            destination=output/f'{name}_seed{seed}'
            if not (destination/'metrics.json').exists():
                command=['pi_experiments.train','--pack',str(pack),'--output',str(destination),
                         '--mode',mode,'--conditioning',conditioning,'--site',site,'--seed',str(seed),
                         '--device','cuda','--refit-epochs',str(refit),
                         '--concept-weight',str(overrides.get('concept_weight',selected['concept_weight'])),
                         '--energy-weight',str(overrides.get('energy_weight',selected['energy_weight']))]
                runner.command(f'{dataset}_{name}_seed{seed}',command)
            if not (destination/'analysis.json').exists():
                runner.command(f'{dataset}_{name}_seed{seed}_analysis',
                               ['pi_experiments.analyze','--pack',str(pack),'--run',str(destination)])
            if dataset=='cub' and not (destination/'cub_concepts.json').exists():
                runner.command(f'cub_{name}_seed{seed}_concepts',
                               ['pi_experiments.cub_metrics','--pack',str(pack),'--run',str(destination)])
        runner.command(f'{dataset}_{name}_summary',['pi_experiments.summarize'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=list(ROOTS),default=['cifar10','cifar100','cub'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    parser.add_argument('--logs',default='artifacts/logs/study_v1')
    parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    runner=Runner(args.logs)
    if args.wait_pid:
        identity=pid_identity(args.wait_pid)
        while identity is not None and pid_identity(args.wait_pid)==identity:
            runner.state.update({'phase':'waiting_for_existing_proposal_process','waiting_pid':args.wait_pid,
                                 'waiting_pid_start_ticks':identity,'checked_at':time.time()})
            runner.save();time.sleep(10)
        runner.state.pop('waiting_pid',None)
    for dataset in args.datasets:
        study_dataset(runner,dataset,args)
    runner.state.update({'phase':'complete','finished_at':time.time()});runner.save()


if __name__=='__main__':
    main()
