"""Follow the live study with CPU diagnostics, then serialized GPU localization."""

import argparse
import json
import time
from pathlib import Path

from .suite import CLUSTERS, Runner, pid_identity


PRIMARY={'clean','fixed','mean_only','adapter_matched','pi'}


def ready_runs(dataset):
    runs=[]
    for directory in sorted((Path('outputs/study_v1')/dataset).glob('*_seed*')):
        if not all((directory/name).exists() for name in ['analysis.json','predictions.pt','model.pt']):continue
        # The main study writes analysis before predictions. Allow the latter
        # file to finish and close before reading either artifact.
        if time.time()-(directory/'predictions.pt').stat().st_mtime<10:continue
        runs.append(directory)
    return runs


def cpu_work(runner,datasets):
    for dataset in datasets:
        directory=Path('artifacts/datasets')/dataset
        pack=directory/f'baseline_k{CLUSTERS[dataset]}_seed0.pt'
        if not pack.exists():continue
        for run in ready_runs(dataset):
            variant=run.name.rsplit('_seed',1)[0]
            if variant in {'pi','pi_refit'} and not (run/'noise_diagnostics_test.json').exists():
                runner.command(f'{dataset}_{run.name}_noise_decomposition',
                               ['pi_experiments.noise_diagnostics','--pack',str(pack),'--run',str(run),'--split','test','--device','cpu'])
            if variant in PRIMARY and (directory/'corruptions/jpeg_10.pt').exists() and not (run/'corruptions.json').exists():
                runner.command(f'{dataset}_{run.name}_corruptions',
                               ['pi_experiments.corruptions','--stage','evaluate','--directory',str(directory),
                                '--pack',str(pack),'--run',str(run),'--device','cpu'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    parser.add_argument('--datasets',nargs='+',choices=list(CLUSTERS),default=['cifar10','cifar100','cub'])
    parser.add_argument('--logs',default='artifacts/logs/postprocess_v1')
    parser.add_argument('--main-status',default='artifacts/logs/study_v1/status.json')
    args=parser.parse_args()
    runner=Runner(args.logs)
    identity=pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid)==identity:
        cpu_work(runner,args.datasets)
        runner.state.update({'phase':'waiting_for_completed_models','waiting_pid':args.wait_pid,
                             'waiting_pid_start_ticks':identity,'checked_at':time.time()})
        runner.save();time.sleep(10)
    # A vanished manager is terminal, but only its successful final state
    # authorizes treating its dataset packs and final models as complete.
    main_state=json.loads(Path(args.main_status).read_text())
    if main_state['phase']!='complete':
        runner.state.update({'phase':'upstream_failed','upstream_state':main_state});runner.save()
        raise RuntimeError('Main study did not complete; repair it before running queued GPU evaluation')
    cpu_work(runner,args.datasets)
    for dataset in args.datasets:
        directory=Path('artifacts/datasets')/dataset
        pack=directory/f'baseline_k{CLUSTERS[dataset]}_seed0.pt'
        if dataset in {'cifar10','cifar100'}:
            runner.command(f'{dataset}_corruption_features',
                           ['pi_experiments.corruptions','--stage','prepare','--directory',str(directory)])
            cpu_work(runner,[dataset])
        if dataset=='cub':
            for run in ready_runs(dataset):
                if run.name.rsplit('_seed',1)[0] not in PRIMARY or (run/'localization.json').exists():continue
                runner.command(f'{dataset}_{run.name}_localization',
                               ['pi_experiments.localization','--directory',str(directory),'--pack',str(pack),'--run',str(run)])
    runner.command('extended_summary',['pi_experiments.summarize'])
    runner.state.update({'phase':'complete','finished_at':time.time()});runner.save()


if __name__=='__main__':main()
