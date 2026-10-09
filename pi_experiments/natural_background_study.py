"""Evaluate natural background patches without replacing prior pressure-test artifacts."""

import argparse
import json
import os
from pathlib import Path
import time

from .backgrounds import CASES
from .background_study import summarize
from .suite import Runner,pid_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    parser.add_argument('--upstream-status',required=True)
    parser.add_argument('--prepare-pid',type=int,required=True)
    parser.add_argument('--source-runs',required=True)
    parser.add_argument('--model-directory',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--logs',required=True)
    parser.add_argument('--reports',required=True)
    args=parser.parse_args()
    runner=Runner(args.logs)
    identity=pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid)==identity:
        runner.state.update(phase='waiting_for_cub_models',waiting_pid=args.wait_pid,checked_at=time.time());runner.save();time.sleep(10)
    upstream=json.loads(Path(args.upstream_status).read_text())
    if upstream['manager_pid']!=args.wait_pid or upstream['phase']!='complete':
        runner.state.update(phase='upstream_failed',upstream_state=upstream);runner.save()
        raise RuntimeError('Expected CUB model study did not complete successfully')
    identity=pid_identity(args.prepare_pid)
    while identity is not None and pid_identity(args.prepare_pid)==identity:
        runner.state.update(phase='waiting_for_natural_backgrounds',waiting_pid=args.prepare_pid,checked_at=time.time());runner.save();time.sleep(10)
    directory=Path('artifacts/datasets/cub_background_natural')
    assert all((directory/'backgrounds'/f'{case}.pt').exists() for case in CASES),'Natural background preparation is incomplete'
    pack=Path(args.model_directory)/'baseline_k2048_seed0.pt'
    root=Path(args.output)/'cub';root.mkdir(parents=True,exist_ok=True)
    for variant in ['clean','fixed','mean_only','adapter_matched','pi']:
        for seed in range(5):
            source=Path(args.source_runs)/'cub'/f'{variant}_seed{seed}'
            run=root/source.name;run.mkdir(exist_ok=True)
            for name in ['model.pt','analysis.json','predictions.pt','cub_concepts.json']:
                target=run/name
                if not target.exists():target.symlink_to(os.path.relpath((source/name).resolve(),run.resolve()))
                else:assert target.resolve()==(source/name).resolve()
            if not (run/'backgrounds.json').exists():
                runner.command(run.name+'_natural_backgrounds',['pi_experiments.natural_backgrounds','--stage','evaluate',
                               '--pack',str(pack),'--run',str(run)])
        summarize(root,Path(args.reports))
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
