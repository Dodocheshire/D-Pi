"""Wait for a successful model study and finish full CUB localization."""

import argparse
import json
from pathlib import Path
import time

from .suite import Runner, pid_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--upstream-status', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--logs', required=True)
    parser.add_argument('--reports', required=True)
    args = parser.parse_args()
    upstream = json.loads(Path(args.upstream_status).read_text())
    assert upstream['manager_pid'] == args.wait_pid
    runner = Runner(args.logs)
    identity = pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid) == identity:
        runner.state.update(phase='waiting_for_model_study', waiting_pid=args.wait_pid,
                            waiting_pid_start_ticks=identity, checked_at=time.time())
        runner.save()
        time.sleep(10)
    upstream = json.loads(Path(args.upstream_status).read_text())
    if upstream['manager_pid'] != args.wait_pid or upstream['phase'] != 'complete':
        runner.state.update(phase='upstream_failed', upstream_state=upstream)
        runner.save()
        raise RuntimeError('Expected upstream study did not complete successfully')
    directory = Path(args.directory)
    pack = directory / 'baseline_k2048_seed0.pt'
    runs = Path(args.runs)
    for variant in ['clean', 'fixed', 'mean_only', 'adapter_matched', 'pi']:
        for seed in range(5):
            run = runs / 'cub' / f'{variant}_seed{seed}'
            assert all((run / f).exists() for f in ['model.pt', 'analysis.json', 'cub_concepts.json'])
            if not (run / 'localization.json').exists():
                runner.command(f'{run.name}_localization',
                               ['pi_experiments.localization', '--directory', str(directory),
                                '--pack', str(pack), '--run', str(run)])
        runner.command(f'{variant}_summary', ['pi_experiments.summarize', '--root', str(runs),
                                            '--output', args.reports])
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
