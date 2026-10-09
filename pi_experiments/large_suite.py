"""Follow the first study with Places365 and the documented clean ImageNet subset."""

import json
from pathlib import Path
import time

from .resources import gpu_lease
from .suite import Runner, pid_identity, study_dataset


class LargeRunner(Runner):
    def command(self, name, args):
        args = list(args)
        if args[0] == 'pi_experiments.prepare' and '--stage' in args and args[args.index('--stage') + 1] == 'baseline':
            # Official DCBM uses 512 for these datasets, versus 32 for CUB/CIFAR.
            args += ['--head-batch-size', '512']
        if args[0] == 'pi_experiments.train':
            args += ['--batch-size', '512']
            self.state.update(phase='waiting_for_gpu', next_command=name)
            self.save()
            with gpu_lease(name):
                return super().command(name, args)
        return super().command(name, args)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--upstream-status', default='artifacts/logs/study_v1/status.json')
    args = parser.parse_args()
    runner = LargeRunner('artifacts/logs/large_v1')
    identity = pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid) == identity:
        runner.state.update(phase='waiting_for_main_study', waiting_pid=args.wait_pid,
                            waiting_pid_start_ticks=identity, checked_at=time.time())
        runner.save()
        time.sleep(10)
    upstream = json.loads(Path(args.upstream_status).read_text())
    if upstream['manager_pid'] != args.wait_pid or upstream['phase'] != 'complete':
        runner.state.update(phase='upstream_failed', upstream_state=upstream)
        runner.save()
        raise RuntimeError('Expected first study did not complete')
    args.seeds = list(range(5))
    runner.state['protocol'] = {
        'places365': 'Full train-standard pool, fixed stratified 90/10 split, official validation split as held-out test.',
        'imagenet': 'Existing clean 200 train + 10 adapter_val per class pool; fixed 90/10 split. Official ImageNet validation is test. This is not full ImageNet training.',
        'head_batch_size': 512, 'adaptation_batch_size': 512,
        'adaptation_note': 'Larger fixed batch for both noise adaptation and matched controls; comparisons are within dataset.'}
    runner.save()
    for dataset in ['places365', 'imagenet']:
        study_dataset(runner, dataset, args)
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
