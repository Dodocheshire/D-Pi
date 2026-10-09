"""Prepare independent CPU attribute caches alongside the existing follower."""

import argparse
import fcntl
from pathlib import Path
import time

from .background_semantic_uncertainty import VARIANTS, locations
from .large_proposal_prewarm import file_sha256
from .suite import Runner
from .visible_semantics import ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--bank', choices=['RN50_GDINO', 'RN50_SAM2'], required=True)
    parser.add_argument('--version', choices=['natural_background', 'background'], required=True)
    args = parser.parse_args()
    _, _, sources, outputs = locations(args.bank, args.version)
    root = Path('artifacts/logs/background_semantic_prewarm_v1') / args.bank / args.version / args.variant
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'manager.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Runner(root)
        runner.state['code_sha256'] = file_sha256(__file__);runner.save()
        for seed in range(5):
            source = sources / f'{args.variant}_seed{seed}'
            assert ready(source / 'backgrounds.json'), source
            if (outputs / source.name / 'semantic_predictions.pt').exists():
                continue
            # The cache writer shares an exclusive file lock with the follower.
            runner.command(source.name, ['pi_experiments.background_semantic_uncertainty',
                           '--bank', args.bank, '--version', args.version,
                           '--variant', args.variant, '--seed', str(seed)])
        runner.state.update(phase='complete', finished_at=time.time());runner.save()


if __name__ == '__main__':
    main()
