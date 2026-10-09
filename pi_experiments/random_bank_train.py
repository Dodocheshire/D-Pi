"""Random conditioning bank with parameters and training RNG matched to real-bank Pi."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import DPi as BaseDPi


class MatchedBankDPi(BaseDPi):
    def __init__(self, projection, weight, bias, feature_scale, mode='mean_scale',
                 conditioning='concept', site='feature', hidden=128, initial_scale=.01):
        seed = torch.initial_seed()
        super().__init__(projection, weight, bias, feature_scale, mode,
                         'concept' if conditioning == 'random_concept' else conditioning,
                         site, hidden, initial_scale)
        if conditioning == 'random_concept' and self.generator is not None:
            rng = torch.Generator(device='cpu').manual_seed(seed + 10000)
            prototypes = torch.randn(self.generator.prototypes.shape, generator=rng)
            self.generator.prototypes.copy_(F.normalize(prototypes, dim=-1))
            self.generator.conditioning = conditioning


def main():
    from . import train
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', required=True)
    args, _ = parser.parse_known_args()
    train.DPi = MatchedBankDPi
    train.main()
    path = Path(args.output) / 'metrics.json'
    report = json.loads(path.read_text())
    report['experiment_code_sha256']['random_bank_train.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['initialization_control'] = {
        'parameters': 'Same initial trainable tensors and global RNG state as real-bank Pi with the same adaptation seed.',
        'random_bank_seed': report['config']['seed'] + 10000,
        'bank_rng': 'Separate CPU torch.Generator; original DProjection bank remains unchanged.'}
    path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
