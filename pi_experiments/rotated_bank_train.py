"""Change conditioning alignment while retaining prototype Gram geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .model import DPi as BaseDPi


class RotatedBankDPi(BaseDPi):
    def __init__(self, projection, weight, bias, feature_scale, mode='mean_scale',
                 conditioning='concept', site='feature', hidden=128, initial_scale=.01):
        seed = torch.initial_seed()
        super().__init__(projection, weight, bias, feature_scale, mode,
                         'concept' if conditioning=='random_concept' else conditioning,
                         site, hidden, initial_scale)
        if conditioning=='random_concept' and self.generator is not None:
            bank = self.generator.prototypes
            rng = torch.Generator(device='cpu').manual_seed(seed+20000)
            permutation = torch.randperm(bank.shape[1], generator=rng).to(bank.device)
            signs = (2*torch.randint(2,(bank.shape[1],),generator=rng)-1).to(bank)
            bank.copy_(bank[:,permutation]*signs)
            self.generator.conditioning = conditioning


def main():
    from . import train
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',required=True)
    args,_ = parser.parse_known_args()
    train.DPi = RotatedBankDPi
    train.main()
    path = Path(args.output)/'metrics.json'; report = json.loads(path.read_text())
    report['experiment_code_sha256']['rotated_bank_train.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['conditioning_control'] = {
        'transformation':'One common signed dimension permutation; preserves within-bank Gram matrix and norm spectrum.',
        'seed':report['config']['seed']+20000,
        'initialization':'Real-bank trainable initialization and global training RNG preserved; separate CPU generator.',
        'projection':'Original classification prototypes, scaler and head unchanged.',
        'limits':'Changes alignment to image coordinates. Trainable query and context network can learn to realign; not a pure semantic causal intervention.'}
    path.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
