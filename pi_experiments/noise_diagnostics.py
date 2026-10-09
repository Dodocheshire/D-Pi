"""Causal inference-time mean/scale ablations on an unchanged Pi checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .analyze import collect, load_model, paired_bootstrap, task_metrics


class NoiseOverride(nn.Module):
    def __init__(self,generator,mean_multiplier=1,scale_multiplier=1):
        super().__init__()
        self.base=generator
        self.mean_multiplier=mean_multiplier
        self.scale_multiplier=scale_multiplier

    @property
    def query(self):return self.base.query

    @property
    def prototypes(self):return self.base.prototypes

    def parameters_for(self,features):
        mean,scale=self.base.parameters_for(features)
        return mean*self.mean_multiplier,scale*self.scale_multiplier


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--samples',type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(8)
    directory=Path(args.run)
    pack=torch.load(args.pack,map_location='cpu',weights_only=True)
    model,config=load_model(pack,directory/'model.pt',args.device)
    if config.get('validation_only') and args.split=='test':
        parser.error('A validation-search checkpoint cannot be used for held-out test diagnostics')
    if config['mode']!='mean_scale':parser.error('Mean/scale diagnostics require a learned Gaussian model')
    original=model.generator
    labels=pack[args.split]['labels']
    mean=collect(model,pack[args.split],samples=1,mean=True)
    mean_metrics=task_metrics(mean['probabilities'],labels)
    scenarios=[('no_noise',0,0),('scale_only',0,1),('mean_only',1,0)]
    scenarios.extend((f'mean_plus_scale_x{factor}',1,factor) for factor in [0.25,0.5,1,2,4])
    rows=[]
    for name,mean_factor,scale_factor in scenarios:
        model.generator=NoiseOverride(original,mean_factor,scale_factor)
        result=collect(model,pack[args.split],samples=args.samples,seed=3010)
        metrics=task_metrics(result['probabilities'],labels)
        mu_energy=result['mu_rms'].square().mean().item()
        sigma_energy=result['sigma_rms'].square().mean().item()
        rows.append({'scenario':name,'mean_multiplier':mean_factor,'scale_multiplier':scale_factor,
                     **metrics,'mean_energy':mu_energy,'scale_energy':sigma_energy,
                     'scale_energy_fraction':sigma_energy/(mu_energy+sigma_energy) if mu_energy+sigma_energy else 0,
                     'standardized_concept_mse':(result['concepts']-result['clean_concepts']).square().mean().item(),
                     'concept_sampling_variance':result['concept_sampling_variance'].mean().item(),
                     'paired_vs_mean':paired_bootstrap(mean['probabilities'].numpy(),result['probabilities'].numpy(),labels.numpy()),
                     'nll_delta_vs_mean':metrics['nll']-mean_metrics['nll']})
        print(json.dumps(rows[-1]),flush=True)
    report={'protocol':'unchanged selected checkpoint; fixed paired random draws; no retraining or threshold selection',
            'evaluation_split':args.split,'mc_samples':args.samples,'sampling_seed':3010,
            'deterministic_mean':mean_metrics,'scenarios':rows}
    (directory/f'noise_diagnostics_{args.split}.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
