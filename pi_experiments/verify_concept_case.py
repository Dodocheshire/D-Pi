"""Independently audit a completed five-method concept-construction condition."""

import argparse
import json
from pathlib import Path
import time

import torch

from . import export_major_results as evidence
from .corruption_uncertainty import MODES
from .large_proposal_prewarm import file_sha256
from .visible_semantics import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='m50_k1024')
    args=parser.parse_args();torch.set_num_threads(8)
    config=json.loads(Path('artifacts/concept_ablation_v1/cifar10_cases.json').read_text())
    case=next(r for r in config['cases'] if r['name']==args.case)
    root=Path('outputs/concept_study_v1/cifar10')/args.case
    pack=Path(case['directory'])/f'baseline_k{case["clusters"]}_seed0.pt'
    for variant in evidence.VARIANTS:
        for seed in range(5):
            state=torch.load(root/f'{variant}_seed{seed}/model.pt',weights_only=True,map_location='cpu')
            assert state['config']['mode']==MODES[variant] and state['config']['seed']==seed
            assert not state['config'].get('validation_only') and not state['config'].get('refit_epochs')
    groups=evidence.verified_task(root,pack,evidence.VARIANTS)
    summary=evidence.read('reports/concept_study_v1/cifar10/results.json')
    contrasts=evidence.find_contrasts(summary,args.case);assert len(contrasts)==5
    evidence.verify_contrasts(root,contrasts)
    result={'case':case,'groups':groups,'contrasts':contrasts,'verified_models':25,
            'verified_inference_arrays':50,'reproduced_paired_intervals':len(evidence.VERIFIED_INTERVALS),
            'source_sha256':evidence.SOURCES,'code_sha256':file_sha256(__file__),
            'verification_helper_sha256':file_sha256(evidence.__file__),'checked_at':time.time(),
            'limits':'Native saved labels, classification probabilities/metrics, frozen prototypes/scalers/initial heads '
                     'and five preset paired intervals independently checked. Fixed reference validation regularization; '
                     'not individually optimal hyperparameters or prototype-building uncertainty.'}
    write_json(Path('reports/concept_study_v1/cifar10')/f'{args.case}_verified.json',result)
    print(json.dumps({'case':args.case,'models':25,'paired_intervals':5,
                     'accuracy':{k:v['accuracy']['mean']*100 for k,v in groups.items()},'contrasts':contrasts},indent=2))


if __name__=='__main__':main()
