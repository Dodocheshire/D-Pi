"""Audit the complete RN50 matrix and isolate the original head-refit stage."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from . import export_major_results as evidence
from .large_proposal_prewarm import file_sha256
from .statistics import hierarchical_paired_bootstrap
from .visible_semantics import write_json


VARIANTS=['clean','fixed','mean_only','adapter_matched','pi','pi_mlp','pi_random_concepts',
          'pi_concept_scores','pi_no_preservation','pi_no_energy','clean_refit','pi_refit']


def main():
    torch.set_num_threads(8)
    groups={};comparisons=[];refit_checks=[]
    for dataset,clusters in [('cifar10',256),('cifar100',2048),('cub',2048)]:
        root=Path('outputs/study_v1')/dataset
        pack=Path('artifacts/datasets')/dataset/f'baseline_k{clusters}_seed0.pt'
        groups[dataset]=evidence.verified_task(root,pack,VARIANTS)
        reference=evidence.pack_reference(pack)
        predictions={v:[evidence.load(root/f'{v}_seed{s}'/'predictions.pt') for s in range(5)]
                     for v in ['clean','pi','clean_refit','pi_refit']}
        for variant in ['clean_refit','pi_refit']:
            base='clean' if variant=='clean_refit' else 'pi'
            for seed in range(5):
                original=evidence.load(root/f'{base}_seed{seed}'/'model.pt')
                refitted=evidence.load(root/f'{variant}_seed{seed}'/'model.pt')
                assert refitted['config']['refit_epochs']==200
                assert refitted['state_dict'].keys()==original['state_dict'].keys()
                unchanged=[key for key in original['state_dict'] if not key.startswith('head.')]
                assert all(torch.equal(original['state_dict'][key],refitted['state_dict'][key]) for key in unchanged),(
                    dataset,variant,seed,'Refitting changed a generator or frozen preprocessing tensor')
                refit_checks.append({'dataset':dataset,'variant':variant,'seed':seed,
                                     'all_non_head_tensors_exact':True,'refit_epochs':200,
                                     'head_changed':any(not torch.equal(original['state_dict'][key],refitted['state_dict'][key])
                                                        for key in ['head.weight','head.bias'])})
        for control,adapted in [('clean','clean_refit'),('pi','pi_refit'),('clean_refit','pi_refit')]:
            a,b=predictions[control],predictions[adapted]
            assert all(torch.equal(p['labels'],reference['labels']) for p in a+b)
            interval=hierarchical_paired_bootstrap(np.stack([p['mc8'].numpy() for p in a]),
                                                  np.stack([p['mc8'].numpy() for p in b]),reference['labels'].numpy())
            interval['protocol']='Paired adaptation/head-fit seeds and common test images; original validation-selected 200-epoch head stage, fixed bank and matched non-head tensors. MC8 probabilities from saved full-test predictions.'
            comparisons.append({'dataset':dataset,'control':control,'adapted':adapted,**interval})
        print('VERIFIED',dataset,12*5,'models and ten frozen-generator head refits',flush=True)
    snapshots=Path('artifacts/report_assets/primary_matrix_source_snapshots');snapshots.mkdir(exist_ok=True)
    for (source,digest),raw in evidence.SNAPSHOTS.items():
        (snapshots/f'{Path(source).stem}_{digest}.json').write_bytes(raw)
    result={'datasets':list(groups),'variants':VARIANTS,'adaptation_seeds':list(range(5)),
            'groups':groups,'verified_classification_checkpoints':evidence.VERIFIED,
            'original_head_refit_checks':refit_checks,'head_refit_comparisons':comparisons,
            'verified_models':len(evidence.VERIFIED),'verified_inference_arrays':2*len(evidence.VERIFIED),
            'source_sha256':evidence.SOURCES,'source_snapshots_directory':str(snapshots.resolve()),
            'code_sha256':file_sha256(__file__),'verification_helper_sha256':file_sha256(evidence.__file__),
            'checked_at':time.time(),
            'limits':'Complete RN50 matrix on three small datasets, not the full research matrix. '
                     'Frozen-head checks reproduce native saved test probabilities/metrics and initial heads. '
                     'Original head refits change only the head; proximal sparse-head experiments remain separate. '
                     'Nine head-refit intervals are descriptive, unadjusted and conditional on a fixed bank, '
                     'selected generators/heads and one MC8 realization. Initial random-conditioning control '
                     'retains its documented initialization/geometry confounds; do not infer pure semantic causality.'}
    assert result['verified_models']==180 and len(refit_checks)==30 and len(comparisons)==9
    write_json('reports/primary_matrix_verification_v1/results.json',result)
    print(json.dumps({'verified_models':180,'verified_inference_arrays':360,'frozen_non_head_refits':30,
                      'head_refit_paired_intervals':9},indent=2))


if __name__=='__main__':main()
