"""Precompute existing RN50 full-image caches under the original GPU lease."""

import hashlib
import json
from pathlib import Path
import time

import torch

from .suite import ROOTS, Runner
from .visible_semantics import write_json


def main():
    torch.set_num_threads(8)
    runner=Runner('artifacts/logs/large_feature_prewarm_v1')
    for dataset in ['places365','imagenet']:
        directory=Path('artifacts/datasets')/dataset
        manifest=json.loads((directory/'manifest.json').read_text())
        verified=json.loads((directory/'manifest_preflight.json').read_text())
        digest=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()
        assert digest==verified['manifest_sha256'] and manifest['backbone']=='RN50'
        command=['pi_experiments.prepare','--dataset',dataset,'--root',ROOTS[dataset],
                 '--output',str(directory),'--stage','features']
        if manifest.get('official_test_root'):command+=['--test-root',manifest['official_test_root']]
        runner.command(dataset+'_features',command)
        checks={}
        for split in ['train','val','test']:
            path=directory/f'{split}_features.pt'
            pack=torch.load(path,weights_only=True,map_location='cpu')
            assert pack['features'].shape==(len(manifest[f'{split}_indices']),1024)
            assert torch.equal(pack['indices'],torch.tensor(manifest[f'{split}_indices']))
            assert torch.isfinite(pack['features']).all()
            labels=pack['labels'];counts=torch.bincount(labels,minlength=verified['classes']).tolist()
            assert counts==verified['class_counts'][split]
            checks[split]={'images':len(labels),'dimensions':1024,'finite':True,'indices_exact':True,
                           'class_counts_exact':True,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            del pack
        write_json(directory/'feature_prewarm_verification.json',{
            'dataset':dataset,'manifest_sha256':digest,'splits':checks,'backbone':'RN50',
            'protocol':'Original prepare.features stage, original CLIP transforms and weights; only preparation starts earlier. '
                       'All future stages reuse the same atomic caches. Original stage lock and GPU lease apply.',
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checked_at':time.time()})
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
