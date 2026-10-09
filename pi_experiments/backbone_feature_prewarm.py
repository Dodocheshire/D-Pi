"""Prepare remaining large ViT image features while original regions are pending."""

import hashlib
import json
from pathlib import Path
import time

import torch

from .prepare import load_sources
from .suite import Runner
from .visible_semantics import write_json


CASES=[('ViT-B/16','imagenet'),('ViT-L/14','imagenet'),('ViT-L/14','places365')]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    torch.set_num_threads(8);runner=Runner('artifacts/logs/backbone_feature_prewarm_v1')
    for backbone,dataset in CASES:
        slug=backbone.replace('/','');directory=Path('artifacts/backbone_ablation_v1')/slug/dataset
        parent=Path('artifacts/datasets')/dataset
        preflight=json.loads((parent/'manifest_preflight.json').read_text())
        assert preflight['manifest_sha256']==sha(parent/'manifest.json')
        runner.command(f'{slug}_{dataset}_features',[
            'pi_experiments.backbone_prepare','--dataset',dataset,'--backbone',backbone,
            '--stage','features','--output',str(directory)])
        manifest=json.loads((directory/'manifest.json').read_text())
        assert manifest['proposal_geometry_parent_manifest_sha256']==preflight['manifest_sha256']
        train,test=load_sources(dataset,manifest['root'],manifest.get('official_test_root'))
        dimension=512 if backbone=='ViT-B/16' else 768;checks={}
        for split in ['train','val','test']:
            path=directory/f'{split}_features.pt';pack=torch.load(path,weights_only=True,map_location='cpu')
            indices=torch.tensor(manifest[f'{split}_indices']);source=train if split!='test' else test
            labels=torch.tensor(source.targets)[indices]
            assert pack['features'].shape==(len(indices),dimension) and torch.isfinite(pack['features']).all()
            assert torch.equal(pack['indices'],indices) and torch.equal(pack['labels'],labels)
            assert torch.bincount(labels,minlength=preflight['classes']).tolist()==preflight['class_counts'][split]
            checks[split]={'images':len(indices),'feature_dimension':dimension,'features_finite':True,
                           'indices_exact':True,'all_source_labels_exact':True,'sha256':sha(path)}
            del pack
        write_json(Path('reports/preparation_verification_v1')/f'{slug}_{dataset}.json',{
            'dataset':dataset,'backbone':backbone,'image_count':sum(r['images'] for r in checks.values()),
            'splits':checks,'manifest_sha256':sha(directory/'manifest.json'),
            'parent_manifest_sha256':preflight['manifest_sha256'],
            'scope':preflight['source_scope']+'; global features only. Region banks and model evaluations remain incomplete.',
            'protocol':'Original backbone_prepare feature entry, verified original weights/transforms and global GPU lease. '
                       'Only preparation starts earlier; later studies reuse these atomic caches.',
            'code_sha256':sha(__file__),'checked_at':time.time()})
        runner.state.update(phase=f'{slug}_{dataset}_verified');runner.save()
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
