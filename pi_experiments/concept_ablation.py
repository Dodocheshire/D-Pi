"""Nested proposal-image and prototype-count ablations from existing SAM2 crops."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .suite import ROOTS, Runner


def setup(dataset,base,root):
    base,root=Path(base),Path(root)
    manifest=json.loads((base/'manifest.json').read_text())
    records={index:torch.load(base/'proposals'/f'{index:08d}.pt',weights_only=True)
             for index in manifest['concept_source_indices']}
    by_class={}
    for index,record in records.items():by_class.setdefault(int(record['label']),[]).append(index)
    rng=np.random.default_rng(727)
    permutations={label:rng.permutation(sorted(indices)).tolist() for label,indices in sorted(by_class.items())}
    assert all(len(ids)>=50 for ids in permutations.values()),'This ablation needs 50 existing sources per class'
    cases=sorted(set([(m,64) for m in [5,10,20,50]]+[(50,k) for k in [64,128,256,512,1024]]))
    output=[]
    for m,k in cases:
        ids=sorted(index for indices in permutations.values() for index in indices[:m])
        count=sum(len(records[index]['features']) for index in ids)
        if count<k:raise ValueError(f'm{m}_k{k}: {count} regions cannot support {k} prototypes')
        destination=root/dataset/f'm{m}_k{k}'
        destination.mkdir(parents=True,exist_ok=True)
        current=dict(manifest)
        current.update(images_per_class=m,concept_source_indices=ids,
                       parent_manifest_sha256=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest(),
                       ablation={'nested_source_seed':727,'images_per_class':m,'clusters':k,'region_count':count,
                                 'selection':'fixed per-class permutation of the original 50 training sources'})
        manifest_path=destination/'manifest.json'
        if manifest_path.exists():assert json.loads(manifest_path.read_text())==current
        else:manifest_path.write_text(json.dumps(current,indent=2))
        links=[(base/f'{split}_features.pt',destination/f'{split}_features.pt') for split in ['train','val','test']]
        links.append((base/'proposal_config.json',destination/'proposal_config.json'))
        (destination/'proposals').mkdir(exist_ok=True)
        links.extend((base/'proposals'/f'{i:08d}.pt',destination/'proposals'/f'{i:08d}.pt') for i in ids)
        baseline=base/f'baseline_k{k}_seed0.pt'
        if m==50 and baseline.exists():
            links.extend([(baseline,destination/baseline.name),(baseline.with_suffix('.json'),destination/baseline.with_suffix('.json').name)])
        for source,target in links:
            if target.exists():assert target.resolve()==source.resolve()
            else:target.symlink_to(os.path.relpath(source.resolve(),target.parent.resolve()))
        output.append({'name':destination.name,'directory':str(destination),'images_per_class':m,'clusters':k,
                       'source_images':len(ids),'regions':count})
    root.mkdir(parents=True,exist_ok=True)
    (root/f'{dataset}_cases.json').write_text(json.dumps({'dataset':dataset,'nested_source_seed':727,'cases':output},indent=2))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['cifar10','cifar100'],default='cifar10')
    parser.add_argument('--base')
    parser.add_argument('--root',default='artifacts/concept_ablation_v1')
    parser.add_argument('--stage',choices=['setup','baselines'],default='setup')
    args=parser.parse_args()
    torch.set_num_threads(8)
    cases=setup(args.dataset,args.base or f'artifacts/datasets/{args.dataset}',args.root)
    print(json.dumps(cases,indent=2),flush=True)
    if args.stage=='baselines':
        runner=Runner(f'artifacts/logs/concept_ablation_v1/{args.dataset}')
        for case in cases:
            directory=Path(case['directory'])
            if (directory/f'baseline_k{case["clusters"]}_seed0.pt').exists():continue
            runner.command(case['name']+'_baseline',['pi_experiments.prepare','--dataset',args.dataset,'--root',ROOTS[args.dataset],
                                                    '--output',str(directory),'--stage','baseline','--images-per-class',str(case['images_per_class']),
                                                    '--clusters',str(case['clusters'])])
        runner.state.update(phase='complete');runner.save()


if __name__=='__main__':main()
