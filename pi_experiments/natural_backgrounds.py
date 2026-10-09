"""Context changes using unmodified, annotation-empty photographic background patches."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image

from .backgrounds import CASES,MASK_MD5,evaluate,replace_background
from .prepare import atomic_save,encode_dataset,load_sources
from .resources import gpu_lease


def empty_rectangle(mask,minimum=64,cells=32):
    """Largest conservative coarse-grid rectangle, aspect ratio at most two."""
    height,width=mask.shape
    ys=np.linspace(0,height,min(cells,height)+1,dtype=int)
    xs=np.linspace(0,width,min(cells,width)+1,dtype=int)
    occupied=np.array([[mask[ys[i]:ys[i+1],xs[j]:xs[j+1]].any() for j in range(len(xs)-1)] for i in range(len(ys)-1)])
    histogram=np.zeros(len(xs)-1,dtype=int)
    best=None;area=0
    for row,line in enumerate(occupied):
        histogram=np.where(line,0,histogram+1)
        stack=[]
        for col in range(len(histogram)+1):
            current=int(histogram[col]) if col<len(histogram) else 0
            left=col
            while stack and stack[-1][1]>current:
                start,count=stack.pop();left=start
                x1,x2=int(xs[start]),int(xs[col]);y1,y2=int(ys[row-count+1]),int(ys[row+1])
                w,h=x2-x1,y2-y1
                if w>2*h:
                    x1+=(w-2*h)//2;x2=x1+2*h
                elif h>2*w:
                    y1+=(h-2*w)//2;y2=y1+2*w
                w,h=x2-x1,y2-y1
                if min(w,h)>=minimum and w*h>area:
                    assert not mask[y1:y2,x1:x2].any()
                    best=[x1,y1,x2,y2];area=w*h
            if current and (not stack or stack[-1][1]<current):stack.append((left,current))
    return best


class NaturalSource:
    def __init__(self,source,train,donors,mask_root,case):
        self.source,self.train,self.donors,self.mask_root,self.case=source,train,donors,mask_root,case

    def __len__(self):return len(self.source)

    def donor(self,index):
        seed=int(self.case.rsplit('_',1)[1])
        rng=np.random.default_rng(np.random.SeedSequence([611,seed,index]))
        return self.donors[int(rng.integers(len(self.donors)))]

    def __getitem__(self,index):
        image,label=self.source[index]
        donor=self.donor(index)
        background=self.train[donor['index']][0].crop(tuple(donor['box']))
        path=Path(self.source.samples[index][0])
        mask=np.asarray(Image.open(Path(self.mask_root)/path.parent.name/(path.stem+'.png')).convert('L'))>=128
        return replace_background(image,mask,background),label


def prepare(args):
    base=Path('artifacts/datasets/cub');directory=Path(args.directory)
    directory.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json','train_features.pt','val_features.pt','test_features.pt','attributes.pt']:
        target=directory/name
        if not target.exists():target.symlink_to(os.path.relpath((base/name).resolve(),directory.resolve()))
        else:assert target.resolve()==(base/name).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    assert hashlib.md5(Path('artifacts/annotations/segmentations.tgz.partial').read_bytes()).hexdigest()==MASK_MD5
    train,test=load_sources('cub',manifest['root'])
    cache=directory/'backgrounds';cache.mkdir(exist_ok=True)
    donor_path=cache/'donor_patches.json'
    if donor_path.exists():donors=json.loads(donor_path.read_text())['donors']
    else:
        donors=[]
        for index in manifest['train_indices']:
            path=Path(train.samples[index][0])
            # Union of all positive mask votes removes uncertain boundary bird pixels too.
            mask=np.asarray(Image.open(Path(args.masks)/path.parent.name/(path.stem+'.png')).convert('L'))>0
            box=empty_rectangle(mask)
            if box is not None:donors.append({'index':index,'box':box})
        donor_path.write_text(json.dumps({'minimum_side':64,'max_aspect_ratio':2,'grid_cells':32,
                                         'candidate_training_images':len(manifest['train_indices']),'donors':donors},indent=2))
    assert donors and {r['index'] for r in donors}<=set(manifest['train_indices'])
    protocol={'cases':CASES,'type':'Natural photographic background patches; no pixel inpainting',
              'mask_md5':MASK_MD5,'foreground_threshold':128,'donor_mask_threshold':0,
              'donor_patches_sha256':hashlib.sha256(donor_path.read_bytes()).hexdigest(),
              'backbone':manifest['backbone'],'encoder_sha256':manifest['encoder_checkpoint_sha256'],
              'source_manifest_sha256':hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest(),
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits':'A rectangular photo patch is resized to the target frame; foreground boundary compositing remains synthetic. '
                       'Empty means no annotated donor bird pixel, rather than a manually certified bird-free scene.'}
    path=cache/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2))
    for case in CASES:
        target=cache/f'{case}.pt'
        if target.exists():continue
        if case in {'identity','gray'}:
            values=torch.load(base/'backgrounds'/target.name,weights_only=True)
            values['protocol_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            atomic_save(values,target);continue
        source=NaturalSource(test,train,donors,args.masks,case)
        with gpu_lease(f'Natural CUB background features {case}',args.device):
            encoder,transform=clip.load(manifest['backbone'],device=args.device)
            encoder.eval().requires_grad_(False)
            values=encode_dataset(source,manifest['test_indices'],encoder,transform,args)
            assert torch.isfinite(values['features']).all()
            values['protocol_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            values['donor_indices']=torch.tensor([source.donor(index)['index'] for index in manifest['test_indices']])
            atomic_save(values,target)
            del encoder
            if str(args.device).startswith('cuda'):torch.cuda.empty_cache()
        print('Prepared natural backgrounds',case,len(values['labels']),'donor patches',len(donors),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','evaluate'],required=True)
    parser.add_argument('--directory',default='artifacts/datasets/cub_background_natural')
    parser.add_argument('--masks',default='artifacts/annotations/segmentations')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--encode-batch-size',type=int,default=128)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--pack');parser.add_argument('--run')
    args=parser.parse_args();args.cases=CASES
    torch.set_num_threads(8)
    if args.stage=='prepare':prepare(args)
    else:
        if not args.pack or not args.run:parser.error('Evaluation requires --pack and --run')
        manifest=json.loads((Path(args.directory)/'manifest.json').read_text())
        with gpu_lease('Natural CUB background evaluation',args.device):evaluate(args,manifest)


if __name__=='__main__':main()
