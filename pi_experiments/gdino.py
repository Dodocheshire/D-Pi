"""Grounding DINO proposals paired to the frozen SAM2 experiment split."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import clip
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .prepare import atomic_save, load_sources
from .resources import gpu_lease


def crop_bounds(box,image_size):
    width,height=image_size
    x1,y1,x2,y2=box
    bounds=(max(0,int(x1)),max(0,int(y1)),min(width,int(x2)),min(height,int(y2)))
    return bounds if bounds[2]>bounds[0] and bounds[3]>bounds[1] else None


def setup(base,destination):
    base,destination=Path(base),Path(destination)
    manifest=json.loads((base/'manifest.json').read_text())
    assert manifest['dataset']=='cub'
    annotations=torch.load(base/'attributes.pt',weights_only=True)
    prompts=annotations['part_names']
    modified=dict(manifest)
    modified.update(proposal_method='grounding_dino',proposal_prompts=prompts,
                    parent_manifest_sha256=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest())
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==modified
    else:path.write_text(json.dumps(modified,indent=2))
    for name in ['train_features.pt','val_features.pt','test_features.pt','attributes.pt']:
        target=destination/name;source=base/name
        if target.exists():assert target.resolve()==source.resolve()
        else:target.symlink_to(os.path.relpath(source.resolve(),target.parent.resolve()))
    return modified,prompts


def proposals(args,manifest,prompts):
    destination=Path(args.output)
    checkpoint=Path(args.model)
    provenance=json.loads((checkpoint/'PROVENANCE.json').read_text())
    caption=' . '.join(prompts)+' .'
    config={'method':'grounding_dino','model_id':provenance['model_id'],'revision':provenance['revision'],
            'weights_sha256':provenance['files_sha256']['model.safetensors'],
            'prompts':prompts,'caption':caption,'box_threshold':0.35,'text_threshold':0.25,
            'backbone':manifest['backbone'],'box_coordinates':'native-image xyxy, correctly converted from normalized detector cxcywh',
            'label_conditioning':'one fixed caption for every training source; no image class or attribute label supplied'}
    path=destination/'proposal_config.json'
    if path.exists():assert json.loads(path.read_text())==config
    else:path.write_text(json.dumps(config,indent=2))
    processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint,local_files_only=True).to(args.device).eval().requires_grad_(False)
    encoder,transform=clip.load(manifest['backbone'],device=args.device)
    encoder.eval().requires_grad_(False)
    source,_=load_sources('cub',manifest['root'])
    ids=manifest['concept_source_indices'][:args.limit] if args.limit else manifest['concept_source_indices']
    shards=destination/'proposals';shards.mkdir(exist_ok=True)
    started=time.monotonic()
    for number,index in enumerate(ids):
        path=shards/f'{index:08d}.pt'
        if path.exists():continue
        image,label=source[index]
        inputs=processor(images=image,text=caption,return_tensors='pt').to(args.device)
        with torch.inference_mode():outputs=detector(**inputs)
        results=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=0.35,
                                                                 text_threshold=0.25,target_sizes=[image.size[::-1]])[0]
        boxes=[];crops=[];scores=[];phrases=[]
        texts=results.get('text_labels',results.get('labels',[]))
        for box,score,text in zip(results['boxes'].cpu().tolist(),results['scores'].cpu().tolist(),texts):
            bounds=crop_bounds(box,image.size)
            if bounds is None:continue
            boxes.append(bounds);crops.append(transform(image.crop(bounds)));scores.append(score);phrases.append(str(text))
        with torch.no_grad():
            features=torch.cat([encoder.encode_image(torch.stack(crops[i:i+64]).to(args.device)).float().cpu()
                                for i in range(0,len(crops),64)]) if crops else torch.empty(0,encoder.visual.output_dim)
        atomic_save({'features':features,'boxes':torch.tensor(boxes).reshape(-1,4),'source_index':index,'label':label,
                     'image_size':list(image.size),'detection_scores':torch.tensor(scores),'detection_phrases':phrases},path)
        print(f'GDINO {number+1}/{len(ids)} source={index} regions={len(boxes)} elapsed={time.monotonic()-started:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='artifacts/datasets/cub')
    parser.add_argument('--output',default='artifacts/datasets/cub_gdino')
    parser.add_argument('--model',default='artifacts/models/grounding-dino-tiny')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--limit',type=int,default=0,help='Debug extraction only; formal baseline requires all source shards')
    args=parser.parse_args()
    torch.set_num_threads(8)
    manifest,prompts=setup(args.base,args.output)
    with gpu_lease('CUB Grounding DINO proposals',args.device):proposals(args,manifest,prompts)


if __name__=='__main__':main()
