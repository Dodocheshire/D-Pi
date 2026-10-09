"""Fixed, label-independent GDINO region pool from every held-out CUB image."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import clip
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .gdino import crop_bounds
from .prepare import atomic_save, load_sources
from .resources import gpu_lease


def visible_window(box):
    x1,y1,x2,y2=box
    width,height=x2-x1,y2-y1
    if width<=height:new_width,new_height=224,int(224*height/width)
    else:new_width,new_height=int(224*width/height),224
    left,top=round((new_width-224)/2),round((new_height-224)/2)
    return [x1+left*width/new_width,y1+top*height/new_height,
            x1+(left+224)*width/new_width,y1+(top+224)*height/new_height]


def run(args):
    directory=Path(args.directory)
    parent=Path('artifacts/datasets/cub')
    manifest=json.loads((parent/'manifest.json').read_text())
    annotations=torch.load(parent/'attributes.pt',weights_only=True)
    _,source=load_sources('cub',manifest['root'])
    checkpoint=Path('artifacts/models/grounding-dino-tiny')
    model_info=json.loads((checkpoint/'PROVENANCE.json').read_text())
    prompts=annotations['part_names']
    caption=' . '.join(prompts)+' .'
    protocol={'evaluation_only':True,'region_backend':'fixed Grounding DINO tiny; same 15-part caption for all images',
              'caption':caption,'box_threshold':.35,'text_threshold':.25,
              'model_revision':model_info['revision'],'model_sha256':model_info['files_sha256']['model.safetensors'],
              'encoder_backbone':manifest['backbone'],'encoder_sha256':manifest['encoder_checkpoint_sha256'],
              'parent_manifest_sha256':hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest(),
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'labels':'No per-image class, attributes, or part coordinates supplied to region detection or encoding.'}
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2))
    shards=directory/'shards';shards.mkdir(exist_ok=True)
    processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint,local_files_only=True).to(args.device).eval().requires_grad_(False)
    encoder,transform=clip.load(manifest['backbone'],device=args.device)
    encoder.eval().requires_grad_(False)
    started=time.monotonic()
    for row,index in enumerate(manifest['test_indices']):
        path=shards/f'{row:08d}.pt'
        if path.exists():continue
        image,label=source[index]
        inputs=processor(images=image,text=caption,return_tensors='pt').to(args.device)
        with torch.inference_mode():outputs=detector(**inputs)
        results=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=.35,text_threshold=.25,
                                                                 target_sizes=[image.size[::-1]])[0]
        boxes=[];crops=[]
        for box in results['boxes'].cpu().tolist():
            bounds=crop_bounds(box,image.size)
            if bounds is not None:boxes.append(bounds);crops.append(transform(image.crop(bounds)))
        with torch.no_grad():
            features=torch.cat([encoder.encode_image(torch.stack(crops[i:i+64]).to(args.device)).float().cpu()
                                for i in range(0,len(crops),64)]) if crops else torch.empty(0,encoder.visual.output_dim)
        assert torch.isfinite(features).all()
        atomic_save({'features':features,'boxes':torch.tensor(boxes).reshape(-1,4),
                     'visible_windows':torch.tensor([visible_window(box) for box in boxes]).reshape(-1,4),
                     'image_position':row,'source_index':index,'label':label},path)
        if (row+1)%100==0:print('Held-out region images',row+1,'/',len(manifest['test_indices']),'elapsed',round(time.monotonic()-started,1),flush=True)
    records=[torch.load(shards/f'{row:08d}.pt',weights_only=True) for row in range(len(manifest['test_indices']))]
    values={'features':torch.cat([r['features'] for r in records]),
            'image_positions':torch.cat([torch.full((len(r['features']),),r['image_position'],dtype=torch.long) for r in records]),
            'boxes':torch.cat([r['boxes'] for r in records]),
            'visible_windows':torch.cat([r['visible_windows'] for r in records]),
            'images':len(records),'empty_images':sum(len(r['features'])==0 for r in records),
            'protocol_sha256':hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest()}
    atomic_save(values,directory/'regions.pt')
    print('Completed held-out region pool',len(values['features']),'regions, empty images',values['empty_images'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default='artifacts/datasets/cub_retrieval')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(8)
    with gpu_lease('Held-out CUB retrieval regions',args.device):run(args)


if __name__=='__main__':main()
