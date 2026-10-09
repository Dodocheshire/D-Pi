"""Paired RN50 Grad-CAM localization: paper-style grids and annotated CUB parts."""

import argparse
import hashlib
import json
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .analyze import load_model
from .cub_metrics import MIN_CERTAINTY, PARTS
from .prepare import load_sources
from .resources import gpu_lease


def grid_plan(labels,seed=720):
    """Different-class distractors and balanced target positions, fixed across runs."""
    labels=np.asarray(labels)
    rng=np.random.default_rng(seed)
    classes=np.unique(labels)
    if len(classes)<4:
        raise ValueError('Grid evaluation needs at least four classes')
    by_class={int(c):np.flatnonzero(labels==c) for c in classes}
    positions=np.arange(len(labels))%4
    rng.shuffle(positions)
    plan=[]
    for index,label in enumerate(labels):
        distractors=rng.choice(classes[classes!=label],3,replace=False)
        ids=[int(rng.choice(by_class[int(c)])) for c in distractors]
        ids.insert(int(positions[index]),index)
        plan.append({'indices':ids,'target_quadrant':int(positions[index])})
    return plan


def grid_image(images,transform):
    # Crop/resize each tile exactly as CLIP does before assembling the grid.
    tiles=[TF.center_crop(TF.resize(image,224,interpolation=InterpolationMode.BICUBIC),224).convert('RGB') for image in images]
    canvas=Image.new('RGB',(448,448))
    for i,tile in enumerate(tiles):canvas.paste(tile,((i%2)*224,(i//2)*224))
    return canvas


def quadrant_metrics(cam,quadrant):
    cam=torch.as_tensor(cam).double().clamp_min(0)
    if not torch.isfinite(cam).all() or cam.sum()<=0:
        return None
    h,w=cam.shape
    masses=torch.stack([cam[:h//2,:w//2].sum(),cam[:h//2,w//2:].sum(),
                        cam[h//2:,:w//2].sum(),cam[h//2:,w//2:].sum()])
    gini=(masses[:,None]-masses[None,:]).abs().sum()/(8*masses.sum())
    # Tie credit prevents fixed argmax order from favoring the first tile.
    maxima=torch.stack([cam[:h//2,:w//2].max(),cam[:h//2,w//2:].max(),
                        cam[h//2:,:w//2].max(),cam[h//2:,w//2:].max()])
    tied=maxima==maxima.max()
    return {'relative_attribution':float(masses[quadrant]/masses.sum()),
            'max_point_accuracy':float(tied[quadrant]/tied.sum()),
            'quadrant_gini':float(gini),'quadrant_gini_normalized':float(gini/0.75)}


def foreground_metrics(cam,mask):
    cam=torch.as_tensor(cam).double().clamp_min(0)
    mask=torch.as_tensor(mask).bool()
    if cam.shape!=mask.shape:raise ValueError('CAM and foreground mask shapes must agree')
    if not torch.isfinite(cam).all() or cam.sum()<=0 or not mask.any():return None
    tied=cam==cam.max()
    return {'foreground_attribution':float(cam[mask].sum()/cam.sum()),
            'foreground_max_point':float(mask[tied].double().mean()),
            'foreground_area_fraction':float(mask.double().mean())}


def transformed_foreground(path,root):
    path=Path(path)
    mask=Image.open(Path(root)/path.parent.name/(path.stem+'.png')).convert('L')
    mask=Image.fromarray((np.asarray(mask)>=128).astype(np.uint8)*255)
    mask=TF.center_crop(TF.resize(mask,224,interpolation=InterpolationMode.NEAREST),224)
    return np.asarray(mask)>0


def part_points(parts,attribute_name,image_size):
    """Map original CUB coordinates through CLIP Resize + CenterCrop."""
    width,height=image_size
    if width<=height:new_width,new_height=224,int(224*height/width)
    else:new_width,new_height=int(224*width/height),224
    left,top=round((new_width-224)/2),round((new_height-224)/2)
    kind=attribute_name.split('::')[0].removeprefix('has_').split('_')[0]
    points=[]
    for index in PARTS.get(kind,[]):
        x,y,visible=parts[index].tolist()
        x,y=x*new_width/width-left,y*new_height/height-top
        if visible and 0<=x<224 and 0<=y<224:points.append([x,y])
    return points


class ConceptCAM:
    def __init__(self,encoder,model,samples=8):
        self.encoder,self.model,self.samples=encoder,model,samples
        self.activation=None
        self.hook=encoder.visual.layer4.register_forward_hook(self.capture)

    def capture(self,module,inputs,output):self.activation=output

    def close(self):self.hook.remove()

    def __call__(self,image,targets,seed):
        device=self.model.feature_scale.device
        image=image[None].to(device).float().requires_grad_(True)
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(seed)
            features=self.encoder.encode_image(image).float()
            concepts=torch.stack([self.model(features)[1] for _ in range(self.samples)]).mean(0)
            activations=self.activation
            maps=[]
            for target in targets:
                gradient=torch.autograd.grad(concepts[0,int(target)],activations,retain_graph=True)[0]
                weights=gradient.float().mean((-2,-1),keepdim=True)
                cam=(weights*activations.float()).sum(1,keepdim=True).relu()
                maps.append(F.interpolate(cam,size=(224,224),mode='bilinear',align_corners=False)[0,0].detach().cpu())
        self.activation=None
        return maps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--masks',default='artifacts/annotations/segmentations')
    parser.add_argument('--limit',type=int,default=0,help='Debug only; formal evaluation uses the entire test split')
    args=parser.parse_args()
    with gpu_lease('RN50 Grad-CAM localization',args.device):
        run(args,parser)


def run(args,parser):
    torch.set_num_threads(8)
    directory=Path(args.directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['backbone']!='RN50':parser.error('This CAM protocol targets CLIP RN50 layer4')
    pack=torch.load(args.pack,map_location='cpu',weights_only=True)
    if args.samples<1 or args.limit<0 or args.limit>len(pack['test']['labels']):
        parser.error('samples must be positive; debug limit must be between zero and test split size')
    model,config=load_model(pack,Path(args.run)/'model.pt',args.device)
    if config.get('validation_only'):parser.error('Localization requires a final selected model')
    encoder,transform=clip.load('RN50',device=args.device)
    encoder=encoder.float().eval().requires_grad_(False)
    _,source=load_sources(manifest['dataset'],manifest['root'],manifest.get('official_test_root'))
    labels=pack['test']['labels'].numpy()
    assert len(labels)==len(manifest['test_indices'])
    assert pack['metadata']['dataset']==manifest['dataset']
    plan=grid_plan(labels)
    plan_digest=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    annot=torch.load(directory/'attributes.pt',weights_only=True) if (directory/'attributes.pt').exists() else None
    mapping_path=directory/f'attribute_mapping_k{len(pack["centers"])}.json'
    reverse={}
    if annot is not None and not mapping_path.exists():
        parser.error('Run cub_metrics first to establish the fixed training-supported semantic mapping')
    if annot is not None and mapping_path.exists():
        mapping=json.loads(mapping_path.read_text())['assignments']
        reverse={int(concept):int(attribute) for attribute,concepts in mapping.items() for concept in concepts}
    foreground_enabled=manifest['dataset']=='cub' and Path(args.masks).exists()
    segmentation_md5=None
    if foreground_enabled:
        archive=Path(args.masks).parent/'segmentations.tgz.partial'
        segmentation_md5=hashlib.md5(archive.read_bytes()).hexdigest()
        assert segmentation_md5=='4d47ba1228eae64f2fa547c47bc65255','Official segmentation checksum differs'
    signature={'grid_plan_sha256':plan_digest,'model_sha256':hashlib.sha256((Path(args.run)/'model.pt').read_bytes()).hexdigest(),
               'pack_sha256':hashlib.sha256(Path(args.pack).read_bytes()).hexdigest(),'samples':args.samples,
               'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'mapping_sha256':hashlib.sha256(mapping_path.read_bytes()).hexdigest() if mapping_path.exists() else None,
               'segmentation_archive_md5':segmentation_md5,'segmentation_threshold':128 if foreground_enabled else None}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    cache=Path(args.run)/'localization_cache'/digest[:16]
    cache.mkdir(parents=True,exist_ok=True)
    (cache/'protocol.json').write_text(json.dumps(signature,indent=2))
    cam=ConceptCAM(encoder,model,args.samples)
    rows=[]
    for row in range(args.limit or len(labels)):
        shard=cache/f'{row:08d}.json'
        if shard.exists():
            rows.append(json.loads(shard.read_text()))
            continue
        image,_=source[manifest['test_indices'][row]]
        with torch.no_grad():
            features=encoder.encode_image(transform(image)[None].to(args.device)).float()
            clean=model.projection(features)
            targets=clean.topk(min(5,len(pack['centers'])),-1).indices[0].tolist()
        images=[source[manifest['test_indices'][i]][0] for i in plan[row]['indices']]
        maps=cam(transform(grid_image(images,transform)),targets,seed=9180+row)
        sample={'row':row,'source_index':manifest['test_indices'][row],'target_quadrant':plan[row]['target_quadrant'],
                'concepts':targets,'grid':[],'parts':[],'part_candidates':0,'part_localizable':0,
                'foreground':[],'foreground_candidates':0,'foreground_localizable':0}
        for concept,heatmap in zip(targets,maps):
            sample['grid'].append({'concept':concept,'metrics':quadrant_metrics(heatmap,sample['target_quadrant'])})
        original_maps=None
        if foreground_enabled:
            foreground=transformed_foreground(source.samples[manifest['test_indices'][row]][0],args.masks)
            original_maps=cam(transform(image),targets,seed=9180+row)
            sample['foreground_candidates']=len(targets)
            sample['foreground_localizable']=len(targets) if foreground.any() else 0
            sample['foreground']=[{'concept':concept,'metrics':foreground_metrics(heatmap,foreground)}
                                  for concept,heatmap in zip(targets,original_maps)]
        if annot is not None and reverse:
            eligible=[c for c in targets if c in reverse and annot['test']['attributes'][row,reverse[c]]>0
                      and annot['test']['certainty'][row,reverse[c]]>=MIN_CERTAINTY]
            sample['part_candidates']=len(eligible)
            if eligible:
                part_maps=[original_maps[targets.index(concept)] for concept in eligible] if original_maps is not None else cam(transform(image),eligible,seed=9180+row)
                for concept,heatmap in zip(eligible,part_maps):
                    attribute=reverse[concept]
                    points=part_points(annot['test']['parts'][row],annot['attribute_names'][attribute],image.size)
                    if not points:continue
                    sample['part_localizable']+=1
                    if heatmap.sum()<=0:continue
                    flat=int(heatmap.argmax());x,y=flat%224,flat//224
                    distance=min(np.hypot(x-px,y-py) for px,py in points)/(224*np.sqrt(2))
                    sample['parts'].append({'concept':concept,'attribute':attribute,'distance_diagonal':float(distance),
                                            'hit_radius_0_1_diagonal':bool(distance<=0.1)})
        temporary=shard.with_suffix('.partial')
        temporary.write_text(json.dumps(sample,allow_nan=False))
        temporary.replace(shard)
        rows.append(sample)
        if (row+1)%50==0:print(f'localization {row+1}/{args.limit or len(labels)}',flush=True)
    cam.close()
    valid=[c['metrics'] for r in rows for c in r['grid'] if c['metrics'] is not None]
    parts=[p for r in rows for p in r['parts']]
    grid_count=sum(len(r['grid']) for r in rows)
    localizable=sum(r['part_localizable'] for r in rows)
    foreground_valid=[c['metrics'] for r in rows for c in r['foreground'] if c['metrics'] is not None]
    foreground_count=sum(r['foreground_candidates'] for r in rows)
    report={'protocol':'fixed clean top-5 standardized responses; paired 2x2 grid; three distinct other classes; RN50 layer4 Grad-CAM through complete Pi path',
            'encoder_precision':'float32 for stable gradients','samples':args.samples,'grid_plan_sha256':plan_digest,
            'cache_signature':signature,'cache_directory':str(cache),
            'evaluated_images':len(rows),'test_images':len(labels),'debug_limit':args.limit,
            'valid_grid_concepts':len(valid),'total_grid_concepts':grid_count,
            'grid_mean':{key:float(np.mean([r[key] for r in valid])) for key in valid[0]} if valid else {},
            'grid_all_concepts_mean':{key:sum(r[key] for r in valid)/grid_count if grid_count else 0
                                      for key in ['relative_attribution','max_point_accuracy','quadrant_gini','quadrant_gini_normalized']},
            'grid_zero_heatmap_policy':'conditional means exclude zero maps; all-concept means score zero maps as zero',
            'part_candidate_concepts':sum(r['part_candidates'] for r in rows),'part_localizable_concepts':localizable,
            'part_evaluated_concepts':len(parts),
            'part_hit_rate':float(np.mean([p['hit_radius_0_1_diagonal'] for p in parts])) if parts else None,
            'part_hit_rate_all_localizable':sum(p['hit_radius_0_1_diagonal'] for p in parts)/localizable if localizable else None,
            'part_mean_distance':float(np.mean([p['distance_diagonal'] for p in parts])) if parts else None,
            'foreground_candidate_concepts':foreground_count,
            'foreground_localizable_concepts':sum(r['foreground_localizable'] for r in rows),
            'foreground_valid_concepts':len(foreground_valid),
            'foreground_mean':{key:float(np.mean([r[key] for r in foreground_valid])) for key in foreground_valid[0]} if foreground_valid else {},
            'foreground_all_candidate_mean':{key:sum(r[key] for r in foreground_valid)/foreground_count if foreground_count else None
                                             for key in ['foreground_attribution','foreground_max_point']},
            'foreground_policy':'Majority mask >=128, nearest resize and same center crop; tied maxima receive fractional credit. Conditional means exclude zero CAMs and entirely cropped foreground; all-candidate means assign zero.',
            'rows':rows}
    destination=Path(args.run)/('localization_debug.json' if args.limit else 'localization.json')
    destination.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
