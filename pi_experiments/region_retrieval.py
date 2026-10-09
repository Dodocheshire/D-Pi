"""Annotation-grounded regional Precision@5 with fixed training prototype identities."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .analyze import collect,load_model
from .cub_metrics import MIN_CERTAINTY,PARTS
from .resources import gpu_lease


def regional_truth(attribute,name,annotations,regions):
    positions=regions['image_positions']
    known=annotations['test']['certainty'][positions,attribute]>=MIN_CERTAINTY
    present=annotations['test']['attributes'][positions,attribute]>0
    kind=name.split('::')[0].removeprefix('has_').split('_')[0]
    parts=annotations['test']['parts'][positions][:,PARTS.get(kind,[])]
    boxes=regions['visible_windows']
    inside=((parts[...,0]>=boxes[:,None,0])&(parts[...,0]<boxes[:,None,2])&
            (parts[...,1]>=boxes[:,None,1])&(parts[...,1]<boxes[:,None,3])&(parts[...,2]>0)).any(-1)
    return known,present&inside


def precision_at_five(scores,known,truth,image_positions):
    eligible=torch.nonzero(known).flatten()
    if len(eligible)<5:return None
    order=eligible[torch.argsort(scores[eligible],descending=True,stable=True)]
    top=order[:5]
    distinct=[];seen=set()
    for row in order.tolist():
        image=int(image_positions[row])
        if image not in seen:
            seen.add(image);distinct.append(row)
        if len(distinct)==5:break
    return {'precision_at_5':truth[top].float().mean().item(),
            'precision_at_5_distinct_images':truth[distinct].float().mean().item() if len(distinct)==5 else None,
            'eligible_regions':len(eligible),'positive_regions':int(truth[eligible].sum()),
            'chance_precision':truth[eligible].float().mean().item(),'top_region_indices':top.tolist(),
            'top_distinct_image_region_indices':distinct}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--model-directory',required=True)
    parser.add_argument('--regions',default='artifacts/datasets/cub_retrieval')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(8)
    run=Path(args.run);directory=Path(args.model_directory);pool=Path(args.regions)
    pack=torch.load(args.pack,weights_only=True)
    annotations=torch.load(directory/'attributes.pt',weights_only=True)
    regions=torch.load(pool/'regions.pt',weights_only=True)
    assert pack['metadata']['backbone']=='RN50'
    assert regions['images']==len(pack['test']['labels'])==len(annotations['test']['image_ids'])
    assert regions['protocol_sha256']==hashlib.sha256((pool/'protocol.json').read_bytes()).hexdigest()
    map_path=directory/f'attribute_mapping_k{len(pack["centers"])}.json'
    mapping=json.loads(map_path.read_text())
    assert mapping['min_certainty']==MIN_CERTAINTY
    assignments={int(attribute):concepts for attribute,concepts in mapping['assignments'].items()}
    with gpu_lease('CUB regional retrieval inference',args.device):
        model,config=load_model(pack,run/'model.pt',args.device)
        assert not config.get('validation_only')
        responses=collect(model,regions,samples=8,seed=7200+config['seed'])['concepts']
        del model
        if str(args.device).startswith('cuda'):torch.cuda.empty_cache()
    queries=[]
    for attribute,concepts in sorted(assignments.items()):
        known,truth=regional_truth(attribute,annotations['attribute_names'][attribute],annotations,regions)
        for concept in concepts:
            metrics=precision_at_five(responses[:,concept],known,truth,regions['image_positions'])
            if metrics is not None:queries.append({'concept':concept,'attribute':attribute,'name':annotations['attribute_names'][attribute],**metrics})
    result={'protocol':'Fixed held-out GDINO region pool, posterior mean standardized responses, training-supported prototype mapping; '
                       'region truth requires known original attribute and visible matching part inside effective CLIP crop window.',
            'interpretation':'Annotation-grounded regional retrieval proxy; no additional human judgment of crop semantics. '
                             'The shared GDINO evaluation pool can favor that crop distribution; use within-bank paired comparisons.',
            'evaluation_split':'test','min_certainty':MIN_CERTAINTY,'region_images':regions['images'],
            'regions':len(regions['features']),'empty_region_images':regions['empty_images'],
            'mapped_queries':sum(map(len,assignments.values())),'evaluated_queries':len(queries),'total_prototypes':len(pack['centers']),
            'precision_at_5_macro':float(np.mean([q['precision_at_5'] for q in queries])) if queries else None,
            'precision_at_5_distinct_images_macro':float(np.mean([q['precision_at_5_distinct_images'] for q in queries if q['precision_at_5_distinct_images'] is not None])) if any(q['precision_at_5_distinct_images'] is not None for q in queries) else None,
            'chance_precision_macro':float(np.mean([q['chance_precision'] for q in queries])) if queries else None,
            'excess_precision_over_chance':float(np.mean([q['precision_at_5']-q['chance_precision'] for q in queries])) if queries else None,
            'pool_protocol_sha256':regions['protocol_sha256'],'mapping_sha256':hashlib.sha256(map_path.read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),'queries':queries}
    temporary=run/'region_retrieval.partial.json'
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(run/'region_retrieval.json')
    print(json.dumps({k:v for k,v in result.items() if k!='queries'},indent=2),flush=True)


if __name__=='__main__':main()
