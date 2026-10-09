"""Ground visual prototypes in training CUB part/attribute annotations, then evaluate."""

import argparse
import json
from pathlib import Path

import clip
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from .analyze import collect, load_model, task_metrics


PARTS={'bill':[1],'wing':[8,12],'tail':[13],'back':[0],'breast':[3],'belly':[2],
       'forehead':[5],'crown':[4],'nape':[9],'eye':[6,10],'leg':[7,11],'throat':[14],
       'head':[4,5,9],'underparts':[2,3],'upperparts':[0,8,12]}
MIN_CERTAINTY=3  # Original annotations: 2=guessing, 3=probably, 4=definitely.


def crop_contains_part(part,box):
    x,y,visible=part
    x1,y1,x2,y2=box
    return bool(visible>0 and x1<=x<x2 and y1<=y<y2)


def mapping(pack,annotations,manifest,device):
    names=annotations['attribute_names']
    train_lookup={int(index):position for position,index in enumerate(manifest['train_indices'])}
    eligibility=torch.zeros(len(pack['centers']),len(names),dtype=torch.bool)
    for concept,(source,box) in enumerate(pack['metadata']['concept_sources']):
        row=train_lookup[source]
        attributes=annotations['train']['attributes'][row]
        certainty=annotations['train']['certainty'][row]
        parts=annotations['train']['parts'][row]
        x1,y1,x2,y2=box
        for attribute,name in enumerate(names):
            kind=name.split('::')[0].removeprefix('has_').split('_')[0]
            indices=PARTS.get(kind,[])
            visible=any(crop_contains_part(parts[p],box) for p in indices)
            eligibility[concept,attribute]=visible and attributes[attribute]>0 and certainty[attribute]>=MIN_CERTAINTY
    assert pack['metadata']['backbone']==manifest['backbone']
    model,_=clip.load(manifest['backbone'],device=device)
    model.eval().requires_grad_(False)
    prompts=[]
    for name in names:
        category,value=name.split('::')
        category=category.removeprefix('has_').replace('_',' ')
        prompts.append(f'a photo of a bird with {value.replace("_"," ")} {category}')
    with torch.no_grad():
        text=model.encode_text(clip.tokenize(prompts).to(device)).float()
        assert text.shape[-1]==pack['centers'].shape[-1]
        scores=torch.nn.functional.normalize(pack['centers'].to(device),dim=-1) @ torch.nn.functional.normalize(text,dim=-1).T
    scores=scores.cpu()
    scores[~eligibility]=-torch.inf
    assignments=scores.argmax(-1)
    assignments[~eligibility.any(-1)]=-1
    del model
    mapping={int(attribute):(assignments==attribute).nonzero().flatten().tolist() for attribute in assignments.unique().tolist() if attribute>=0}
    return mapping


def attribute_scores(concepts,assigned):
    if not assigned:
        return np.empty((len(concepts),0),dtype=np.float32)
    return torch.stack([concepts[:,assigned[a]].mean(-1) for a in sorted(assigned)],dim=-1).numpy()


def attribute_metrics(val_scores,test_scores,annotations,assigned,min_certainty=MIN_CERTAINTY):
    details=[]
    for col,attribute in enumerate(sorted(assigned)):
        vy=annotations['val']['attributes'][:,attribute].numpy()
        vm=(annotations['val']['certainty'][:,attribute]>=min_certainty).numpy()
        if (vy[vm]==0).sum()<5 or (vy[vm]==1).sum()<5:
            continue
        false_positive,true_positive,thresholds=roc_curve(vy[vm],val_scores[vm,col])
        index=(true_positive-false_positive).argmax()
        threshold=float(thresholds[index])
        if not np.isfinite(threshold):
            # A constant/uninformative score can select the all-negative ROC
            # endpoint; encode it as a finite threshold for portable JSON.
            maximum=val_scores[vm,col].max()
            threshold=float(np.nextafter(maximum,np.array(np.inf,dtype=val_scores.dtype)))
        ty=annotations['test']['attributes'][:,attribute].numpy()
        tm=(annotations['test']['certainty'][:,attribute]>=min_certainty).numpy()
        if len(np.unique(ty[tm]))<2:
            continue
        details.append({'attribute':attribute,'name':annotations['attribute_names'][attribute],
                        'prototypes':assigned[attribute],'threshold':threshold,'test_known_samples':int(tm.sum()),
                        'balanced_accuracy':float(balanced_accuracy_score(ty[tm],test_scores[tm,col]>=threshold)),
                        'auroc':float(roc_auc_score(ty[tm],test_scores[tm,col]))})
    return {'attribute_macro_balanced_accuracy':float(np.mean([d['balanced_accuracy'] for d in details])) if details else None,
            'attribute_macro_auroc':float(np.mean([d['auroc'] for d in details])) if details else None,
            'evaluated_attributes':len(details),'mapped_attributes':len(assigned),'total_attributes':312,
            'min_certainty':min_certainty,'details':details}


def intervention_curve(model,clean_train,test_concepts,annotations,assigned):
    references={}
    attributes=annotations['train']['attributes']
    certainty=annotations['train']['certainty']
    for attribute,columns in assigned.items():
        negative=(attributes[:,attribute]==0)&(certainty[:,attribute]>=MIN_CERTAINTY)
        positive=(attributes[:,attribute]>0)&(certainty[:,attribute]>=MIN_CERTAINTY)
        if negative.sum()<5 or positive.sum()<5:continue
        references[attribute]=(clean_train[negative][:,columns].mean(0),clean_train[positive][:,columns].mean(0))
    keys=sorted(references)
    rng=np.random.default_rng(731)
    orders=[rng.permutation(keys) for _ in range(len(test_concepts))]
    head_weight=model.head.weight.detach().cpu();bias=model.head.bias.detach().cpu()
    results=[]
    test_labels=annotations['test']['task_labels']
    for fraction in [0,0.05,0.1,0.2,0.4,0.8,1]:
        modified=test_concepts.clone()
        number=int(round(len(keys)*fraction))
        corrected=0
        for row,order in enumerate(orders):
            for attribute in order[:number]:
                if annotations['test']['certainty'][row,attribute]<MIN_CERTAINTY:continue
                columns=assigned[attribute]
                value=int(annotations['test']['attributes'][row,attribute])
                modified[row,columns]=references[attribute][value]
                corrected+=1
        probabilities=(modified @ head_weight.T+bias).softmax(-1)
        results.append({'fraction':fraction,'attribute_groups':len(keys),'protocol':'posterior mean concepts; train positive/negative references',
                        'corrected_groups_per_image':corrected/len(test_concepts),
                        **task_metrics(probabilities,test_labels)})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--directory',default='artifacts/datasets/cub')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(8)
    pack=torch.load(args.pack,map_location='cpu',weights_only=True)
    annotations=torch.load(Path(args.directory)/'attributes.pt',weights_only=True)
    manifest=json.loads((Path(args.directory)/'manifest.json').read_text())
    map_path=Path(args.directory)/f'attribute_mapping_k{pack["centers"].shape[0]}.json'
    if map_path.exists():
        saved=json.loads(map_path.read_text())
        assert saved.get('min_certainty')==MIN_CERTAINTY,'Mapping confidence protocol changed; preserve the old mapping separately'
        assigned={int(k):v for k,v in saved['assignments'].items()}
    else:
        assigned=mapping(pack,annotations,manifest,args.device)
        map_path.write_text(json.dumps({'protocol':'training-source positive attribute with certainty >=3; visible matching part inside prototype crop; nearest CLIP text within eligible attributes',
                                        'min_certainty':MIN_CERTAINTY,
                                        'assignments':assigned,'attribute_names':annotations['attribute_names']},indent=2))
    model,config=load_model(pack,Path(args.run)/'model.pt',args.device)
    assert not config.get('validation_only')
    val=collect(model,pack['val'],samples=8,seed=2110)
    test=collect(model,pack['test'],samples=8,seed=2110)
    val_scores=attribute_scores(val['concepts'],assigned);test_scores=attribute_scores(test['concepts'],assigned)
    report=attribute_metrics(val_scores,test_scores,annotations,assigned)
    report['certainty_sensitivity']={str(level):attribute_metrics(val_scores,test_scores,annotations,assigned,min_certainty=level)
                                     for level in [2,4]}
    report['task_metrics']=task_metrics(test['probabilities'],pack['test']['labels'])
    train=collect(model,pack['train'],samples=1,mean=True)
    annotations['test']['task_labels']=pack['test']['labels']
    report['interventions']=intervention_curve(model,train['clean_concepts'],test['concepts'],annotations,assigned)
    report['mapping_file']=str(map_path)
    (Path(args.run)/'cub_concepts.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {'details','interventions'}},indent=2),flush=True)


if __name__=='__main__':main()
