"""Evaluate task calibration, noise diagnostics, sparsity and concept faithfulness."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F

from .model import DPi, DProjection


def load_model(pack, checkpoint, device='cuda'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    args = state['config']
    projection = DProjection(pack['centers'], pack['mean'], pack['scale'])
    model = DPi(projection, pack['weight'], pack['bias'],
                pack['train']['features'].square().mean(0).sqrt().clamp_min(1e-6),
                args['mode'], args['conditioning'], args['site'], args['hidden'], args['initial_scale'])
    model.load_state_dict(state['state_dict'])
    return model.to(device).eval(), args


def task_metrics(probabilities, labels):
    probs = torch.as_tensor(probabilities).float()
    y = torch.as_tensor(labels).long()
    confidence, predicted = probs.max(-1)
    correct = predicted.eq(y)
    ece = 0.0
    for i in range(15):
        mask = (confidence > i/15) & (confidence <= (i+1)/15)
        if mask.any():
            ece += mask.float().mean().item() * abs(correct[mask].float().mean().item()-confidence[mask].mean().item())
    return {'accuracy':correct.float().mean().item(),
            'macro_f1':float(f1_score(y.numpy(),predicted.numpy(),average='macro')),
            'nll':F.nll_loss(probs.clamp_min(1e-12).log(),y).item(),
            'ece_15_bins':ece,
            'brier':(probs-F.one_hot(y,probs.shape[1])).square().sum(-1).mean().item()}


@torch.no_grad()
def collect(model, split, samples=8, seed=0, mean=False, batch_size=512):
    device=model.feature_scale.device
    arrays={key:[] for key in ['probabilities','concepts','clean_concepts','top5_agreement','sampling_top5_stability','concept_sampling_variance','noise_energy','mu_rms','sigma_rms','attention_entropy']}
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(seed)
        for features in split['features'].split(batch_size):
            features=features.to(device)
            draws=[];probs=[];clean=None;energies=[]
            for _ in range(samples if not mean else 1):
                logits, concepts, clean, energy=model(features,sample=not mean)
                draws.append(concepts);probs.append(logits.softmax(-1));energies.append(energy)
            average=torch.stack(draws).mean(0)
            clean_top=clean.topk(min(5,clean.shape[-1]),dim=-1).indices
            current_top=average.topk(min(5,average.shape[-1]),dim=-1).indices
            overlap=(current_top[:,:,None]==clean_top[:,None,:]).any(-1).float().mean(-1)
            draw_tops=torch.stack(draws).topk(min(5,average.shape[-1]),dim=-1).indices
            stability=(draw_tops[:,:,:,None]==current_top[None,:,None,:]).any(-1).float().mean((0,2))
            arrays['probabilities'].append(torch.stack(probs).mean(0).cpu())
            arrays['concepts'].append(average.cpu())
            arrays['clean_concepts'].append(clean.cpu())
            arrays['top5_agreement'].append(overlap.cpu())
            arrays['sampling_top5_stability'].append(stability.cpu())
            arrays['concept_sampling_variance'].append(torch.stack(draws).var(0,unbiased=False).mean(-1).cpu())
            # Energy is a batch mean. Repeat it per image so a short final batch
            # receives its actual sample weight in the dataset summary.
            arrays['noise_energy'].append(torch.stack(energies).mean().cpu().expand(len(features)))
            if model.generator is not None:
                mu,sigma=model.generator.parameters_for(features)
                arrays['mu_rms'].append(mu.square().mean(-1).sqrt().cpu())
                arrays['sigma_rms'].append(sigma.square().mean(-1).sqrt().cpu())
                if model.generator.query is not None:
                    logits=model.generator.query(F.normalize(features,dim=-1)) @ model.generator.prototypes.T
                    attention=logits.softmax(-1)
                    entropy=-(attention*attention.clamp_min(1e-12).log()).sum(-1)/np.log(attention.shape[-1])
                    arrays['attention_entropy'].append(entropy.cpu())
    return {key:torch.cat(values) for key,values in arrays.items() if values}


def deletion_curve(model, concepts, labels, fractions=(0.0,0.01,0.05,0.1,0.2,0.4), seed=991):
    # Use the training-centered baseline u=0; this evaluates concept contribution,
    # and does not interpret zero standardized response as semantic absence.
    weight=model.head.weight.detach().cpu()
    bias=model.head.bias.detach().cpu()
    logits=concepts @ weight.T+bias
    predicted=logits.argmax(-1)
    contribution=concepts*weight[predicted]
    ranked=contribution.argsort(-1,descending=True)
    rng=torch.Generator().manual_seed(seed)
    random_order=torch.rand(len(concepts),concepts.shape[1],generator=rng).argsort(-1)
    curves=[]
    for order,name in [(ranked,'contribution'),(random_order,'random')]:
        for fraction in fractions:
            removed=int(round(concepts.shape[1]*fraction))
            modified=concepts.clone()
            modified.scatter_(1,order[:,:removed],0)
            probabilities=(modified @ weight.T+bias).softmax(-1)
            curves.append({'policy':name,'fraction':fraction,
                           'original_class_probability':probabilities[torch.arange(len(concepts)),predicted].mean().item(),
                           **task_metrics(probabilities,labels)})
    return curves


def pruning_curve(model, concepts, labels):
    weight=model.head.weight.detach().cpu()
    bias=model.head.bias.detach().cpu()
    order=weight.abs().argsort(-1,descending=True)
    points=[]
    for k in sorted({min(count,weight.shape[1]) for count in [1,5,10,20,40,80,160,weight.shape[1]]}):
        mask=torch.zeros_like(weight).scatter_(1,order[:,:k],1)
        probs=(concepts @ (weight*mask).T+bias).softmax(-1)
        points.append({'concepts_per_class':k,**task_metrics(probs,labels)})
    return points


def paired_bootstrap(clean, adapted, labels, seed=31, draws=2000):
    labels=np.asarray(labels)
    difference=(np.asarray(adapted).argmax(-1)==labels).astype(float)-(np.asarray(clean).argmax(-1)==labels).astype(float)
    rng=np.random.default_rng(seed)
    values=np.empty(draws)
    for start in range(0,draws,50):
        indices=rng.integers(0,len(labels),size=(min(50,draws-start),len(labels)))
        values[start:start+len(indices)]=difference[indices].mean(-1)
    return {'accuracy_delta':float(difference.mean()),'ci95':np.quantile(values,[0.025,0.975]).tolist(),
            'draws':draws,'seed':seed,'wins':int((difference>0).sum()),'losses':int((difference<0).sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--split',choices=['val','test'],default='test')
    args=parser.parse_args()
    torch.set_num_threads(8)
    pack=torch.load(args.pack,weights_only=True,map_location='cpu')
    directory=Path(args.run)
    model,config=load_model(pack,directory/'model.pt',args.device)
    if config.get('validation_only') and args.split=='test':
        parser.error('Validation-search models must be analyzed with --split val; run the selected final experiment before testing')
    baseline=copy.deepcopy(model)
    baseline.generator=None;baseline.mode='none'
    with torch.no_grad():
        baseline.head.weight.copy_(pack['weight'].to(args.device))
        baseline.head.bias.copy_(pack['bias'].to(args.device))
    labels=pack[args.split]['labels']
    clean=collect(baseline,pack[args.split],samples=1,seed=1900)
    report={'baseline':task_metrics(clean['probabilities'],labels),'inference':{},
            'baseline_pruning':pruning_curve(baseline,clean['concepts'],labels),
            'baseline_deletion':deletion_curve(baseline,clean['concepts'],labels)}
    predictions={'labels':labels,'baseline':clean['probabilities']}
    for count in [1,4,8,16]:
        outputs=collect(model,pack[args.split],samples=count,seed=1900+config['seed'])
        metrics=task_metrics(outputs['probabilities'],labels)
        metrics['standardized_concept_mse']=(outputs['concepts']-outputs['clean_concepts']).square().mean().item()
        metrics['top5_clean_agreement']=outputs['top5_agreement'].mean().item()
        metrics['noise_energy']=outputs['noise_energy'].mean().item()
        metrics['sampling_top5_stability']=outputs['sampling_top5_stability'].mean().item()
        metrics['concept_sampling_variance']=outputs['concept_sampling_variance'].mean().item()
        if 'mu_rms' in outputs:
            metrics['mu_rms']=outputs['mu_rms'].mean().item();metrics['sigma_rms']=outputs['sigma_rms'].mean().item()
        if 'attention_entropy' in outputs:
            metrics['attention_entropy_normalized']=outputs['attention_entropy'].mean().item()
        report['inference'][f'mc{count}']=metrics
        predictions[f'mc{count}']=outputs['probabilities']
        if count==8:
            report['deletion']=deletion_curve(model,outputs['concepts'],labels)
            report['pruning']=pruning_curve(model,outputs['concepts'],labels)
            report['paired_accuracy']=paired_bootstrap(clean['probabilities'].numpy(),outputs['probabilities'].numpy(),labels.numpy())
            predictions['mean_concepts_mc8']=outputs['concepts']
    mean=collect(model,pack[args.split],mean=True,seed=1900)
    predictions['mean']=mean['probabilities']
    report['inference']['mean']=task_metrics(mean['probabilities'],labels)
    report['nec_threshold_1e-5']=(model.head.weight.abs()>1e-5).float().sum(-1).mean().item()
    suffix='' if args.split=='test' else '_validation'
    report['evaluation_split']=args.split
    (directory/f'analysis{suffix}.json').write_text(json.dumps(report,indent=2))
    torch.save(predictions,directory/f'predictions{suffix}.pt')
    print(json.dumps({'run':str(directory),'baseline':report['baseline'],'mc8':report['inference']['mc8'],
                      'paired_accuracy':report['paired_accuracy']},indent=2),flush=True)


if __name__=='__main__':
    main()
