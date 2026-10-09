"""Train internal Pi with frozen RN50 pooling, shared prototypes and head."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from clip.model import AttentionPool2d
from torch.nn import functional as F

from .analyze import deletion_curve, paired_bootstrap, pruning_curve, task_metrics
from .internal_model import InternalDPi
from .model import DProjection
from .prepare import atomic_save
from .resources import gpu_lease


def load_pool(directory,device):
    record=torch.load(Path(directory)/'pool.pt',weights_only=True,map_location='cpu')
    dtype={'torch.float16':torch.float16,'torch.float32':torch.float32}[record['dtype']]
    pool=AttentionPool2d(**record['kwargs']).to(dtype=dtype)
    pool.load_state_dict(record['state_dict'])
    return pool.to(device).eval().requires_grad_(False)


def load_split(pack,split,directory,device,resident):
    maps=np.load(Path(directory)/f'{split}_maps.npy',mmap_mode='r')
    data=pack[split]
    assert len(maps)==len(data['features'])
    if resident:maps=torch.from_numpy(np.array(maps,copy=True)).to(device)
    return {'maps':maps,'features':data['features'].to(device),'labels':data['labels'].to(device)}


def batches(split,batch_size,shuffle=False):
    device=split['features'].device
    order=torch.randperm(len(split['labels']),device=device) if shuffle else torch.arange(len(split['labels']),device=device)
    for ids in order.split(batch_size):
        if isinstance(split['maps'],torch.Tensor):maps=split['maps'][ids]
        else:maps=torch.from_numpy(np.array(split['maps'][ids.cpu().numpy()],copy=True)).to(device)
        yield maps,split['features'][ids],split['labels'][ids]


@torch.no_grad()
def evaluate(model,split,samples=8,seed=1700,mean=False):
    device=model.channel_scale.device
    probabilities=[];concepts=[];clean=[];variance=[];draw_mse=[];top_agreement=[];stability=[];mu_energy=[];sigma_energy=[]
    model.eval()
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(seed)
        for maps,z,labels in batches(split,256):
            ps=[];cs=[]
            for _ in range(1 if mean else samples):
                logits,current,original,_=model(maps,z,sample=not mean)
                ps.append(logits.softmax(-1));cs.append(current)
            stacked=torch.stack(cs)
            average=stacked.mean(0)
            width=min(5,average.shape[-1])
            current_top=average.topk(width,-1).indices
            clean_top=original.topk(width,-1).indices
            top_agreement.append((current_top[:,:,None]==clean_top[:,None,:]).any(-1).float().mean(-1).cpu())
            draw_top=stacked.topk(width,-1).indices
            stability.append((draw_top[:,:,:,None]==current_top[None,:,None,:]).any(-1).float().mean((0,2)).cpu())
            if model.generator is not None:
                mu,sigma=model.generator.parameters_for(z)
            else:
                mu=z.new_zeros(len(z),len(model.channel_scale))
                sigma=torch.full_like(mu,model.initial_scale) if model.mode=='fixed' else torch.zeros_like(mu)
            mu_energy.append(mu.square().mean(-1).cpu());sigma_energy.append(sigma.square().mean(-1).cpu())
            probabilities.append(torch.stack(ps).mean(0).cpu())
            concepts.append(stacked.mean(0).cpu());clean.append(original.cpu())
            variance.append(stacked.var(0,unbiased=False).mean(-1).cpu())
            draw_mse.append((stacked-original).square().mean((0,2)).cpu())
    result={'probabilities':torch.cat(probabilities),'concepts':torch.cat(concepts),'clean_concepts':torch.cat(clean)}
    metrics=task_metrics(result['probabilities'],split['labels'].cpu())
    metrics.update(standardized_concept_draw_mse=torch.cat(draw_mse).mean().item(),
                   standardized_concept_mean_mse=(result['concepts']-result['clean_concepts']).square().mean().item(),
                   concept_sampling_variance=torch.cat(variance).mean().item(),
                   top5_clean_agreement=torch.cat(top_agreement).mean().item(),
                   sampling_top5_stability=torch.cat(stability).mean().item(),
                   mean_energy=torch.cat(mu_energy).mean().item(),scale_energy=torch.cat(sigma_energy).mean().item())
    return metrics,result


def run(pack,args):
    torch.set_num_threads(8);torch.manual_seed(args.seed)
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['internal_model.py','internal_train.py','model.py','resources.py','analyze.py']}
    for name,digest in hashes.items():
        snapshot=Path('artifacts/code_snapshots')/digest
        snapshot.mkdir(parents=True,exist_ok=True)
        (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    directory=Path(args.directory)/'internal'
    manifest=json.loads((Path(args.directory)/'manifest.json').read_text())
    cache_config=json.loads((directory/'config.json').read_text())
    assert cache_config['manifest_sha256']==hashlib.sha256((Path(args.directory)/'manifest.json').read_bytes()).hexdigest()
    for split in ['train','val']:
        assert pack[split]['indices'].tolist()==manifest[f'{split}_indices']
    channel_scale=torch.load(directory/'channel_scale.pt',weights_only=True)['channel_scale']
    projection=DProjection(pack['centers'],pack['mean'],pack['scale'])
    model=InternalDPi(projection,pack['weight'],pack['bias'],load_pool(directory,args.device),channel_scale,
                      args.mode,args.conditioning,args.hidden,args.initial_scale).to(args.device)
    train=load_split(pack,'train',directory,args.device,args.resident_maps)
    val=load_split(pack,'val',directory,args.device,args.resident_maps)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    history=[];best=None;best_nll=float('inf')
    started=time.monotonic()
    if model.generator is not None:
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
        for epoch in range(args.epochs):
            model.train()
            for maps,z,labels in batches(train,args.batch_size,shuffle=True):
                optimizer.zero_grad()
                logits,concepts,clean,energy=model(maps,z)
                loss=F.cross_entropy(logits,labels)+args.concept_weight*F.mse_loss(concepts,clean)+args.energy_weight*energy
                loss.backward();optimizer.step()
            metrics,_=evaluate(model,val,samples=args.mc_samples,seed=args.seed+1000)
            history.append({'epoch':epoch+1,**metrics})
            if metrics['nll']<best_nll:
                best_nll=metrics['nll'];best=copy.deepcopy(model.state_dict())
            print(f"internal epoch={epoch+1}/{args.epochs} val_acc={metrics['accuracy']:.5f} val_nll={metrics['nll']:.5f} elapsed={time.monotonic()-started:.1f}s",flush=True)
        model.load_state_dict(best)
    split='val' if args.validation_only else 'test'
    if split=='test':
        assert pack[split]['indices'].tolist()==manifest[f'{split}_indices']
        data=load_split(pack,split,directory,args.device,args.resident_maps)
    else:data=val
    labels=data['labels'].cpu()
    clean_concepts=projection.to(args.device)(data['features']).detach().cpu()
    clean=(clean_concepts @ pack['weight'].T+pack['bias']).softmax(-1)
    report={'evaluation_split':split,'config':vars(args),'cache_config':cache_config,'trainable_parameters':trainable,
            'experiment_code_sha256':hashes,'history':history,'baseline':task_metrics(clean,labels),'inference':{}}
    report['baseline_pruning']=pruning_curve(model,clean_concepts,labels)
    report['baseline_deletion']=deletion_curve(model,clean_concepts,labels)
    report['nec_threshold_1e-5']=(model.head.weight.abs()>1e-5).float().sum(-1).mean().item()
    predictions={'labels':labels,'baseline':clean}
    for samples in [1,4,8,16]:
        metrics,outputs=evaluate(model,data,samples=samples,seed=args.seed+2000)
        report['inference'][f'mc{samples}']=metrics;predictions[f'mc{samples}']=outputs['probabilities']
        if samples==8:
            report['paired_accuracy']=paired_bootstrap(clean.numpy(),outputs['probabilities'].numpy(),labels.numpy())
            predictions['mean_concepts_mc8']=outputs['concepts']
            report['pruning']=pruning_curve(model,outputs['concepts'],labels)
            report['deletion']=deletion_curve(model,outputs['concepts'],labels)
    report['inference']['mean'],outputs=evaluate(model,data,mean=True,seed=args.seed+2000)
    predictions['mean']=outputs['probabilities']
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    atomic_save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'config':vars(args)},output/'model.pt')
    atomic_save(predictions,output/'predictions.pt')
    temporary=output/'metrics.partial.json';temporary.write_text(json.dumps(report,indent=2,allow_nan=False));temporary.replace(output/'metrics.json')
    print(json.dumps({'evaluation_split':split,'baseline':report['baseline'],'mc8':report['inference']['mc8'],'mean':report['inference']['mean']},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--pack',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['none','fixed','mean_only','mean_scale','adapter'],default='mean_scale')
    parser.add_argument('--conditioning',choices=['concept','random_concept','mlp'],default='concept')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--mc-samples',type=int,default=8)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--hidden',type=int,default=128)
    parser.add_argument('--initial-scale',type=float,default=0.01)
    parser.add_argument('--concept-weight',type=float,default=1)
    parser.add_argument('--energy-weight',type=float,default=0.1)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--validation-only',action='store_true')
    parser.add_argument('--resident-maps',action='store_true',help='Load maps into GPU memory for faster training')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.mc_samples<1:parser.error('epochs, batch size and MC samples must be positive')
    with gpu_lease('RN50 internal Pi training',args.device):
        run(torch.load(args.pack,map_location='cpu',weights_only=True),args)


if __name__=='__main__':main()
