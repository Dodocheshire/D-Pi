"""Resumable layer4 caches; exact frozen RN50 pooling and training-only RMS."""

import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .prepare import ImageDataset, atomic_save, load_sources
from .resources import gpu_lease


def prepare(args):
    directory=Path(args.directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['backbone']=='RN50'
    output=directory/'internal'
    output.mkdir(parents=True,exist_ok=True)
    config={'encoder_sha256':manifest['encoder_checkpoint_sha256'],'layer':'visual.layer4',
            'shape':[2048,7,7],'cache_dtype':'float16','torch':torch.__version__,
            'manifest_sha256':hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()}
    config_path=output/'config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config
    else:config_path.write_text(json.dumps(config,indent=2))
    encoder,transform=clip.load('RN50',device=args.device)
    encoder.eval().requires_grad_(False)
    pool=encoder.visual.attnpool
    pool_record={'state_dict':{k:v.cpu() for k,v in pool.state_dict().items()},
                 'kwargs':{'spacial_dim':7,'embed_dim':2048,'num_heads':32,'output_dim':1024},
                 'dtype':str(pool.q_proj.weight.dtype),'encoder_sha256':config['encoder_sha256']}
    atomic_save(pool_record,output/'pool.pt')
    captured={}
    hook=encoder.visual.layer4.register_forward_hook(lambda module,inputs,value:captured.update(maps=value.detach()))
    train,test=load_sources(manifest['dataset'],manifest['root'],manifest.get('official_test_root'))
    for split in ['train','val','test']:
        final=output/f'{split}_maps.npy'
        if final.exists():
            assert np.load(final,mmap_mode='r').shape==(len(manifest[f'{split}_indices']),2048,7,7)
            print('reuse',final,flush=True);continue
        ids=manifest[f'{split}_indices']
        partial=output/f'{split}_maps.partial.npy'
        progress_path=output/f'{split}_progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'completed_rows':0,'max_embedding_error':0.0}
        mode='r+' if partial.exists() else 'w+'
        if mode=='w+':assert progress['completed_rows']==0
        maps=np.lib.format.open_memmap(partial,mode=mode,dtype=np.float16,shape=(len(ids),2048,7,7))
        baseline=torch.load(directory/f'{split}_features.pt',weights_only=True)
        assert baseline['indices'].tolist()==ids
        source=test if split=='test' else train
        start=progress['completed_rows']
        loader=DataLoader(Subset(ImageDataset(source,transform),ids[start:]),batch_size=128,
                          num_workers=args.workers,pin_memory=True)
        started=time.monotonic()
        for images,labels in loader:
            with torch.no_grad():
                features=encoder.encode_image(images.to(args.device)).float().cpu()
            count=len(images)
            expected=baseline['features'][start:start+count]
            error=(features-expected).abs().max().item()
            torch.testing.assert_close(features,expected,rtol=0,atol=0.002)
            assert torch.equal(labels,baseline['labels'][start:start+count])
            maps[start:start+count]=captured.pop('maps').cpu().numpy()
            start+=count
            # Persist data before claiming progress; interruption may safely
            # recompute the last batch, but never skip unwritten activations.
            maps.flush()
            progress.update(completed_rows=start,max_embedding_error=max(progress['max_embedding_error'],error))
            temporary=progress_path.with_suffix('.partial')
            temporary.write_text(json.dumps(progress));temporary.replace(progress_path)
            if start%6400==0 or start==len(ids):
                print(f'{split} layer4 {start}/{len(ids)} elapsed={time.monotonic()-started:.1f}s max_z_error={progress["max_embedding_error"]}',flush=True)
        del maps
        partial.replace(final)
    hook.remove()
    train_maps=np.load(output/'train_maps.npy',mmap_mode='r')
    sum_squares=torch.zeros(2048,dtype=torch.float64)
    for start in range(0,len(train_maps),256):
        values=torch.from_numpy(np.array(train_maps[start:start+256],copy=True)).float()
        sum_squares+=values.double().square().sum((0,2,3))
    scale=(sum_squares/(len(train_maps)*49)).sqrt().float().clamp_min(1e-6)
    atomic_save({'channel_scale':scale,'training_images':len(train_maps),'spatial_cells':49,
                 'protocol':'per-channel RMS from training layer4 activations only'},output/'channel_scale.pt')
    print('completed internal caches and training-only channel RMS',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    torch.set_num_threads(8)
    output=Path(args.directory)/'internal';output.mkdir(parents=True,exist_ok=True)
    with (output/'prepare.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with gpu_lease('RN50 layer4 cache',args.device):
            prepare(args)


if __name__=='__main__':main()
