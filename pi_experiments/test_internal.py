"""Check internal perturbations preserve the original clean path and frozen pool."""

import torch
from torch import nn

from .internal_model import InternalDPi
from .model import DProjection


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(4,6)
    def forward(self,maps):return self.linear(maps.mean((-2,-1)))


def fixture(mode):
    torch.manual_seed(11)
    pool=MeanPool()
    maps=torch.randn(13,4,3,3)
    features=pool(maps).detach()
    centers=torch.randn(7,6)
    raw=(features @ centers.T/centers.square().sum(-1)).softmax(-1)
    projection=DProjection(centers,raw.mean(0),raw.std(0).clamp_min(1e-4))
    model=InternalDPi(projection,torch.randn(3,7),torch.randn(3),pool,
                      maps.square().mean((0,2,3)).sqrt(),mode=mode,hidden=8)
    return model,maps,features


def test_zero_internal_noise_preserves_clean_concepts_and_logits():
    for mode in ['none','mean_only','adapter']:
        model,maps,features=fixture(mode)
        logits,concepts,clean,energy=model(maps,features)
        torch.testing.assert_close(concepts,clean)
        torch.testing.assert_close(logits,model.head(clean))
        assert energy<1e-12


def test_internal_gradient_flows_through_frozen_pool():
    model,maps,features=fixture('mean_scale')
    before={k:v.clone() for k,v in model.pool.state_dict().items()}
    logits,concepts,clean,energy=model(maps,features)
    loss=nn.functional.cross_entropy(logits,torch.arange(13)%3)+0.1*(concepts-clean).square().mean()+0.01*energy
    loss.backward()
    assert model.generator.mean.weight.grad.abs().sum()>0
    assert model.generator.log_scale.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.pool.parameters())
    assert model.head.weight.grad is None
    for name,value in model.pool.state_dict().items():torch.testing.assert_close(value,before[name])


def test_internal_adapter_and_gaussian_have_equal_capacity():
    gaussian,_,_=fixture('mean_scale')
    adapter,maps,features=fixture('adapter')
    count=lambda model:sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count(gaussian)==count(adapter)
    torch.testing.assert_close(adapter(maps,features,sample=True)[0],adapter(maps,features,sample=False)[0])


def test_internal_validation_does_not_open_test_data(tmp_path,monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace
    import numpy as np
    from . import internal_train

    class GuardedPack(dict):
        def __getitem__(self,key):
            assert key!='test','internal validation accessed held-out test data'
            return super().__getitem__(key)

    model,maps,features=fixture('mean_scale')
    directory=tmp_path/'dataset';internal=directory/'internal';internal.mkdir(parents=True)
    train_ids=list(range(13));val_ids=list(range(13,26))
    manifest=directory/'manifest.json'
    manifest.write_text(json.dumps({'train_indices':train_ids,'val_indices':val_ids}))
    (internal/'config.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    torch.save({'channel_scale':model.channel_scale},internal/'channel_scale.pt')
    for split in ['train','val']:np.save(internal/f'{split}_maps.npy',maps.numpy())
    monkeypatch.setattr(internal_train,'load_pool',lambda *args:model.pool)
    pack=GuardedPack(centers=model.projection.centers,mean=model.projection.mean,scale=model.projection.scale,
                     weight=model.head.weight,bias=model.head.bias,
                     train={'features':features,'indices':torch.tensor(train_ids),'labels':torch.arange(13)%3},
                     val={'features':features,'indices':torch.tensor(val_ids),'labels':torch.arange(13)%3})
    args=SimpleNamespace(seed=0,directory=str(directory),device='cpu',mode='mean_scale',conditioning='concept',
                         hidden=8,initial_scale=0.01,resident_maps=False,lr=3e-4,epochs=1,batch_size=8,
                         concept_weight=1,energy_weight=0.1,mc_samples=2,validation_only=True,output=str(tmp_path/'run'))
    report=internal_train.run(pack,args)
    assert report['evaluation_split']=='val'
