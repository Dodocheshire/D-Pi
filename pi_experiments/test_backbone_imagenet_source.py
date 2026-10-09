"""Exercise ImageNet crop re-encoding with a separate official-test root."""

from types import SimpleNamespace
import json

from PIL import Image
import torch

from . import backbone_prepare


def test_imagefolder_crop_reencoding_preserves_explicit_test_root(tmp_path, monkeypatch):
    root=tmp_path/'clean_subset';test_root=tmp_path/'official_validation'
    for split in ['train','adapter_val']:
        for name in ['class_a','class_b']:
            path=root/split/name;path.mkdir(parents=True)
            Image.new('RGB',(4,4),(20,50,80)).save(path/'sample.png')
    for name in ['class_a','class_b']:
        path=test_root/name;path.mkdir(parents=True)
        Image.new('RGB',(4,4)).save(path/'sample.png')
    source=tmp_path/'rn50';destination=tmp_path/'vit'
    (source/'proposals').mkdir(parents=True);destination.mkdir()
    (source/'proposal_config.json').write_text(json.dumps({'max_long_side':1024}))
    manifest={'root':str(root),'official_test_root':str(test_root),'concept_source_indices':[0,3]}
    for index,label in [(0,0),(3,1)]:
        torch.save({'source_index':index,'label':label,'image_size':[4,4],
                    'boxes':torch.tensor([[0.,0.,4.,4.]]),'features':torch.zeros(1,3)},
                   source/'proposals'/f'{index:08d}.pt')
    class Encoder:
        visual=SimpleNamespace(output_dim=2)
        def eval(self):return self
        def requires_grad_(self,enabled):return self
        def encode_image(self,inputs):return torch.ones(len(inputs),2)
    monkeypatch.setattr(backbone_prepare.clip,'load',lambda *args,**kwargs:(Encoder(),lambda image:torch.ones(3,4,4)))
    args=SimpleNamespace(dataset='imagenet',backbone='ViT-B/16',device='cpu')
    backbone_prepare.reencode_crops(args,manifest,source,destination)
    for index,label in [(0,0),(3,1)]:
        result=torch.load(destination/'proposals'/f'{index:08d}.pt',weights_only=True)
        assert result['label']==label and result['source_index']==index
        assert result['features'].shape==(1,2) and torch.equal(result['boxes'],torch.tensor([[0.,0.,4.,4.]]))
