"""Align original per-image CUB attributes and parts to the frozen experiment split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .prepare import atomic_save, load_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default='artifacts/datasets/cub')
    parser.add_argument('--annotations',default='artifacts/annotations')
    args=parser.parse_args()
    directory=Path(args.directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    root=Path(args.annotations)/'CUB_200_2011'
    image_names={int(line.split()[0]):line.split()[1] for line in (root/'images.txt').read_text().splitlines()}
    name_to_id={name:image_id for image_id,name in image_names.items()}
    official_split={int(line.split()[0]):int(line.split()[1]) for line in (root/'train_test_split.txt').read_text().splitlines()}
    raw=np.loadtxt(root/'attributes/image_attribute_labels.txt',usecols=(0,1,2,3))
    attributes=torch.zeros(len(image_names),312,dtype=torch.uint8)
    certainty=torch.zeros_like(attributes)
    attributes[torch.tensor(raw[:,0]-1,dtype=torch.long),torch.tensor(raw[:,1]-1,dtype=torch.long)]=torch.tensor(raw[:,2],dtype=torch.uint8)
    certainty[torch.tensor(raw[:,0]-1,dtype=torch.long),torch.tensor(raw[:,1]-1,dtype=torch.long)]=torch.tensor(raw[:,3],dtype=torch.uint8)
    part_data=np.loadtxt(root/'parts/part_locs.txt')
    parts=torch.zeros(len(image_names),15,3)
    parts[torch.tensor(part_data[:,0]-1,dtype=torch.long),torch.tensor(part_data[:,1]-1,dtype=torch.long)]=torch.tensor(part_data[:,2:5]).float()
    names=[line.split(' ',1)[1] for line in (Path(args.annotations)/'attributes.txt').read_text().splitlines()]
    part_names=[line.split(' ',1)[1] for line in (root/'parts/parts.txt').read_text().splitlines()]
    train,test=load_sources('cub',manifest['root'])
    output={'attribute_names':names,'part_names':part_names,'metadata':{'source':'Caltech CUB-200-2011 original per-image annotations',
             'archive_md5':'97eceeb196236b17998738112f37df78',
             'attribute_file_sha256':hashlib.sha256((root/'attributes/image_attribute_labels.txt').read_bytes()).hexdigest()}}
    for split in ('train','val','test'):
        source=test if split=='test' else train
        ids=[]
        for index in manifest[f'{split}_indices']:
            image_path=Path(source.samples[index][0])
            name='/'.join(image_path.parts[-2:])
            image_id=name_to_id[name]
            assert official_split[image_id]==(0 if split=='test' else 1)
            ids.append(image_id-1)
        ids=torch.tensor(ids)
        output[split]={'image_ids':ids+1,'attributes':attributes[ids],'certainty':certainty[ids],'parts':parts[ids]}
    assert not set(output['train']['image_ids'].tolist()) & set(output['val']['image_ids'].tolist())
    assert not set(output['train']['image_ids'].tolist()) & set(output['test']['image_ids'].tolist())
    atomic_save(output,directory/'attributes.pt')
    print(json.dumps({'output':str(directory/'attributes.pt'),'split_sizes':{s:len(output[s]['image_ids']) for s in ('train','val','test')},
                      'attributes':312,'parts':15},indent=2))


if __name__=='__main__':
    main()
