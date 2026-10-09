"""Prepare and verify large-dataset split manifests before GPU preparation."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from .prepare import create_manifest, load_sources
from .suite import ROOTS, Runner
from .visible_semantics import write_json


IMAGENET_TEST='/home/pengcheng/VLG-CBM/datasets/imagenet/ILSVRC/Data/CLS-LOC/ImageNet_val'
EXPECTED={'places365':(1803460,36500,365),'imagenet':(210000,50000,1000)}


def main():
    runner=Runner('artifacts/logs/large_manifest_preflight_v1')
    for dataset in ['places365','imagenet']:
        runner.state.update(phase=f'{dataset}_source_and_split');runner.save()
        test_root=IMAGENET_TEST if dataset=='imagenet' else None
        args=SimpleNamespace(dataset=dataset,root=ROOTS[dataset],test_root=test_root,
                             backbone='RN50',split_seed=310,images_per_class=50)
        manifest=create_manifest(args)
        train,test=load_sources(dataset,args.root,args.test_root)
        labels=np.asarray(train.targets);test_labels=np.asarray(test.targets)
        expected_train,expected_test,classes=EXPECTED[dataset]
        assert len(train)==expected_train and len(test)==expected_test and len(manifest['classes'])==classes
        assert list(train.classes)==list(test.classes)
        if hasattr(train,'class_to_idx'):assert train.class_to_idx==test.class_to_idx
        tr=np.asarray(manifest['train_indices']);va=np.asarray(manifest['val_indices'])
        concepts=np.asarray(manifest['concept_source_indices'])
        assert np.array_equal(np.sort(np.concatenate([tr,va])),np.arange(len(train)))
        assert not np.intersect1d(tr,va).size and np.isin(concepts,tr).all()
        assert len(concepts)==len(np.unique(concepts))==classes*50
        train_counts=np.bincount(labels[tr],minlength=classes)
        val_counts=np.bincount(labels[va],minlength=classes)
        test_counts=np.bincount(test_labels,minlength=classes)
        concept_counts=np.bincount(labels[concepts],minlength=classes)
        assert (train_counts>0).all() and (val_counts>0).all() and (test_counts>0).all() and (concept_counts==50).all()
        if dataset=='imagenet':assert (train_counts==189).all() and (val_counts==21).all() and (test_counts==50).all()
        directory=Path('artifacts/datasets')/dataset;directory.mkdir(parents=True,exist_ok=True)
        path=directory/'manifest.json'
        if path.exists():assert json.loads(path.read_text())==manifest
        else:write_json(path,manifest)
        record={'dataset':dataset,'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'split_seed':310,'images_per_class':50,'classes':classes,
                'train_images':len(tr),'validation_images':len(va),'test_images':len(test),
                'concept_source_images':len(concepts),'train_val_disjoint':True,
                'source_pool_covered_exactly_once':True,'concept_sources_training_only':True,
                'class_alignment_exact':True,'class_counts':{'train':train_counts.tolist(),'val':val_counts.tolist(),
                                                          'test':test_counts.tolist(),'concept_sources':concept_counts.tolist()},
                'source_scope':'Full train-standard' if dataset=='places365' else 'Clean 200+10/class subset; not full ImageNet training',
                'test_scope':'Official validation as held-out test','device':'cpu',
                'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checked_at':time.time()}
        write_json(directory/'manifest_preflight.json',record)
        runner.state['completed_commands'].append({'name':dataset,'returncode':0,'finished_at':time.time()});runner.save()
        print(dataset,{k:record[k] for k in ['train_images','validation_images','test_images','concept_source_images']},flush=True)
    runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
