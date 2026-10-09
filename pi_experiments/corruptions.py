"""Custom, paired image corruption stress tests; not the CIFAR-C benchmark."""

import argparse
import io
import hashlib
import inspect
import json
from pathlib import Path

import clip
import numpy as np
import PIL
import torch
from PIL import Image, ImageEnhance, ImageFilter

from .analyze import collect, load_model, paired_bootstrap, task_metrics
from .prepare import atomic_save, encode_dataset, load_sources
from .resources import gpu_lease


SETTINGS={'gaussian_noise':[0.05,0.1,0.2],'gaussian_blur':[0.5,1,2],
          'brightness':[0.5,0.75,1.5],'contrast':[0.5,0.75,1.5],'jpeg':[50,25,10]}


def corrupt(image,kind,parameter,index,seed=909):
    image=image.convert('RGB')
    if kind=='gaussian_noise':
        # Same standard-normal field at all severities and in every model.
        rng=np.random.default_rng(np.random.SeedSequence([seed,index]))
        pixels=np.asarray(image).astype(np.float32)
        noisy=np.clip(pixels+rng.normal(size=pixels.shape)*255*parameter,0,255).round().astype(np.uint8)
        return Image.fromarray(noisy)
    if kind=='gaussian_blur':return image.filter(ImageFilter.GaussianBlur(parameter))
    if kind=='brightness':return ImageEnhance.Brightness(image).enhance(parameter)
    if kind=='contrast':return ImageEnhance.Contrast(image).enhance(parameter)
    if kind=='jpeg':
        stream=io.BytesIO()
        image.save(stream,format='JPEG',quality=int(parameter))
        stream.seek(0)
        return Image.open(stream).convert('RGB')
    raise ValueError(kind)


class CorruptedSource:
    def __init__(self,source,kind,parameter):self.source,self.kind,self.parameter=source,kind,parameter
    def __len__(self):return len(self.source)
    def __getitem__(self,index):
        image,label=self.source[index]
        return corrupt(image,self.kind,self.parameter,index),label


def cache_cpu_generator(model, device):
    if str(device) != 'cpu' or model.generator is None:
        return False
    from .noise_branch_evidence import CachedParameters
    model.generator = CachedParameters(model.generator).eval()
    return True


def prepare(args,manifest,directory):
    config={'protocol':'custom native-resolution image corruptions, not CIFAR-C','settings':SETTINGS,
            'test_indices_sha256':hashlib.sha256(json.dumps(manifest['test_indices']).encode()).hexdigest(),
            'encoder_sha256':manifest['encoder_checkpoint_sha256'],'backbone':manifest['backbone'],
            'Pillow':PIL.__version__,'numpy':np.__version__,
            'corruption_function_sha256':hashlib.sha256(inspect.getsource(corrupt).encode()).hexdigest()}
    config_path=directory/'corruptions'/'protocol.json'
    config_path.parent.mkdir(parents=True,exist_ok=True)
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Corruption cache settings changed'
    else:config_path.write_text(json.dumps(config,indent=2))
    _,source=load_sources(manifest['dataset'],manifest['root'],manifest.get('official_test_root'))
    encoder,transform=clip.load(manifest['backbone'],device=args.device)
    encoder.eval().requires_grad_(False)
    for kind in args.kinds:
        for parameter in SETTINGS[kind]:
            path=directory/'corruptions'/f'{kind}_{parameter}.pt'
            if path.exists():
                print('reuse',path,flush=True)
                continue
            values=encode_dataset(CorruptedSource(source,kind,parameter),manifest['test_indices'],encoder,transform,args)
            values['metadata']={'protocol':'native-resolution corruption before frozen CLIP preprocessing; custom stress test',
                                'kind':kind,'parameter':parameter,'noise_seed':909,'Pillow':PIL.__version__,
                                'encoder_sha256':manifest['encoder_checkpoint_sha256'],'backbone':manifest['backbone']}
            atomic_save(values,path)
            print('saved',path,flush=True)


def evaluate(args,manifest,directory):
    pack=torch.load(args.pack,map_location='cpu',weights_only=True)
    runs=[Path(args.run)] if args.run else sorted((Path(args.runs_root)/manifest['dataset']).glob('*_seed*'))
    labels=pack['test']['labels']
    for run in runs:
        if not (run/'analysis.json').exists():continue
        model,config=load_model(pack,run/'model.pt',args.device)
        if config.get('validation_only'):raise ValueError('Search checkpoints cannot evaluate corruptions on the test set')
        cached_generator = cache_cpu_generator(model, args.device)
        clean_predictions=torch.load(run/'predictions.pt',weights_only=True)
        assert torch.equal(clean_predictions['labels'],labels)
        results=[]
        predictions={'labels':labels}
        for kind in args.kinds:
            for parameter in SETTINGS[kind]:
                case=f'{kind}_{parameter}'
                values=torch.load(directory/'corruptions'/f'{case}.pt',map_location='cpu',weights_only=True)
                assert torch.equal(values['labels'],labels)
                assert torch.equal(values['indices'],pack['test']['indices'])
                outputs=collect(model,values,samples=8,seed=4090+config['seed'])
                predictions[case]=outputs['probabilities']
                drift=(outputs['concepts']-clean_predictions['mean_concepts_mc8']).square().mean().item()
                results.append({'case':case,'metadata':values['metadata'],**task_metrics(outputs['probabilities'],labels),
                                'corruption_concept_drift_mse':drift,
                                'paired_accuracy_vs_same_model_clean':paired_bootstrap(clean_predictions['mc8'].numpy(),outputs['probabilities'].numpy(),labels.numpy())})
                print(run.name,case,results[-1]['accuracy'],flush=True)
        report={'protocol':'all fixed corruptions; no retraining, refitting or tuning on corrupted test images',
                'evaluation_split':'test','mc_samples':8,'cases':results,
                'deterministic_generator_cache':cached_generator,
                'evaluation_entry_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (run/'corruptions.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        atomic_save(predictions,run/'corruption_predictions.pt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','evaluate'],required=True)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--pack')
    parser.add_argument('--run')
    parser.add_argument('--runs-root',default='outputs/study_v1')
    parser.add_argument('--kinds',nargs='+',choices=list(SETTINGS),default=list(SETTINGS))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--encode-batch-size',type=int,default=128)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    torch.set_num_threads(8)
    if args.stage=='evaluate' and not args.pack:parser.error('--pack is required for evaluation')
    directory=Path(args.directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    with gpu_lease(f'image corruptions:{args.stage}',args.device):
        (prepare if args.stage=='prepare' else evaluate)(args,manifest,directory)


if __name__=='__main__':main()
