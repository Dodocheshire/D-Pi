"""CUB foreground-preserving background changes with frozen attribute thresholds."""

import argparse
import hashlib
import json
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .analyze import collect, load_model, paired_bootstrap, task_metrics
from .cub_metrics import MIN_CERTAINTY
from .prepare import atomic_save, encode_dataset, load_sources
from .resources import gpu_lease


CASES = ['identity', 'gray', 'train_background_17', 'train_background_29', 'train_background_43']
MASK_MD5 = '4d47ba1228eae64f2fa547c47bc65255'


def mask_for(path, root):
    path = Path(path)
    mask_path = Path(root) / path.parent.name / (path.stem + '.png')
    return np.asarray(Image.open(mask_path).convert('L')) >= 128


def remove_foreground(image, mask):
    pixels = np.asarray(image.convert('RGB'))
    if pixels.shape[:2] != mask.shape or not mask.any() or mask.all():
        raise ValueError('Donor needs an aligned, nonempty mask and some background')
    nearest = distance_transform_edt(mask, return_distances=False, return_indices=True)
    result = pixels.copy()
    result[mask] = pixels[nearest[0][mask], nearest[1][mask]]
    return Image.fromarray(result)


def replace_background(image, mask, background):
    pixels = np.asarray(image.convert('RGB'))
    if pixels.shape[:2] != mask.shape or not mask.any() or mask.all():
        raise ValueError('Foreground needs an aligned, nonempty majority mask')
    replacement = np.asarray(background.convert('RGB').resize(image.size, Image.Resampling.BICUBIC)).copy()
    replacement[mask] = pixels[mask]
    return Image.fromarray(replacement)


class BackgroundSource:
    def __init__(self, source, train, donor_indices, mask_root, case):
        self.source, self.train, self.donors = source, train, donor_indices
        self.mask_root, self.case = mask_root, case

    def __len__(self):
        return len(self.source)

    def donor(self, index):
        seed = int(self.case.rsplit('_', 1)[1])
        rng = np.random.default_rng(np.random.SeedSequence([611, seed, index]))
        return int(self.donors[int(rng.integers(len(self.donors)))])

    def __getitem__(self, index):
        image, label = self.source[index]
        if self.case == 'identity':
            return image.convert('RGB'), label
        foreground = mask_for(self.source.samples[index][0], self.mask_root)
        if self.case == 'gray':
            background = Image.new('RGB', image.size, (128, 128, 128))
        else:
            donor = self.donor(index)
            raw, _ = self.train[donor]
            background = remove_foreground(raw, mask_for(self.train.samples[donor][0], self.mask_root))
        return replace_background(image, foreground, background), label


def frozen_attribute_metrics(concepts, original, annotations, certainty=MIN_CERTAINTY):
    details = []
    for reference in original['details']:
        attribute = reference['attribute']
        known = (annotations['test']['certainty'][:, attribute] >= certainty).numpy()
        labels = annotations['test']['attributes'][:, attribute].numpy()[known]
        if len(np.unique(labels)) != 2:
            continue
        scores = concepts[:, reference['prototypes']].mean(-1).numpy()[known]
        details.append({'attribute': attribute, 'name': reference['name'], 'threshold': reference['threshold'],
                        'known_images': int(known.sum()),
                        'balanced_accuracy': float(balanced_accuracy_score(labels, scores >= reference['threshold'])),
                        'auroc': float(roc_auc_score(labels, scores))})
    return {'min_certainty': certainty, 'evaluated_attributes': len(details),
            'attribute_macro_balanced_accuracy': float(np.mean([d['balanced_accuracy'] for d in details])) if details else None,
            'attribute_macro_auroc': float(np.mean([d['auroc'] for d in details])) if details else None,
            'threshold_policy': 'Reuse this model original-validation thresholds; no calibration on changed backgrounds',
            'details': details}


def prepare(args, manifest):
    directory = Path(args.directory)
    cache = directory / 'backgrounds'
    cache.mkdir(parents=True, exist_ok=True)
    archive = Path('artifacts/annotations/segmentations.tgz.partial')
    assert hashlib.md5(archive.read_bytes()).hexdigest() == MASK_MD5
    train, test = load_sources('cub', manifest['root'])
    donors = manifest['train_indices']
    protocol = {'cases': CASES, 'mask_md5': MASK_MD5, 'mask_threshold': 128,
                'mask_source': 'https://data.caltech.edu/records/w9d68-gec53',
                'donor_indices_sha256': hashlib.sha256(json.dumps(donors).encode()).hexdigest(),
                'manifest_sha256': hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest(),
                'encoder_sha256': manifest['encoder_checkpoint_sha256'], 'backbone': manifest['backbone'],
                'foreground': 'Original RGB values retained exactly wherever official mask >=128 (majority annotation).',
                'donor': 'Uniform training-split donor, independent of class; remove donor bird with nearest background pixels.',
                'test_use': 'Evaluation only, no refitting or hyperparameter selection.',
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = cache / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        path.write_text(json.dumps(protocol, indent=2))
    for case in args.cases:
        target = cache / f'{case}.pt'
        if target.exists():
            continue
        source = BackgroundSource(test, train, donors, args.masks, case)
        with gpu_lease(f'CUB backgrounds feature cache {case}', args.device):
            encoder, transform = clip.load(manifest['backbone'], device=args.device)
            encoder.eval().requires_grad_(False)
            values = encode_dataset(source, manifest['test_indices'], encoder, transform, args)
            assert torch.isfinite(values['features']).all()
            if case == 'identity':
                original = torch.load(directory / 'test_features.pt', weights_only=True)
                assert torch.equal(original['labels'], values['labels'])
                assert torch.equal(original['indices'], values['indices'])
                torch.testing.assert_close(original['features'], values['features'], rtol=0, atol=0)
            values['protocol_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if case.startswith('train_background'):
                values['donor_indices'] = torch.tensor([source.donor(i) for i in manifest['test_indices']])
                assert set(values['donor_indices'].tolist()) <= set(donors)
            atomic_save(values, target)
            del encoder
            if str(args.device).startswith('cuda'):
                torch.cuda.empty_cache()
        print('Prepared backgrounds', case, len(values['labels']), flush=True)


def evaluate(args, manifest):
    directory = Path(args.directory)
    run = Path(args.run)
    pack = torch.load(args.pack, weights_only=True)
    assert pack['metadata']['backbone'] == manifest['backbone']
    original = json.loads((run / 'cub_concepts.json').read_text())
    assert original['min_certainty'] == MIN_CERTAINTY
    annotations = torch.load(directory / 'attributes.pt', weights_only=True)
    model, config = load_model(pack, run / 'model.pt', args.device)
    assert not config.get('validation_only')
    cases = []
    predictions = {'labels': pack['test']['labels']}
    identity = collect(model, pack['test'], samples=8, seed=config['seed'] + 6090)
    for case in args.cases:
        values = torch.load(directory / 'backgrounds' / f'{case}.pt', weights_only=True)
        assert values['protocol_sha256'] == hashlib.sha256((directory / 'backgrounds/protocol.json').read_bytes()).hexdigest()
        assert torch.equal(values['labels'], pack['test']['labels'])
        assert torch.equal(values['indices'], pack['test']['indices'])
        outputs = collect(model, values, samples=8, seed=config['seed'] + 6090)
        mean = collect(model, values, samples=1, seed=config['seed'] + 6090, mean=True)
        predictions[case] = {'mc8': outputs['probabilities'], 'mean': mean['probabilities']}
        report = {'case': case, 'mc8': task_metrics(outputs['probabilities'], values['labels']),
                  'mean': task_metrics(mean['probabilities'], values['labels']),
                  'background_concept_drift_mse': (outputs['concepts'] - identity['concepts']).square().mean().item(),
                  'paired_vs_original': paired_bootstrap(identity['probabilities'].numpy(), outputs['probabilities'].numpy(), values['labels'].numpy()),
                  'semantic': frozen_attribute_metrics(outputs['concepts'], original, annotations),
                  'certainty_sensitivity': {str(level): frozen_attribute_metrics(outputs['concepts'], original['certainty_sensitivity'][str(level)], annotations, level)
                                           for level in [2, 4]}}
        cases.append(report)
        print(run.name, case, report['mc8']['accuracy'], flush=True)
    atomic_save(predictions, run / 'background_predictions.pt')
    report = {'protocol': 'Official majority foreground unchanged; paired context changes with fixed validation attribute thresholds',
              'evaluation_split': 'test', 'images': len(pack['test']['labels']), 'cases': cases,
              'original_attribute_report_sha256': hashlib.sha256((run / 'cub_concepts.json').read_bytes()).hexdigest(),
              'background_protocol_sha256': hashlib.sha256((directory / 'backgrounds/protocol.json').read_bytes()).hexdigest()}
    temporary = run / 'backgrounds.partial.json'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(run / 'backgrounds.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'evaluate'], required=True)
    parser.add_argument('--directory', default='artifacts/datasets/cub')
    parser.add_argument('--masks', default='artifacts/annotations/segmentations')
    parser.add_argument('--cases', choices=CASES, nargs='+', default=CASES)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--encode-batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pack')
    parser.add_argument('--run')
    args = parser.parse_args()
    torch.set_num_threads(8)
    manifest = json.loads((Path(args.directory) / 'manifest.json').read_text())
    assert manifest['dataset'] == 'cub'
    if args.stage == 'prepare':
        prepare(args, manifest)
    else:
        if not args.pack or not args.run:
            parser.error('Evaluation requires --pack and --run')
        with gpu_lease('CUB background evaluation', args.device):
            evaluate(args, manifest)


if __name__ == '__main__':
    main()
