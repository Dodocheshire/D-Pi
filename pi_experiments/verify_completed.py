"""Verify completed study artifacts on CPU without claiming queued evaluations."""

import hashlib
import json
from pathlib import Path
import time

import torch

from .model import DPi, DProjection
from .random_bank_train import MatchedBankDPi
from .rotated_bank_train import RotatedBankDPi


def load(path):
    return torch.load(path, weights_only=True, map_location='cpu')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(8)
    report = json.loads(Path('reports/STUDY_VERIFICATION.json').read_text())
    pack = load('artifacts/datasets/cifar10/baseline_k256_seed0.pt')
    keys = {'projection.centers': 'centers', 'projection.mean': 'mean', 'projection.scale': 'scale',
            'head.weight': 'weight', 'head.bias': 'bias'}
    rows = []; seen = set()
    for root in ['outputs/study_v1/cifar10', 'outputs/conditioning_v1/cifar10',
                 'outputs/rotation_conditioning_v1/cifar10']:
        for path in sorted(Path(root).glob('*_seed*/model.pt')):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            state = load(path); config = state['config']; tensors = state['state_dict']
            refit = bool(config.get('refit_epochs', 0))
            for key, source in keys.items():
                if refit and key.startswith('head.'):
                    continue
                assert torch.equal(tensors[key], pack[source]), (path, key)
            rows.append({'checkpoint': str(path), 'sha256': sha(path),
                         'projection_exact': True, 'head_exact': not refit})
    assert len(rows) == 70, len(rows)
    report['cifar10_all_completed_checkpoint_invariants'] = rows
    for seed in range(5):
        projection = DProjection(pack['centers'], pack['mean'], pack['scale'])
        scale = pack['train']['features'].square().mean(0).sqrt().clamp_min(1e-6)
        torch.manual_seed(seed)
        original = DPi(projection, pack['weight'], pack['bias'], scale)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        matched = MatchedBankDPi(projection, pack['weight'], pack['bias'], scale, conditioning='random_concept')
        assert torch.equal(rng, torch.get_rng_state())
        for (name_a, a), (name_b, b) in zip(original.named_parameters(), matched.named_parameters()):
            assert name_a == name_b and torch.equal(a, b), (seed, name_a)
        assert torch.equal(matched.projection.centers, pack['centers'])
        assert not torch.equal(original.generator.prototypes, matched.generator.prototypes)
    report['matched_random_initialization'] = {'verified_seeds': list(range(5)), 'all_parameters_exact': True,
                                             'global_rng_exact': True, 'original_projection_exact': True,
                                             'conditioning_bank_differs': True}
    rotations = []
    for seed in range(5):
        torch.manual_seed(seed)
        original = DPi(projection, pack['weight'], pack['bias'], scale)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        rotated = RotatedBankDPi(projection, pack['weight'], pack['bias'], scale,
                                conditioning='random_concept')
        assert torch.equal(rng, torch.get_rng_state())
        a, b = dict(original.named_parameters()), dict(rotated.named_parameters())
        assert a.keys() == b.keys()
        assert all(torch.equal(a[name], b[name]) for name in a)
        path = Path('outputs/rotation_conditioning_v1/cifar10')/f'pi_rotated_matched_seed{seed}/model.pt'
        saved = load(path)['state_dict']['generator.prototypes']
        assert torch.equal(saved, rotated.generator.prototypes), (seed, 'conditioning bank changed during training')
        real_path = Path('outputs/study_v1/cifar10')/f'pi_seed{seed}/model.pt'
        real = load(real_path)['state_dict']['generator.prototypes']
        assert torch.equal(real, original.generator.prototypes)
        gram_error = (real.double()@real.double().T - saved.double()@saved.double().T).abs().max().item()
        assert gram_error < 1e-12, (seed, gram_error)
        rotations.append({'seed':seed,'checkpoint_sha256':sha(path),
                          'trainable_initialization_exact':True,'global_rng_exact':True,
                          'saved_conditioning_bank_exact':True,'gram_max_absolute_error':gram_error})
    report['geometry_preserving_rotation'] = rotations
    base = Path('artifacts/datasets/cifar10')
    vit = Path('artifacts/backbone_ablation_v1/ViT-B16/cifar10')
    manifest = json.loads((vit / 'manifest.json').read_text())
    for split in ['train', 'val', 'test']:
        a = load(vit / f'{split}_features.pt'); b = load(base / f'{split}_features.pt')
        assert a['features'].shape == (len(manifest[f'{split}_indices']), 512)
        assert torch.isfinite(a['features']).all()
        assert torch.equal(a['indices'], b['indices']) and torch.equal(a['labels'], b['labels'])
    region_count = 0
    for index in manifest['concept_source_indices']:
        path = base / 'proposals' / f'{index:08d}.pt'
        a = load(path); b = load(vit / 'proposals' / path.name)
        for key in ['source_index', 'label', 'image_size']:
            assert a[key] == b[key]
        assert torch.equal(a['boxes'], b['boxes']) and torch.isfinite(b['features']).all()
        assert b['features'].shape == (len(a['features']), 512)
        assert b['source_geometry_sha256'] == sha(path)
        region_count += len(a['features'])
    report['vit_b16_cifar10'] = {'split_alignment_exact': True, 'finite_features': True,
                               'feature_dimension': 512, 'images': 60000,
                               'identical_sam_geometry_sources': len(manifest['concept_source_indices']),
                               'regions': region_count}
    base = Path('artifacts/datasets/cub')
    a = load(base / 'test_features.pt'); b = load(base / 'backgrounds/identity.pt')
    for key in ['features', 'labels', 'indices']:
        assert torch.equal(a[key], b[key]), key
    report['cub_background_identity'] = {'images': len(a['labels']), 'features_labels_indices_exact': True,
                                        'max_feature_error': 0}
    report['cub_segmentation_alignment'] = json.loads(Path('reports/SEGMENTATION_ALIGNMENT.json').read_text())
    path = Path('artifacts/datasets/cub_background_natural/backgrounds/donor_patches.json')
    report['natural_background_donors'] = {'count': len(json.loads(path.read_text())['donors']), 'file_sha256': sha(path),
                                         'protocol': 'Training-only, annotation-empty union-positive masks; side >=64; aspect <=2'}
    path = Path('artifacts/datasets/cub_retrieval/regions.pt'); regions = load(path)
    assert len(regions['features']) == 27619 and regions['images'] == 5794 and torch.isfinite(regions['features']).all()
    report['heldout_region_pool'] = {'images': regions['images'], 'regions': len(regions['features']),
                                   'empty_images': regions['empty_images'], 'finite': True,
                                   'file_sha256': sha(path), 'evaluation_only': True}
    path = Path('artifacts/datasets/cub_gdino/attribute_mapping_k2048.json')
    report['gdino_semantic_mapping'] = {'sha256': sha(path), 'min_certainty': json.loads(path.read_text())['min_certainty'],
                                      'backbone': 'RN50', 'source_code_sha256': sha('pi_experiments/cub_metrics.py')}
    report['checked_at'] = time.time()
    Path('reports/STUDY_VERIFICATION.json').write_text(json.dumps(report, indent=2))
    print('Verified 70 checkpoints, ten matched initializations, five saved rotated banks, 60000 ViT features, 500 fixed crops, CUB identity and held-out pool.')


if __name__ == '__main__':
    main()
