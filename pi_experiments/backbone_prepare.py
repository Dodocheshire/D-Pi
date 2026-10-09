"""Re-encode identical SAM2 geometry with ViT CLIP backbones and fixed splits."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import clip
import torch

from .prepare import atomic_save, features_stage, load_sources, resize_sam_image
from .resources import gpu_lease
from .suite import pid_identity


def proposal_image(image, record, config):
    if 'source_image_size' in record:
        assert record['source_image_size'] == list(image.size), 'Original source coordinates changed'
        image = resize_sam_image(image, config['max_long_side'])
    assert record['image_size'] == list(image.size), 'Cannot reuse boxes in different image coordinates'
    return image


def reencode_crops(args, manifest, source_directory, destination):
    source, _ = load_sources(args.dataset, manifest['root'], manifest.get('official_test_root'))
    encoder, transform = clip.load(args.backbone, device=args.device)
    encoder.eval().requires_grad_(False)
    target = destination / 'proposals'
    target.mkdir(exist_ok=True)
    config = json.loads((source_directory / 'proposal_config.json').read_text())
    for number, index in enumerate(manifest['concept_source_indices']):
        path = target / f'{index:08d}.pt'
        if path.exists():
            continue
        record = torch.load(source_directory / 'proposals' / path.name, weights_only=True)
        image, label = source[index]
        assert record['source_index'] == index and record['label'] == label
        image = proposal_image(image, record, config)
        crops = [transform(image.crop(tuple(box))) for box in record['boxes'].tolist()]
        with torch.no_grad():
            values = torch.cat([encoder.encode_image(torch.stack(crops[i:i + 64]).to(args.device)).float().cpu()
                                for i in range(0, len(crops), 64)]) if crops else torch.empty(0, encoder.visual.output_dim)
        assert torch.isfinite(values).all()
        atomic_save({**record, 'features': values,
                     'source_geometry_sha256': hashlib.sha256((source_directory / 'proposals' / path.name).read_bytes()).hexdigest()}, path)
        if (number + 1) % 100 == 0:
            print('Re-encoded fixed SAM2 crops', number + 1, '/', len(manifest['concept_source_indices']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'cub', 'places365', 'imagenet'], required=True)
    parser.add_argument('--backbone', choices=['ViT-B/16', 'ViT-L/14'], required=True)
    parser.add_argument('--stage', choices=['features', 'crops'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--encode-batch-size', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(8)
    parent = Path('artifacts/datasets') / args.dataset
    source = json.loads((parent / 'manifest.json').read_text())
    provenance = json.loads((Path('artifacts/models/clip') / (args.backbone.replace('/', '') + '.json')).read_text())
    assert provenance['verified'] and provenance['backbone'] == args.backbone
    manifest = {**source, 'backbone': args.backbone, 'encoder_checkpoint_sha256': provenance['sha256'],
                'proposal_geometry_parent_manifest_sha256': hashlib.sha256((parent / 'manifest.json').read_bytes()).hexdigest(),
                'proposal_geometry': 'Original SAM2 boxes and source indices, re-encoded with the selected backbone'}
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2))
    if (parent / 'attributes.pt').exists():
        target = destination / 'attributes.pt'
        if not target.exists():
            target.symlink_to(os.path.relpath((parent / 'attributes.pt').resolve(), destination.resolve()))
        else:
            assert target.resolve() == (parent / 'attributes.pt').resolve()
    args.root, args.test_root = manifest['root'], manifest.get('official_test_root')
    if args.stage == 'features':
        with gpu_lease(f'{args.dataset} {args.backbone} global features', args.device):
            features_stage(args, destination, manifest)
        return
    required = [parent / 'proposals' / f'{index:08d}.pt' for index in manifest['concept_source_indices']]
    identity = pid_identity(args.wait_pid) if args.wait_pid else None
    while not all(path.exists() for path in required):
        if identity is None or pid_identity(args.wait_pid) != identity:
            raise RuntimeError('Original proposal manager is terminal before required geometry is complete')
        time.sleep(10)
    config = json.loads((parent / 'proposal_config.json').read_text())
    config.update(backbone=args.backbone,
                  geometry_source_config_sha256=hashlib.sha256((parent / 'proposal_config.json').read_bytes()).hexdigest(),
                  reuse='Identical SAM2 geometry; only region/image encoder changes')
    path = destination / 'proposal_config.json'
    if path.exists():
        assert json.loads(path.read_text()) == config
    else:
        path.write_text(json.dumps(config, indent=2))
    with gpu_lease(f'{args.dataset} {args.backbone} region features', args.device):
        reencode_crops(args, manifest, parent, destination)


if __name__ == '__main__':
    main()
