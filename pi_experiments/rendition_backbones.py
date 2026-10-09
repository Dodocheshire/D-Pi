"""Cache ViT rendition features, then evaluate the completed ImageNet controls."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import clip
import torch
from torchvision.datasets import ImageFolder

from .prepare import atomic_save, encode_dataset
from .resources import gpu_lease
from .suite import Runner, pid_identity


def prepare(backbone, device):
    parent = Path('artifacts/datasets/imagenet_r')
    original = json.loads((parent / 'manifest.json').read_text())
    slug = backbone.replace('/', '')
    provenance = json.loads((Path('artifacts/models/clip') / f'{slug}.json').read_text())
    assert provenance['verified'] and provenance['backbone'] == backbone
    manifest = {**original, 'backbone': backbone, 'encoder_checkpoint_sha256': provenance['sha256'],
                'parent_manifest_sha256': hashlib.sha256((parent / 'manifest.json').read_bytes()).hexdigest()}
    directory = Path('artifacts/backbone_ablation_v1') / slug / 'imagenet_r'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    output = directory / 'test_features.pt'
    if output.exists():
        cached = torch.load(output, weights_only=True)
        assert cached['manifest_sha256'] == digest
        return directory
    source = ImageFolder(original['root'])
    assert source.classes == original['restricted_classes'] and len(source) == 30000
    assert [str(Path(p).relative_to(original['root'])) for p, _ in source.samples] == original['paths']
    labels = torch.tensor(source.targets)
    allowed = torch.tensor(original['restricted_class_indices'])
    source.samples = [(p, int(allowed[label])) for p, label in source.samples]
    source.targets = allowed[labels].tolist()
    with gpu_lease(f'{backbone} ImageNet-R features', device):
        encoder, transform = clip.load(backbone, device=device)
        encoder.eval().requires_grad_(False)
        values = encode_dataset(source, original['test_indices'], encoder, transform,
                                SimpleNamespace(device=device, encode_batch_size=128, workers=8))
        assert torch.equal(values['labels'], allowed[labels]) and torch.isfinite(values['features']).all()
        values.update(restricted_labels=labels, restricted_class_indices=allowed, manifest_sha256=digest)
        atomic_save(values, output)
        del encoder
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--upstream-status', default='artifacts/logs/backbone_large_v1/status.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(8)
    runner = Runner('artifacts/logs/rendition_backbones_v1')
    identity = pid_identity(args.wait_pid)
    while identity is not None and pid_identity(args.wait_pid) == identity:
        runner.state.update(phase='waiting_for_vit_imagenet_models', waiting_pid=args.wait_pid, checked_at=time.time())
        runner.save()
        time.sleep(10)
    upstream = json.loads(Path(args.upstream_status).read_text())
    assert upstream['manager_pid'] == args.wait_pid and upstream['phase'] == 'complete'
    for backbone in ['ViT-B/16', 'ViT-L/14']:
        slug = backbone.replace('/', '')
        runner.state.update(phase=f'{slug}_features'); runner.save()
        directory = prepare(backbone, args.device)
        runner.command(f'{slug}_evaluation', ['pi_experiments.rendition_eval',
                       '--wait-pid', str(args.wait_pid), '--upstream-status', args.upstream_status,
                       '--device', args.device, '--directory', str(directory),
                       '--pack', f'artifacts/backbone_ablation_v1/{slug}/imagenet/baseline_k4096_seed0.pt',
                       '--runs', f'outputs/backbone_v1/{slug}/imagenet',
                       '--output', f'outputs/rendition_backbones_v1/{slug}/imagenet',
                       '--reports', f'reports/rendition_backbones_v1/{slug}',
                       '--logs', f'artifacts/logs/rendition_backbones_v1/{slug}'])
    runner.state.update(phase='complete', finished_at=time.time()); runner.save()


if __name__ == '__main__':
    main()
