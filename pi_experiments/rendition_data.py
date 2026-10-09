"""Validate the official ImageNet-R archive and cache frozen evaluation features."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace

import clip
import torch
from torchvision.datasets import ImageFolder

from .prepare import atomic_save, encode_dataset
from .resources import gpu_lease
from .suite import ROOTS, Runner, pid_identity


SOURCE = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
DOCUMENTATION = 'https://github.com/hendrycks/imagenet-r'


def prepare_archive(archive, directory):
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        files = [m for m in members if m.isfile() and Path(m.name).suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        assert len(files) == 30000, f'Expected 30,000 images, found {len(files)}'
        classes = {Path(m.name).parts[1] for m in files}
        assert len(classes) == 200
        assert all(Path(m.name).parts[0] == 'imagenet-r' for m in members)
        images = directory / 'images'
        images.mkdir(parents=True, exist_ok=True)
        # The standard data filter rejects links escaping the destination.
        stream.extractall(images, filter='data')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    provenance = {'source': SOURCE, 'documentation': DOCUMENTATION, 'archive_sha256': digest,
                  'checksum_status': 'Computed locally; publisher does not provide an expected hash here.',
                  'images': len(files), 'classes': len(classes), 'evaluation_only': True}
    (directory / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2))
    return images / 'imagenet-r'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='artifacts/archives/imagenet-r.tar.partial')
    parser.add_argument('--directory', default='artifacts/datasets/imagenet_r')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backbone', default='RN50')
    args = parser.parse_args()
    runner = Runner('artifacts/logs/imagenet_r_data_v1')
    if args.wait_pid:
        identity = pid_identity(args.wait_pid)
        while identity is not None and pid_identity(args.wait_pid) == identity:
            runner.state.update(phase='waiting_for_archive_download', waiting_pid=args.wait_pid, checked_at=time.time())
            runner.save()
            time.sleep(10)
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    runner.state.update(phase='validating_archive')
    runner.save()
    archive = Path(args.archive)
    root = prepare_archive(archive, directory)
    source = ImageFolder(root)
    assert len(source) == 30000 and len(source.classes) == 200
    imagenet = ImageFolder(Path(ROOTS['imagenet']) / 'train')
    assert len(imagenet.classes) == 1000 and set(source.classes) <= set(imagenet.classes)
    allowed = [imagenet.class_to_idx[name] for name in source.classes]
    original_labels = source.targets.copy()
    source.samples = [(path, allowed[label]) for path, label in source.samples]
    source.targets = [allowed[label] for label in source.targets]
    manifest = {'dataset': 'imagenet_r', 'root': str(root.resolve()), 'backbone': args.backbone,
                'classes_1000': imagenet.classes, 'restricted_classes': source.classes,
                'restricted_class_indices': allowed, 'test_indices': list(range(len(source))),
                'paths': [str(Path(path).relative_to(root)) for path, _ in source.samples],
                'evaluation_only': True, 'protocol': 'Restrict each draw of ImageNet logits to these 200 classes before softmax.'}
    path = directory / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2))
    runner.state.update(phase='waiting_for_gpu')
    runner.save()
    with gpu_lease('ImageNet-R evaluation feature cache', args.device):
        output = directory / 'test_features.pt'
        if not output.exists():
            torch.set_num_threads(8)
            model, transform = clip.load(args.backbone, device=args.device)
            model.eval().requires_grad_(False)
            runner.state.update(phase='encoding_evaluation_images')
            runner.save()
            values = encode_dataset(source, manifest['test_indices'], model, transform,
                                    SimpleNamespace(device=args.device, encode_batch_size=128, workers=8))
            values['restricted_labels'] = torch.tensor(original_labels)
            values['restricted_class_indices'] = torch.tensor(allowed)
            values['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert torch.isfinite(values['features']).all()
            atomic_save(values, output)
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
