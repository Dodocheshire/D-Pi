"""Independently align RN50 full-image caches with native source labels."""

import argparse
import json
from pathlib import Path
import time

import torch

from .large_proposal_prewarm import file_sha256
from .prepare import load_sources
from .visible_semantics import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['places365', 'imagenet'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    directory = Path('artifacts/datasets') / args.dataset
    manifest = json.loads((directory / 'manifest.json').read_text())
    preflight = json.loads((directory / 'manifest_preflight.json').read_text())
    warm = json.loads((directory / 'feature_prewarm_verification.json').read_text())
    manifest_digest = file_sha256(directory / 'manifest.json')
    assert manifest_digest == preflight['manifest_sha256'] == warm['manifest_sha256']
    assert manifest['backbone'] == warm['backbone'] == 'RN50'
    train, test = load_sources(args.dataset, manifest['root'], manifest.get('official_test_root'))
    assert list(train.classes) == list(test.classes) == manifest['classes']
    checks = {}
    for split in ['train', 'val', 'test']:
        path = directory / f'{split}_features.pt'
        digest = file_sha256(path)
        assert digest == warm['splits'][split]['file_sha256']
        pack = torch.load(path, weights_only=True, map_location='cpu')
        indices = torch.tensor(manifest[f'{split}_indices'])
        source = test if split == 'test' else train
        labels = torch.tensor(source.targets)[indices]
        assert torch.equal(pack['indices'], indices)
        assert torch.equal(pack['labels'], labels), 'Cached labels differ from native source labels'
        assert pack['features'].shape == (len(indices), 1024)
        assert torch.isfinite(pack['features']).all()
        assert torch.bincount(labels, minlength=preflight['classes']).tolist() == preflight['class_counts'][split]
        checks[split] = {'images': len(indices), 'dimensions': 1024,
                         'indices_exact': True, 'native_source_labels_exact': True,
                         'finite': True, 'feature_sha256': digest}
        del pack
    record = {'dataset': args.dataset, 'backbone': 'RN50',
              'images': sum(row['images'] for row in checks.values()), 'splits': checks,
              'manifest_sha256': manifest_digest,
              'prewarm_verification_sha256': file_sha256(directory / 'feature_prewarm_verification.json'),
              'verification_code_sha256': file_sha256(__file__), 'checked_at': time.time(),
              'scope': preflight['source_scope'] + '; source alignment of global caches only, not Pi performance.'}
    write_json(Path('reports/preparation_verification_v1') / f'RN50_{args.dataset}_native_source_audit.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
