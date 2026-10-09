"""Advance the original CIFAR corruption caches before CUB training finishes."""

import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import PIL
import torch

from . import corruptions
from .large_proposal_prewarm import file_sha256
from .prepare import load_sources
from .suite import Runner
from .visible_semantics import write_json


def main():
    torch.set_num_threads(8)
    runner = Runner('artifacts/logs/corruption_prewarm_v1')
    entry_digest = file_sha256(corruptions.__file__)
    cases = [(kind, parameter) for kind, values in corruptions.SETTINGS.items() for parameter in values]
    runner.state['protocol'] = {
        'entry_code_sha256': entry_digest, 'datasets': ['cifar10', 'cifar100'],
        'settings': corruptions.SETTINGS, 'noise_seed': 909, 'cases_per_dataset': len(cases),
        'note': 'Original corruption.prepare stage and GPU lease; only preparation starts earlier. '
                'Existing CPU postprocess manager evaluates the original completed checkpoints.',
    }
    runner.save()
    for dataset in ['cifar10', 'cifar100']:
        directory = Path('artifacts/datasets') / dataset
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert manifest['dataset'] == dataset and manifest['backbone'] == 'RN50'
        assert file_sha256(corruptions.__file__) == entry_digest
        if any(not (directory / 'corruptions' / f'{kind}_{parameter}.pt').exists() for kind, parameter in cases):
            runner.command(dataset + '_corruption_features', [
                'pi_experiments.corruptions', '--stage', 'prepare', '--directory', str(directory),
            ])
        runner.state.update(phase=dataset + '_corruption_verification')
        runner.save()
        config = json.loads((directory / 'corruptions' / 'protocol.json').read_text())
        expected_config = {
            'protocol': 'custom native-resolution image corruptions, not CIFAR-C',
            'settings': corruptions.SETTINGS,
            'test_indices_sha256': hashlib.sha256(json.dumps(manifest['test_indices']).encode()).hexdigest(),
            'encoder_sha256': manifest['encoder_checkpoint_sha256'], 'backbone': 'RN50',
            'Pillow': PIL.__version__, 'numpy': np.__version__,
            'corruption_function_sha256': hashlib.sha256(inspect.getsource(corruptions.corrupt).encode()).hexdigest(),
        }
        assert config == expected_config
        checkpoint = Path.home() / '.cache/clip/RN50.pt'
        assert file_sha256(checkpoint) == manifest['encoder_checkpoint_sha256']
        _, source = load_sources(dataset, manifest['root'])
        indices = torch.tensor(manifest['test_indices'])
        labels = torch.tensor(source.targets)[indices]
        checks = {}
        for kind, parameter in cases:
            name = f'{kind}_{parameter}'
            path = directory / 'corruptions' / f'{name}.pt'
            values = torch.load(path, weights_only=True, map_location='cpu')
            assert values['features'].shape == (len(indices), 1024)
            assert torch.isfinite(values['features']).all()
            assert torch.equal(values['indices'], indices) and torch.equal(values['labels'], labels)
            metadata = values['metadata']
            assert metadata['kind'] == kind and metadata['parameter'] == parameter
            assert metadata['noise_seed'] == 909 and metadata['backbone'] == 'RN50'
            assert metadata['encoder_sha256'] == manifest['encoder_checkpoint_sha256']
            assert metadata['Pillow'] == PIL.__version__
            checks[name] = {'images': len(indices), 'dimensions': 1024, 'finite': True,
                            'indices_exact': True, 'native_labels_exact': True,
                            'metadata_exact': True, 'sha256': file_sha256(path)}
            del values
        assert file_sha256(corruptions.__file__) == entry_digest
        write_json(directory / 'corruptions' / 'prewarm_verification.json', {
            'dataset': dataset, 'cases': checks, 'total_encoded_images': len(indices) * len(cases),
            'manifest_sha256': file_sha256(directory / 'manifest.json'),
            'protocol_sha256': file_sha256(directory / 'corruptions' / 'protocol.json'),
            'entry_code_sha256': entry_digest, 'verification_code_sha256': file_sha256(__file__),
            'checked_at': time.time(),
            'scope': 'Original native-resolution corruption cache only; custom stress test, not CIFAR-C. '
                     'Model robustness conclusions require the separate completed five-seed evaluations.',
        })
        runner.state.update(phase=dataset + '_verified')
        runner.save()
        print('VERIFIED', dataset, len(cases), 'cases', flush=True)
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
