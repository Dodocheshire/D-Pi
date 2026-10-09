"""Run the original large-dataset SAM2 stage early, then audit every shard."""

import hashlib
import json
from pathlib import Path
import time

import torch

from . import prepare
from .suite import ROOTS, Runner
from .visible_semantics import write_json


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_record(record, index, label, source_size, cap=1024):
    """Reject mixed source identities, uncapped coordinates and invalid crops."""
    assert record['source_index'] == index, 'Incorrect proposal source index'
    assert record['label'] == label, 'Incorrect native source label'
    assert list(record['source_image_size']) == list(source_size), 'Incorrect source frame'
    width, height = source_size
    factor = min(1., cap / max(width, height))
    frame = [max(1, int(width * factor)), max(1, int(height * factor))]
    assert list(record['image_size']) == frame, 'Incorrect capped proposal frame'
    features, boxes = record['features'], record['boxes']
    assert features.ndim == 2 and features.shape[1] == 1024
    assert features.dtype == torch.float32 and torch.isfinite(features).all()
    assert boxes.shape == (len(features), 4) and torch.isfinite(boxes).all()
    assert torch.equal(boxes, boxes.round()), 'Crop bounds must be integer coordinates'
    if len(boxes):
        assert (boxes[:, :2] >= 0).all(), 'Negative crop origin'
        assert (boxes[:, 2:] > boxes[:, :2]).all(), 'Degenerate crop'
        assert (boxes[:, 2] <= frame[0]).all() and (boxes[:, 3] <= frame[1]).all(), 'Crop exceeds proposal frame'
    return len(features)


def verify_dataset(dataset, directory, manifest, manifest_digest, prepare_digest, checkpoint_digest):
    assert file_sha256(directory / 'manifest.json') == manifest_digest
    assert file_sha256(Path(prepare.__file__)) == prepare_digest
    expected_config = {
        'sam_checkpoint_sha256': checkpoint_digest, 'points_per_side': 64,
        'min_region_area': 500, 'resize_short_side': 0, 'backbone': 'RN50',
        'postprocessing': 'sam2_cuda_connected_components', 'max_long_side': 1024,
        'resampling': 'OpenCV INTER_LINEAR; floor dimensions',
    }
    assert json.loads((directory / 'proposal_config.json').read_text()) == expected_config
    indices = manifest['concept_source_indices']
    assert len(indices) == len(set(indices)) == len(manifest['classes']) * 50
    assert set(indices) <= set(manifest['train_indices'])
    assert {path.name for path in (directory / 'proposals').glob('*.pt')} == {
        f'{index:08d}.pt' for index in indices
    }, 'Missing or extra proposal sources'
    source, _ = prepare.load_sources(dataset, manifest['root'], manifest.get('official_test_root'))
    assert list(source.classes) == manifest['classes']
    records_digest = hashlib.sha256()
    regions, empty, resized = 0, 0, 0
    class_counts = [0] * len(manifest['classes'])
    for position, index in enumerate(indices):
        path = directory / 'proposals' / f'{index:08d}.pt'
        record = torch.load(path, weights_only=True, map_location='cpu')
        image, label = source[index]
        count = verify_record(record, index, label, image.size)
        class_counts[label] += 1
        regions += count
        empty += count == 0
        resized += max(image.size) > 1024
        image.close()
        records_digest.update(f'{path.name}:{file_sha256(path)}\n'.encode())
        if (position + 1) % 1000 == 0:
            print(f'VERIFIED {dataset} {position + 1}/{len(indices)}', flush=True)
    assert all(count == 50 for count in class_counts)
    write_json(directory / 'proposal_prewarm_verification.json', {
        'dataset': dataset, 'manifest_sha256': manifest_digest,
        'prepare_code_sha256': prepare_digest, 'proposal_config': expected_config,
        'source_images': len(indices), 'regions': regions, 'empty_source_images': empty,
        'resized_source_images': resized, 'concept_sources_training_only': True,
        'native_labels_exact': True, 'source_and_proposal_frames_exact': True,
        'crop_bounds_valid': True, 'finite_RN50_embeddings': True,
        'class_counts': class_counts, 'ordered_shard_hashes_sha256': records_digest.hexdigest(),
        'verification_code_sha256': file_sha256(Path(__file__)), 'checked_at': time.time(),
        'scope': 'Complete original training-only SAM2 proposal preparation; not Pi performance evidence.',
    })


def main():
    torch.set_num_threads(8)
    runner = Runner('artifacts/logs/large_proposal_prewarm_v1')
    prepare_digest = file_sha256(Path(prepare.__file__))
    checkpoint_digest = file_sha256('artifacts/models/sam2.1_hiera_large.pt')
    runner.state['protocol'] = {
        'prepare_code_sha256': prepare_digest,
        'sam_checkpoint_sha256': checkpoint_digest,
        'order': ['places365', 'imagenet'], 'proposal_chunk_size': 100,
        'points_per_side': 64, 'max_long_side': 1024,
        'note': 'Original prepare.proposals stage and stage lock; chunks share the original GPU lease. '
                'Only preparation starts earlier. Later study managers reuse the same atomic shards.',
    }
    runner.save()
    for dataset in ['places365', 'imagenet']:
        directory = Path('artifacts/datasets') / dataset
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest_digest = file_sha256(directory / 'manifest.json')
        preflight = json.loads((directory / 'manifest_preflight.json').read_text())
        assert manifest_digest == preflight['manifest_sha256']
        assert manifest['root'] == ROOTS[dataset] and manifest['backbone'] == 'RN50'
        assert manifest['split_seed'] == 310 and manifest['images_per_class'] == 50
        assert file_sha256(Path(prepare.__file__)) == prepare_digest
        command = ['pi_experiments.prepare', '--dataset', dataset, '--root', ROOTS[dataset],
                   '--output', str(directory), '--stage', 'proposals']
        if manifest.get('official_test_root'):
            command += ['--test-root', manifest['official_test_root']]
        runner.command(dataset + '_proposals', command)
        runner.state.update(phase=dataset + '_proposal_verification')
        runner.save()
        verify_dataset(dataset, directory, manifest, manifest_digest, prepare_digest, checkpoint_digest)
    runner.state.update(phase='complete', finished_at=time.time())
    runner.save()


if __name__ == '__main__':
    main()
