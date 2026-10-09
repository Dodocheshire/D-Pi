"""Audit native annotation grounding of saved regional retrieval selections.

This checks selected regions and their metrics, not their model-score ranking.
The original inference/rankings remain untouched and are not recomputed here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .cub_metrics import PARTS
from .large_proposal_prewarm import file_sha256
from .prepare import load_sources
from .retrieval_regions import visible_window
from .visible_semantics import write_json


VARIANTS = ['clean', 'fixed', 'mean_only', 'adapter_matched', 'pi']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', choices=['RN50_GDINO', 'RN50_SAM2'], default='RN50_GDINO')
    args = parser.parse_args();torch.set_num_threads(8)
    directory = Path('artifacts/datasets') / ('cub_gdino' if args.bank == 'RN50_GDINO' else 'cub')
    root = Path('outputs/visible_semantics_v1') / args.bank
    raw_root = Path('outputs/gdino_v1/cub' if args.bank == 'RN50_GDINO' else 'outputs/study_v1/cub')
    pool_path = Path('artifacts/datasets/cub_retrieval/regions.pt')
    protocol_path = pool_path.parent / 'protocol.json'
    hashes, snapshots = {}, {}

    def read(path):
        b = path.read_bytes();sha = hashlib.sha256(b).hexdigest()
        hashes[str(path)] = sha;snapshots[(path.stem, sha)] = b
        return json.loads(b)

    def load(path):
        hashes[str(path)] = file_sha256(path)
        return torch.load(path, weights_only=True, map_location='cpu')

    manifest_path = directory / 'manifest.json'
    manifest = read(manifest_path)
    pack = load(directory / 'baseline_k2048_seed0.pt')
    annotations = load(directory / 'attributes.pt')
    regions = load(pool_path);protocol = read(protocol_path)
    mapping_path = directory / 'attribute_mapping_visible_k2048.json'
    mapping = read(mapping_path)
    assert mapping['min_certainty'] == 3
    original_mapping_path = directory / 'attribute_mapping_k2048.json'
    original_mapping = read(original_mapping_path)
    assert mapping['original_mapping_sha256'] == hashes[str(original_mapping_path)]
    assert regions['protocol_sha256'] == hashes[str(protocol_path)]
    assert protocol['evaluation_only'] and protocol['encoder_backbone'] == 'RN50'
    assert regions['images'] == len(manifest['test_indices']) == len(pack['test']['indices']) == 5794
    assert torch.equal(pack['test']['indices'], torch.tensor(manifest['test_indices']))
    assert torch.isfinite(regions['features']).all() and regions['features'].shape == (27619, 1024)
    assert (regions['image_positions'] >= 0).all() and (regions['image_positions'] < 5794).all()
    _, native = load_sources('cub', manifest['root'])
    offset, empty = 0, 0
    for row, source_index in enumerate(manifest['test_indices']):
        shard = load(pool_path.parent / 'shards' / f'{row:08d}.pt')
        count = len(shard['features']);subset = slice(offset, offset + count)
        assert shard['image_position'] == row and shard['source_index'] == source_index
        assert shard['label'] == native.samples[source_index][1] == pack['test']['labels'][row].item()
        assert torch.equal(regions['features'][subset], shard['features'])
        assert torch.equal(regions['boxes'][subset], shard['boxes'])
        assert torch.equal(regions['visible_windows'][subset], shard['visible_windows'])
        assert torch.equal(regions['image_positions'][subset], torch.full((count,), row, dtype=torch.long))
        expected = torch.tensor([visible_window(box) for box in shard['boxes'].tolist()]).reshape(-1, 4)
        assert torch.equal(expected, shard['visible_windows'])
        empty += count == 0;offset += count
    assert offset == len(regions['features']) and empty == regions['empty_images']
    reverse = {c: int(a) for a, concepts in mapping['assignments'].items() for c in concepts}
    assert len(reverse) == sum(map(len, mapping['assignments'].values()))
    # Independently derive known/positive masks from native attributes and crop-visible parts.
    truth = {}
    positions, boxes = regions['image_positions'], regions['visible_windows']
    for attribute in sorted(set(reverse.values())):
        name = annotations['attribute_names'][attribute]
        kind = name.split('::')[0].removeprefix('has_').split('_')[0]
        parts = annotations['test']['parts'][positions][:, PARTS.get(kind, [])]
        within = ((parts[..., 0] >= boxes[:, None, 0]) & (parts[..., 0] < boxes[:, None, 2])
                  & (parts[..., 1] >= boxes[:, None, 1]) & (parts[..., 1] < boxes[:, None, 3])
                  & (parts[..., 2] > 0)).any(-1)
        known = annotations['test']['certainty'][positions, attribute] >= 3
        positive = (annotations['test']['attributes'][positions, attribute] > 0) & within
        truth[attribute] = (known, positive)
    groups, records = {}, []
    for variant in VARIANTS:
        values = []
        for seed in range(5):
            run = root / f'{variant}_seed{seed}'
            report_path = run / 'region_retrieval.json';report = read(report_path)
            raw_path = raw_root / run.name / 'region_retrieval.json';raw = read(raw_path)
            checkpoint_path = run / 'model.pt';state = load(checkpoint_path)
            assert state['config']['seed'] == seed and not state['config'].get('validation_only')
            for key, original in [('projection.centers', 'centers'), ('projection.mean', 'mean'),
                                  ('projection.scale', 'scale'), ('head.weight', 'weight'), ('head.bias', 'bias')]:
                assert torch.equal(state['state_dict'][key], pack[original])
            assert report['checkpoint_sha256'] == hashes[str(checkpoint_path)]
            assert report['source_report_sha256'] == hashes[str(raw_path)]
            assert raw['mapping_sha256'] == hashes[str(original_mapping_path)]
            assert report['mapping_sha256'] == hashes[str(mapping_path)]
            assert report['pool_protocol_sha256'] == hashes[str(protocol_path)]
            assert report['evaluation_split'] == 'test' and report['min_certainty'] == 3
            assert report['region_images'] == 5794 and report['regions'] == offset and report['empty_region_images'] == empty
            selected = [q for q in raw['queries'] if q['concept'] in reverse]
            assert selected == report['queries'], 'Filtering altered the saved selection or scores'
            assert len(selected) == report['evaluated_queries'] == len({q['concept'] for q in selected})
            assert report['mapped_queries'] == len(reverse)
            for q in selected:
                attribute = reverse[q['concept']]
                assert q['attribute'] == attribute and q['name'] == annotations['attribute_names'][attribute]
                known, positive = truth[attribute]
                top = torch.tensor(q['top_region_indices'])
                distinct = torch.tensor(q['top_distinct_image_region_indices'])
                assert len(top) == len(top.unique()) == 5 and len(distinct) == len(distinct.unique()) == 5
                assert ((top >= 0) & (top < offset)).all() and ((distinct >= 0) & (distinct < offset)).all()
                assert known[top].all() and known[distinct].all()
                assert len(positions[distinct].unique()) == 5
                assert q['eligible_regions'] == known.sum().item()
                assert q['positive_regions'] == positive[known].sum().item()
                actual = {'precision_at_5': positive[top].float().mean().item(),
                          'precision_at_5_distinct_images': positive[distinct].float().mean().item(),
                          'chance_precision': positive[known].float().mean().item()}
                for key, value in actual.items():
                    assert abs(q[key] - value) < 1e-12, (run, q['concept'], key)
            metrics = {key: float(np.mean([q[field] for q in selected])) for key, field in [
                ('precision_at_5_macro', 'precision_at_5'),
                ('precision_at_5_distinct_images_macro', 'precision_at_5_distinct_images'),
                ('chance_precision_macro', 'chance_precision')]}
            metrics['excess_precision_over_chance'] = float(np.mean([
                q['precision_at_5'] - q['chance_precision'] for q in selected]))
            for key, value in metrics.items():
                assert abs(value - report[key]) < 1e-12
            values.append(metrics)
            records.append({'variant': variant, 'seed': seed, 'queries_verified': len(selected),
                            'native_crop_grounding_exact': True, 'ranking_recomputed': False})
            print('VERIFIED native retrieval selections', args.bank, run.name, len(selected), flush=True)
        groups[variant] = {key: {'mean': float(np.mean([r[key] for r in values])),
                                'std': float(np.std([r[key] for r in values], ddof=1))} for key in values[0]}
    snapshot = Path('artifacts/report_assets/retrieval_evidence_source_snapshots');snapshot.mkdir(parents=True, exist_ok=True)
    for (stem, sha), raw_bytes in snapshots.items():
        (snapshot / f'{stem}_{sha}.json').write_bytes(raw_bytes)
    result = {'bank': args.bank, 'groups': groups, 'verified_checkpoints': records,
              'native_test_images_verified': 5794, 'regions_verified': offset, 'empty_region_images': empty,
              'retained_prototype_queries': len(reverse), 'verified_query_selections': sum(r['queries_verified'] for r in records),
              'source_sha256': hashes, 'source_snapshots_directory': str(snapshot.resolve()),
              'code_sha256': file_sha256(__file__), 'checked_at': time.time(),
              'limits': 'Native held-out image labels, merged region-shard identity and exact CLIP crop windows verified. '
                        'Recomputes annotation grounding and macro metrics of all saved top-five region selections. '
                        'Does not recompute model responses or verify model-score ranking. Annotation-supported '
                        'proxy rather than manual semantic judgments; fixed GDINO test-pool geometry can favor GDINO. '
                        'Compare paired within-bank results, not absolute performance across different attribute/query cohorts.'}
    write_json(Path('reports/retrieval_evidence_verification_v1') / f'{args.bank}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k in ['bank', 'native_test_images_verified', 'regions_verified',
                     'retained_prototype_queries', 'verified_query_selections']}, indent=2))


if __name__ == '__main__':
    main()
