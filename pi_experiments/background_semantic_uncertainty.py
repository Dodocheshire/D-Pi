"""Preserve native CUB attribute evidence on the existing background stress tests."""

import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from .analyze import collect, load_model, task_metrics
from .backgrounds import CASES
from .corruption_uncertainty import MODES
from .large_proposal_prewarm import file_sha256
from .noise_branch_evidence import CachedParameters
from .prepare import atomic_save
from .semantic_uncertainty import paired_attribute_bootstrap
from .statistics import hierarchical_paired_bootstrap
from .suite import Runner, pid_identity
from .visible_semantics import ready, write_json


VARIANTS = ['clean', 'fixed', 'mean_only', 'adapter_matched', 'pi']
GROUPS = [(bank, version) for bank in ['RN50_GDINO', 'RN50_SAM2']
          for version in ['natural_background', 'background']]
OUTPUT = Path('outputs/background_semantic_uncertainty_v1')
REPORTS = Path('reports/background_semantic_uncertainty_v1')


def locations(bank, version):
    directory = Path('artifacts/datasets') / ('cub_gdino' if bank == 'RN50_GDINO' else 'cub')
    features = Path('artifacts/datasets') / ('cub_background_natural' if version == 'natural_background' else 'cub')
    source = Path('outputs') / f'visible_{version}_v1' / bank
    return directory, features, source, OUTPUT / bank / version


def attribute_evidence(scores, thresholds, positive, known):
    predicted = scores >= thresholds
    pos, neg = known & positive, known & ~positive
    assert (pos.sum(0) > 0).all() and (neg.sum(0) > 0).all()
    balanced = .5 * ((predicted & pos).sum(0).double() / pos.sum(0)
                      + (~predicted & neg).sum(0).double() / neg.sum(0))
    auc = [roc_auc_score(positive[:, i][known[:, i]].numpy(), scores[:, i][known[:, i]].numpy())
           for i in range(scores.shape[1])]
    return predicted, balanced, auc


def cache_scores(bank, version, variant, seed):
    directory, feature_root, source_root, destination_root = locations(bank, version)
    source = source_root / f'{variant}_seed{seed}'
    destination = destination_root / source.name
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'semantic_predictions.pt'
    with (destination / 'cache.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return
        pack_path = directory / 'baseline_k2048_seed0.pt'
        pack = torch.load(pack_path, weights_only=True, map_location='cpu')
        annotations_path = directory / 'attributes.pt'
        annotations = torch.load(annotations_path, weights_only=True, map_location='cpu')
        report = json.loads((source / 'backgrounds.json').read_text())
        reference = json.loads((source / 'cub_concepts.json').read_text())
        saved = torch.load(source / 'background_predictions.pt', weights_only=True, map_location='cpu')
        assert report['device'] == 'cpu' and report['evaluation_split'] == 'test' and report['images'] == 5794
        assert report['original_attribute_report_sha256'] == file_sha256(source / 'cub_concepts.json')
        assert report['visible_mapping_sha256'] == reference['mapping_sha256']
        mapping_path = Path(reference['mapping_file'])
        mapping = json.loads(mapping_path.read_text())
        assert file_sha256(mapping_path) == reference['mapping_sha256']
        assert reference['source_checkpoint_sha256'] == file_sha256(source / 'model.pt')
        assert report['background_protocol_sha256'] == file_sha256(feature_root / 'backgrounds/protocol.json')
        assert reference['min_certainty'] == 3
        details = sorted(reference['details'], key=lambda r: r['attribute'])
        ids = torch.tensor([r['attribute'] for r in details])
        assert ids.numel() and torch.equal(ids, ids.unique(sorted=True))
        for row in details:
            assert row['prototypes'] == mapping['assignments'][str(row['attribute'])]
        thresholds = torch.tensor([r['threshold'] for r in details], dtype=torch.float32)
        positive = annotations['test']['attributes'][:, ids] > 0
        known = annotations['test']['certainty'][:, ids] >= 3
        assert torch.equal(saved['labels'], pack['test']['labels'])
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert torch.equal(pack['test']['indices'], torch.tensor(manifest['test_indices']))
        model, config = load_model(pack, source / 'model.pt', 'cpu')
        assert config['seed'] == seed and config['mode'] == MODES[variant]
        assert not config.get('validation_only') and not config.get('refit_epochs')
        for key, original in [('projection.centers', 'centers'), ('projection.mean', 'mean'),
                              ('projection.scale', 'scale'), ('head.weight', 'weight'), ('head.bias', 'bias')]:
            assert torch.equal(model.state_dict()[key], pack[original]), (source, key)
        if model.generator is not None:
            model.generator = CachedParameters(model.generator).eval()
        case_reports = {r['case']: r for r in report['cases']}
        assert len(case_reports) == len(report['cases']) == len(CASES) and set(case_reports) == set(CASES)
        metadata = {'bank': bank, 'version': version, 'variant': variant, 'seed': seed,
                    'mc_samples': 8, 'sampling_seed': seed + 6090, 'device': 'cpu', 'min_certainty': 3,
                    'checkpoint_sha256': file_sha256(source / 'model.pt'),
                    'background_report_sha256': file_sha256(source / 'backgrounds.json'),
                    'background_predictions_sha256': file_sha256(source / 'background_predictions.pt'),
                    'attribute_report_sha256': file_sha256(source / 'cub_concepts.json'),
                    'mapping_sha256': file_sha256(mapping_path),
                    'baseline_sha256': file_sha256(pack_path), 'annotations_sha256': file_sha256(annotations_path),
                    'protocol_sha256': report['background_protocol_sha256'],
                    'code_sha256': file_sha256(__file__),
                    'collect_code_sha256': file_sha256(Path(__file__).with_name('analyze.py')),
                    'cache_helper_sha256': file_sha256(Path(__file__).with_name('noise_branch_evidence.py'))}
        result = {'attribute_ids': ids, 'thresholds': thresholds, 'positive': positive, 'known': known,
                  'indices': pack['test']['indices'], 'labels': pack['test']['labels'], 'metadata': metadata,
                  'cases': {}}
        for case in CASES:
            values_path = feature_root / 'backgrounds' / f'{case}.pt'
            values = torch.load(values_path, weights_only=True, map_location='cpu')
            assert torch.equal(values['indices'], result['indices']) and torch.equal(values['labels'], result['labels'])
            assert values['protocol_sha256'] == metadata['protocol_sha256'] and torch.isfinite(values['features']).all()
            row = {'feature_sha256': file_sha256(values_path)}
            for mode in ['mc8', 'mean']:
                outputs = collect(model, values, samples=8 if mode == 'mc8' else 1,
                                  seed=seed + 6090, mean=mode == 'mean')
                # This cache preserves the exact native CPU inference and RNG schedule.
                assert torch.equal(outputs['probabilities'], saved[case][mode]), (source, case, mode)
                metrics = task_metrics(outputs['probabilities'], result['labels'])
                for key, value in metrics.items():
                    assert abs(value - case_reports[case][mode][key]) < 1e-7
                scores = torch.stack([outputs['concepts'][:, r['prototypes']].mean(-1) for r in details], dim=-1)
                predicted, balanced, auc = attribute_evidence(scores, thresholds, positive, known)
                semantic_metrics = {'balanced_accuracy': balanced.mean().item(), 'auroc': float(np.mean(auc))}
                if mode == 'mc8':
                    expected = case_reports[case]['semantic']
                    native = sorted(expected['details'], key=lambda r: r['attribute'])
                    assert [r['attribute'] for r in native] == ids.tolist()
                    assert [r['threshold'] for r in native] == [r['threshold'] for r in details]
                    assert abs(semantic_metrics['balanced_accuracy'] - expected['attribute_macro_balanced_accuracy']) < 1e-10
                    assert abs(semantic_metrics['auroc'] - expected['attribute_macro_auroc']) < 1e-10
                    for i, item in enumerate(native):
                        assert item['known_images'] == known[:, i].sum().item()
                        assert abs(balanced[i].item() - item['balanced_accuracy']) < 1e-10
                        assert abs(auc[i] - item['auroc']) < 1e-10
                row[mode] = {'scores': scores, 'predicted': predicted, 'semantic_metrics': semantic_metrics,
                             'task_metrics': metrics, 'probabilities_exact': True}
                del outputs
            result['cases'][case] = row
            print('CACHED', bank, version, source.name, case, row['mc8']['semantic_metrics'], flush=True)
            if model.generator is not None:
                model.generator.cache.clear()
        assert metadata['checkpoint_sha256'] == file_sha256(source / 'model.pt')
        assert metadata['background_report_sha256'] == file_sha256(source / 'backgrounds.json')
        atomic_save(result, path)


def summarize(bank, version):
    directory, features, source, destination = locations(bank, version)
    paths = {v: [destination / f'{v}_seed{s}/semantic_predictions.pt' for s in range(5)] for v in VARIANTS}
    if not all(p.exists() for rows in paths.values() for p in rows):
        return False
    result_path = REPORTS / bank / f'{version}.json'
    digests = {v: [file_sha256(p) for p in rows] for v, rows in paths.items()}
    if result_path.exists():
        assert json.loads(result_path.read_text())['prediction_sha256'] == digests
        return True
    caches = {v: [torch.load(p, weights_only=True, map_location='cpu') for p in rows] for v, rows in paths.items()}
    reference = caches['pi'][0]
    annotations = torch.load(directory / 'attributes.pt', weights_only=True, map_location='cpu')
    assert torch.equal(reference['positive'], annotations['test']['attributes'][:, reference['attribute_ids']] > 0)
    assert torch.equal(reference['known'], annotations['test']['certainty'][:, reference['attribute_ids']] >= 3)
    common_fields = ['attribute_ids', 'indices', 'labels', 'positive', 'known']
    frozen_sources = {}
    for variant, rows in caches.items():
        for seed, item in enumerate(rows):
            assert all(torch.equal(item[key], reference[key]) for key in common_fields)
            assert item['metadata']['mapping_sha256'] == reference['metadata']['mapping_sha256']
            assert item['metadata']['protocol_sha256'] == reference['metadata']['protocol_sha256']
            run = source / f'{variant}_seed{seed}'
            for filename, field in [('model.pt', 'checkpoint_sha256'), ('backgrounds.json', 'background_report_sha256'),
                                    ('background_predictions.pt', 'background_predictions_sha256'),
                                    ('cub_concepts.json', 'attribute_report_sha256')]:
                frozen_sources[str(run / filename)] = file_sha256(run / filename)
                assert frozen_sources[str(run / filename)] == item['metadata'][field]
    groups = {}
    for variant, rows in caches.items():
        groups[variant] = {case: {metric: {'mean': float(np.mean([r['cases'][case]['mc8']['semantic_metrics'][metric] for r in rows])),
                                          'std': float(np.std([r['cases'][case]['mc8']['semantic_metrics'][metric] for r in rows], ddof=1))}
                                  for metric in ['balanced_accuracy', 'auroc']} for case in CASES}
    probabilities = {v: [torch.load(source / f'{v}_seed{s}/background_predictions.pt', weights_only=True,
                                    map_location='cpu') for s in range(5)] for v in VARIANTS}
    comparisons = []
    for case in CASES:
        for control in VARIANTS[:-1] + ['same_checkpoint_mean']:
            v, mode = ('pi', 'mean') if control == 'same_checkpoint_mean' else (control, 'mc8')
            actual = paired_attribute_bootstrap(
                np.stack([r['cases'][case][mode]['predicted'].numpy() for r in caches[v]]),
                np.stack([r['cases'][case]['mc8']['predicted'].numpy() for r in caches['pi']]),
                reference['positive'].numpy(), reference['known'].numpy())
            task = hierarchical_paired_bootstrap(
                np.stack([r[case][mode].numpy() for r in probabilities[v]]),
                np.stack([r[case]['mc8'].numpy() for r in probabilities['pi']]), reference['labels'].numpy())
            comparisons.append({'case': case, 'control': control, 'semantic': actual, 'task': task})
            print('INTERVAL', bank, version, case, control, actual['balanced_accuracy_delta'], actual['ci95'], flush=True)
    result = {'bank': bank, 'version': version, 'groups': groups, 'comparisons': comparisons,
              'seeds': list(range(5)), 'case_order': CASES, 'test_images': 5794,
              'evaluated_attributes': len(reference['attribute_ids']), 'attribute_ids': reference['attribute_ids'].tolist(),
              'prediction_sha256': digests, 'source_sha256': frozen_sources,
              'mapping_sha256': reference['metadata']['mapping_sha256'],
              'background_protocol_sha256': reference['metadata']['protocol_sha256'],
              'code_sha256': file_sha256(__file__), 'checked_at': time.time(),
              'verified_models': 25, 'verified_prediction_cases': 125, 'verified_inference_arrays': 250,
              'native_attribute_truth_exact': True, 'per_attribute_report_metrics_reproduced': True,
              'original_cpu_probabilities_bitwise_exact': True,
              'limits': 'Original models, context images, training-only mapping and original-validation thresholds fixed. '
                        'Native certainty>=3 labels and the same attribute cohort used for every condition. '
                        'Twenty-five semantic and twenty-five task intervals are descriptive, unadjusted, paired '
                        'adaptation seeds/common source images and conditional on this fixed background protocol '
                        'and one MC8 realization. No test recalibration; mean inference does not retrain the model. '
                        'Majority foreground pixels are preserved, but compositing/resizing remain synthetic. '
                        'Banks have different attribute cohorts; do not compare absolute BA across banks. '
                        'Identity uses the background evaluation sampling seed, not the seed of the clean attribute report.'}
    write_json(result_path, result)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', choices=['RN50_GDINO', 'RN50_SAM2'])
    parser.add_argument('--version', choices=['natural_background', 'background'])
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--seed', type=int, choices=range(5))
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if not args.follow:
        assert args.bank and args.version
        if args.variant is not None:
            assert args.seed is not None
            cache_scores(args.bank, args.version, args.variant, args.seed)
        else:
            print('Complete comparison:', summarize(args.bank, args.version))
        return
    state_path = Path('artifacts/logs/visible_semantics_v1/status.json')
    assert args.wait_pid == json.loads(state_path.read_text())['manager_pid']
    identity = pid_identity(args.wait_pid)
    assert identity is not None, 'Follow mode requires an actually live upstream process'
    log_root = Path('artifacts/logs/background_semantic_uncertainty_v1')
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / 'manager.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Runner(log_root)
        try:
            while True:
                completed = []
                for bank, version in GROUPS:
                    _, _, sources, destination = locations(bank, version)
                    for variant in VARIANTS:
                        for seed in range(5):
                            source = sources / f'{variant}_seed{seed}'
                            cache = destination / source.name / 'semantic_predictions.pt'
                            if not ready(source / 'backgrounds.json') or cache.exists():
                                continue
                            runner.command(f'{bank}_{version}_{source.name}', [
                                'pi_experiments.background_semantic_uncertainty', '--bank', bank,
                                '--version', version, '--variant', variant, '--seed', str(seed)])
                    runner.state.update(phase=f'{bank}_{version}_paired_intervals');runner.save()
                    if summarize(bank, version):
                        completed.append(f'{bank}/{version}')
                runner.state.update(phase='waiting_for_visible_backgrounds', completed_groups=completed,
                                    checked_at=time.time());runner.save()
                if len(completed) == len(GROUPS):
                    break
                if pid_identity(args.wait_pid) != identity:
                    upstream = json.loads(state_path.read_text())
                    assert upstream['manager_pid'] == args.wait_pid and upstream['phase'] == 'complete'
                    raise RuntimeError('Upstream ended without all required background semantic caches')
                time.sleep(30)
            runner.state.update(phase='complete', finished_at=time.time());runner.save()
        except Exception:
            runner.state.update(phase='failed', failed_at=time.time());runner.save()
            raise


if __name__ == '__main__':
    main()
