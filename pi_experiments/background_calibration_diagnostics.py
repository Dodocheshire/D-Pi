"""Describe fixed-threshold behavior without calibrating on changed test images."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from .large_proposal_prewarm import file_sha256
from .visible_semantics import write_json


def main():
    torch.set_num_threads(8)
    completed = []
    for path in sorted(Path('reports/background_semantic_uncertainty_v1').glob('*/*.json')):
        evidence = json.loads(path.read_text())
        if 'prediction_sha256' not in evidence:
            continue
        bank, version = evidence['bank'], evidence['version']
        root = Path('outputs/background_semantic_uncertainty_v1') / bank / version
        groups = {}
        for variant, digests in evidence['prediction_sha256'].items():
            records = []
            for seed, digest in enumerate(digests):
                cache_path = root / f'{variant}_seed{seed}/semantic_predictions.pt'
                assert file_sha256(cache_path) == digest
                cache = torch.load(cache_path, weights_only=True, map_location='cpu')
                positive, known = cache['positive'], cache['known']
                pos, neg = positive & known, ~positive & known
                row = {}
                for case in evidence['case_order']:
                    values = cache['cases'][case]['mc8']
                    predicted = values['predicted']
                    assert torch.equal(predicted, values['scores'] >= cache['thresholds'])
                    tpr = (predicted & pos).sum(0).double() / pos.sum(0)
                    tnr = (~predicted & neg).sum(0).double() / neg.sum(0)
                    rate = (predicted & known).sum(0).double() / known.sum(0)
                    metrics = values['semantic_metrics']
                    assert abs(.5 * (tpr + tnr).mean().item() - metrics['balanced_accuracy']) < 1e-10
                    row[case] = {'macro_positive_recall': tpr.mean().item(),
                                 'macro_negative_recall': tnr.mean().item(),
                                 'macro_predicted_positive_fraction': rate.mean().item(),
                                 'attributes_below_one_percent_positive_predictions': (rate < .01).double().mean().item(),
                                 **metrics}
                records.append(row)
            groups[variant] = {case: {metric: {'mean': float(np.mean([r[case][metric] for r in records])),
                                               'std': float(np.std([r[case][metric] for r in records], ddof=1))}
                                     for metric in records[0][case]} for case in evidence['case_order']}
        result = {'bank': bank, 'version': version, 'groups': groups, 'seeds': list(range(5)),
                  'test_images': evidence['test_images'], 'attributes': evidence['evaluated_attributes'],
                  'source_evidence_sha256': file_sha256(path),
                  'prediction_sha256': evidence['prediction_sha256'],
                  'code_sha256': file_sha256(__file__), 'checked_at': time.time(),
                  'limits': 'Descriptive decomposition of the saved MC8 predictions, with unchanged original-validation '
                            'thresholds and native known-label masks. BA equals half the macro positive and negative recalls. '
                            'AUROC describes ranking, whereas fixed-threshold recall describes decisions; this diagnostic '
                            'does not identify a causal mechanism or fit any test threshold. Means/SD over the same five '
                            'adaptation seeds are not confidence intervals. Different attributes/banks are not pooled.'}
        output = Path('reports/background_calibration_diagnostics_v1') / bank / f'{version}.json'
        write_json(output, result);completed.append(f'{bank}/{version}')
        print(bank, version, 'Pi identity/gray',
              {case: {key: round(value['mean'] * 100, 5) for key, value in groups['pi'][case].items()}
               for case in ['identity', 'gray']}, flush=True)
    assert completed, 'No completed native background attribute group'
    print('Completed diagnostics:', completed)


if __name__ == '__main__':
    main()
