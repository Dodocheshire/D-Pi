"""Export finished internal evaluations through the shared statistical report."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from .summarize import summarize
from .suite import pid_identity


def export(root, output):
    count = 0
    for path in Path(root).glob('*/*_seed*/metrics.json'):
        record = json.loads(path.read_text())
        if record['evaluation_split'] != 'test':
            continue
        if not all((path.parent / name).exists() for name in ['model.pt', 'predictions.pt']):
            continue
        # Global analysis reports MSE of mean concept response. Preserve both
        # internal mean and draw MSE, and alias only that same mean quantity.
        for metrics in record['inference'].values():
            metrics['standardized_concept_mse'] = metrics['standardized_concept_mean_mse']
        record['source_metrics_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        record['injection'] = 'RN50 layer4 channels before frozen attention pooling'
        target = path.parent / 'analysis.json'
        text = json.dumps(record, indent=2, allow_nan=False)
        if not target.exists() or target.read_text() != text:
            target.write_text(text)
        count += 1
    summarize(root, output)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='outputs/internal_v1')
    parser.add_argument('--output', default='reports/internal_v1')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    identity = pid_identity(args.wait_pid) if args.wait_pid else None
    last_count = -1
    while True:
        count = len(list(Path(args.root).glob('*/*_seed*/metrics.json')))
        if count != last_count:
            print('Finished internal evaluations:', export(args.root, args.output), flush=True)
            last_count = count
        if identity is None or pid_identity(args.wait_pid) != identity:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
