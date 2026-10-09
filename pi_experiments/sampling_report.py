"""Paired uncertainty for completed nested-budget evaluations, without selection."""

import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .sampling_budget import BUDGETS
from .sampling_study import specifications
from .statistics import hierarchical_paired_bootstrap


def main():
    torch.set_num_threads(8)
    groups = []
    for name, root, _ in specifications():
        runs = [root/f'pi_seed{seed}' for seed in range(5)]
        if not all(all((run/file).exists() for file in ['sampling_budget.json',
                      'sampling_budget_predictions.pt', 'model.pt']) for run in runs):
            continue
        records = [json.loads((run/'sampling_budget.json').read_text()) for run in runs]
        packs = [torch.load(run/'sampling_budget_predictions.pt', weights_only=True,
                           map_location='cpu') for run in runs]
        labels = packs[0]['labels']
        for run, record, pack in zip(runs, records, packs):
            assert record['evaluation_split'] == 'test' and record['budgets'] == BUDGETS
            assert record['checkpoint_sha256'] == hashlib.sha256((run/'model.pt').read_bytes()).hexdigest()
            assert torch.equal(labels, pack['labels'])
            for key in ['mean', *map(str, BUDGETS)]:
                assert torch.isfinite(pack[key]).all() and (pack[key] >= 0).all()
                assert torch.allclose(pack[key].sum(-1), torch.ones(len(labels)), atol=1e-5)
        assert len({r['sampling_seed'] for r in records}) == 1
        comparisons = [('mean', str(n)) for n in BUDGETS]
        comparisons += [(str(a), str(b)) for i, a in enumerate(BUDGETS) for b in BUDGETS[i+1:]]
        rows = []
        for control, adapted in comparisons:
            rows.append({'control':control, 'adapted':adapted,
                         **hierarchical_paired_bootstrap(
                             np.stack([p[control].numpy() for p in packs]),
                             np.stack([p[adapted].numpy() for p in packs]), labels.numpy())})
        groups.append({'case':name,'seeds':list(range(5)),'sampling_seed':records[0]['sampling_seed'],
                       'test_images':len(labels),
                       'checkpoint_sha256':[r['checkpoint_sha256'] for r in records],
                       'predictions_sha256':[hashlib.sha256((run/'sampling_budget_predictions.pt').read_bytes()).hexdigest()
                                             for run in runs],
                       'comparisons':rows})
    destination = Path('reports/sampling_budget_v1/paired_results.json')
    record = {'checked_at':time.time(),'groups':groups,
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits':'All preset contrasts are descriptive, with unadjusted 95% intervals. '
                       'No test-based budget selection. Only complete five-seed groups are included. '
                       'Conditional on fixed prototypes, initial classifier, selected checkpoints and one sampling realization.'}
    temporary = destination.with_suffix('.partial.json')
    temporary.write_text(json.dumps(record,indent=2)); temporary.replace(destination)
    if groups:
        contrasts = [(group['case'], row) for group in groups
                     for row in group['comparisons'] if row['control'] == 'mean']
        fig, axis = plt.subplots(figsize=(8, max(3.5, .36*len(contrasts))), constrained_layout=True)
        for index, (_, row) in enumerate(contrasts):
            point = row['accuracy_delta']*100
            lo, hi = [value*100 for value in row['ci95']]
            axis.errorbar(point, index, xerr=np.array([[point-lo],[hi-point]]),
                          fmt='o', color='tab:blue', capsize=3)
        axis.set_yticks(range(len(contrasts)), [f'{name}: MC{row["adapted"]}' for name,row in contrasts])
        axis.invert_yaxis(); axis.axvline(0,color='gray',linestyle='--')
        axis.set_xlabel('Accuracy difference from mean inference (percentage points)')
        axis.set_title('Paired adaptation-seed and image bootstrap: 95% intervals')
        axis.grid(axis='x',alpha=.2)
        for extension in ['svg','pdf','png']:
            fig.savefig(destination.parent/f'paired_accuracy_vs_mean.{extension}',dpi=160)
        plt.close(fig)
    print(json.dumps({'completed_groups':len(groups),'comparisons':sum(len(g['comparisons']) for g in groups)},indent=2))


if __name__ == '__main__':
    main()
