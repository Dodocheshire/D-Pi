"""Audit complete four-method exact-NEC predictions and paired uncertainty."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from .analyze import task_metrics
from .large_proposal_prewarm import file_sha256
from .matched_nec import common_targets, truncate_exact, THRESHOLD
from .proximal_sparsity import LAMBDAS
from .statistics import hierarchical_paired_bootstrap
from .visible_semantics import write_json


VARIANTS = ['clean','mean_only','adapter_matched','pi']
MODES = dict(zip(VARIANTS,['none','mean_only','adapter','mean_scale']))


def main():
    torch.set_num_threads(8)
    root = Path('outputs/sparse_capacity_v1/cifar10')
    destination = Path('reports/sparse_capacity_v1/cifar10')
    hashes = {}

    def read(path):
        hashes[str(path)] = file_sha256(path)
        return json.loads(path.read_text())

    def load(path):
        hashes[str(path)] = file_sha256(path)
        return torch.load(path,weights_only=True,map_location='cpu')

    pack = load(Path('artifacts/datasets/cifar10/baseline_k256_seed0.pt'))
    labels = pack['test']['labels']
    assert len(labels)==10000
    feature_scale = pack['train']['features'].square().mean(0).sqrt().clamp_min(1e-6)
    protocol = read(root/'protocol.json')
    exact = read(root/'exact_protocol.json')
    result = read(destination/'results.json')
    assert protocol['variants']==VARIANTS and protocol['adaptation_seeds']==list(range(5))
    assert protocol['max_iterations']==20000 and protocol['lambdas']==LAMBDAS
    targets = exact['targets']
    assert result['targets']==targets
    assert len(result['pi_minus_control'])==3*len(targets)
    assert len(result['pi_mc8_minus_mean'])==len(targets)
    predictions = {};selected_weights = [];rows = [];curves = {}
    for variant in VARIANTS:
        predictions[variant] = []
        curves[variant] = []
        for seed in range(5):
            run = root/f'{variant}_seed{seed}'
            heads = load(run/'heads.pt')
            grid = read(run/'head_grid.json')
            selection = read(run/'selection.json')
            assert heads['selection']==selection==grid['selection']
            assert selection['seed']==seed and selection['variant']==variant
            assert selection['config']['epochs']==20000 and selection['lambdas']==LAMBDAS
            index = selection['selected_index']
            assert index==int(np.argmin(selection['val_nll']))
            assert selection['selected_l1']==LAMBDAS[index]
            assert grid['selected']==grid['candidates'][index]
            assert grid==result['groups'][variant][seed]
            checkpoint_path = Path(selection['source_checkpoint'])
            checkpoint = load(checkpoint_path)
            assert hashes[str(checkpoint_path)]==selection['source_sha256']
            assert checkpoint['config']['seed']==seed and checkpoint['config']['mode']==MODES[variant]
            assert not checkpoint['config'].get('validation_only') and not checkpoint['config'].get('refit_epochs')
            state = checkpoint['state_dict']
            for key,original in [('projection.centers','centers'),('projection.mean','mean'),
                                 ('projection.scale','scale'),('head.weight','weight'),('head.bias','bias')]:
                assert torch.equal(state[key],pack[original]),(run,key)
            assert torch.equal(state['feature_scale'],feature_scale)
            assert hashes[str(run/'heads.pt')]==exact['head_sha256'][run.name]
            if variant in ['clean','pi']:
                assert all(file_sha256(run/file)==expected
                           for file,expected in protocol['reused_heads'][run.name].items())
            weight,bias = heads['weights'][index],heads['biases'][index]
            assert weight.shape==(10,256) and bias.shape==(10,)
            assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
            selected_weights.append(weight)
            probabilities = load(run/'exact_predictions.pt')
            metrics = read(run/'exact_metrics.json')
            assert metrics['variant']==variant and metrics['seed']==seed
            assert [p['nec_per_class'] for p in metrics['points']]==targets
            assert torch.equal(probabilities['labels'],labels)
            accuracy = []
            for i,k in enumerate(targets):
                truncated = truncate_exact(weight,k)
                assert (truncated.ne(0).sum(-1)==k).all()
                assert ((truncated.abs()>THRESHOLD).sum(-1)==k).all()
                for mode in ['mc8','mean']:
                    assert set(probabilities[mode])==set(map(str,targets))
                    p = probabilities[mode][str(k)]
                    assert p.shape==(10000,10) and torch.isfinite(p).all() and (p>=0).all()
                    assert torch.allclose(p.sum(-1),torch.ones(10000),atol=1e-5)
                    actual = task_metrics(p,labels)
                    for key,value in actual.items():
                        assert abs(value-metrics['points'][i][mode][key])<1e-7,(run,k,mode,key)
                accuracy.append(metrics['points'][i]['mc8']['accuracy'])
            curves[variant].append(accuracy)
            predictions[variant].append(probabilities)
            rows.append({'variant':variant,'seed':seed,'frozen_source_invariants_exact':True,
                         'validation_selection_verified':True,'strict_nonzeros_verified_in_every_class':True,
                         'prediction_metrics_reproduced':True,'budget_points':len(targets)})
        print('VERIFIED',variant,5*len(targets),'budget points',flush=True)
    assert common_targets(selected_weights)==targets
    intervals = []
    for row in result['pi_minus_control']+result['pi_mc8_minus_mean']:
        k = str(row['nec_per_class'])
        control = row.get('control','same_checkpoint_mean')
        a = predictions['pi'] if control=='same_checkpoint_mean' else predictions[control]
        mode = 'mean' if control=='same_checkpoint_mean' else 'mc8'
        actual = hierarchical_paired_bootstrap(np.stack([p[mode][k].numpy() for p in a]),
                                               np.stack([p['mc8'][k].numpy() for p in predictions['pi']]),
                                               labels.numpy(),seed=row['seed'],draws=row['draws'])
        assert abs(actual['accuracy_delta']-row['accuracy_delta'])<1e-12
        assert np.allclose(actual['ci95'],row['ci95'],rtol=0,atol=1e-12)
        intervals.append({'control':control,'nec_per_class':int(k),'paired_interval_reproduced':True})
    curve_summary = {v:{'mean':np.mean(values,0).tolist(),'std':np.std(values,0,ddof=1).tolist()}
                     for v,values in curves.items()}
    audit = {'targets':targets,'checkpoints':rows,'verified_models':len(rows),
             'verified_exact_budget_points':len(rows)*len(targets),
             'verified_probability_arrays':2*len(rows)*len(targets),
             'reproduced_paired_intervals':intervals,'curves':curve_summary,
             'source_sha256':hashes,'code_sha256':file_sha256(__file__),'checked_at':time.time(),
             'limits':'Audits frozen source models, validation-selected heads, exact per-class support, saved '
                      'probabilities and all paired intervals. Does not re-run GPU sampling or optimize each NEC; '
                      'biases remain validation-selected. Intervals are descriptive, unadjusted and conditional '
                      'on one fixed bank, selected generators/heads and MC8 realization.'}
    write_json(destination/'exact_capacity_verification.json',audit)
    print(json.dumps({k:audit[k] for k in ['verified_models','verified_exact_budget_points','verified_probability_arrays']}
                     | {'paired_intervals':len(intervals),'targets':targets},indent=2))


if __name__=='__main__':
    main()
