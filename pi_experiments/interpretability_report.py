"""Render completed full-test localization against measured perturbation energy."""

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .suite import pid_identity


def render(root,destination):
    destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in root.glob('*/*_seed*/localization.json'):
        report=json.loads(path.read_text())
        if report['debug_limit'] or report['evaluated_images']!=report['test_images']:continue
        analysis=json.loads((path.parent/'analysis.json').read_text())
        assert analysis['evaluation_split']=='test'
        variant,seed=path.parent.name.rsplit('_seed',1)
        rows.append({'dataset':path.parent.parent.name,'variant':variant,'seed':int(seed),
                     'noise_energy':analysis['inference']['mc8']['noise_energy'],
                     'grid_attribution':report['grid_all_concepts_mean']['relative_attribution'],
                     'grid_max_point':report['grid_all_concepts_mean']['max_point_accuracy'],
                     'foreground_attribution':report.get('foreground_all_candidate_mean',{}).get('foreground_attribution'),
                     'part_hit_rate':report['part_hit_rate_all_localizable'],
                     'grid_valid_coverage':report['valid_grid_concepts']/report['total_grid_concepts'],
                     'foreground_valid_concepts':report.get('foreground_valid_concepts'),
                     'part_localizable_concepts':report['part_localizable_concepts']})
    (destination/'localization_energy.json').write_text(json.dumps({
        'rows':rows,'noise_energy':'Expected coordinate-mean relative energy mu^2 + sigma^2, averaged over test images',
        'localization':'Full-test evaluation only; grid and foreground include zero maps as zero; parts use all localizable concepts.'},indent=2))
    for dataset in sorted({r['dataset'] for r in rows}):
        subset=[r for r in rows if r['dataset']==dataset]
        fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for variant in sorted({r['variant'] for r in subset}):
            values=[r for r in subset if r['variant']==variant]
            x=np.array([r['noise_energy'] for r in values])
            for ax,key,label in zip(axes,['grid_attribution','grid_max_point','foreground_attribution'],
                                     ['Target quadrant attribution','Target quadrant max-point credit','Bird foreground attribution']):
                y=np.array([r[key] for r in values if r[key] is not None])
                if len(y)!=len(x):continue
                ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1) if len(x)>1 else 0,
                            yerr=y.std(ddof=1) if len(y)>1 else 0,fmt='o',capsize=3,
                            label=f'{variant} (n={len(values)})')
                ax.set_ylabel(label)
                ax.set_xlabel('Expected relative perturbation energy')
                ax.set_xscale('symlog',linthresh=1e-5)
                ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        axes[0].axhline(.25,color='gray',linestyle=':',alpha=.7)
        axes[1].axhline(.25,color='gray',linestyle=':',alpha=.7)
        fig.suptitle(f'{dataset}: full-test localization and noise energy (mean ± SD)')
        for extension in ['svg','pdf','png']:fig.savefig(destination/f'{dataset}_localization_energy.{extension}',dpi=180)
        plt.close(fig)
    return len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--wait-pid',type=int,required=True)
    args=parser.parse_args()
    root,destination=Path(args.root),Path(args.output)
    identity=pid_identity(args.wait_pid)
    previous=None
    while True:
        signature=sorted((str(p),p.stat().st_mtime_ns) for p in root.glob('*/*_seed*/localization.json'))
        if signature!=previous:
            print('Completed full-test localization runs:',render(root,destination),flush=True);previous=signature
        if identity is None or pid_identity(args.wait_pid)!=identity:break
        time.sleep(30)


if __name__=='__main__':main()
