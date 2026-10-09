"""Render the mean/scale decomposition without combining validation and test data."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic',required=True)
    parser.add_argument('--output',required=True,help='Output filename prefix')
    args=parser.parse_args()
    report=json.loads(Path(args.diagnostic).read_text())
    rows=sorted([r for r in report['scenarios'] if r['mean_multiplier']==1],key=lambda r:r['scale_multiplier'])
    scales=[r['scale_multiplier'] for r in rows]
    differences=np.array([r['paired_vs_mean']['accuracy_delta']*100 for r in rows])
    bounds=np.array([r['paired_vs_mean']['ci95'] for r in rows])*100
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    axes[0].errorbar(scales,differences,yerr=[differences-bounds[:,0],bounds[:,1]-differences],
                     fmt='o-',color='#2563EB',capsize=4)
    axes[0].axhline(0,color='#64748B',linestyle='--',linewidth=1)
    axes[0].set_ylabel('Accuracy change vs mean (percentage points)')
    axes[0].set_title('Paired sample bootstrap: 95% interval')
    axes[1].plot(scales,[r['nll_delta_vs_mean'] for r in rows],'o-',color='#7C3AED')
    axes[1].axhline(0,color='#64748B',linestyle='--',linewidth=1)
    axes[1].set_ylabel('NLL change vs mean')
    axes[1].set_title('Fixed random draws; no refitting')
    axes[2].plot(scales,[r['concept_sampling_variance'] for r in rows],'o-',color='#EA580C')
    axes[2].set_ylabel('Standardized concept sampling variance')
    axes[2].set_title('Mean held constant')
    for ax in axes:
        ax.set_xlabel('Multiplier of learned standard deviation')
        ax.grid(alpha=0.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"Pi mean/scale decomposition | {report['evaluation_split']} split | one fixed checkpoint | MC{report['mc_samples']}")
    fig.tight_layout()
    prefix=Path(args.output)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['svg','pdf','png']:fig.savefig(prefix.with_suffix('.'+extension),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
