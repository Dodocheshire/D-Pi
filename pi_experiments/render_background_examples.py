"""Show real paired background changes for representative held-out CUB images."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .backgrounds import BackgroundSource,CASES,mask_for
from .natural_backgrounds import NaturalSource
from .prepare import load_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural',action='store_true')
    args=parser.parse_args()
    directory=Path('artifacts/datasets/cub')
    manifest=json.loads((directory/'manifest.json').read_text())
    train,test=load_sources('cub',manifest['root'])
    sources=[BackgroundSource(test,train,manifest['train_indices'],'artifacts/annotations/segmentations',case) for case in CASES]
    if args.natural:
        donors=json.loads(Path('artifacts/datasets/cub_background_natural/backgrounds/donor_patches.json').read_text())['donors']
        sources=[NaturalSource(test,train,donors,'artifacts/annotations/segmentations',case)
                 if case.startswith('train_background') else source for case,source in zip(CASES,sources)]
    selected=[manifest['test_indices'][i] for i in np.linspace(0,len(manifest['test_indices'])-1,4,dtype=int)]
    fig,axes=plt.subplots(len(selected),len(CASES),figsize=(12,9),constrained_layout=True)
    metadata=[]
    for row,index in enumerate(selected):
        original=np.asarray(test[index][0].convert('RGB'))
        foreground=mask_for(test.samples[index][0],'artifacts/annotations/segmentations')
        record={'test_index':index,'source_path':test.samples[index][0],'foreground_pixels':int(foreground.sum()),'cases':[]}
        for col,(case,source) in enumerate(zip(CASES,sources)):
            image,label=source[index]
            assert np.array_equal(np.asarray(image)[foreground],original[foreground])
            axes[row,col].imshow(image)
            axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(case.replace('train_background_','Train BG seed '))
            record['cases'].append({'case':case,'foreground_exact':True,
                                    'donor':source.donor(index) if case.startswith('train_background') else None})
        metadata.append(record)
    fig.suptitle('CUB held-out images: original majority foreground pixels preserved exactly')
    destination=Path('reports/natural_background_examples' if args.natural else 'reports/background_examples');destination.mkdir(parents=True,exist_ok=True)
    for extension in ['svg','pdf','png']:fig.savefig(destination/f'foreground_preserved.{extension}',dpi=160)
    (destination/'examples.json').write_text(json.dumps(metadata,indent=2))
    plt.close(fig)


if __name__=='__main__':main()
