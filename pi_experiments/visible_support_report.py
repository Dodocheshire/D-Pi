"""Compare original and visible-support CUB metrics on identical attribute sets."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .visible_semantics import BANKS, write_json


def main():
    groups = []
    for bank,root,_,variants in BANKS:
        for variant in variants:
            rows = []
            for seed in range(5):
                original = Path(root)/f'{variant}_seed{seed}/cub_concepts.json'
                visible = Path('outputs/visible_semantics_v1')/bank/f'{variant}_seed{seed}/cub_concepts.json'
                if not original.exists() or not visible.exists():continue
                a,b = json.loads(original.read_text()),json.loads(visible.read_text())
                assert b['original_attribute_report_sha256'] == hashlib.sha256(original.read_bytes()).hexdigest()
                comparisons = {}
                for level in [3,2,4]:
                    av,bv = (a,b) if level==3 else (a['certainty_sensitivity'][str(level)],b['certainty_sensitivity'][str(level)])
                    ad,bd = ({r['attribute']:r for r in record['details']} for record in [av,bv])
                    common = sorted(ad.keys() & bd.keys())
                    assert all(ad[i]['test_known_samples']==bd[i]['test_known_samples'] for i in common)
                    comparisons[str(level)] = {
                        'attributes':common,'original_evaluated_attributes':len(ad),'visible_evaluated_attributes':len(bd),
                        'metrics':{key:{'original':float(np.mean([ad[i][key] for i in common])),
                                        'visible':float(np.mean([bd[i][key] for i in common])),
                                        'delta':float(np.mean([bd[i][key]-ad[i][key] for i in common]))}
                                   for key in ['balanced_accuracy','auroc']} if common else {}}
                rows.append({'seed':seed,'comparisons':comparisons,
                             'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                             'visible_sha256':hashlib.sha256(visible.read_bytes()).hexdigest()})
            if rows:groups.append({'bank':bank,'variant':variant,'rows':rows})
    write_json('reports/visible_semantics_v1/support_comparison.json',{
        'groups':groups,'limits':'Original and filtered aggregate macros can cover different attributes. '
                               'These descriptive comparisons use their intersection and identical known test annotations. '
                               'No model, prototype name or test-derived choice is changed. '
                               'The difference diagnoses mapping sensitivity, not a Pi treatment effect.'})
    print('Compared identical attribute sets in',sum(len(g['rows']) for g in groups),'completed report pairs')


if __name__=='__main__':main()
