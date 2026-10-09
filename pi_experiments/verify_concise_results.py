"""Validate six-figure delivery against its audited full-result snapshot."""

import argparse
import csv
import json
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader

from .export_major_results import LABELS
from .verify_major_results import Links, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('../Essay/D/report'))
    dest=parser.parse_args().output
    manifest=json.loads((dest/'manifest.json').read_text())
    source=Path(manifest['source_directory'])
    checks={'manifest.json':'source_manifest_sha256','plot_data.json':'source_plot_data_sha256',
            '交付核验.json':'source_delivery_verification_sha256'}
    for name,key in checks.items():assert digest(source/name)==manifest[key],name
    audit=json.loads((source/'交付核验.json').read_text())
    for name in ['manifest.json','plot_data.json']:
        assert audit['delivery_file_sha256'][name]==digest(source/name)
    assert all(audit[key] for key in ['local_links_valid','purpose_conclusion_per_group',
                                      'plot_data_hash_exact','export_source_hash_exact'])
    assert digest(dest/'plot_data.json')==manifest['plot_data_sha256']
    assert digest(Path(__file__).with_name('export_concise_results.py'))==manifest['export_code_sha256']
    assert digest(Path(__file__).with_name('export_major_results.py'))==manifest['plot_helper_sha256']
    assert digest(Path('reports/ONE_HOUR_CLOSEOUT.json'))==manifest['one_hour_closeout_sha256']
    full=json.loads((source/'plot_data.json').read_text())
    data=json.loads((dest/'plot_data.json').read_text())
    figures=manifest['figures'];assert len(figures)==6
    assert set(data)=={s['file'] for s in figures}
    document=(dest/'主要实验结果.md').read_text();page=(dest/'index.html').read_text()
    sizes={};source_groups=0
    for section in figures:
        assert all(section[k] for k in ['purpose','conclusion','note'])
        assert section['purpose'] in document and section['conclusion'] in document
        for key,value in data[section['file']].items():
            assert value==full[key],(section['file'],key);source_groups+=1
        for ext in ['png','svg','pdf']:assert (dest/f'{section["file"]}.{ext}').stat().st_size>1000
        with Image.open(dest/f'{section["file"]}.png') as im:
            sizes[section['file']]=list(im.size);assert min(im.size)>=600;im.verify()
        assert ET.parse(dest/f'{section["file"]}.svg').getroot().tag.endswith('svg')
        assert len(PdfReader(dest/f'{section["file"]}.pdf').pages)==1
    assert len(PdfReader(dest/'主要实验图表合集.pdf').pages)==6
    links=Links();links.feed(page)
    for link in links.links+re.findall(r'\]\(([^)]+)\)',document):
        url=urlsplit(link);assert not url.scheme and not url.netloc
        if url.path:
            path=dest/unquote(url.path);assert path.is_file() and path.resolve().is_relative_to(dest.resolve())
        if url.fragment:assert url.fragment in links.ids
    core=full['01_task_accuracy'];expected={}
    for case in core:
        for method,row in case['groups'].items():expected[(case['case'],LABELS[method])]=row
    with (dest/'核心分类指标.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==len(expected)
    for row in rows:
        metrics=expected.pop((row['实验'],row['方法']));assert int(row['适配种子数'])==5
        for column,key,factor in [('准确率均值%','accuracy',100),('NLL','nll',1),('ECE','ece_15_bins',1)]:
            assert abs(float(row[column])-metrics[key]['mean']*factor)<1e-10
        assert abs(float(row['准确率SD%'])-metrics['accuracy']['std']*100)<1e-10
    assert not expected
    result={'main_figures':6,'combined_pdf_pages':6,'source_groups_exact':source_groups,
            'source_full_delivery_verified':True,'csv_rows_reproduced':len(rows),
            'core_cases':[r['case'] for r in core],'local_links_valid':True,
            'purpose_conclusion_per_group':True,'figure_sizes':sizes,
            'checked_at':time.time(),'verifier_code_sha256':digest(Path(__file__)),
            'delivery_file_sha256':{p.name:digest(p) for p in sorted(dest.iterdir())
                                   if p.is_file() and p.name!='交付核验.json'}}
    (dest/'交付核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['figure_sizes','delivery_file_sha256']},ensure_ascii=False))


if __name__=='__main__':main()
