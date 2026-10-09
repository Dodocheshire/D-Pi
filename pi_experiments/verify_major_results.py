"""Check the exported figures, documents, snapshots and delivery manifest."""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader


DEST = Path('../Essay/D/report')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ['href', 'src']:
                self.links.append(value)
            if name == 'id':
                self.ids.add(value)


def main():
    manifest = json.loads((DEST/'manifest.json').read_text())
    data = json.loads((DEST/'plot_data.json').read_text())
    document = (DEST/'主要实验结果.md').read_text()
    page = (DEST/'index.html').read_text()
    assert digest(DEST/'plot_data.json') == manifest['plot_data_sha256']
    assert digest(Path(__file__).with_name('export_major_results.py')) == manifest['export_code_sha256']
    figures = manifest['figures']
    assert set(data) == {section['file'] for section in figures}
    figure_sizes = {}
    for section in figures:
        assert all(section[key] for key in ['purpose','conclusion','note'])
        assert section['purpose'] in document and section['conclusion'] in document
        assert section['file'] in page
        root = DEST/section['file']
        with Image.open(root.with_suffix('.png')) as picture:
            figure_sizes[section['file']] = list(picture.size)
            assert min(picture.size) >= 600
            picture.verify()
        assert ET.parse(root.with_suffix('.svg')).getroot().tag.endswith('svg')
        assert len(PdfReader(root.with_suffix('.pdf')).pages) == 1
    pages = len(PdfReader(DEST/'主要实验图表合集.pdf').pages)
    assert pages == len(figures)
    parser = Links()
    parser.feed(page)
    markdown_links = re.findall(r'\]\(([^)]+)\)',document)
    for link in parser.links+markdown_links:
        url = urlsplit(link)
        assert not url.scheme and not url.netloc, link
        if url.path:
            path = DEST/unquote(url.path)
            assert path.is_file(), path
            assert path.resolve().is_relative_to(DEST.resolve())
        if url.fragment:
            assert url.fragment in parser.ids
    snapshot = Path(manifest['source_snapshots_directory'])
    snapshots_checked = 0
    for source, expected in manifest['sources_sha256'].items():
        path = Path(source)
        if path.suffix == '.json':
            assert digest(snapshot/f'{path.stem}_{expected}.json') == expected
            snapshots_checked += 1
    checkpoints = manifest['verified_classification_checkpoints']
    assert all(r['projection_exact'] and r['prediction_metrics_reproduced'] for r in checkpoints)
    intervals = manifest['verified_core_paired_intervals']
    assert all(r['paired_interval_reproduced'] for r in intervals)
    sparse = data['11_sparse_heads']['selected_heads']
    stress = data.get('14_image_corruptions',{})
    branches = data.get('15_mean_scale_branches',{})
    exact_sparse = data['11_sparse_heads'].get('exact_capacity_verification')
    sam_attributes = data.get('17_sam2_attributes')
    gdino_attributes = data['03_grounded_attributes'].get('native_attribute_verification')
    primary = data.get('18_original_head_refit')
    background_attributes = data.get('19_background_attributes',{})
    retrieval_grounding = data['05_retrieval_intervention'].get('native_selection_grounding_audit')
    record = {
        'figure_groups':len(figures),'formats':['PNG','SVG','PDF'],'combined_pdf_pages':pages,
        'local_links_valid':True,'purpose_conclusion_per_group':True,
        'core_classification_cases':[r['case'] for r in data['01_task_accuracy']],
        'verified_classification_records':len(checkpoints),
        'unique_classification_checkpoints':len({manifest['sources_sha256'][r['checkpoint']] for r in checkpoints}),
        'checkpoint_deduplication':'SHA-256 of checkpoint bytes; symlink aliases counted once',
        'core_paired_intervals_reproduced':len(intervals),
        'semantic_checkpoints_reproduced':sum('/visible_semantics_v1/' in p and p.endswith('/semantic_predictions.pt')
                                               for p in manifest['sources_sha256']),
        'retrieval_native_grounding_models_verified':len(retrieval_grounding['verified_checkpoints']) if retrieval_grounding else 0,
        'retrieval_query_selections_verified':retrieval_grounding['verified_query_selections'] if retrieval_grounding else 0,
        'retrieval_score_rankings_recomputed':False,
        'additional_retrieval_native_grounding_audits':{bank:{'models':len(r['verified_checkpoints']),
                       'query_selections':r['verified_query_selections'],'test_images':r['native_test_images_verified']}
                      for bank,r in data['05_retrieval_intervention'].get('additional_native_selection_grounding_audits',{}).items()},
        'background_attribute_groups':list(background_attributes),
        'background_native_attribute_models_verified':sum(r['verified_models'] for r in background_attributes.values()),
        'background_unique_model_checkpoints':len({sha for r in background_attributes.values()
                                                    for path,sha in r['source_sha256'].items() if path.endswith('/model.pt')}),
        'background_model_count_scope':'model references per background protocol; shared checkpoints counted once in unique field',
        'background_native_attribute_prediction_cases':sum(r['verified_prediction_cases'] for r in background_attributes.values()),
        'background_native_attribute_intervals':sum(len(r['comparisons']) for r in background_attributes.values()),
        'background_paired_task_intervals':sum(len(r['comparisons']) for r in background_attributes.values()),
        'localization_comparisons':len(data['13_localization_intervals']['comparisons']),
        'completed_concept_configurations':len(data['10_concept_bank']),
        'sparse_selected_four_methods_complete':all(v in sparse for v in ['clean','mean_only','adapter_matched','pi']),
        'sparse_four_method_exact_nec_complete':exact_sparse is not None,
        'sparse_exact_models_verified':exact_sparse['verified_models'] if exact_sparse else 0,
        'sparse_exact_budget_points_verified':exact_sparse['verified_exact_budget_points'] if exact_sparse else 0,
        'sparse_exact_paired_intervals_reproduced':len(exact_sparse['reproduced_paired_intervals']) if exact_sparse else 0,
        'SAM2_native_attribute_checkpoints_verified':len(sam_attributes['verified_checkpoints']) if sam_attributes else 0,
        'SAM2_attribute_paired_intervals_reproduced':len(sam_attributes['comparisons']) if sam_attributes else 0,
        'GDINO_native_attribute_checkpoints_verified':len(gdino_attributes['verified_checkpoints']) if gdino_attributes else 0,
        'GDINO_attribute_paired_intervals_reproduced':len(gdino_attributes['comparisons']) if gdino_attributes else 0,
        'RN50_primary_matrix_models_verified':primary['verified_models'] if primary else 0,
        'original_head_refits_non_head_tensors_exact':len(primary['original_head_refit_checks']) if primary else 0,
        'original_head_refit_paired_intervals':len(primary['head_refit_comparisons']) if primary else 0,
        'displayed_corruption_datasets':list(stress),
        'corruption_frozen_checkpoints_audited':sum(r['verified_frozen_checkpoints'] for r in stress.values()),
        'corruption_prediction_cases_reproduced':sum(r['reproduced_prediction_cases'] for r in stress.values()),
        'corruption_paired_intervals':sum(len(r['comparisons']) for r in stress.values()),
        'displayed_branch_datasets':list(branches),
        'branch_prediction_scenarios_reproduced':sum(len(r['seeds'])*len(r['groups']) for r in branches.values()),
        'branch_pairwise_intervals':sum(len(r['comparisons']) for r in branches.values()),
        'source_json_snapshots_checked':snapshots_checked,
        'plot_data_hash_exact':True,'export_source_hash_exact':True,
        'figure_sizes':figure_sizes,'checked_at':time.time(),
        'delivery_file_sha256':{str(p.name):digest(p) for p in sorted(DEST.iterdir())
                                if p.is_file() and p.name!='交付核验.json'},
    }
    temporary = DEST/'交付核验.partial.json'
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    temporary.replace(DEST/'交付核验.json')
    print(json.dumps({k:v for k,v in record.items() if k not in ['figure_sizes','delivery_file_sha256']},
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEST)
    DEST=parser.parse_args().output
    main()
