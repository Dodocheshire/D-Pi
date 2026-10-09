"""Reevaluate CUB semantics after removing support cropped out of a prototype.

Original names and checkpoints remain fixed. Only training-supported assignments
whose named part survives the actual CLIP crop are retained. Original reports
are preserved; downstream retrieval and CAM annotations can be filtered without
rerunning their unchanged model responses.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .analyze import collect, load_model, task_metrics
from .cub_metrics import (MIN_CERTAINTY, PARTS, attribute_metrics, attribute_scores,
                          crop_contains_part, intervention_curve)
from .retrieval_regions import visible_window
from .suite import Runner, pid_identity


CORE = ['clean', 'fixed', 'mean_only', 'adapter_matched', 'pi']
COMPONENTS = ['pi_mlp','pi_random_concepts','pi_concept_scores','pi_no_preservation',
              'pi_no_energy','clean_refit','pi_refit']
BANKS = [
    ('RN50_SAM2', 'outputs/study_v1/cub', 'artifacts/datasets/cub', CORE+COMPONENTS),
    ('RN50_GDINO', 'outputs/gdino_v1/cub', 'artifacts/datasets/cub_gdino', CORE),
    ('ViT-B16_SAM2', 'outputs/backbone_v1/ViT-B16/cub', 'artifacts/backbone_ablation_v1/ViT-B16/cub', CORE),
    ('ViT-L14_SAM2', 'outputs/backbone_v1/ViT-L14/cub', 'artifacts/backbone_ablation_v1/ViT-L14/cub', CORE),
]
UPSTREAM = ['study_v1','gdino_v1','backbone_v1','postprocess_v1','gdino_localization_v1',
            'retrieval_gdino_v1','retrieval_sam2_v1','background_gdino_v1','background_sam2_v1',
            'natural_background_gdino_v1','natural_background_sam2_v1']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)); temporary.replace(path)


def link(source, target):
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        assert target.resolve() == source, (source, target)
    else:
        target.symlink_to(os.path.relpath(source, target.parent.resolve()))


def filtered_assignments(pack, annotations, manifest, assignments):
    """Use source training parts only, with the exact region encoding window."""
    lookup = {int(index):row for row,index in enumerate(manifest['train_indices'])}
    retained, removed = {}, []
    for attribute, concepts in sorted(assignments.items()):
        attribute = int(attribute)
        name = annotations['attribute_names'][attribute]
        kind = name.split('::')[0].removeprefix('has_').split('_')[0]
        for concept in concepts:
            source, box = pack['metadata']['concept_sources'][concept]
            row = lookup[source]  # A nontraining source is an error, not an eligible candidate.
            assert annotations['train']['attributes'][row,attribute] > 0
            assert annotations['train']['certainty'][row,attribute] >= MIN_CERTAINTY
            parts = annotations['train']['parts'][row]
            assert any(crop_contains_part(parts[p],box) for p in PARTS.get(kind,[]))
            window = visible_window([round(v) for v in box])
            if any(crop_contains_part(parts[p],window) for p in PARTS.get(kind,[])):
                retained.setdefault(attribute, []).append(concept)
            else:
                removed.append({'concept':concept,'attribute':attribute,'source_index':source,
                                'proposal_box':box,'visible_window':window})
    return retained, removed


def establish_mapping(directory, pack, annotations, manifest):
    original = directory/f'attribute_mapping_k{len(pack["centers"])}.json'
    source = json.loads(original.read_text())
    assert source['min_certainty'] == MIN_CERTAINTY
    assigned, removed = filtered_assignments(pack, annotations, manifest, source['assignments'])
    result = {'protocol':'Retain original training-supported names only when their supporting '
                        'part survives the actual Pillow region crop and CLIP resize/center crop. '
                        'Conservative filtering; no reassignment and no test labels used.',
              'min_certainty':MIN_CERTAINTY,'assignments':assigned,
              'attribute_names':annotations['attribute_names'], 'removed':removed,
              'original_mapping_sha256':sha(original),
              'manifest_sha256':sha(directory/'manifest.json'),
              'code_sha256':sha(__file__)}
    path = directory/f'attribute_mapping_visible_k{len(pack["centers"])}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        # JSON converts attribute keys to strings.
        assert saved == json.loads(json.dumps(result)), 'Visible-support protocol changed'
    else:
        write_json(path,result)
    return assigned, path


def evaluate(source, output, directory, device='cpu'):
    pack_path = directory/'baseline_k2048_seed0.pt'
    pack = torch.load(pack_path,weights_only=True,map_location='cpu')
    annotations = torch.load(directory/'attributes.pt',weights_only=True,map_location='cpu')
    manifest = json.loads((directory/'manifest.json').read_text())
    assigned, mapping_path = establish_mapping(directory,pack,annotations,manifest)
    for split in ['train','val','test']:
        assert torch.equal(pack[split]['indices'],torch.tensor(manifest[f'{split}_indices']))
        assert len(annotations[split]['attributes']) == len(pack[split]['labels'])
    model, config = load_model(pack,source/'model.pt',device)
    assert not config.get('validation_only')
    val = collect(model,pack['val'],samples=8,seed=2110)
    test = collect(model,pack['test'],samples=8,seed=2110)
    val_scores,test_scores = attribute_scores(val['concepts'],assigned),attribute_scores(test['concepts'],assigned)
    report = attribute_metrics(val_scores,test_scores,annotations,assigned)
    report['certainty_sensitivity'] = {str(level):attribute_metrics(val_scores,test_scores,annotations,assigned,level)
                                       for level in [2,4]}
    report['task_metrics'] = task_metrics(test['probabilities'],pack['test']['labels'])
    train = collect(model,pack['train'],samples=1,mean=True)
    annotations['test']['task_labels'] = pack['test']['labels']
    report['interventions'] = intervention_curve(model,train['clean_concepts'],test['concepts'],annotations,assigned)
    report.update(mapping_file=str(mapping_path),mapping_sha256=sha(mapping_path),
                  source_checkpoint_sha256=sha(source/'model.pt'),
                  original_attribute_report_sha256=sha(source/'cub_concepts.json'),
                  code_sha256=sha(__file__),evaluation_split='test',device=device,
                  interpretation='Training-only support filtering of the original mapping; '
                                 'new thresholds from unchanged validation images; no model retraining. '
                                 'Remaining mappings are annotation-supported proxies, not human crop judgments.')
    for file in ['model.pt','analysis.json','predictions.pt']:
        link(source/file,output/file)
    write_json(output/'cub_concepts.json',report)
    print(output.name,'visible attribute BA',report['attribute_macro_balanced_accuracy'],flush=True)


def derive_retrieval(source, output, assigned, mapping_path):
    original = json.loads((source/'region_retrieval.json').read_text())
    reverse = {c:int(a) for a,concepts in assigned.items() for c in concepts}
    kept = [q for q in original['queries'] if q['concept'] in reverse]
    assert all(q['attribute'] == reverse[q['concept']] for q in kept)
    result = copy.deepcopy(original)
    result.update(queries=kept,mapped_queries=len(reverse),evaluated_queries=len(kept),
                  mapping_sha256=sha(mapping_path),source_report_sha256=sha(source/'region_retrieval.json'),
                  support_filter='Remove unsupported training prototype names; unchanged test region scores and ranking.')
    for field,key in [('precision_at_5_macro','precision_at_5'),
                      ('precision_at_5_distinct_images_macro','precision_at_5_distinct_images'),
                      ('chance_precision_macro','chance_precision')]:
        values = [q[key] for q in kept if q[key] is not None]
        result[field] = float(np.mean(values)) if values else None
    result['excess_precision_over_chance'] = float(np.mean([
        q['precision_at_5']-q['chance_precision'] for q in kept])) if kept else None
    write_json(output/'region_retrieval.json',result)


def derive_localization(source, output, assigned, mapping_path, directory):
    from .localization import part_points
    from .prepare import load_sources
    original = json.loads((source/'localization.json').read_text())
    assert not original['debug_limit'] and original['evaluated_images'] == original['test_images']
    annotations = torch.load(directory/'attributes.pt',weights_only=True,map_location='cpu')
    manifest = json.loads((directory/'manifest.json').read_text())
    _, images = load_sources('cub',manifest['root'])
    reverse = {c:int(a) for a,concepts in assigned.items() for c in concepts}
    result = copy.deepcopy(original)
    for row in result['rows']:
        position = row['row']
        eligible = [c for c in row['concepts'] if c in reverse
                    and annotations['test']['attributes'][position,reverse[c]] > 0
                    and annotations['test']['certainty'][position,reverse[c]] >= MIN_CERTAINTY]
        with Image.open(images.samples[manifest['test_indices'][position]][0]) as image:
            size = image.size
        row['part_candidates'] = len(eligible)
        row['part_localizable'] = sum(bool(part_points(annotations['test']['parts'][position],
                                      annotations['attribute_names'][reverse[c]],size)) for c in eligible)
        row['parts'] = [p for p in row['parts'] if p['concept'] in eligible]
        assert all(p['attribute'] == reverse[p['concept']] for p in row['parts'])
    parts = [p for row in result['rows'] for p in row['parts']]
    localizable = sum(row['part_localizable'] for row in result['rows'])
    result.update(part_candidate_concepts=sum(row['part_candidates'] for row in result['rows']),
                  part_localizable_concepts=localizable,part_evaluated_concepts=len(parts),
                  part_hit_rate=float(np.mean([p['hit_radius_0_1_diagonal'] for p in parts])) if parts else None,
                  part_hit_rate_all_localizable=sum(p['hit_radius_0_1_diagonal'] for p in parts)/localizable if localizable else None,
                  part_mean_distance=float(np.mean([p['distance_diagonal'] for p in parts])) if parts else None,
                  source_report_sha256=sha(source/'localization.json'),visible_mapping_sha256=sha(mapping_path),
                  support_filter='Filter part eligibility only; unchanged CAMs, grids, foregrounds and surviving part distances.')
    write_json(output/'localization.json',result)


def ready(path):
    return path.exists() and time.time()-path.stat().st_mtime > 10


def summarize():
    groups = []
    for name,_,_,variants in BANKS:
        for variant in variants:
            paths = [Path('outputs/visible_semantics_v1')/name/f'{variant}_seed{s}/cub_concepts.json' for s in range(5)]
            rows = [json.loads(p.read_text()) for p in paths if p.exists()]
            if not rows:continue
            metrics = {}
            for key in ['attribute_macro_balanced_accuracy','attribute_macro_auroc','mapped_attributes','evaluated_attributes']:
                values = [r[key] for r in rows if r[key] is not None]
                metrics[key] = {'mean':float(np.mean(values)) if values else None,
                                'std':float(np.std(values,ddof=1)) if len(values)>1 else None}
            groups.append({'bank':name,'variant':variant,'seeds':[s for s,p in enumerate(paths) if p.exists()],
                           'metrics':metrics})
    write_json('reports/visible_semantics_v1/results.json',{'groups':groups,
                'limits':'Original task checkpoints and names fixed. Training support filtered, validation thresholds recalibrated. '
                         'Partial seed groups explicitly identified. Coverage must accompany semantic macro averages.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follow',action='store_true')
    parser.add_argument('--source');parser.add_argument('--output');parser.add_argument('--directory')
    args = parser.parse_args();torch.set_num_threads(8)
    if not args.follow:
        assert args.source and args.output and args.directory
        evaluate(Path(args.source),Path(args.output),Path(args.directory));return
    runner = Runner('artifacts/logs/visible_semantics_v1')
    dependencies = []
    for name in UPSTREAM:
        path = Path('artifacts/logs')/name/'status.json';state = json.loads(path.read_text())
        dependencies.append((path,state['manager_pid'],pid_identity(state['manager_pid'])))
    while True:
        for name,root,directory,variants in BANKS:
            root,directory = Path(root),Path(directory)
            if not (directory/'attribute_mapping_k2048.json').exists():continue
            for variant in variants:
                for seed in range(5):
                    source = root/f'{variant}_seed{seed}'
                    output = Path('outputs/visible_semantics_v1')/name/source.name
                    if ready(source/'cub_concepts.json') and not (output/'cub_concepts.json').exists():
                        runner.state.update(phase=f'{name}_{source.name}_attributes');runner.save()
                        evaluate(source,output,directory)
                        runner.state['completed_commands'].append({'name':f'{name}_{source.name}_attributes','returncode':0,'finished_at':time.time()})
                        runner.save();summarize()
                    if not (output/'cub_concepts.json').exists() or variant not in CORE or not name.startswith('RN50'):continue
                    mapping_path = directory/'attribute_mapping_visible_k2048.json'
                    assigned = json.loads(mapping_path.read_text())['assignments']
                    for file,derive in [('region_retrieval.json',derive_retrieval),('localization.json',derive_localization)]:
                        if ready(source/file) and not (output/file).exists():
                            if derive is derive_localization:derive(source,output,assigned,mapping_path,directory)
                            else:derive(source,output,assigned,mapping_path)
                    for version in ['background','natural_background']:
                        bank = 'sam2' if name=='RN50_SAM2' else 'gdino'
                        raw = source if version=='background' else Path('outputs')/f'natural_background_{bank}_v1/cub'/source.name
                        destination = Path('outputs')/f'visible_{version}_v1'/name/source.name
                        if not ready(raw/'backgrounds.json') or (destination/'backgrounds.json').exists():continue
                        for file in ['model.pt','analysis.json','predictions.pt','cub_concepts.json']:
                            link(output/file,destination/file)
                        cache = 'artifacts/datasets/cub' if version=='background' else 'artifacts/datasets/cub_background_natural'
                        runner.command(f'{name}_{source.name}_{version}',[
                            'pi_experiments.backgrounds','--stage','evaluate','--device','cpu',
                            '--directory',cache,'--pack',str(directory/'baseline_k2048_seed0.pt'),'--run',str(destination)])
                        report = json.loads((destination/'backgrounds.json').read_text())
                        report.update(visible_mapping_sha256=sha(mapping_path),device='cpu',
                                      source_background_report_sha256=sha(raw/'backgrounds.json'),
                                      support_filter='Unchanged models and images, original-validation thresholds from the visible-support mapping.')
                        write_json(destination/'backgrounds.json',report)
        live = [identity is not None and pid_identity(pid)==identity for _,pid,identity in dependencies]
        if not any(live):break
        runner.state.update(phase='waiting_for_original_evaluations',checked_at=time.time());runner.save();time.sleep(10)
    for path,pid,_ in dependencies:
        state = json.loads(path.read_text());assert state['manager_pid']==pid and state['phase']=='complete',(path,state['phase'])
    missing = []
    for name,_,_,variants in BANKS:
        for variant in variants:
            for seed in range(5):
                run = f'{variant}_seed{seed}'
                files = ['cub_concepts.json'] + (['region_retrieval.json','localization.json'] if name.startswith('RN50') and variant in CORE else [])
                missing.extend(str(p) for file in files if not (p:=Path('outputs/visible_semantics_v1')/name/run/file).exists())
                if name.startswith('RN50') and variant in CORE:
                    missing.extend(str(p) for version in ['background','natural_background']
                                   if not (p:=Path('outputs')/f'visible_{version}_v1'/name/run/'backgrounds.json').exists())
    assert not missing,missing
    summarize();runner.state.update(phase='complete',finished_at=time.time());runner.save()


if __name__=='__main__':main()
