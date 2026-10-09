"""Track artifact coverage of the research plan without declaring scientific completion."""

import json
from pathlib import Path
import time

from .sampling_study import specifications
from .visible_semantics import BANKS


CORE = ['clean','fixed','mean_only','adapter_matched','pi']
COMPONENTS = ['pi_mlp','pi_random_concepts','pi_concept_scores','pi_no_preservation','pi_no_energy','clean_refit','pi_refit']


def case(name, root, variants, files):
    required = [Path(root)/f'{variant}_seed{seed}'/file for variant in variants for seed in range(5) for file in files]
    return file_case(name,required)


def file_case(name, required):
    missing = [str(path) for path in required if not path.exists()]
    return {'name':name,'required_artifacts':len(required),'present_artifacts':len(required)-len(missing),
            'all_files_present':not missing,'missing':missing}


def main():
    rows = []
    rows.append(file_case('Large-dataset source and split verification',
                         [Path('artifacts/datasets')/dataset/file
                          for dataset in ['places365','imagenet']
                          for file in ['manifest.json','manifest_preflight.json']]))
    rows.append(file_case('RN50 large-dataset full-image feature preparation',
                         [Path('artifacts/datasets')/dataset/file
                          for dataset in ['places365','imagenet']
                          for file in ['train_features.pt','val_features.pt','test_features.pt',
                                       'feature_prewarm_verification.json']]))
    rows.append(file_case('ViT large-dataset full-image feature verification',
                         [Path('reports/preparation_verification_v1')/f'{backbone}_{dataset}.json'
                          for backbone in ['ViT-B16','ViT-L14'] for dataset in ['places365','imagenet']]))
    rows.append(file_case('Large-dataset complete SAM2 proposal verification',
                         [Path('artifacts/datasets')/dataset/'proposal_prewarm_verification.json'
                          for dataset in ['places365','imagenet']]))
    rows.append(file_case('RN50 large-dataset source label alignment', [
        Path('reports/preparation_verification_v1/RN50_places365_source_alignment.json'),
        Path('reports/preparation_verification_v1/RN50_imagenet_native_source_audit.json')]))
    rows.append(file_case('Complete custom corruption cache verification', [
        Path('artifacts/datasets')/dataset/'corruptions/prewarm_verification.json'
        for dataset in ['cifar10','cifar100']]))
    rows.append(file_case('Paired custom corruption intervals', [
        Path('reports/corruption_uncertainty_v1')/f'{dataset}.json'
        for dataset in ['cifar10','cifar100']]))
    for dataset in ['cifar10','cifar100','cub']:
        rows.append(case(f'Original mean/scale branch interventions {dataset}',
                         Path('outputs/noise_branch_evidence_v1')/dataset,
                         ['pi','pi_refit'],['metrics.json','predictions.pt']))
        rows.append(file_case(f'Paired branch intervention intervals {dataset}',
                             [Path('reports/noise_branch_evidence_v1')/dataset/f'{variant}.json'
                              for variant in ['pi','pi_refit']]))
    for name, root, _ in specifications():
        rows.append(case(f'Core paired models {name}',root,CORE,['model.pt','analysis.json','predictions.pt']))
        rows.append(case(f'Nested sampling budgets {name}',root,['pi'],['sampling_budget.json','sampling_budget_predictions.pt']))
        if name.endswith('/cub') or name.endswith('/cub_gdino'):
            rows.append(case(f'True attributes and interventions {name}',root,CORE,['cub_concepts.json']))
    for dataset in ['cifar10','cifar100','cub']:
        root=Path('outputs/study_v1')/dataset
        rows.append(case(f'Condition, loss, score injection and head refit {dataset}',root,COMPONENTS,
                         ['model.pt','analysis.json','predictions.pt']))
    rows.append(case('Expanded RN50 internal injection', 'outputs/internal_v2/cifar10', CORE,
                     ['model.pt','analysis.json','predictions.pt']))
    rows.append(case('Initialization-matched random conditioning','outputs/conditioning_v1/cifar10',
                     ['pi_random_matched'],['model.pt','analysis.json','predictions.pt']))
    rows.append(case('Geometry-preserving conditioning rotation','outputs/rotation_conditioning_v1/cifar10',
                     ['pi_rotated_matched'],['model.pt','analysis.json','predictions.pt']))
    configuration = json.loads(Path('artifacts/concept_ablation_v1/cifar10_cases.json').read_text())['cases']
    for item in configuration:
        rows.append(case(f'Concept source/K {item["name"]}',Path('outputs/concept_study_v1/cifar10')/item['name'],
                         CORE,['model.pt','analysis.json','predictions.pt']))
    for version in ['head_sparsity_v1','proximal_sparsity_v1','proximal_sparsity_v2']:
        rows.append(case(version,Path('outputs')/version/'cifar10',['clean','pi'],['head_grid.json','heads.pt','predictions.pt']))
    rows.append(case('Exactly matched per-class NEC', 'outputs/matched_nec_v2/cifar10',
                     ['clean','pi'], ['metrics.json','predictions.pt']))
    rows.append(case('Sparse deterministic capacity controls', 'outputs/sparse_capacity_v1/cifar10',
                     ['clean','mean_only','adapter_matched','pi'], ['selection.json','head_grid.json','heads.pt','predictions.pt']))
    rows.append(case('Exact NEC deterministic and sampling comparisons', 'outputs/sparse_capacity_v1/cifar10',
                     ['clean','mean_only','adapter_matched','pi'], ['exact_metrics.json','exact_predictions.pt']))
    rows.append(file_case('Joint sparse capacity intervals', [Path('reports/sparse_capacity_v1/cifar10/results.json')]))
    rows.append(file_case('Independent exact NEC four-method audit', [
        Path('reports/sparse_capacity_v1/cifar10/exact_capacity_verification.json')]))
    rows.append(file_case('Complete RN50 primary matrix and frozen-generator head-refit audit', [
        Path('reports/primary_matrix_verification_v1/results.json')]))
    for bank in ['RN50_SAM2','RN50_GDINO']:
        rows.append(file_case(f'Independent native attribute truth and interval audit {bank}', [
            Path('reports/attribute_evidence_verification_v1')/f'{bank}.json']))
        rows.append(file_case(f'Independent native retrieval selection grounding audit {bank}', [
            Path('reports/retrieval_evidence_verification_v1')/f'{bank}.json']))
    for bank in ['RN50_SAM2','RN50_GDINO']:
        rows.append(file_case(f'Paired full-test localization {bank}',
                             [Path('reports/localization_uncertainty_v1')/bank/f'{variant}_vs_pi.json'
                              for variant in CORE if variant!='pi']))
    for dataset in ['cifar10','cifar100']:
        rows.append(case(f'Image corruptions {dataset}',Path('outputs/study_v1')/dataset,CORE,['corruptions.json']))
    for bank, root in [('SAM2',Path('outputs/study_v1/cub')),('GDINO',Path('outputs/gdino_v1/cub'))]:
        rows.append(case(f'Full-test localization {bank}',root,CORE,['localization.json']))
        rows.append(case(f'Annotation-grounded region retrieval {bank}',root,CORE,['region_retrieval.json']))
        for version in ['background','natural_background']:
            background_root = root if version == 'background' else Path('outputs')/f'{version}_{bank.lower()}_v1'/'cub'
            rows.append(case(f'{version} {bank}',background_root,
                             CORE,['backgrounds.json']))
    rows.append(case('ImageNet-R RN50','outputs/rendition_eval_v1/imagenet',CORE,['rendition.json','rendition_predictions.pt']))
    for backbone in ['ViT-B16','ViT-L14']:
        rows.append(case(f'ImageNet-R {backbone}',Path('outputs/rendition_backbones_v1')/backbone/'imagenet',
                         CORE,['rendition.json','rendition_predictions.pt']))
    for name,_,_,variants in BANKS:
        root=Path('outputs/visible_semantics_v1')/name
        rows.append(case(f'Visible-support true attributes {name}',root,variants,['cub_concepts.json']))
        rows.append(case(f'Per-image true attribute predictions {name}',root,variants,['semantic_predictions.pt']))
        rows.append(file_case(f'Paired true attribute balanced accuracy {name}',
                             [Path('reports/semantic_uncertainty_v1')/name/f'{variant}_vs_pi.json'
                              for variant in variants if variant!='pi']))
        if name.startswith('RN50'):
            rows.append(case(f'Visible-support localization and retrieval {name}',root,CORE,
                             ['localization.json','region_retrieval.json']))
            for version in ['background','natural_background']:
                rows.append(case(f'Visible-support {version} {name}',
                                 Path('outputs')/f'visible_{version}_v1'/name,CORE,['backgrounds.json']))
                rows.append(case(f'Native per-image background attribute evidence {version} {name}',
                                 Path('outputs/background_semantic_uncertainty_v1')/name/version,
                                 CORE,['semantic_predictions.pt']))
                rows.append(file_case(f'Paired task and native attribute background intervals {version} {name}',
                                      [Path('reports/background_semantic_uncertainty_v1')/name/f'{version}.json']))
    report={'checked_at':time.time(),'source_plan':'Essay/D/研究方案.md','cases':rows,
            'limits':'This inventories expected artifacts. File presence is not proof of scientific validity, '
                     'complete test coverage, frozen invariants, or goal completion. Inspect contents and corresponding verification.'}
    destination=Path('reports/RESEARCH_COVERAGE.json');destination.write_text(json.dumps(report,indent=2))
    print(json.dumps({'cases':len(rows),'cases_with_all_files':sum(r['all_files_present'] for r in rows),
                      'present_artifacts':sum(r['present_artifacts'] for r in rows),
                      'required_artifacts':sum(r['required_artifacts'] for r in rows)},indent=2))


if __name__=='__main__':main()
