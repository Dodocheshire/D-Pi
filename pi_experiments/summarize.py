"""Aggregate completed real-data experiments and export research figures."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .analyze import paired_bootstrap
from .statistics import hierarchical_paired_bootstrap


def summarize(root,destination):
    root,destination=Path(root),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for path in root.glob('*/*_seed*/analysis.json'):
        report=json.loads(path.read_text())
        if report.get('evaluation_split')!='test':
            continue
        dataset=path.parent.parent.name
        name,seed=path.parent.name.rsplit('_seed',1)
        groups[dataset,name].append((int(seed),report,path.parent))
    output=[]
    for (dataset,name),rows in sorted(groups.items()):
        record={'dataset':dataset,'variant':name,'seeds':sorted(r[0] for r in rows),'metrics':{}}
        for key in rows[0][1]['inference']['mc8']:
            values=np.array([r[1]['inference']['mc8'][key] for r in rows])
            record['metrics'][key]={'mean':float(values.mean()),'std':float(values.std(ddof=1)) if len(values)>1 else None}
        semantic=[]
        for _,_,directory in rows:
            path=directory/'cub_concepts.json'
            if path.exists():
                semantic.append(json.loads(path.read_text()))
        if semantic:
            record['semantic_runs']=len(semantic)
            record['semantic_metrics']={}
            for key in ['attribute_macro_balanced_accuracy','attribute_macro_auroc','evaluated_attributes','mapped_attributes']:
                values=[r[key] for r in semantic if r[key] is not None]
                record['semantic_metrics'][key]={'mean':float(np.mean(values)) if values else None,
                                               'std':float(np.std(values,ddof=1)) if len(values)>1 else None}
        corruptions=[]
        localizations=[]
        for seed,_,directory in rows:
            path=directory/'corruptions.json'
            if path.exists():corruptions.append((seed,json.loads(path.read_text())))
            path=directory/'localization.json'
            if path.exists():
                localization=json.loads(path.read_text())
                if not localization['debug_limit'] and localization['evaluated_images']==localization['test_images']:
                    localizations.append(localization)
        if corruptions:
            record['corruption_seeds']=[seed for seed,_ in corruptions]
            record['corruption_metrics']={}
            for case in corruptions[0][1]['cases']:
                records=[next(c for c in report['cases'] if c['case']==case['case']) for _,report in corruptions]
                record['corruption_metrics'][case['case']]={}
                for key in ['accuracy','nll','ece_15_bins','corruption_concept_drift_mse']:
                    values=[r[key] for r in records]
                    record['corruption_metrics'][case['case']][key]={'mean':float(np.mean(values)),
                                                                   'std':float(np.std(values,ddof=1)) if len(values)>1 else None}
        if localizations:
            record['localization_runs']=len(localizations)
            record['localization_metrics']={}
            for key in ['relative_attribution','max_point_accuracy','quadrant_gini_normalized']:
                values=[r['grid_all_concepts_mean'][key] for r in localizations]
                record['localization_metrics'][key]={'mean':float(np.mean(values)),
                                                     'std':float(np.std(values,ddof=1)) if len(values)>1 else None}
            record['localization_valid_coverage']=float(np.mean([r['valid_grid_concepts']/r['total_grid_concepts'] for r in localizations]))
        output.append(record)
    comparisons=[]
    stochastic_comparisons=[]
    hierarchical=[]
    for dataset in sorted({d for d,n in groups}):
        pi={r[0]:r for r in groups.get((dataset,'pi'),[])}
        for seed,row in sorted(pi.items()):
            predictions=torch.load(row[2]/'predictions.pt',weights_only=True)
            inference=row[1]['inference']
            stochastic_comparisons.append({'dataset':dataset,'seed':seed,'comparison':'MC8 minus deterministic mean',
                                          'nll_delta':inference['mc8']['nll']-inference['mean']['nll'],
                                          'ece_delta':inference['mc8']['ece_15_bins']-inference['mean']['ece_15_bins'],
                                          **paired_bootstrap(predictions['mean'].numpy(),predictions['mc8'].numpy(),predictions['labels'].numpy())})
        for control in ['clean','fixed','mean_only','adapter_matched']:
            others={r[0]:r for r in groups.get((dataset,control),[])}
            shared=sorted(pi.keys() & others.keys())
            if shared:
                a=[torch.load(others[seed][2]/'predictions.pt',weights_only=True) for seed in shared]
                b=[torch.load(pi[seed][2]/'predictions.pt',weights_only=True) for seed in shared]
                for control_predictions,pi_predictions in zip(a,b):torch.testing.assert_close(control_predictions['labels'],pi_predictions['labels'])
                hierarchical.append({'dataset':dataset,'control':control,'seeds':shared,
                                     **hierarchical_paired_bootstrap(np.stack([r['mc8'].numpy() for r in a]),
                                                                   np.stack([r['mc8'].numpy() for r in b]),a[0]['labels'].numpy())})
            for seed in sorted(pi.keys() & others.keys()):
                a=torch.load(others[seed][2]/'predictions.pt',weights_only=True)
                b=torch.load(pi[seed][2]/'predictions.pt',weights_only=True)
                torch.testing.assert_close(a['labels'],b['labels'])
                comparisons.append({'dataset':dataset,'control':control,'seed':seed,
                                    **paired_bootstrap(a['mc8'].numpy(),b['mc8'].numpy(),a['labels'].numpy())})
        if pi:
            predictions=[torch.load(row[2]/'predictions.pt',weights_only=True) for _,row in sorted(pi.items())]
            hierarchical.append({'dataset':dataset,'control':'same_checkpoint_mean','seeds':sorted(pi),
                                 **hierarchical_paired_bootstrap(np.stack([r['mean'].numpy() for r in predictions]),
                                                               np.stack([r['mc8'].numpy() for r in predictions]),predictions[0]['labels'].numpy())})
    (destination/'results.json').write_text(json.dumps({'groups':output,'pi_paired_comparisons':comparisons,
                                                      'pi_stochastic_vs_mean':stochastic_comparisons,
                                                      'pi_hierarchical_comparisons':hierarchical},indent=2))
    lines=['# D-Pi 实验结果（自动更新）','',
           '仅汇总已完成的独立测试集评估。每组列出实际完成的种子；未完成五种子的结果仍为阶段性结果。',
           '干净基线共享固定原型与初始分类头，五个适配种子不代表五次独立概念构建。','',
           '| 数据集 | 方法 | 已完成种子 | Top-1 (%) | NLL | ECE | 概念漂移 MSE |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for row in output:
        def cell(key,factor=1):
            metric=row['metrics'][key]
            value=f"{metric['mean']*factor:.4f}"
            return value if metric['std'] is None else value+f" ± {metric['std']*factor:.4f}"
        lines.append(f"| {row['dataset']} | {row['variant']} | {','.join(map(str,row['seeds']))} | {cell('accuracy',100)} | {cell('nll')} | {cell('ece_15_bins')} | {cell('standardized_concept_mse')} |")
    if not output:
        lines.extend(['','目前尚无完成的测试集评估；准备缓存和运行中的任务不构成性能证据。'])
    lines.extend(['','## Pi 与容量、随机性对照','',
                  '以下区间分层重采样适配种子与测试图像，种子间共享同一图像重采样，保留配对关系。这些区间以固定原型和初始头为条件，不反映原型构建的不确定性。逐种子样本区间同时保存在结果 JSON。','',
                  '| 数据集 | 共同种子 | 比较（Pi 减对照） | 准确率差（百分点） | 95% 区间 |',
                  '| --- | --- | --- | ---: | --- |'])
    for comparison in hierarchical:
        lo,hi=comparison['ci95']
        label='确定性均值推理（同一 Pi 检查点）' if comparison['control']=='same_checkpoint_mean' else comparison['control']
        lines.append(f"| {comparison['dataset']} | {','.join(map(str,comparison['seeds']))} | {label} | {comparison['accuracy_delta']*100:.3f} | [{lo*100:.3f}, {hi*100:.3f}] |")
    lines.extend(['','## 解释指标与结论边界','',
                  '概念漂移是与原模型标准化响应的差异，不等同于真实概念准确率。CUB 属性评估采用训练标注支撑的固定映射，需同时报告覆盖率。',
                  '准确率—稀疏度图使用每类保留相同数量的概念；统计区间和全部指标见 [results.json](results.json)。',
                  '只有在完成仅均值、同容量适配及同一检查点的均值推理比较后，才能判断随机分布是否带来额外收益。',''])
    if any(r.get('semantic_metrics') for r in output):
        lines.extend(['## CUB 真实属性评估','',
                      '| 方法 | 语义评估次数 | 属性 balanced accuracy (%) | AUROC | 属性覆盖 |',
                      '| --- | ---: | ---: | ---: | --- |'])
        for row in output:
            if not row.get('semantic_metrics'):continue
            semantic=row['semantic_metrics']
            accuracy=semantic['attribute_macro_balanced_accuracy']['mean']
            auroc=semantic['attribute_macro_auroc']['mean']
            lines.append(f"| {row['variant']} | {row['semantic_runs']} | {accuracy*100 if accuracy is not None else '不可用'} | {auroc} | {semantic['evaluated_attributes']['mean']:.0f}/312 |")
        lines.append('')
    (destination/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines))
    for dataset in sorted({d for d,n in groups}):
        subset=[r for r in output if r['dataset']==dataset]
        order=['clean','fixed','mean_only','adapter_matched','pi','pi_mlp','pi_random_concepts','pi_concept_scores','pi_refit','clean_refit']
        subset.sort(key=lambda r:order.index(r['variant']) if r['variant'] in order else 99)
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for ax,key,label in zip(axes,['accuracy','nll','standardized_concept_mse'],['Task accuracy (%)','Test NLL','Concept drift (standardized MSE)']):
            factor=100 if key=='accuracy' else 1
            heights=[r['metrics'][key]['mean']*factor for r in subset]
            errors=[(r['metrics'][key]['std'] or 0)*factor for r in subset]
            ax.bar(np.arange(len(subset)),heights,yerr=errors,color=['#E11D48' if r['variant']=='pi' else '#2563EB' for r in subset],alpha=0.8,capsize=3)
            ax.set_xticks(np.arange(len(subset)),[f"{r['variant']} (n={len(r['seeds'])})" for r in subset],rotation=60,ha='right')
            ax.set_ylabel(label);ax.grid(axis='y',alpha=0.2)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle(f'{dataset} | Completed D-Pi experiments | mean and standard deviation across seeds')
        fig.tight_layout()
        for ext in ['svg','pdf','png']:
            fig.savefig(destination/f'{dataset}_task_concept.{ext}',dpi=180)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5))
        for variant,color in [('clean','#2563EB'),('mean_only','#EA580C'),('pi','#E11D48'),('pi_refit','#7C3AED')]:
            if (dataset,variant) not in groups:continue
            rows=groups[dataset,variant]
            ks=[p['concepts_per_class'] for p in rows[0][1]['pruning']]
            values=np.array([[p['accuracy']*100 for p in row[1]['pruning']] for row in rows])
            ax.plot(ks,values.mean(0),'-o',label=f'{variant} (n={len(rows)})',color=color)
            if len(rows)>1:
                ax.fill_between(ks,values.mean(0)-values.std(0,ddof=1),values.mean(0)+values.std(0,ddof=1),color=color,alpha=0.12)
        ax.set_xscale('log');ax.set_xlabel('Retained concepts per class');ax.set_ylabel('Test accuracy (%)')
        ax.set_title(f'{dataset} | Matched sparsity comparison');ax.legend();ax.grid(alpha=0.2)
        fig.tight_layout()
        for ext in ['svg','pdf','png']:fig.savefig(destination/f'{dataset}_sparsity.{ext}',dpi=180)
        plt.close(fig)
        corrupted=[r for r in subset if r.get('corruption_metrics')]
        if corrupted:
            cases=list(corrupted[0]['corruption_metrics'])
            fig,ax=plt.subplots(figsize=(13,5))
            for row in corrupted:
                means=[row['corruption_metrics'][case]['accuracy']['mean']*100 for case in cases]
                ax.plot(range(len(cases)),means,'-o',label=f"{row['variant']} (n={len(row['corruption_seeds'])})")
            ax.set_xticks(range(len(cases)),cases,rotation=55,ha='right')
            ax.set_ylabel('Test accuracy (%)');ax.set_title(f'{dataset} | Custom image corruption stress tests (not CIFAR-C)')
            ax.legend();ax.grid(alpha=0.2);fig.tight_layout()
            for ext in ['svg','pdf','png']:fig.savefig(destination/f'{dataset}_corruptions.{ext}',dpi=180)
            plt.close(fig)
        semantic=[r for r in subset if r.get('semantic_metrics',{}).get('attribute_macro_balanced_accuracy',{}).get('mean') is not None]
        if semantic:
            fig,ax=plt.subplots(figsize=(8,6))
            for row in semantic:
                concept=row['semantic_metrics']['attribute_macro_balanced_accuracy']
                task=row['metrics']['accuracy']
                coverage=row['semantic_metrics']['evaluated_attributes']['mean']
                ax.errorbar(concept['mean']*100,task['mean']*100,
                            xerr=(concept['std'] or 0)*100,yerr=(task['std'] or 0)*100,
                            fmt='o',capsize=3,label=f"{row['variant']} ({coverage:.0f}/312 attributes)")
            ax.set_xlabel('Attribute macro balanced accuracy (%)');ax.set_ylabel('Task accuracy (%)')
            ax.set_title(f'{dataset} | Grounded semantic evaluation');ax.legend(fontsize=8);ax.grid(alpha=0.2)
            fig.tight_layout()
            for ext in ['svg','pdf','png']:fig.savefig(destination/f'{dataset}_semantic_task.{ext}',dpi=180)
            plt.close(fig)
            fig,ax=plt.subplots(figsize=(8,5))
            for row in semantic:
                curves=[]
                for _,_,directory in groups[dataset,row['variant']]:
                    path=directory/'cub_concepts.json'
                    if path.exists():curves.append(json.loads(path.read_text())['interventions'])
                fractions=[p['fraction'] for p in curves[0]]
                accuracies=np.array([[p['accuracy']*100 for p in curve] for curve in curves])
                ax.plot(fractions,accuracies.mean(0),'-o',label=row['variant'])
                if len(curves)>1:
                    ax.fill_between(fractions,accuracies.mean(0)-accuracies.std(0,ddof=1),
                                    accuracies.mean(0)+accuracies.std(0,ddof=1),alpha=0.12)
            ax.set_xlabel('Fraction of attribute groups selected for correction');ax.set_ylabel('Task accuracy (%)')
            ax.set_title(f'{dataset} | Attribute interventions (known labels only)');ax.legend(fontsize=8);ax.grid(alpha=0.2)
            fig.tight_layout()
            for ext in ['svg','pdf','png']:fig.savefig(destination/f'{dataset}_interventions.{ext}',dpi=180)
            plt.close(fig)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='outputs/study_v1')
    parser.add_argument('--output',default='reports/study_v1')
    args=parser.parse_args()
    rows=summarize(args.root,args.output)
    print(f'Summarized {len(rows)} completed dataset/variant groups',flush=True)


if __name__=='__main__':main()
