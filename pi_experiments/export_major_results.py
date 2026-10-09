"""Export verified five-seed findings to the requested Essay/D/report directory."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .analyze import task_metrics
from .matched_nec import truncate_exact
from .statistics import hierarchical_paired_bootstrap


DEST = Path('../Essay/D/report')
VARIANTS = ['clean', 'fixed', 'mean_only', 'adapter_matched', 'pi']
MAIN = ['clean', 'mean_only', 'adapter_matched', 'pi']
LABELS = {'clean':'干净模型', 'fixed':'固定噪声', 'mean_only':'仅均值',
          'adapter_matched':'等容量适配器', 'pi':'Pi（MC8）',
          'same_checkpoint_mean':'同一 Pi 的均值推理', 'pi_no_preservation':'去掉概念保持',
          'pi_no_energy':'去掉能量约束', 'pi_concept_scores':'概念分数注入',
          'clean_refit':'干净重拟合头','pi_refit':'Pi 重拟合头（MC8）'}
COLORS = {'clean':'#616161', 'fixed':'#56B4E9', 'mean_only':'#E69F00',
          'adapter_matched':'#0072B2', 'pi':'#CC3377', 'same_checkpoint_mean':'#009E73',
          'pi_no_preservation':'#D55E00', 'pi_no_energy':'#009E73', 'pi_concept_scores':'#7B61A8',
          'clean_refit':'#616161','pi_refit':'#CC3377'}
MARKERS = {'clean':'o','fixed':'x','mean_only':'s','adapter_matched':'^','pi':'D','clean_refit':'s','pi_refit':'P'}
STYLES = {'clean':'-','mean_only':'--','adapter_matched':':','pi':'-.'}
SOURCES = {}
SNAPSHOTS = {}
VERIFIED = []
VERIFIED_INTERVALS = []
DATA = {}
SECTIONS = []
PACKS = {}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path)] = sha_bytes(data)
    SNAPSHOTS[(str(path),sha_bytes(data))] = data
    return json.loads(data)


def load(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path)] = sha_bytes(raw)
    return torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu')


def pack_reference(path):
    path = Path(path)
    key = str(path.resolve())
    if key not in PACKS:
        pack = torch.load(path, weights_only=True, map_location='cpu')
        PACKS[key] = {k:pack[k] for k in ['centers','mean','scale','weight','bias']}
        PACKS[key]['labels'] = pack['test']['labels']
        SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return PACKS[key]


def verified_task(root, pack_path, variants):
    ref = pack_reference(pack_path)
    groups = {}
    for variant in variants:
        records = []
        for seed in range(5):
            run = Path(root)/f'{variant}_seed{seed}'
            state = load(run/'model.pt')
            config = state['config']
            assert config['seed'] == seed
            refit = bool(config.get('refit_epochs',0))
            for tensor, original in [('projection.centers','centers'),('projection.mean','mean'),
                                     ('projection.scale','scale'),('head.weight','weight'),('head.bias','bias')]:
                if refit and tensor.startswith('head.'): continue
                assert torch.equal(state['state_dict'][tensor],ref[original]), (run,tensor)
            predictions = load(run/'predictions.pt')
            report = read(run/'analysis.json')
            assert report['evaluation_split']=='test'
            assert torch.equal(predictions['labels'],ref['labels'])
            for mode in ['mc8','mean']:
                probability = predictions[mode]
                assert torch.isfinite(probability).all() and (probability>=0).all()
                assert torch.allclose(probability.sum(-1),torch.ones(len(ref['labels'])),atol=1e-5)
                for key,value in task_metrics(probability,ref['labels']).items():
                    assert abs(value-report['inference'][mode][key])<1e-7,(run,mode,key)
            records.append(report['inference']['mc8'])
            VERIFIED.append({'checkpoint':str(run/'model.pt'),'projection_exact':True,
                             'frozen_head_exact':not refit,'prediction_metrics_reproduced':True,
                             'test_images':len(ref['labels'])})
        groups[variant] = {key:{'mean':float(np.mean([r[key] for r in records])),
                                 'std':float(np.std([r[key] for r in records],ddof=1))}
                           for key in records[0] if isinstance(records[0][key],(int,float))}
    return groups


def stats(values):
    return {'mean':float(np.mean(values)), 'std':float(np.std(values,ddof=1))}


def decorate(axis, ylabel):
    axis.set_ylabel(ylabel)
    axis.spines[['top','right']].set_visible(False)
    axis.grid(axis='y',color='#E6EAF0',linewidth=.7)
    axis.set_axisbelow(True)


def legend(fig, variants=MAIN):
    handles = [plt.Line2D([],[],color=COLORS[v],marker=MARKERS.get(v,'o'),lw=2,label=LABELS[v]) for v in variants]
    fig.legend(handles=handles, loc='outside lower center', ncol=min(len(handles),5), frameon=False)


def save(pdf, name, title, fig, purpose, conclusion, note, data):
    fig.suptitle(title,fontsize=16,fontweight='bold')
    for ext in ['png','svg','pdf']:
        fig.savefig(DEST/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    pdf.savefig(fig,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    DATA[name] = data
    SECTIONS.append({'file':name,'title':title,'purpose':purpose,'conclusion':conclusion,'note':note})
    print('Exported',name,flush=True)


def draw_method_points(axis, groups, metric='accuracy', variants=MAIN, factor=100, annotate=False):
    values = []
    for i,v in enumerate(variants):
        m = groups[v][metric]
        mean,error = m['mean']*factor,m['std']*factor
        axis.errorbar(i,mean,yerr=error,fmt=MARKERS.get(v,'o'),color=COLORS[v],markersize=7,capsize=4,lw=1.8)
        if annotate: axis.annotate(f'{mean:.3f}',(i,mean+error),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        values.extend([mean-error,mean+error])
    axis.set_xticks(range(len(variants)),[LABELS[v] for v in variants],rotation=25,ha='right')
    span = max(max(values)-min(values),.1)
    axis.set_ylim(min(values)-span*.18,max(values)+span*.4)


def interval(axis, row, y, color, key='accuracy_delta'):
    point = row[key]*100
    lo,hi = np.array(row['ci95'])*100
    axis.plot([lo,hi],[y,y],color=color,lw=2)
    axis.plot([lo,lo],[y-.07,y+.07],color=color,lw=1.4)
    axis.plot([hi,hi],[y-.07,y+.07],color=color,lw=1.4)
    axis.scatter([point],[y],color=color,s=35,zorder=3)


def curve(axis, x, y, error, variant):
    axis.errorbar(x,y,yerr=error,color=COLORS[variant],marker=MARKERS.get(variant,'o'),
                  linestyle=STYLES.get(variant,'-'),markersize=8 if variant=='mean_only' else 6,
                  markerfacecolor='none' if variant=='pi' else COLORS[variant],
                  markeredgewidth=1.4,capsize=3,linewidth=1.7)


def find_contrasts(summary,dataset):
    rows = [r for r in summary['pi_hierarchical_comparisons'] if r['dataset']==dataset]
    rows = {r['control']:r for r in rows}
    for r in rows.values(): assert r['seeds']==list(range(5)) and r['adaptation_seeds']==5
    return rows


def verify_contrasts(root, rows):
    predictions = {v:[load(Path(root)/f'{v}_seed{s}'/'predictions.pt') for s in range(5)]
                   for v in VARIANTS}
    labels = predictions['pi'][0]['labels'].numpy()
    adapted = np.stack([r['mc8'].numpy() for r in predictions['pi']])
    for control,row in rows.items():
        records = predictions['pi'] if control=='same_checkpoint_mean' else predictions[control]
        mode = 'mean' if control=='same_checkpoint_mean' else 'mc8'
        assert all(np.array_equal(r['labels'].numpy(),labels) for r in records)
        result = hierarchical_paired_bootstrap(np.stack([r[mode].numpy() for r in records]),adapted,labels,
                                              seed=row['seed'],draws=row['draws'])
        assert abs(result['accuracy_delta']-row['accuracy_delta'])<1e-12
        assert np.allclose(result['ci95'],row['ci95'],rtol=0,atol=1e-12)
        VERIFIED_INTERVALS.append({'root':str(root),'control':control,'draws':row['draws'],
                                   'paired_interval_reproduced':True})


def main():
    torch.set_num_threads(8)
    DEST.mkdir(parents=True,exist_ok=True)
    font = Path('artifacts/report_assets/NotoSansCJKsc-Regular.otf')
    assert font.exists(),'A Chinese font is required for the exported figures'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path',
                         'pdf.fonttype':42,'axes.titleweight':'bold','figure.facecolor':'white'})
    specs = [
        ('RN50 / CIFAR-10','outputs/study_v1/cifar10','artifacts/datasets/cifar10/baseline_k256_seed0.pt','reports/study_v1/results.json','cifar10'),
        ('RN50 / CIFAR-100','outputs/study_v1/cifar100','artifacts/datasets/cifar100/baseline_k2048_seed0.pt','reports/study_v1/results.json','cifar100'),
        ('RN50 / CUB-GDINO','outputs/gdino_v1/cub','artifacts/datasets/cub_gdino/baseline_k2048_seed0.pt','reports/gdino_v1/results.json','cub'),
        ('ViT-B/16 / CIFAR-10','outputs/backbone_v1/ViT-B16/cifar10','artifacts/backbone_ablation_v1/ViT-B16/cifar10/baseline_k256_seed0.pt','reports/backbone_v1/ViT-B16/results.json','cifar10'),
        ('ViT-L/14 / CIFAR-10','outputs/backbone_v1/ViT-L14/cifar10','artifacts/backbone_ablation_v1/ViT-L14/cifar10/baseline_k256_seed0.pt','reports/backbone_v1/ViT-L14/results.json','cifar10')]
    candidates = [('RN50 / CUB-SAM2','outputs/study_v1/cub','artifacts/datasets/cub/baseline_k2048_seed0.pt',
                   'reports/study_v1/results.json','cub')]
    for dataset in ['cifar100','cub']:
        for slug,label in [('ViT-B16','ViT-B/16'),('ViT-L14','ViT-L/14')]:
            dataset_label='CIFAR-100' if dataset=='cifar100' else 'CUB-SAM2'
            candidates.append((f'{label} / {dataset_label}',f'outputs/backbone_v1/{slug}/{dataset}',
                               f'artifacts/backbone_ablation_v1/{slug}/{dataset}/baseline_k2048_seed0.pt',
                               f'reports/backbone_v1/{slug}/results.json',dataset))
    pending_core = []
    for spec in candidates:
        name,root,pack,summary,dataset = spec
        paths = [Path(root)/f'{v}_seed{s}'/file for v in VARIANTS for s in range(5)
                 for file in ['model.pt','predictions.pt','analysis.json']]
        if not Path(summary).exists() or not Path(pack).exists() or not all(p.exists() for p in paths):
            pending_core.append(name);continue
        original = read(summary)
        complete = all(any(g['dataset']==dataset and g['variant']==v and g['seeds']==list(range(5))
                           for g in original['groups']) for v in VARIANTS)
        if not complete or len(find_contrasts(original,dataset))!=5:
            pending_core.append(name);continue
        specs.append(spec)
    core = []
    for name,root,pack,summary,dataset in specs:
        print('Verifying',name,flush=True)
        groups = verified_task(root,pack,VARIANTS)
        original = read(summary)
        for v in VARIANTS:
            row = next(g for g in original['groups'] if g['dataset']==dataset and g['variant']==v)
            assert row['seeds']==list(range(5))
            for key in ['accuracy','nll','ece_15_bins','standardized_concept_mse']:
                assert abs(row['metrics'][key]['mean']-groups[v][key]['mean'])<1e-10
        contrasts = find_contrasts(original,dataset)
        verify_contrasts(root,contrasts)
        core.append({'case':name,'groups':groups,'contrasts':contrasts})
    with PdfPages(DEST/'主要实验图表合集.pdf') as pdf:
        columns = 4 if len(core) in [7,8] else min(3,len(core));nrows = (len(core)+columns-1)//columns
        fig,axes = plt.subplots(nrows,columns,figsize=(5.4*columns,5*nrows),layout='constrained',squeeze=False)
        for axis,case in zip(axes.flat,core):
            draw_method_points(axis,case['groups'],variants=VARIANTS,annotate=True)
            axis.set_title(case['case'].replace(' / ','\n'))
            decorate(axis,'测试准确率（%；纵轴局部放大）')
        for axis in list(axes.flat)[len(core):]:axis.set_visible(False)
        legend(fig,VARIANTS)
        accuracies = '；'.join(f"{r['case']}：{r['groups']['clean']['accuracy']['mean']*100:.3f}% → {r['groups']['pi']['accuracy']['mean']*100:.3f}%" for r in core)
        save(pdf,'01_task_accuracy','分类性能：跨数据集与骨干的核心五种子对照',fig,
             '检验学习式适配能否改善分类，并与固定噪声、仅均值和等容量适配器比较。',
             accuracies+f'。已完成的 {len(core)} 组中 Pi 均改善干净参考；固定噪声没有接近学习式适配的提升。',
             '点为五个适配种子的均值，误差线为样本标准差（SD），不是置信区间。每个小图独立放大纵轴，禁止跨图比较视觉高度。共享原型和初始头的干净模型是确定性重复。',core)

        fig,axes = plt.subplots(1,2,figsize=(14,max(6.5,len(core)*.85+2)),layout='constrained')
        for i,case in enumerate(core):
            interval(axes[0],case['contrasts']['clean'],i,COLORS['pi'])
            for j,c in enumerate(['mean_only','adapter_matched','same_checkpoint_mean']):
                interval(axes[1],case['contrasts'][c],i+(j-1)*.2,COLORS[c])
        for axis in axes:
            axis.axvline(0,color='#353B45',ls='--',lw=1)
            axis.set_yticks(range(len(core)),[r['case'] for r in core]);axis.invert_yaxis()
            axis.set_xlabel('Pi − 对照：准确率差（百分点）');axis.grid(axis='x',alpha=.2)
            axis.spines[['top','right']].set_visible(False)
        axes[0].set_title('Pi 相对干净模型');axes[1].set_title('随机性与容量：细尺度配对比较')
        legend(fig,['mean_only','adapter_matched','same_checkpoint_mean'])
        positive_clean=sum(c['contrasts']['clean']['ci95'][0]>1e-12 for c in core)
        detailed_controls=['mean_only','adapter_matched','same_checkpoint_mean']
        zero_controls=all(c['contrasts'][control]['ci95'][0]<=1e-12 and c['contrasts'][control]['ci95'][1]>=-1e-12
                          for c in core for control in detailed_controls)
        control_findings=('Pi−仅均值、Pi−等容量以及 MC8−同检查点均值的区间均包含零或触及零边界。'
                          if zero_controls else '各确定性与同检查点推理对照的差异见右图；不将不同对照混为单一随机噪声效果。')
        positive_deterministic=[(c['case'],control) for c in core for control in ['mean_only','adapter_matched']
                                if c['contrasts'][control]['ci95'][0]>1e-12]
        core_finding=('当前结果支持适配有效，尚未证明随机噪声额外提高分类准确率；这也不是等价性证明。'
                      if not positive_deterministic else
                      '部分 Pi−独立确定性对照的描述性区间高于零；是否由随机采样带来，需结合同检查点均值/尺度干预，不能只凭训练方案差异作因果归因。')
        boundaries = [c for c in core if abs(c['contrasts']['same_checkpoint_mean']['ci95'][0])<1e-12
                      and c['contrasts']['same_checkpoint_mean']['accuracy_delta']>1e-12]
        boundary_note = ''.join(f"{c['case']} 的 MC8−同检查点均值为 {c['contrasts']['same_checkpoint_mean']['accuracy_delta']*100:+.3f} 个百分点，区间下界贴近零，不能视为稳健正增益。" for c in boundaries)
        save(pdf,'02_randomness_controls','适配收益与随机噪声收益的区分',fig,
             '通过独立训练的确定性对照及同一 Pi 检查点的均值推理，检验随机分布是否带来额外准确率收益。',
             f'{positive_clean}/{len(core)} 组 Pi−干净参考的 95% 区间高于零。'+control_findings+boundary_note+core_finding,
             '误差线为 2,000 次配对适配种子与共同测试图的分层 bootstrap 95% 区间，条件于固定原型和初始头；区间未作多重比较校正。两个面板的横轴尺度不同。',core)

        semantic_summary = read('reports/visible_semantics_v1/results.json')
        semantic_ci = read('reports/semantic_uncertainty_v1/results.json')
        semantics = {}
        for v in MAIN:
            row = next(r for r in semantic_summary['groups'] if r['bank']=='RN50_GDINO' and r['variant']==v)
            assert row['seeds']==list(range(5))
            results = []
            for s in range(5):
                run = Path('outputs/visible_semantics_v1/RN50_GDINO')/f'{v}_seed{s}'
                r = read(run/'cub_concepts.json');p = load(run/'semantic_predictions.pt')
                assert r['evaluated_attributes']==139 and r['min_certainty']==3
                assert r['source_checkpoint_sha256']==SOURCES[f'outputs/gdino_v1/cub/{v}_seed{s}/model.pt']
                scores = p['scores'].numpy() if 'scores' in p else None
                assert scores is not None and scores.shape==(5794,139) and np.isfinite(scores).all()
                assert p['metadata']['attribute_report_sha256']==SOURCES[str(run/'cub_concepts.json')]
                assert p['metadata']['checkpoint_sha256']==r['source_checkpoint_sha256']
                assert torch.equal(p['predicted'],p['scores']>=p['thresholds'])
                positive,known=p['positive'],p['known'];predicted=p['predicted']
                pos,neg=positive&known,~positive&known
                actual=.5*((predicted&pos).sum(0).double()/pos.sum(0)+(~predicted&neg).sum(0).double()/neg.sum(0)).mean().item()
                auroc=float(np.mean([roc_auc_score(positive[:,i][known[:,i]].numpy(),scores[:,i][known[:,i].numpy()]) for i in range(139)]))
                assert abs(actual-r['attribute_macro_balanced_accuracy'])<1e-10
                assert abs(auroc-r['attribute_macro_auroc'])<1e-10
                results.append(r)
            semantics[v] = row['metrics']
            for key in ['attribute_macro_balanced_accuracy','attribute_macro_auroc']:
                assert abs(semantics[v][key]['mean']-np.mean([r[key] for r in results]))<1e-12
        srows = [r for r in semantic_ci['comparisons'] if r['bank']=='RN50_GDINO' and r['control'] in MAIN[:-1]]
        assert len(srows)==3
        fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
        for i,key in enumerate(['attribute_macro_balanced_accuracy','attribute_macro_auroc']):
            for j,v in enumerate(MAIN):
                m=semantics[v][key];axes[0].errorbar(i+(j-1.5)*.13,m['mean']*100,yerr=m['std']*100,
                                                   fmt=MARKERS[v],color=COLORS[v],capsize=4,markersize=7)
        axes[0].set_xticks([0,1],['Macro balanced accuracy','Macro AUROC'])
        decorate(axes[0],'属性指标（%；纵轴局部放大）')
        for i,r in enumerate(srows):interval(axes[1],r,i,COLORS[r['control']],key='balanced_accuracy_delta')
        axes[1].set_yticks(range(3),['Pi − '+LABELS[r['control']] for r in srows]);axes[1].invert_yaxis()
        axes[1].axvline(0,ls='--',color='gray');axes[1].set_xlabel('Balanced accuracy 差（百分点）；95% 区间')
        axes[1].grid(axis='x',alpha=.2);legend(fig)
        sc=next(r for r in srows if r['control']=='clean')
        gdino_attribute_audit=None
        gdino_attribute_path=Path('reports/attribute_evidence_verification_v1/RN50_GDINO.json')
        if gdino_attribute_path.exists():
            gdino_attribute_audit=read(gdino_attribute_path)
            assert len(gdino_attribute_audit['verified_checkpoints'])==20 and gdino_attribute_audit['evaluated_attributes']==139
            for source,digest in gdino_attribute_audit['source_sha256'].items():
                actual=sha_bytes(Path(source).read_bytes())
                assert actual==digest,(source,'audited GDINO attribute source changed')
                SOURCES[source]=actual
            for v in MAIN:
                for key in ['attribute_macro_balanced_accuracy','attribute_macro_auroc']:
                    assert abs(semantics[v][key]['mean']-gdino_attribute_audit['groups'][v][key]['mean'])<1e-12
            assert srows==gdino_attribute_audit['comparisons']
        save(pdf,'03_grounded_attributes','CUB-GDINO 真实属性：可见支持映射的五种子结果',fig,
             '用训练标注支撑、实际 CLIP 裁剪仍可见的原型命名，检验任务提升是否伴随真实属性改善。',
             f"Pi 的 balanced accuracy 为 {semantics['pi']['attribute_macro_balanced_accuracy']['mean']*100:.3f}%，相对干净模型 +{sc['balanced_accuracy_delta']*100:.3f} 个百分点，95% 区间 [{sc['ci95'][0]*100:.3f}, {sc['ci95'][1]*100:.3f}]；相对仅均值和等容量对照的区间均包含零。",
             '主置信度 certainty≥3；映射 148/312、可评估 139/312 属性，测试图 5,794 张。左图误差线为 SD，右图为配对分层 95% 区间。固定验证阈值；未知标签排除。属性覆盖与旧映射不同，因此不混用旧版宏平均。',{'groups':semantics,'contrasts':srows,'native_attribute_verification':gdino_attribute_audit})

        loc = {};retrieval={};interventions={};bg={}
        for v in MAIN:
            lrows=[];rrows=[];brows=[];irows=[]
            for s in range(5):
                run=Path('outputs/visible_semantics_v1/RN50_GDINO')/f'{v}_seed{s}'
                l=read(run/'localization.json');r=read(run/'region_retrieval.json')
                b=read(Path('outputs/visible_natural_background_v1/RN50_GDINO')/run.name/'backgrounds.json')
                i=read(run/'cub_concepts.json')['interventions']
                assert l['evaluated_images']==l['test_images']==5794 and not l['debug_limit']
                assert r['region_images']==5794 and r['regions']==27619 and r['evaluated_queries']==1647
                assert b['images']==5794 and b['evaluation_split']=='test'
                lrows.append(l);rrows.append(r);brows.append(b);irows.append(i)
            loc[v]={
                'grid_attribution':stats([r['grid_all_concepts_mean']['relative_attribution'] for r in lrows]),
                'grid_point':stats([r['grid_all_concepts_mean']['max_point_accuracy'] for r in lrows]),
                'foreground_attribution':stats([r['foreground_all_candidate_mean']['foreground_attribution'] for r in lrows]),
                'part_hit':stats([r['part_hit_rate_all_localizable'] for r in lrows]),
                'part_localizable_concepts':stats([r['part_localizable_concepts'] for r in lrows]),
                'grid_plan_sha256':lrows[0]['grid_plan_sha256']}
            assert all(r['grid_plan_sha256']==loc[v]['grid_plan_sha256'] for r in lrows)
            retrieval[v]={key:stats([r[key] for r in rrows]) for key in ['precision_at_5_macro','precision_at_5_distinct_images_macro','chance_precision_macro']}
            bg[v]={case['case']:{key:stats([r['cases'][j]['mc8'][key] for r in brows]) for key in ['accuracy','nll']}
                   for j,case in enumerate(brows[0]['cases'])}
            interventions[v]={'fractions':[r['fraction'] for r in irows[0]],
                              'accuracy':[stats([r[j]['accuracy'] for r in irows]) for j in range(len(irows[0]))]}
        assert len({loc[v]['grid_plan_sha256'] for v in MAIN})==1
        fig,axes=plt.subplots(1,4,figsize=(16,5.3),layout='constrained')
        for axis,key,label in zip(axes,['grid_attribution','grid_point','foreground_attribution','part_hit'],
                                  ['网格：目标象限归因','网格：最大响应落点','分割：鸟体内归因','部位：0.1 对角线命中']):
            draw_method_points(axis,loc,key,annotate=True);axis.set_title(label);decorate(axis,'比例（%；纵轴局部放大）')
        legend(fig)
        save(pdf,'04_localization','CUB 定位：相同干净 top-5 概念与完整测试集',fig,
             '检验适配是否改善概念空间定位，并区分任务准确率与解释定位质量。',
             f"Pi 的目标象限归因为 {loc['pi']['grid_attribution']['mean']*100:.3f}%（干净 {loc['clean']['grid_attribution']['mean']*100:.3f}%），鸟体内归因为 {loc['pi']['foreground_attribution']['mean']*100:.3f}%；三种学习式适配数值接近。定位变化幅度远小于分类提升，图中 SD 不能证明定位差异显著。",
             '正式 5,794 图，固定干净 top-5、共同网格；零热图按全候选口径计分，部位按全部可定位概念计分。采用可见支持映射过滤部位；网格、前景和原 CAM 保留。误差线为 SD。',loc)

        fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
        for i,key in enumerate(['precision_at_5_macro','precision_at_5_distinct_images_macro']):
            for j,v in enumerate(MAIN):
                m=retrieval[v][key];axes[0].errorbar(i+(j-1.5)*.14,m['mean']*100,yerr=m['std']*100,
                                                    fmt=MARKERS[v],color=COLORS[v],capsize=4,markersize=7)
        chance=retrieval['clean']['chance_precision_macro']['mean']*100
        axes[0].axhline(chance,color='gray',ls='--',label=f'机会精度 {chance:.2f}%')
        axes[0].set_xticks([0,1],['区域 P@5','不同图像 P@5']);decorate(axes[0],'标注代理精度（%）');axes[0].legend(frameon=False)
        for v in MAIN:
            xs=interventions[v]['fractions'];means=np.array([m['mean'] for m in interventions[v]['accuracy']])*100
            errors=np.array([m['std'] for m in interventions[v]['accuracy']])*100
            curve(axes[1],np.array(xs)*100,means,errors,v)
        axes[1].set_xlabel('选择纠正的属性组比例（%）');decorate(axes[1],'均值概念推理：分类准确率（%）')
        axes[1].set_title('训练正负参考下的属性干预');legend(fig)
        retrieval_audit=None
        retrieval_audit_path=Path('reports/retrieval_evidence_verification_v1/RN50_GDINO.json')
        if retrieval_audit_path.exists():
            retrieval_audit=read(retrieval_audit_path)
            assert len(retrieval_audit['verified_checkpoints'])==25
            assert retrieval_audit['verified_query_selections']==41175
            for source,digest in retrieval_audit['source_sha256'].items():
                actual=sha_bytes(Path(source).read_bytes())
                assert actual==digest,(source,'retrieval annotation source changed')
                SOURCES[source]=actual
            for v in MAIN:
                for key in retrieval[v]:
                    assert abs(retrieval[v][key]['mean']-retrieval_audit['groups'][v][key]['mean'])<1e-12
        additional_retrieval_audits={}
        sam_retrieval_path=Path('reports/retrieval_evidence_verification_v1/RN50_SAM2.json')
        sam_retrieval_note=''
        if sam_retrieval_path.exists():
            sam_retrieval_audit=read(sam_retrieval_path)
            assert len(sam_retrieval_audit['verified_checkpoints'])==25
            assert sam_retrieval_audit['native_test_images_verified']==5794
            for source,digest in sam_retrieval_audit['source_sha256'].items():
                actual=sha_bytes(Path(source).read_bytes())
                assert actual==digest,(source,'SAM2 retrieval annotation source changed')
                SOURCES[source]=actual
            additional_retrieval_audits['RN50_SAM2']=sam_retrieval_audit
            sam_retrieval_note=(f"另已核验SAM2全部25模型、{sam_retrieval_audit['verified_query_selections']}组保存选择；"
                                f"其Pi/干净区域P@5为{sam_retrieval_audit['groups']['pi']['precision_at_5_macro']['mean']*100:.3f}/"
                                f"{sam_retrieval_audit['groups']['clean']['precision_at_5_macro']['mean']*100:.3f}%。"
                                '两库查询集合不同，不能跨库比较绝对P@5。')
        save(pdf,'05_retrieval_intervention','概念解释：标注区域检索与属性干预',fig,
             '从区域检索和属性干预两条路径检查概念解释，避免只用任务准确率评价解释有效性。',
             f"图中GDINO Pi 的区域 P@5 为 {retrieval['pi']['precision_at_5_macro']['mean']*100:.3f}%，干净模型为 {retrieval['clean']['precision_at_5_macro']['mean']*100:.3f}%；学习式对照接近。用训练正负参考纠正属性并未形成随纠正比例增加而单调提升的分类效果，因此不能把现有概念当作可靠的可编辑因果变量。"+sam_retrieval_note,
             '图示GDINO检索池 27,619 区域、5,794 图、1,647 个命名查询。P@5 是属性与可见部位标注代理，非人工语义判断；共用 GDINO 区域池有后端偏向。保存的前五区域及其标注精度已独立核验，不重新计算模型排序。干预使用已知测试属性标签、训练参考和均值概念，属于解释诊断。误差线为 SD。',{'retrieval':retrieval,'interventions':interventions,'native_selection_grounding_audit':retrieval_audit,'additional_native_selection_grounding_audits':additional_retrieval_audits})

        fig,axes=plt.subplots(1,2,figsize=(13,5.6),layout='constrained')
        names=list(bg['clean']);x=np.arange(len(names))
        for v in MAIN:
            means=np.array([bg[v][n]['accuracy']['mean'] for n in names])*100
            errors=np.array([bg[v][n]['accuracy']['std'] for n in names])*100
            curve(axes[0],x,means,errors,v)
            drops=means[0]-means[1:]
            curve(axes[1],x[1:],drops,None,v)
        ticks=['原图','灰背景','训练供体 17','训练供体 29','训练供体 43']
        axes[0].set_xticks(x,ticks,rotation=20);axes[1].set_xticks(x[1:],ticks[1:],rotation=20)
        decorate(axes[0],'分类准确率（%）');decorate(axes[1],'相对该方法原图的下降（百分点）')
        axes[0].set_title('鸟体保持，改变背景');axes[1].set_title('剩余背景敏感性');legend(fig)
        save(pdf,'06_background_stress','CUB 背景压力测试：训练照片供体与固定鸟体',fig,
             '在多数前景像素不变时改变背景，检验适配后模型的背景敏感性。',
             f"Pi 在原图为 {bg['pi']['identity']['accuracy']['mean']*100:.3f}%，三个训练供体背景为 "+
             '/'.join(f"{bg['pi'][n]['accuracy']['mean']*100:.3f}%" for n in names[2:])+
             '。适配模型在改背景后优于干净模型，但仍出现明显下降；Pi 与确定性对照接近，尚未证明随机尺度带来独立背景鲁棒性收益。',
             '使用训练图中的无鸟体照片块、固定供体种子 17/29/43，完整 5,794 图；不是自然分布的无偏估计。缩放和拼接仍属合成变换。左图误差线为 SD；右图只显示平均下降，不提供区间。',bg)

        components={}
        comp_variants=['pi','pi_no_preservation','pi_no_energy','pi_concept_scores']
        component_specs=list(specs[:2])
        cub_spec=next((spec for spec in specs if spec[0]=='RN50 / CUB-SAM2'),None)
        if cub_spec and all((Path(cub_spec[1])/f'{v}_seed{s}'/file).exists()
                            for v in comp_variants for s in range(5) for file in ['model.pt','analysis.json','predictions.pt']):
            component_specs.append(cub_spec)
        for name,root,pack,_,dataset in component_specs:
            components[dataset]=verified_task(root,pack,comp_variants)
        fig,axes=plt.subplots(len(components),2,figsize=(13,4.5*len(components)),layout='constrained')
        for row,d in enumerate(components):
            for axis,key,label in zip(axes[row],['accuracy','standardized_concept_mse'],['分类准确率（%；局部放大）','标准化概念漂移 MSE（对数轴）']):
                draw_method_points(axis,components[d],key,comp_variants,factor=100 if key=='accuracy' else 1,annotate=key=='accuracy')
                if key.endswith('mse'):axis.set_yscale('log');axis.set_ylim(bottom=min(components[d][v][key]['mean'] for v in comp_variants)*.6)
                axis.set_title('CUB / SAM2（参考 λe=0）' if d=='cub' else d.upper());decorate(axis,label)
        legend(fig,comp_variants)
        ratio=components['cifar10']['pi_no_preservation']['standardized_concept_mse']['mean']/components['cifar10']['pi']['standardized_concept_mse']['mean']
        cub_component_note=(' CUB/SAM2 去掉概念保持后的漂移约为参考的 '+f"{components['cub']['pi_no_preservation']['standardized_concept_mse']['mean']/components['cub']['pi']['standardized_concept_mse']['mean']:.2f}"+' 倍，分类准确率仍接近。' if 'cub' in components else '')
        save(pdf,'07_loss_components','损失与注入组件：任务性能不能替代概念保持',fig,
             '检验概念保持、能量约束和注入位置的作用，同时观察分类与概念漂移。',
             f'CIFAR-10 去掉概念保持后的漂移约为参考 Pi 的 {ratio:.2f} 倍，分类准确率却接近。'+cub_component_note+'概念保持约束具有保持原响应的作用；高任务准确率本身不能证明概念解释被保持。漂移不是属性识别准确率。',
             f"{len(components)} 个数据集均为各方法五种子；误差线为 SD。漂移纵轴为对数；CUB 参考验证选定 λe=0，去掉能量项与参考配置相同，不能作为独立能量项效果。概念分数注入的输出宽度为 K，特征注入为 1024，生成器容量可能不同，位置差异不能单独作因果归因。不混入存在初始化混杂的旧随机原型版本，也不把不同参数量的 MLP 当作同容量对照。",components)

        sampling=read('reports/sampling_budget_v1/paired_results.json')
        assert len(sampling['groups'])>=5
        fig,axes=plt.subplots(1,2,figsize=(14,6.6),layout='constrained')
        budgetcolors=['#0072B2','#E69F00','#009E73','#CC3377']
        sampling_values=[]
        for i,g in enumerate(sampling['groups']):
            assert g['seeds']==list(range(5))
            comparisons=[r for r in g['comparisons'] if r['control']=='mean']
            assert len(comparisons)==4
            for j,r in enumerate(comparisons): interval(axes[0],r,i+(j-1.5)*.16,budgetcolors[j])
            sampling_values.append(g)
        axes[0].set_yticks(range(len(sampling_values)),[g['case'] for g in sampling_values]);axes[0].invert_yaxis()
        axes[0].axvline(0,ls='--',color='gray');axes[0].set_xlabel('MCn − 同检查点均值：准确率差（百分点）')
        axes[0].set_title('准确率：颜色区分采样次数')
        axes[0].grid(axis='x',alpha=.2)
        sr=read('reports/sampling_budget_v1/results.json')
        for i,g in enumerate(sr['groups']):
            if g['seeds']!=list(range(5)):continue
            axes[1].plot([1,4,8,16],[g['budgets'][str(n)]['nll']['mean']-g['mean']['nll']['mean'] for n in [1,4,8,16]],
                         '-o',label=g['case'],color=['#616161','#0072B2','#E69F00','#CC3377','#009E73','#56B4E9','#882255'][i%7])
        axes[1].axhline(0,ls='--',color='gray');axes[1].set_xscale('log',base=2);axes[1].set_xticks([1,4,8,16],[1,4,8,16])
        axes[1].set_xlabel('前缀耦合采样次数');decorate(axes[1],'NLL − 同检查点均值（越低越好）');axes[1].legend(fontsize=8,frameon=False)
        axes[1].set_title('NLL：颜色区分数据集与骨干')
        fig.legend(handles=[plt.Line2D([],[],color=c,marker='o',label=f'MC{n}') for c,n in zip(budgetcolors,[1,4,8,16])],
                   loc='outside lower center',ncol=4,frameon=False)
        allzero=all(r['ci95'][0]<=0<=r['ci95'][1] for g in sampling_values for r in g['comparisons'] if r['control']=='mean')
        save(pdf,'08_sampling_budget','采样预算：前缀耦合的 MC1/4/8/16 比较',fig,
             '固定已选 Pi 检查点，用同一组采样前缀隔离推理预算的作用。',
             (f'已完成 {len(sampling_values)} 组中，各预算相对均值推理的准确率区间均包含零。' if allzero else '已完成组的预算相对均值差异及区间见左图；不根据测试结果选择预算。')+
             '当前未形成增加采样次数即可稳定提高分类准确率的证据；NLL 的变化另列，不能用准确率结论替代概率质量指标。',
             '左图为 2,000 次分层配对 95% 区间，右图为均值差，无区间。每批次同一 16 次采样构造前缀，种子固定；条件于一次采样实现。计算次数不是实测端到端延迟。',{'paired':sampling,'budget_metrics':sr})

        internal=verified_task('outputs/internal_v2/cifar10','artifacts/datasets/cifar10/baseline_k256_seed0.pt',MAIN)
        ints=read('reports/internal_v2/results.json');ic=find_contrasts(ints,'cifar10')
        fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
        for i,(name,g) in enumerate([('全局嵌入',core[0]['groups']),('内部卷积后',internal)]):
            for j,v in enumerate(MAIN):
                m=g[v]['accuracy'];axes[0].errorbar(i+(j-1.5)*.14,m['mean']*100,yerr=m['std']*100,fmt=MARKERS[v],color=COLORS[v],capsize=4,markersize=7)
        axes[0].set_xticks([0,1],['全局嵌入\n1,575,040 参数','卷积输出→池化\n1,839,232 参数']);decorate(axes[0],'分类准确率（%）')
        for j,c in enumerate(['mean_only','adapter_matched','same_checkpoint_mean']):interval(axes[1],ic[c],j,COLORS[c])
        axes[1].set_yticks(range(3),['Pi − '+LABELS[c] for c in ['mean_only','adapter_matched','same_checkpoint_mean']]);axes[1].invert_yaxis()
        axes[1].axvline(0,ls='--',color='gray');axes[1].set_xlabel('内部注入：准确率差（百分点）；95% 区间');legend(fig)
        save(pdf,'09_internal_injection','内部注入：位置、容量与随机性',fig,
             '在冻结 RN50 的末卷积输出与注意力池化之间注入，检验全局嵌入外的位置作用。',
             f"扩展验证搜索后的内部 Pi 为 {internal['pi']['accuracy']['mean']*100:.3f}%，干净为 {internal['clean']['accuracy']['mean']*100:.3f}%；与仅均值、等容量及同检查点均值的差异区间均包含零。跨位置参数量不同，不能将内部与全局的小幅差异单独归因于注入位置。",
             '使用 internal_v2 完成的五种子，零噪声池化重建已核验。两个位置各自验证选模；误差线左为 SD、右为 95% 区间。',{'groups':internal,'contrasts':ic})

        config=read('artifacts/concept_ablation_v1/cifar10_cases.json')['cases']
        concept_summary=read('reports/concept_study_v1/cifar10/results.json')
        complete={g['dataset'] for g in concept_summary['groups'] if g['variant']=='pi' and g['seeds']==list(range(5))}
        concept={}
        for c in config:
            if c['name'] not in complete:continue
            root=Path('outputs/concept_study_v1/cifar10')/c['name']
            if not all((root/f'{v}_seed{s}/analysis.json').exists() for v in MAIN for s in range(5)):continue
            concept[c['name']]={'images_per_class':c['images_per_class'],'clusters':c['clusters'],
                                'groups':verified_task(root,Path(c['directory'])/f'baseline_k{c["clusters"]}_seed0.pt',MAIN)}
        fig,axes=plt.subplots(1,2,figsize=(13,5.6),layout='constrained')
        for axis,key,condition,label in [(axes[0],'images_per_class',lambda r:r['clusters']==64,'每类训练源图数（K=64）'),
                                         (axes[1],'clusters',lambda r:r['images_per_class']==50,'概念数 K（每类 50 源图）')]:
            rows=sorted([r for r in concept.values() if condition(r)],key=lambda r:r[key])
            for v in MAIN:
                curve(axis,[r[key] for r in rows],[r['groups'][v]['accuracy']['mean']*100 for r in rows],
                      [r['groups'][v]['accuracy']['std']*100 for r in rows],v)
            axis.set_xscale('log',base=2);axis.set_xticks([r[key] for r in rows],[r[key] for r in rows]);axis.set_xlabel(label)
            decorate(axis,'分类准确率（%）')
        legend(fig)
        pending_concept_cases=[c['name'] for c in config if c['name'] not in concept]
        pending_concept_note=('尚未完整的配置：'+', '.join(pending_concept_cases)+'；图中不补值。'
                              if pending_concept_cases else '预设概念构建配置均已齐全。')
        save(pdf,'10_concept_bank','概念构建：嵌套源图数与概念数量',fig,
             '分别固定 K 和训练源图数，检查构建条件对分类性能与适配收益的影响。',
             f'当前展示 {len(concept)} 个完整五种子配置。较少概念/源图时干净性能较低，三种学习式适配明显接近；不能把这一收益单独归因于噪声。{pending_concept_note}不根据部分结果断言随 K 单调变化。',
             '源图类内严格嵌套；全部配置沿用参考验证选定正则，不是逐配置最优。误差线为 SD；两个面板共享 m=50/K=64，m=50/K=256 复用原模型，不是新的独立重复。',concept)

        sparse=read('reports/proximal_sparsity_v2/cifar10/results.json')
        exact=read('reports/matched_nec_v2/cifar10/results.json')
        sparse_selected_path=Path('reports/sparse_capacity_v1/cifar10/selected_head_comparisons.json')
        sparse_selected=read(sparse_selected_path) if sparse_selected_path.exists() else None
        sp={}
        for v in ['clean','pi']:
            acc=[];nec=[];heads=[]
            for s in range(5):
                root=Path('outputs/proximal_sparsity_v2/cifar10')/f'{v}_seed{s}'
                h=load(root/'heads.pt');r=read(root/'head_grid.json');p=load(root/'predictions.pt')
                selected=h['selection']['selected_index'];weight=h['weights'][selected]
                assert torch.equal(p['labels'],PACKS[str(Path('artifacts/datasets/cifar10/baseline_k256_seed0.pt').resolve())]['labels'])
                assert abs(task_metrics(p['mc8'],p['labels'])['accuracy']-r['selected']['mc8']['accuracy'])<1e-7
                acc.append(r['selected']['mc8']['accuracy']);nec.append(float(weight.ne(0).sum(1).float().mean()))
                eroot=Path('outputs/matched_nec_v2/cifar10')/root.name;e=read(eroot/'metrics.json');ep=load(eroot/'predictions.pt')
                assert exact['protocol']['head_sha256'][root.name]==SOURCES[str(root/'heads.pt')]
                for i,k in enumerate(exact['protocol']['targets']):
                    truncate_exact(weight,k)
                    assert abs(task_metrics(ep[str(k)],ep['labels'])['accuracy']-e['points'][i]['accuracy'])<1e-7
                heads.append(e)
            sp[v]={'accuracy':stats(acc),'strict_nec':stats(nec),'exact_curve':heads}
        sparse_variants=['clean','pi']
        if sparse_selected is not None:
            for variant in ['mean_only','adapter_matched']:
                if variant not in sparse_selected['groups']:continue
                row=sparse_selected['groups'][variant];assert row['seeds']==list(range(5))
                for seed,hashes in enumerate(sparse_selected['source_sha256'][variant]):
                    root=Path('outputs/sparse_capacity_v1/cifar10')/f'{variant}_seed{seed}'
                    for file,digest in hashes.items():
                        raw=(root/file).read_bytes();actual=sha_bytes(raw)
                        assert actual==digest,(root,file,'sparse control source changed')
                        SOURCES[str(root/file)]=actual
                sp[variant]={'accuracy':stats([r['accuracy'] for r in row['rows']]),
                             'strict_nec':stats([r['strict_nec'] for r in row['rows']])}
            sparse_variants=[v for v in MAIN if v in sp]
        joint_exact=None;joint_audit=None
        joint_audit_path=Path('reports/sparse_capacity_v1/cifar10/exact_capacity_verification.json')
        if joint_audit_path.exists():
            joint_audit=read(joint_audit_path)
            assert joint_audit['verified_models']==20 and joint_audit['verified_exact_budget_points']==140
            assert len(joint_audit['reproduced_paired_intervals'])==28
            for source,digest in joint_audit['source_sha256'].items():
                actual=sha_bytes(Path(source).read_bytes())
                assert actual==digest,(source,'audited sparse source changed')
                SOURCES[source]=actual
            joint_exact=read('reports/sparse_capacity_v1/cifar10/results.json')
            assert joint_exact['targets']==joint_audit['targets']
            for v in MAIN:
                sp[v]['exact_curve']=[read(Path('outputs/sparse_capacity_v1/cifar10')/f'{v}_seed{s}'/'exact_metrics.json')
                                      for s in range(5)]
        exact_targets=joint_exact['targets'] if joint_exact else exact['protocol']['targets']
        fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
        offsets={'clean':(0,18),'mean_only':(-35,30),'adapter_matched':(40,30),'pi':(5,78)}
        for v in sparse_variants:
            axis=axes[0];a=sp[v]['accuracy'];n=sp[v]['strict_nec']
            axis.errorbar(n['mean'],a['mean']*100,xerr=n['std'],yerr=a['std']*100,fmt=MARKERS[v],color=COLORS[v],capsize=4,markersize=9)
            axis.annotate(f"{LABELS[v]}\n{a['mean']*100:.3f}%；NEC={n['mean']:.2f}",(n['mean'],a['mean']*100),xytext=offsets[v],textcoords='offset points',ha='center',fontsize=10,arrowprops={'arrowstyle':'-','color':COLORS[v]})
            if 'exact_curve' not in sp[v]:continue
            curves=sp[v]['exact_curve'];ks=exact_targets
            values=np.array([[r.get('mc8',r)['accuracy']*100 for r in c['points']] for c in curves])
            curve(axes[1],ks,values.mean(0),values.std(0,ddof=1),v)
        axes[0].set_ylim(86.9,89.5);axes[0].set_xlim(85,300);axes[0].set_xlabel('验证选定头：平均严格非零 NEC');decorate(axes[0],'测试准确率（%；局部放大）')
        axes[1].set_xlabel('每类别严格非零概念数（精确匹配）');decorate(axes[1],'测试准确率（%）')
        axes[1].set_title('选定头后剪枝；各预算未重新优化');legend(fig,sparse_variants)
        sparse_conclusion=f"延长拟合后干净 {sp['clean']['accuracy']['mean']*100:.3f}%，Pi {sp['pi']['accuracy']['mean']*100:.3f}%，但平均 NEC 不同。"
        for control in ['mean_only','adapter_matched']:
            if control not in sp:continue
            contrast=next(r for r in sparse_selected['contrasts'] if r['control']==control)
            sparse_conclusion+=f"{LABELS[control]}为 {sp[control]['accuracy']['mean']*100:.3f}%，Pi−{LABELS[control]} {contrast['accuracy_delta']*100:+.3f} 个百分点，95% [{contrast['ci95'][0]*100:.3f}, {contrast['ci95'][1]*100:.3f}]。"
        sparse_conclusion+='选定头的确定性对照差异区间包含零，尚未证明随机分布额外收益。'
        if joint_exact:
            sparse_conclusion+='四方法严格 NEC 比较已完成：Pi 相对干净头的全部七个预算区间高于零；相对仅均值、等容量及同 Pi 头均值推理的全部区间均包含零。'
        else:
            sparse_conclusion+='严格 NEC 后剪枝下 Pi 相对干净的曲线更高。四方法的选定稀疏头已齐全，严格 NEC 四方法比较仍在运行。'
        sparse_conclusion+='偏置不重校准、各预算不重新拟合，不能解释为每个 NEC 下的最优可达性能。'
        sparse_title='稀疏头：延长拟合与四方法严格 NEC 后剪枝' if joint_exact else '稀疏头：延长拟合与严格 NEC 后剪枝（阶段性）'
        save(pdf,'11_sparse_heads',sparse_title,fig,
             '用能产生零权重的近端求解器重拟合头，并区分实际 NEC 与保留槽位上限。',
             sparse_conclusion,
             f"左图为验证选定 L1/迭代的五种子均值±SD；右图{'四方法' if joint_exact else '仅干净/Pi'}的同一训练选定头按绝对权重剪枝，逐类恰有 k 个严格非零且超过 1e−5 的系数。共同可行预算为 {'/'.join(map(str,exact_targets))}，预算仅由验证选定权重确定。干净五份是确定性重复。",
             {'selected_heads':sp,'exact_comparisons':joint_exact if joint_exact else exact['pi_minus_clean'],
              'previous_clean_pi_only':exact,'selection_comparison':sparse['pi_minus_clean'],
              'deterministic_comparisons':sparse_selected,'exact_capacity_verification':joint_audit})

        fig,axes=plt.subplots(1,3,figsize=(16,5.6),layout='constrained')
        for v in MAIN:
            task=core[2]['groups'][v]['accuracy'];attribute=semantics[v]['attribute_macro_balanced_accuracy']
            energy=core[2]['groups'][v]['noise_energy']
            axes[0].errorbar(attribute['mean']*100,task['mean']*100,xerr=attribute['std']*100,
                             yerr=task['std']*100,fmt=MARKERS[v],color=COLORS[v],capsize=4,markersize=8)
            for axis,key in zip(axes[1:],['grid_attribution','foreground_attribution']):
                m=loc[v][key]
                axis.errorbar(energy['mean'],m['mean']*100,xerr=energy['std'],yerr=m['std']*100,
                              fmt=MARKERS[v],color=COLORS[v],capsize=4,markersize=8)
        axes[0].set_xlabel('真实属性 balanced accuracy（%）');decorate(axes[0],'分类准确率（%）')
        axes[0].set_title('概念准确率—任务准确率')
        for axis,label in zip(axes[1:],['目标象限归因（%）','鸟体内归因（%）']):
            axis.set_xscale('symlog',linthresh=1e-4);axis.set_xlabel('相对扰动能量（均值项 + 尺度项）')
            decorate(axis,label);axis.set_title('定位质量—扰动能量')
        legend(fig)
        save(pdf,'12_joint_hypothesis','研究假设的联合检查：任务、真实属性、定位与能量',fig,
             '按研究方案将任务与真实属性、定位与扰动能量联合展示，配合图 11 的准确率—NEC 曲线。',
             '学习式适配同时改善分类、真实属性与定位；确定性方法与 Pi 位于接近的位置。能量包含均值调整，不能称为纯随机噪声能量；这些观测点不能证明加大能量会因果改善定位。',
             'RN50/CUB-GDINO、完整五种子；属性使用同一 139 属性可见支持映射，定位为正式全测试口径。横纵误差均为 SD；能量横轴为 symlog，干净模型能量为零。',
             {v:{'task':core[2]['groups'][v]['accuracy'],'attribute':semantics[v]['attribute_macro_balanced_accuracy'],
                 'energy':core[2]['groups'][v]['noise_energy'],'localization':loc[v]} for v in MAIN})

        lci_path=Path('reports/localization_uncertainty_v1/results.json')
        if lci_path.exists():
            lci=read(lci_path)
            rows=[r for r in lci['comparisons'] if r['bank']=='RN50_GDINO' and r['control'] in MAIN[:-1]]
            if len(rows)==3:
                fig,axes=plt.subplots(1,5,figsize=(19,5.5),layout='constrained')
                metric_labels=['目标象限归因','象限最大落点','鸟体内归因','鸟体最大落点','可定位部位命中']
                for axis,key,label in zip(axes,['grid_attribution','grid_max_point','foreground_attribution','foreground_max_point','part_hit'],metric_labels):
                    for i,r in enumerate(rows):
                        assert r['adaptation_seeds']==5 and r['test_images']==5794
                        interval(axis,r['metrics'][key],i,COLORS[r['control']],key='delta')
                    axis.set_yticks(range(3),['Pi − '+LABELS[r['control']] for r in rows]);axis.invert_yaxis()
                    axis.axvline(0,ls='--',color='gray');axis.set_title(label);axis.set_xlabel('比例差（百分点）；95% 区间')
                    axis.grid(axis='x',alpha=.2);axis.spines[['top','right']].set_visible(False)
                save(pdf,'13_localization_intervals','定位收益的配对区间：适配与随机性',fig,
                     '用共同测试图和配对适配种子重采样，判断微小定位变化是否有区间支持。',
                     'Pi 相对干净模型的五项定位区间均高于零；相对仅均值、等容量对照的五项区间均包含零。适配的定位改善获得描述性区间支持，但尚未证明随机尺度独立改善定位；部位命中相对干净的区间接近零边界，需保留未校正多重比较的限制。',
                     '2,000 次分层配对 bootstrap。每次按可评估概念数加权，不能平均每张图的部位命中率。零热图记零，未知/不可定位部位排除；所有方法的目标、覆盖和分母逐图一致。固定网格与一次 MC8 CAM，区间不包括网格源图重建和原型构建的不确定性；未作多重比较校正。',lci)

        stress={}
        for dataset in ['cifar10','cifar100']:
            path=Path('reports/corruption_uncertainty_v1')/f'{dataset}.json'
            if not path.exists():continue
            report=read(path)
            assert report['verified_frozen_checkpoints']==25 and report['reproduced_prediction_cases']==375
            assert all(g['seeds']==list(range(5)) for g in report['groups'].values())
            for source,files in report['source_sha256'].items():
                if 'metadata' in files:continue
                for name,digest in files.items():
                    input_path=Path(source)/name
                    actual=sha_bytes(input_path.read_bytes())
                    assert actual==digest,(input_path,'stress-test source changed')
                    SOURCES[str(input_path)]=actual
            stress[dataset]=report
        if stress:
            fig,axes=plt.subplots(len(stress),2,figsize=(13,5.3*len(stress)),layout='constrained',squeeze=False)
            kind_labels={'gaussian_noise':'高斯噪声','gaussian_blur':'高斯模糊','brightness':'亮度','contrast':'对比度','jpeg':'JPEG'}
            stress_conclusions=[]
            for row,(dataset,report) in enumerate(stress.items()):
                groups=report['groups'];kinds=list(kind_labels)
                for v in MAIN:
                    curve(axes[row,0],list(range(len(kinds))),
                          [groups[v]['per_kind_accuracy'][k]['mean']*100 for k in kinds],
                          [groups[v]['per_kind_accuracy'][k]['std']*100 for k in kinds],v)
                axes[row,0].set_xticks(list(range(len(kinds))),[kind_labels[k] for k in kinds])
                axes[row,0].set_title(dataset.upper()+'：各类型三个条件等权平均')
                decorate(axes[row,0],'损坏图像分类准确率（%）')
                contrasts={r['control']:r for r in report['comparisons'] if r['case']=='all_15_equal_weight'}
                axes[row,1].axvline(0,color='#596678',ls='--',lw=1)
                controls=['mean_only','adapter_matched']
                for i,control in enumerate(controls):
                    r=contrasts[control];point=r['accuracy_delta']*100;lo,hi=[v*100 for v in r['ci95']]
                    axes[row,1].errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',color=COLORS[control],capsize=4)
                axes[row,1].set_yticks(list(range(len(controls))),[LABELS[c] for c in controls])
                axes[row,1].set_ylim(-.6,len(controls)-.4)
                axes[row,1].set_title(dataset.upper()+'：15 条件汇总的确定性对照')
                axes[row,1].set_xlabel('Pi − 对照：准确率差（百分点），95% 配对区间')
                axes[row,1].spines[['top','right']].set_visible(False)
                clean=groups['clean']['equal_weight_case_accuracy']['mean']*100
                pi=groups['pi']['equal_weight_case_accuracy']['mean']*100
                zero=all(contrasts[c]['ci95'][0]<=0<=contrasts[c]['ci95'][1] for c in controls)
                conclusion=('相对两个确定性对照的汇总区间均包含零，尚未证明随机分布额外鲁棒性收益。'
                            if zero else '确定性对照的汇总差异见配对区间；这些比较仍不是同检查点的纯尺度因果效果。')
                stress_conclusions.append(f'{dataset.upper()}：15 条件平均干净 {clean:.3f}%、Pi {pi:.3f}%；{conclusion}')
            legend(fig)
            save(pdf,'14_image_corruptions','图像损坏：适配与随机分布的鲁棒性比较',fig,
                 '在原定 15 种图像损坏条件下检验泛化，并区分适配收益与随机分布的额外作用。',
                 ' '.join(stress_conclusions),
                 '自定义原生分辨率压力测试，不是 CIFAR-C。左图为每类型三个固定条件等权平均、误差线为适配种子 SD；右图先在同一种子/源图内平均 15 条件，再重采样配对种子与共同源图，2,000 次 95% 区间。条件和一次噪声场固定，未校正多重比较；只展示完整五种子数据集。',stress)

        branch={}
        scenario_names=['no_noise','scale_only','mean_only','mean_plus_scale_x0.25','mean_plus_scale_x0.5','mean_plus_scale_x1','mean_plus_scale_x2','mean_plus_scale_x4']
        scenario_labels=['0','σξ','μ','μ+0.25σξ','μ+0.5σξ','μ+σξ','μ+2σξ','μ+4σξ']
        for dataset,clusters in [('cifar10',256),('cifar100',2048),('cub',2048)]:
            path=Path('reports/noise_branch_evidence_v1')/dataset/'pi.json'
            if not path.exists():continue
            report=read(path);assert report['seeds']==list(range(5))
            reference=pack_reference(Path('artifacts/datasets')/dataset/f'baseline_k{clusters}_seed0.pt')
            for seed in range(5):
                run=Path('outputs/noise_branch_evidence_v1')/dataset/f'pi_seed{seed}'
                metadata=read(run/'metrics.json');p=load(run/'predictions.pt')
                assert metadata['source_sha256']==report['source_checkpoint_sha256'][seed]
                assert sha_bytes(Path(metadata['source_checkpoint']).read_bytes())==metadata['source_sha256']
                assert SOURCES[str(run/'predictions.pt')]==report['predictions_sha256'][seed]
                assert metadata['sampling_seed']==3010 and metadata['mc_samples']==16
                assert torch.equal(p['labels'],reference['labels'])
                for name in scenario_names:
                    probability=p[name]
                    assert torch.isfinite(probability).all() and (probability>=0).all()
                    assert torch.allclose(probability.sum(-1),torch.ones(len(p['labels'])),atol=1e-5)
                    row=next(r for r in metadata['scenarios'] if r['scenario']==name)
                    for key,value in task_metrics(probability,p['labels']).items():assert abs(value-row[key])<1e-7
            branch[dataset]=report
        if branch:
            fig,axes=plt.subplots(len(branch),3,figsize=(17,5.5*len(branch)),layout='constrained',squeeze=False)
            branch_conclusions=[];multipliers=[.25,.5,1,2,4]
            for row,(dataset,report) in enumerate(branch.items()):
                groups=report['groups'];label='CUB / SAM2' if dataset=='cub' else dataset.upper()
                for i,name in enumerate(scenario_names):
                    color=COLORS['clean'] if name=='no_noise' else '#7B61A8' if name=='scale_only' else COLORS['same_checkpoint_mean'] if name=='mean_only' else COLORS['pi']
                    value=groups[name]['accuracy'];axes[row,0].errorbar(i,value['mean']*100,yerr=value['std']*100,fmt='o',color=color,capsize=3)
                axes[row,0].set_xticks(range(8),scenario_labels,rotation=35,ha='right')
                axes[row,0].set_title(label+'：同一检查点的分支干预')
                axes[row,0].set_xlabel('推理扰动；固定 MC16');decorate(axes[row,0],'分类准确率（%），均值±SD')
                by_name={r['scenario']:r for r in report['comparisons']}
                names=[f'mean_plus_scale_x{factor}' for factor in multipliers]
                axes[row,1].axvline(0,color='#596678',ls='--',lw=1)
                for i,name in enumerate(names):
                    r=by_name[name];point=r['accuracy_delta']*100;lo,hi=[v*100 for v in r['ci95']]
                    axes[row,1].errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',color=COLORS['pi'],capsize=3)
                axes[row,1].set_yticks(range(5),[f'σ × {factor:g}' for factor in multipliers]);axes[row,1].set_ylim(-.6,4.6)
                axes[row,1].set_title('均值保留：随机尺度相对均值推理')
                axes[row,1].set_xlabel('准确率差（百分点），95% 配对区间');axes[row,1].spines[['top','right']].set_visible(False)
                energies=[groups[name]['scale_energy_fraction'] for name in names]
                axes[row,2].errorbar(multipliers,[e['mean']*100 for e in energies],yerr=[e['std']*100 for e in energies],fmt='-o',color='#7B61A8',capsize=3)
                axes[row,2].set_xscale('log',base=2);axes[row,2].set_xticks(multipliers,[f'{factor:g}' for factor in multipliers])
                axes[row,2].set_title('保留均值时的尺度能量占比')
                axes[row,2].set_xlabel('σ 倍率');decorate(axes[row,2],'σ² /（μ²+σ²）（%；特征 RMS 缩放前）')
                positive=[factor for factor,name in zip(multipliers,names) if by_name[name]['ci95'][0]>0]
                finding=('预设尺度倍率均未显示相对均值的正增益区间证据。' if not positive else f'尺度倍率 {positive} 的区间高于零，但未经多重比较校正，不据此选择测试倍率。')
                branch_conclusions.append(f"{label}：仅尺度 {groups['scale_only']['accuracy']['mean']*100:.3f}%、均值推理 {groups['mean_only']['accuracy']['mean']*100:.3f}%；原尺度能量占比 {groups['mean_plus_scale_x1']['scale_energy_fraction']['mean']*100:.3f}%。{finding}")
            save(pdf,'15_mean_scale_branches','均值与随机尺度：同检查点推理分解',fig,
                 '在同一已选 Pi 检查点上关闭分支和改变原定尺度倍率，直接检验推理时均值与随机尺度的作用。',
                 ' '.join(branch_conclusions),
                 '冻结头、原定八种干预、固定 seed3010/MC16、完整五种子；不重新训练或从测试结果选择倍率。左/右图误差为 SD，中图为 2,000 次配对种子与共同图像 bootstrap，未校正多重比较。能量比例在特征 RMS 缩放前；区间条件于一次固定采样实现。关闭分支的推理结果不能推出重新训练时删除该分支的效果，也不是等价性证明。',branch)

        if joint_exact:
            fig,axes=plt.subplots(1,3,figsize=(18,6.7),layout='constrained')
            budgets=joint_exact['targets']
            panels=[(['clean'],'Pi 相对干净稀疏头'),
                    (['mean_only','adapter_matched'],'Pi 相对独立确定性适配器'),
                    (['same_checkpoint_mean'],'同一 Pi 稀疏头：MC8 − 均值')]
            for axis,(controls,title) in zip(axes,panels):
                axis.axvline(0,color='#353B45',ls='--',lw=1)
                for j,control in enumerate(controls):
                    rows=joint_exact['pi_mc8_minus_mean'] if control=='same_checkpoint_mean' else [r for r in joint_exact['pi_minus_control'] if r['control']==control]
                    assert [r['nec_per_class'] for r in rows]==budgets
                    color=COLORS['pi'] if control=='clean' else COLORS[control]
                    for i,row in enumerate(rows):interval(axis,row,i+(j-(len(controls)-1)/2)*.22,color)
                axis.set_yticks(range(len(budgets)),[str(k) for k in budgets]);axis.invert_yaxis()
                axis.set_ylabel('每类严格非零 NEC');axis.set_title(title)
                axis.set_xlabel('准确率差（百分点）；95% 配对区间')
                axis.grid(axis='x',alpha=.2);axis.spines[['top','right']].set_visible(False)
            legend(fig,['mean_only','adapter_matched','same_checkpoint_mean'])
            save(pdf,'16_sparse_paired_intervals','严格 NEC：适配、容量与随机采样的配对证据',fig,
                 '在四方法共同可行的每类严格非零概念数下，区分适配收益、额外容量与同检查点随机推理的作用。',
                 '七个共同预算中，Pi−干净头的区间均高于零；Pi−仅均值、Pi−等容量和 MC8−同 Pi 头均值的全部 21 个区间均包含零。NEC=87 时四方法准确率为干净 39.890%、仅均值 88.450%、等容量 88.414%、Pi 88.410%。当前剪枝优势可由确定性适配实现，尚无随机分布额外稀疏分类收益的区间证据。',
                 '固定完整五种子与验证选定头，预算为 1/5/10/20/40/80/87；逐类核验严格非零数与超过 1e−5 的数均等于预算。2,000 次配对种子/共同图像 bootstrap，未校正多重比较；一次固定 MC8 实现，剪枝后不重拟合头或校准偏置。三个面板横轴尺度不同，不能比较视觉长度；低 NEC 干净头偏置失配会放大性能下降，不表示其最优可达性能。',
                 {'comparisons':joint_exact['pi_minus_control'],'same_checkpoint_sampling':joint_exact['pi_mc8_minus_mean'],
                  'verification':joint_audit})

        sam_attributes=None
        sam_attribute_path=Path('reports/attribute_evidence_verification_v1/RN50_SAM2.json')
        if sam_attribute_path.exists():
            sam_attributes=read(sam_attribute_path)
            assert len(sam_attributes['verified_checkpoints'])==20 and len(sam_attributes['comparisons'])==3
            assert sam_attributes['test_images']==5794 and sam_attributes['evaluated_attributes']==161
            for source,digest in sam_attributes['source_sha256'].items():
                actual=sha_bytes(Path(source).read_bytes())
                assert actual==digest,(source,'audited attribute source changed')
                SOURCES[source]=actual
            fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
            groups=sam_attributes['groups'];comparisons=sam_attributes['comparisons']
            for i,key in enumerate(['attribute_macro_balanced_accuracy','attribute_macro_auroc']):
                for j,v in enumerate(MAIN):
                    value=groups[v][key]
                    axes[0].errorbar(i+(j-1.5)*.13,value['mean']*100,yerr=value['std']*100,
                                     fmt=MARKERS[v],color=COLORS[v],capsize=4,markersize=7)
            axes[0].set_xticks([0,1],['Macro balanced accuracy','Macro AUROC'])
            decorate(axes[0],'属性指标（%；纵轴局部放大）')
            for i,row in enumerate(comparisons):interval(axes[1],row,i,COLORS[row['control']],key='balanced_accuracy_delta')
            axes[1].set_yticks(range(3),['Pi − '+LABELS[r['control']] for r in comparisons]);axes[1].invert_yaxis()
            axes[1].axvline(0,color='#353B45',ls='--',lw=1)
            axes[1].set_xlabel('Balanced accuracy 差（百分点）；95% 配对区间')
            axes[1].grid(axis='x',alpha=.2);axes[1].spines[['top','right']].set_visible(False);legend(fig)
            clean=next(r for r in comparisons if r['control']=='clean')
            save(pdf,'17_sam2_attributes','CUB-SAM2 真实属性：原生标注与完整五种子核验',fig,
                 '独立核对原生属性真值、可见支持映射和固定验证阈值，检验 SAM2 概念库上的任务收益是否伴随真实属性改善。',
                 f"Pi 的 balanced accuracy 为 {groups['pi']['attribute_macro_balanced_accuracy']['mean']*100:.3f}%，相对干净 +{clean['balanced_accuracy_delta']*100:.3f} 个百分点，95% [{clean['ci95'][0]*100:.3f},{clean['ci95'][1]*100:.3f}]；相对仅均值和等容量的区间均包含零。原生真值、20 个完整属性预测缓存的逐属性 BA/AUROC 及三个配对区间均独立重现，仍未显示随机分布额外属性收益。",
                 'RN50/SAM2，certainty≥3，映射 166/312、可评估 161/312 属性，完整 5,794 测试图。左图为五种子均值±SD，右图为 2,000 次配对适配种子/共同图像 bootstrap；固定验证阈值与一次 seed2110/MC8 实现，未校正多重比较。SAM2 与 GDINO 的可评估属性集合不同，不能直接用两库的绝对宏平均判断提案库更优。',sam_attributes)

        primary_matrix=None
        primary_path=Path('reports/primary_matrix_verification_v1/results.json')
        if primary_path.exists():
            primary_matrix=read(primary_path)
            assert primary_matrix['verified_models']==180 and primary_matrix['verified_inference_arrays']==360
            assert len(primary_matrix['original_head_refit_checks'])==30 and len(primary_matrix['head_refit_comparisons'])==9
            assert all(r['all_non_head_tensors_exact'] for r in primary_matrix['original_head_refit_checks'])
            for source,digest in primary_matrix['source_sha256'].items():
                if source in SOURCES:assert SOURCES[source]==digest,(source,'matrix source differs from plotted source')
                else:
                    actual=sha_bytes(Path(source).read_bytes())
                    assert actual==digest,(source,'audited primary matrix source changed')
                    SOURCES[source]=actual
            refit_variants=['clean','clean_refit','pi','pi_refit']
            VERIFIED.extend(r for r in primary_matrix['verified_classification_checkpoints']
                            if Path(r['checkpoint']).parent.name.rsplit('_seed',1)[0] in refit_variants)
            fig,axes=plt.subplots(2,3,figsize=(16,10.2),layout='constrained')
            conclusions=[]
            for column,dataset in enumerate(primary_matrix['datasets']):
                label='CUB / SAM2' if dataset=='cub' else 'CIFAR-10' if dataset=='cifar10' else 'CIFAR-100'
                draw_method_points(axes[0,column],primary_matrix['groups'][dataset],variants=refit_variants,annotate=True)
                axes[0,column].set_title(label);decorate(axes[0,column],'测试准确率（%；纵轴局部放大）')
                comparisons=[r for r in primary_matrix['head_refit_comparisons'] if r['dataset']==dataset]
                for i,row in enumerate(comparisons):
                    interval(axes[1,column],row,i,COLORS['clean'] if row['adapted']=='clean_refit' else COLORS['pi'])
                axes[1,column].axvline(0,color='#353B45',ls='--',lw=1)
                axes[1,column].set_yticks(range(3),['干净重拟合 − 干净','Pi 重拟合 − Pi','Pi 重拟合 − 干净重拟合'])
                axes[1,column].invert_yaxis();axes[1,column].set_xlabel('准确率差（百分点）；95% 配对区间')
                axes[1,column].grid(axis='x',alpha=.2);axes[1,column].spines[['top','right']].set_visible(False)
                row=next(r for r in comparisons if r['control']=='pi')
                conclusions.append(f"{label}：Pi 重拟合−冻结头 {row['accuracy_delta']*100:+.3f} 个百分点，95% [{row['ci95'][0]*100:.3f},{row['ci95'][1]*100:.3f}]。")
            legend(fig,refit_variants)
            save(pdf,'18_original_head_refit','原 200 轮头重拟合：额外训练的收益与代价',fig,
                 '在逐种子生成器和全部非头张量完全相同的条件下，隔离原第二阶段头重拟合的效果，并与干净头重拟合比较。',
                 ' '.join(conclusions)+'Pi 头重拟合改善 CIFAR-10、降低 CIFAR-100 准确率，CUB 的下降区间包含零；不能假定额外头训练一定改善任务。三组 Pi 重拟合相对干净重拟合的区间仍高于零，这项差异不隔离随机尺度。',
                 '原 200 轮 Adam+L1 方案，按验证 NLL 选头；各数据集完整五种子，所有 30 次重拟合的非头张量均与对应冻结头模型逐位一致。上图为均值±SD，下图为 2,000 次配对种子/共同测试图 bootstrap，未校正多重比较。固定原型和一次 MC8 实现；与图 11/16 的近端稀疏头区别。各数据集纵轴局部放大，干净确定性重复不代表独立基线初始化。',primary_matrix)

        background_evidence={}
        for bank in ['RN50_GDINO','RN50_SAM2']:
            for version in ['natural_background','background']:
                path=Path('reports/background_semantic_uncertainty_v1')/bank/f'{version}.json'
                if not path.exists():continue
                row=read(path)
                assert row['seeds']==list(range(5)) and row['test_images']==5794
                assert row['verified_models']==25 and len(row['comparisons'])==25
                assert row['native_attribute_truth_exact'] and row['original_cpu_probabilities_bitwise_exact']
                for source,digest in row['source_sha256'].items():
                    actual=sha_bytes(Path(source).read_bytes());assert actual==digest,(source,'background source changed')
                    SOURCES[source]=actual
                for v,digests in row['prediction_sha256'].items():
                    for seed,digest in enumerate(digests):
                        source=Path('outputs/background_semantic_uncertainty_v1')/bank/version/f'{v}_seed{seed}/semantic_predictions.pt'
                        cache=load(source);assert SOURCES[str(source)]==digest
                        directory=Path('artifacts/datasets')/('cub_gdino' if bank=='RN50_GDINO' else 'cub')
                        features=Path('artifacts/datasets')/('cub_background_natural' if version=='natural_background' else 'cub')
                        bindings=[(directory/'baseline_k2048_seed0.pt',cache['metadata']['baseline_sha256']),
                                  (directory/'attributes.pt',cache['metadata']['annotations_sha256']),
                                  (directory/'attribute_mapping_visible_k2048.json',cache['metadata']['mapping_sha256']),
                                  (features/'backgrounds/protocol.json',cache['metadata']['protocol_sha256'])]
                        bindings.extend((features/'backgrounds'/f'{c}.pt',r['feature_sha256'])
                                        for c,r in cache['cases'].items())
                        for bound,expected in bindings:
                            actual=SOURCES.get(str(bound))
                            if actual is None:actual=sha_bytes(bound.read_bytes());SOURCES[str(bound)]=actual
                            assert actual==expected,(bound,'native background binding changed')
                        assert cache['attribute_ids'].tolist()==row['attribute_ids']
                        assert len(cache['indices'])==5794 and cache['metadata']['sampling_seed']==6090+seed
                        del cache
                background_evidence[f'{bank}/{version}']=row
        if background_evidence:
            fig,axes=plt.subplots(len(background_evidence),3,figsize=(17,5.6*len(background_evidence)),
                                  squeeze=False,layout='constrained')
            captions=[];titles=['原图','灰背景','训练供体17','训练供体29','训练供体43']
            for row_index,(name,r) in enumerate(background_evidence.items()):
                ax=axes[row_index];x=np.arange(5)
                for v in VARIANTS:
                    means=[r['groups'][v][c]['balanced_accuracy']['mean']*100 for c in r['case_order']]
                    errors=[r['groups'][v][c]['balanced_accuracy']['std']*100 for c in r['case_order']]
                    ax[0].errorbar(x,means,yerr=errors,fmt='-'+MARKERS[v],color=COLORS[v],capsize=3,label=LABELS[v])
                ax[0].set_xticks(x,titles,rotation=20);decorate(ax[0],'真实属性 BA（%；均值±SD）')
                ax[0].legend(frameon=False,fontsize=8,ncol=2)
                version='自然照片块' if r['version']=='natural_background' else '最近像素填补'
                ax[0].set_title(f"{r['bank']} / {version}\n{r['evaluated_attributes']} 个固定属性")
                for column,controls in [(1,['clean','fixed']),(2,['mean_only','adapter_matched','same_checkpoint_mean'])]:
                    for i,case in enumerate(r['case_order']):
                        for j,control in enumerate(controls):
                            comp=next(v for v in r['comparisons'] if v['case']==case and v['control']==control)
                            interval(ax[column],comp['semantic'],i+(j-(len(controls)-1)/2)*.20,
                                     COLORS[control],key='balanced_accuracy_delta')
                    ax[column].set_yticks(x,titles);ax[column].invert_yaxis()
                    ax[column].axvline(0,ls='--',color='gray');ax[column].grid(axis='x',alpha=.2)
                    ax[column].set_xlabel('Pi − 对照：属性 BA 差（百分点）；95% 配对区间')
                    ax[column].set_title('与干净/固定噪声比较' if column==1 else '与确定性适配/同头均值比较')
                    ax[column].legend(handles=[plt.Line2D([],[],color=COLORS[c],marker='o',label=LABELS[c])
                                               for c in controls],frameon=False,fontsize=8)
                clean=[c['semantic'] for c in r['comparisons'] if c['control']=='clean']
                deterministic=[c['semantic'] for c in r['comparisons'] if c['control'] in
                               ['mean_only','adapter_matched','same_checkpoint_mean']]
                positive=sum(c['ci95'][0]>1e-12 for c in clean)
                zero=sum(c['ci95'][0]<=1e-12 and c['ci95'][1]>=-1e-12 for c in deterministic)
                captions.append(f'{name}：{positive}/5 条件的 Pi−干净真实属性区间为正；'
                                f'{zero}/15 项确定性适配/同头均值比较区间包含零。')
            save(pdf,'19_background_attributes','背景变化下的真实属性：固定验证阈值与配对证据',fig,
                 '检验背景改变后任务收益是否伴随真实概念识别收益，并隔离随机分布相对确定性适配与同头均值推理的作用。',
                 ' '.join(captions)+'背景变化后的绝对属性准确率见左图；'+
                 ('当前完成组支持适配收益，未显示随机尺度的独立概念鲁棒性增益。' if all(
                     c['semantic']['ci95'][0]<=1e-12 and c['semantic']['ci95'][1]>=-1e-12
                     for r in background_evidence.values() for c in r['comparisons']
                     if c['control'] in ['mean_only','adapter_matched','same_checkpoint_mean']) else
                  '随机分布的额外属性收益须结合各固定条件及配对区间解释，不从压力测试选择条件。'),
                 '每库每版本完整五方法五种子、5,794测试图；certainty≥3、可见支持训练映射、原验证阈值固定。逐属性 BA/AUROC 与原报告重现，全部 MC8/均值概率与原 CPU 输出逐位一致。每条件2,000次配对种子/共同源图重采样，固定属性交集，未定义次数保留于数据；未校正多重比较。身份条件 seed=6090+适配种子，不混用 seed2110 的原属性结果。两个区间面板尺度不同；灰背景/缩放拼接均为合成压力测试。不同概念库的属性集合不可直接比较。',background_evidence)

    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    preface=['# Pi 噪声研究：主要实验结果','',f'更新：{stamp}。图表使用已完成的种子 0–4；结果快照，不代表整个研究矩阵完成。','',
             '当前主要发现：'+core_finding+' CUB 真属性和已完成的稀疏头比较也需结合确定性对照解释。','',
             '统一颜色：灰色＝干净模型，天蓝＝固定噪声，橙色＝仅均值，蓝色＝等容量适配器，玫红＝Pi，绿色＝同一 Pi 检查点的均值推理。SD 与 95% 区间分别标明；“区间包含零”不等于证明完全等价。','',
             '[所有图表合并 PDF](主要实验图表合集.pdf) · [网页浏览](index.html) · [机器可读数据](plot_data.json) · [来源与核验](manifest.json)','']
    for sec in SECTIONS:
        preface.extend([f"## {sec['title']}",'',f"**目的**：{sec['purpose']}",'',f"![{sec['title']}]({sec['file']}.png)",'',
                        f"**结论**：{sec['conclusion']}",'',f"**口径**：{sec['note']}",'',
                        f"[PNG]({sec['file']}.png) · [SVG]({sec['file']}.svg) · [PDF]({sec['file']}.pdf)",''])
    preface.extend(['## 尚待完成','',
                    f"尚未形成完整核心分类对照：{'、'.join(pending_core)}。"+('RN50 三个小数据集的 12 方案五种子主训练矩阵及原头重拟合已核验。' if primary_matrix else '')+('CUB-SAM2 核心真实属性评估已完整；' if sam_attributes else '')+f'CUB-SAM2 的完整定位、检索、背景与扩展语义比较继续完成；Places365、ImageNet 及其 ImageNet-R 扩展仍在进行。{pending_concept_note}'+('四方法选定稀疏头与共同严格 NEC 比较均已完成。' if joint_exact else '四方法严格 NEC 曲线仍在运行，四方法选定稀疏头已齐全。')+'ImageNet 将明确使用既有每类 200+10 图干净训练子集。缺失配置没有填充估计值。','',
                    '本次导出核验了所绘分类检查点的冻结原型、标准化与初始头，以及保存预测的分类指标；定位使用完整测试报告，属性预测来自已进行一致性复算的专用缓存。来源快照及 SHA-256 见 manifest.json。',''])
    (DEST/'主要实验结果.md').write_text('\n'.join(preface))
    cards=[]
    for sec in SECTIONS:
        f=sec['file'];cards.append(f'<article id="{f}"><h2>{html.escape(sec["title"])}</h2><p><b>目的：</b>{html.escape(sec["purpose"])}</p><a href="{f}.svg"><img src="{f}.png" alt="{html.escape(sec["title"])}" loading="lazy"></a><p><b>结论：</b>{html.escape(sec["conclusion"])}</p><p class="note"><b>口径：</b>{html.escape(sec["note"])}</p><p class="links"><a href="{f}.png">PNG</a><a href="{f}.svg">SVG</a><a href="{f}.pdf">PDF</a></p></article>')
    navigation=''.join(f'<a href="#{s["file"]}">{html.escape(s["title"])}</a>' for s in SECTIONS)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pi 噪声研究 · 主要结果</title><style>body{margin:0;background:#f3f5f8;color:#192431;font:16px/1.8 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:32px 24px}header,article{background:white;border-radius:12px;padding:26px 32px;margin-bottom:24px;box-shadow:0 2px 12px #19243108}h1{margin:0}h2{font-size:22px;line-height:1.5}a{color:#0072b2}img{width:100%;height:auto}.note{color:#596678;font-size:14px}.links a{margin-right:18px}nav{display:flex;flex-wrap:wrap;gap:10px}nav a{font-size:13px;background:#edf3f8;padding:3px 10px;border-radius:6px;text-decoration:none}@media(max-width:650px){main{padding:12px}header,article{padding:18px}}@media print{body{background:white}article{break-inside:avoid;box-shadow:none}nav{display:none}}</style><main><header><h1>Pi 噪声研究 · 主要实验结果</h1>'
    page+=f'<p class="note">更新：{stamp} · 正式测试集 · 完整五种子 · 固定颜色</p><p>{html.escape(preface[4])}</p><p><a href="主要实验图表合集.pdf">下载合并 PDF</a> · <a href="主要实验结果.md">说明文档</a> · <a href="plot_data.json">完整绘图数据</a></p><nav>{navigation}</nav></header>'+''.join(cards)+'<footer>研究仍在进行；未完成结果不填值。统计区间条件于固定原型、映射和已选检查点。</footer></main></html>'
    (DEST/'index.html').write_text(page)
    (DEST/'plot_data.json').write_text(json.dumps(DATA,ensure_ascii=False,indent=2,allow_nan=False))
    csv_rows=[]
    for c in core:
        for v,g in c['groups'].items():
            csv_rows.append({'实验':c['case'],'方法':LABELS[v],'适配种子数':5,'准确率均值%':g['accuracy']['mean']*100,
                             '准确率SD%':g['accuracy']['std']*100,'NLL均值':g['nll']['mean'],'ECE均值':g['ece_15_bins']['mean']})
    with (DEST/'核心分类指标.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    snapshot=Path('artifacts/report_assets/major_results_source_snapshots');snapshot.mkdir(exist_ok=True)
    for source,digest in SOURCES.items():
        if Path(source).suffix=='.json' and (source,digest) not in SNAPSHOTS:
            raw=Path(source).read_bytes()
            assert sha_bytes(raw)==digest,(source,'source changed before snapshot')
            SNAPSHOTS[(source,digest)]=raw
    for (source,digest),raw in SNAPSHOTS.items():
        path=Path(source)
        (snapshot/f'{path.stem}_{digest}.json').write_bytes(raw)
    manifest={'generated_at':stamp,'export_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'figures':SECTIONS,'verified_classification_checkpoints':VERIFIED,'sources_sha256':SOURCES,
              'verified_core_paired_intervals':VERIFIED_INTERVALS,
              'source_snapshots_directory':str(snapshot.resolve()),
              'plot_data_sha256':hashlib.sha256((DEST/'plot_data.json').read_bytes()).hexdigest(),
              'limits':'Snapshot of completed five-seed main results, not proof of full research completion. '
                       'Shared deterministic baselines are not independent initializations. SD bars differ from bootstrap intervals.'}
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'figures':len(SECTIONS),'verified_checkpoints':len(VERIFIED),'destination':str(DEST.resolve())},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEST)
    DEST=parser.parse_args().output
    main()
