"""Build six main research figures from a verified full-results snapshot."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from .export_major_results import COLORS, LABELS, MARKERS, interval, decorate


METHODS=['clean','mean_only','adapter_matched','pi']
DATASETS={'cifar10':'CIFAR-10','cifar100':'CIFAR-100','cub':'CUB / SAM2'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def methods_legend(fig,methods=METHODS):
    fig.legend(handles=[plt.Line2D([],[],color=COLORS[v],marker=MARKERS.get(v,'o'),label=LABELS[v])
                        for v in methods],loc='outside lower center',ncol=len(methods),frameon=False)


def forest(axis,rows,labels,controls,key='accuracy_delta'):
    for i,row in enumerate(rows):
        for j,control in enumerate(controls):
            interval(axis,row[control],i+(j-(len(controls)-1)/2)*.18,COLORS[control],key=key)
    axis.set_yticks(range(len(rows)),labels);axis.invert_yaxis()
    axis.axvline(0,color='#596678',ls='--',lw=1);axis.grid(axis='x',alpha=.2)
    axis.spines[['top','right']].set_visible(False)
    axis.set_xlabel('Pi − 对照（百分点）；95% 配对区间')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('../Essay/D/report/详细结果'))
    parser.add_argument('--output',type=Path,default=Path('../Essay/D/report'))
    args=parser.parse_args();source,dest=args.source,args.output
    dest.mkdir(parents=True,exist_ok=True)
    data_path=source/'plot_data.json';manifest_path=source/'manifest.json';audit_path=source/'交付核验.json'
    full=json.loads(data_path.read_text());manifest=json.loads(manifest_path.read_text());audit=json.loads(audit_path.read_text())
    assert digest(data_path)==manifest['plot_data_sha256']
    assert audit['plot_data_hash_exact'] and audit['export_source_hash_exact'] and audit['local_links_valid']
    assert audit['delivery_file_sha256']['plot_data.json']==digest(data_path)
    assert audit['delivery_file_sha256']['manifest.json']==digest(manifest_path)
    font=Path('artifacts/report_assets/NotoSansCJKsc-Regular.otf');font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'font.size':10,
                         'axes.unicode_minus':False,'svg.fonttype':'path','pdf.fonttype':42,'figure.facecolor':'white'})
    sections=[];plot_data={}

    def save(pdf,name,title,fig,purpose,conclusion,note,inputs):
        fig.suptitle(title,fontsize=16)
        for extension in ['png','svg','pdf']:
            fig.savefig(dest/f'{name}.{extension}',dpi=210,bbox_inches='tight',facecolor='white')
        pdf.savefig(fig,bbox_inches='tight',facecolor='white');plt.close(fig)
        sections.append({'file':name,'title':title,'purpose':purpose,'conclusion':conclusion,'note':note})
        plot_data[name]={key:full[key] for key in inputs}
        print('EXPORTED concise',name,flush=True)

    with PdfPages(dest/'主要实验图表合集.pdf') as pdf:
        core=full['01_task_accuracy']
        positive_clean=sum(r['contrasts']['clean']['ci95'][0]>1e-12 for r in core)
        deterministic_zero=all(r['contrasts'][c]['ci95'][0]<=1e-12 and r['contrasts'][c]['ci95'][1]>=-1e-12
                               for r in core for c in ['mean_only','adapter_matched','same_checkpoint_mean'])
        fig,axes=plt.subplots(1,2,figsize=(15,6.5),layout='constrained')
        labels=[r['case']+f"\n{r['groups']['clean']['accuracy']['mean']*100:.2f}% → {r['groups']['pi']['accuracy']['mean']*100:.2f}%" for r in core]
        forest(axes[0],[r['contrasts'] for r in core],labels,['clean'])
        forest(axes[1],[r['contrasts'] for r in core],[r['case'] for r in core],
               ['mean_only','adapter_matched','same_checkpoint_mean'])
        axes[0].set_title('学习式适配相对干净模型');axes[1].set_title('随机分布的额外收益')
        methods_legend(fig,['clean','mean_only','adapter_matched','same_checkpoint_mean'])
        save(pdf,'01_task_and_randomness','分类结果：适配有效，随机性额外收益未证实',fig,
             '用独立确定性对照和同检查点均值推理，区分适配与随机噪声的作用。',
             f'{positive_clean}/{len(core)}组完整对照中，Pi−干净的区间高于零；'+
             ('相对仅均值、等容量及同头均值的区间包含零或触及零边界。' if deterministic_zero else
              '确定性及同头对照的结果逐项见右图；不能将训练方案差异直接归因为随机采样。'),
             '标签为干净→Pi准确率；每组共享原型/初始头。RN50 CUB-SAM2同头比较下界为浮点零边界，不写成稳健正收益。左右尺度不同。',
             ['01_task_accuracy','02_randomness_controls'])

        fig,axes=plt.subplots(2,2,figsize=(14,9.2),layout='constrained')
        gdino=full['03_grounded_attributes']['native_attribute_verification'];sam=full['17_sam2_attributes']
        attrs=[gdino,sam]
        for axis,bank,case in zip(axes[0],attrs,['RN50 / CUB-GDINO','RN50 / CUB-SAM2']):
            task=next(r for r in core if r['case']==case)
            for v in METHODS:
                t=task['groups'][v]['accuracy'];a=bank['groups'][v]['attribute_macro_balanced_accuracy']
                axis.errorbar(t['mean']*100,a['mean']*100,xerr=t['std']*100,yerr=a['std']*100,
                              fmt=MARKERS[v],color=COLORS[v],capsize=3,markersize=7)
            axis.set_title(f"{case}\n{bank['evaluated_attributes']}个固定属性")
            axis.set_xlabel('分类准确率（%；均值±SD）');decorate(axis,'真实属性 BA（%；均值±SD）')
        attribute_rows=[{r['control']:r for r in bank['comparisons']} for bank in attrs]
        forest(axes[1,0],attribute_rows,['CUB-GDINO','CUB-SAM2'],['clean','mean_only','adapter_matched'],
               key='balanced_accuracy_delta');axes[1,0].set_title('真实属性的配对收益')
        comparisons=full['13_localization_intervals']['comparisons']
        metrics=['grid_attribution','grid_max_point','foreground_attribution','foreground_max_point','part_hit']
        loc_rows=[{r['control']:r['metrics'][metric] for r in comparisons if r['control'] in METHODS[:-1]} for metric in metrics]
        forest(axes[1,1],loc_rows,['目标象限归因','象限最大落点','鸟体内归因','鸟体最大落点','可定位部位命中'],
               ['clean','mean_only','adapter_matched'],key='delta');axes[1,1].set_title('CUB-GDINO：完整测试集定位')
        methods_legend(fig)
        save(pdf,'02_attributes_and_localization','解释质量：真实属性与定位改善来自适配',fig,
             '检查分类提升是否伴随真实属性识别和空间解释改善。',
             '两个概念库的Pi属性BA相对干净提高约0.76/0.56个百分点；已完成的GDINO定位也改善。相对确定性对照的区间包含零。',
             '5,794张测试图，certainty≥3。两库属性集合不同，不据绝对BA判断库的优劣；上图为SD，下图为95%配对区间。部位命中区间靠近零；SAM2完整定位未完成。',
             ['03_grounded_attributes','17_sam2_attributes','13_localization_intervals','01_task_accuracy'])

        branches=full['15_mean_scale_branches']
        fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
        scenarios=['no_noise','scale_only','mean_only','mean_plus_scale_x1']
        for column,(dataset,row) in enumerate(branches.items()):
            for i,(scenario,color,marker) in enumerate(zip(scenarios,
                    [COLORS[v] for v in ['clean','fixed','mean_only','pi']],['o','x','s','D'])):
                m=row['groups'][scenario]['accuracy']
                axes[0,column].errorbar(i,m['mean']*100,yerr=m['std']*100,fmt=marker,color=color,capsize=3)
            axes[0,column].set_xticks(range(4),['关闭两支','仅尺度','仅均值','均值+尺度'],rotation=25)
            fraction=row['groups']['mean_plus_scale_x1']['scale_energy_fraction']['mean']*100
            axes[0,column].set_title(f'{DATASETS[dataset]}\n原尺度能量占比 {fraction:.3f}%')
            decorate(axes[0,column],'准确率（%；均值±SD）')
            for i,beta in enumerate([.25,.5,1,2,4]):
                r=next(q for q in row['comparisons'] if q['scenario']==f'mean_plus_scale_x{beta}')
                interval(axes[1,column],r,i,COLORS['pi'])
            axes[1,column].set_yticks(range(5),['σ×0.25','σ×0.5','σ×1','σ×2','σ×4'])
            axes[1,column].invert_yaxis();axes[1,column].axvline(0,ls='--',color='gray')
            axes[1,column].set_xlabel('相对同头均值：准确率差（百分点）');axes[1,column].grid(axis='x',alpha=.2)
        save(pdf,'03_mean_and_scale','同检查点分解：均值是主要收益来源',fig,
             '固定模型与随机数，分别关闭均值/尺度并改变推理尺度，检验分支作用。',
             '仅尺度接近干净水平；保留均值恢复收益。三数据集五个预设尺度倍率的额外准确率区间均包含零或触及零。',
             '原干预seed3010/MC16，完整五种子；不从测试集选倍率。能量占比在特征RMS缩放之前。同头推理干预不替代删除分支后重新训练。',
             ['15_mean_scale_branches'])

        sparse=full['11_sparse_heads']['exact_capacity_verification']
        paired=full['16_sparse_paired_intervals']
        fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
        for v in METHODS:
            values=sparse['curves'][v]
            axes[0].errorbar(sparse['targets'],np.array(values['mean'])*100,yerr=np.array(values['std'])*100,
                            color=COLORS[v],marker=MARKERS[v],mfc='white' if v=='pi' else COLORS[v],capsize=3)
        axes[0].set_xlabel('逐类严格非零概念数 NEC');decorate(axes[0],'分类准确率（%；均值±SD）')
        rows=[{r['control']:r for r in paired['comparisons'] if r['nec_per_class']==k} for k in sparse['targets']]
        forest(axes[1],rows,[f'NEC={k}' for k in sparse['targets']],['mean_only','adapter_matched'])
        methods_legend(fig)
        save(pdf,'04_matched_nec','同稀疏度：Pi与确定性适配的收益接近',fig,
             '在共同可行的严格NEC下比较，排除只比较不同稀疏度造成的混杂。',
             '七个共同预算下，Pi均优于干净剪枝头；相对仅均值和等容量的14个区间均包含零。',
             '每类NEC=1/5/10/20/40/80/87；20个来源头。后剪枝未逐预算优化或重校准偏置，不能称为该NEC最优性能；低预算偏置不匹配可放大差距。',
             ['11_sparse_heads','16_sparse_paired_intervals'])

        stress=full['14_image_corruptions'];backgrounds=full.get('19_background_attributes',{})
        task_rows=[];task_labels=[];attribute_rows=[];attribute_labels=[]
        for dataset,r in stress.items():
            task_rows.append({v['control']:v for v in r['comparisons'] if v['case']=='all_15_equal_weight'})
            task_labels.append(DATASETS[dataset]+' / 15种损坏等权')
        case_names={'gray':'灰背景','train_background_17':'照片供体17',
                    'train_background_29':'照片供体29','train_background_43':'照片供体43'}
        for name,r in backgrounds.items():
            if r['version']!='natural_background':continue
            for case,label in case_names.items():
                matches=[c for c in r['comparisons'] if c['case']==case]
                task_rows.append({c['control']:c['task'] for c in matches})
                attribute_rows.append({c['control']:c['semantic'] for c in matches})
                bank='GDINO' if r['bank']=='RN50_GDINO' else 'SAM2'
                task_labels.append(f'CUB-{bank} / {label}');attribute_labels.append(f'CUB-{bank} / {label}')
        fig,axes=plt.subplots(1,2,figsize=(15,max(6,1.8+.53*len(task_rows))),layout='constrained')
        forest(axes[0],task_rows,task_labels,['clean','mean_only','adapter_matched']);axes[0].set_title('任务准确率')
        forest(axes[1],attribute_rows,attribute_labels,['clean','mean_only','adapter_matched'],key='balanced_accuracy_delta')
        axes[1].set_title('背景变化下真实属性 BA')
        methods_legend(fig,['clean','mean_only','adapter_matched'])
        comparisons=[q for r in backgrounds.values() if r['version']=='natural_background' for q in r['comparisons']
                     if q['case']!='identity' and q['control'] in ['mean_only','adapter_matched']]
        zero=all(q['semantic']['ci95'][0]<=1e-12 and q['semantic']['ci95'][1]>=-1e-12 for q in comparisons)
        save(pdf,'05_generalization','压力测试：适配收益延续，随机增益未获一致证据',fig,
             '在固定损坏与背景变化下，检查任务及真实属性收益是否延续。',
             'CIFAR损坏测试中适配改善任务，相对确定性对照的区间包含零。'+
             ('已完成背景属性比较也未显示随机分布额外收益。' if zero else '背景属性的额外收益按各预设条件及区间分别解释。'),
             '损坏为自定义原生分辨率15条件，不是CIFAR-C；汇总保持共同源图配对。背景为训练照片块合成变换，原验证属性阈值冻结。两面板指标/尺度不同；不跨库比较属性绝对值。',
             ['14_image_corruptions','19_background_attributes'])

        concepts=full['10_concept_bank'];components=full['07_loss_components']
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for axis,key,condition,xlabel in [(axes[0,0],'images_per_class',lambda r:r['clusters']==64,'每类训练源图数（K=64）'),
                                          (axes[0,1],'clusters',lambda r:r['images_per_class']==50,'概念数K（每类50图）')]:
            rows=sorted([r for r in concepts.values() if condition(r)],key=lambda r:r[key])
            for v in METHODS:
                axis.errorbar([r[key] for r in rows],[r['groups'][v]['accuracy']['mean']*100 for r in rows],
                              yerr=[r['groups'][v]['accuracy']['std']*100 for r in rows],marker=MARKERS[v],
                              color=COLORS[v],linestyle={'clean':'-','mean_only':'--','adapter_matched':':','pi':'-.'}[v],
                              mfc='white' if v=='pi' else COLORS[v],capsize=3)
            axis.set_xscale('log',base=2);axis.set_xticks([r[key] for r in rows],[r[key] for r in rows])
            axis.set_xlabel(xlabel);decorate(axis,'CIFAR-10准确率（%；均值±SD）')
        names=list(components);x=np.arange(len(names));ratios={}
        for offset,v,color,label in [(-.12,'pi',COLORS['pi'],'参考Pi'),(.12,'pi_no_preservation','#D55E00','去掉概念保持')]:
            axes[1,0].errorbar(x+offset,[components[d][v]['standardized_concept_mse']['mean'] for d in names],
                              yerr=[components[d][v]['standardized_concept_mse']['std'] for d in names],
                              fmt='o',color=color,capsize=3,label=label)
        for i,d in enumerate(names):
            ratios[d]=components[d]['pi_no_preservation']['standardized_concept_mse']['mean']/components[d]['pi']['standardized_concept_mse']['mean']
            delta=(components[d]['pi_no_preservation']['accuracy']['mean']-components[d]['pi']['accuracy']['mean'])*100
            axes[1,1].scatter(i,delta,color='#D55E00',s=50)
            axes[1,1].annotate(f'{delta:+.3f}',(i,delta),xytext=(0,9),textcoords='offset points',ha='center')
        axes[1,0].set_yscale('log');axes[1,0].set_xticks(x,[DATASETS[d] for d in names]);axes[1,0].legend(frameon=False)
        decorate(axes[1,0],'标准化响应漂移MSE（对数轴；均值±SD）')
        axes[1,1].set_xticks(x,[DATASETS[d] for d in names]);axes[1,1].axhline(0,ls='--',color='gray')
        decorate(axes[1,1],'去掉概念保持 − 参考Pi（准确率百分点）')
        axes[1,1].set_ylim(min(-.1,axes[1,1].get_ylim()[0]),max(.15,axes[1,1].get_ylim()[1]+.10))
        methods_legend(fig)
        save(pdf,'06_concepts_and_preservation','构建与约束：保留概念响应不能只看分类准确率',fig,
             '比较稀少/冗余概念条件，并检验概念保持约束的作用。',
             f'{len(concepts)}个完整配置中学习式对照接近。去掉概念保持后，CIFAR-10/CUB响应漂移约增至{ratios["cifar10"]:.2f}/{ratios["cub"]:.2f}倍，任务准确率变化较小。',
             '源图严格嵌套，沿用参考验证正则；上图与左下误差线为SD，右下为均值差、无区间。响应漂移不是属性准确率。CUB选定λe=0，去能量项是重复条件。',
             ['10_concept_bank','07_loss_components'])

    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    rows=[]
    for case in core:
        for method,g in case['groups'].items():
            rows.append({'实验':case['case'],'方法':LABELS[method],'适配种子数':5,
                         '准确率均值%':g['accuracy']['mean']*100,'准确率SD%':g['accuracy']['std']*100,
                         'NLL':g['nll']['mean'],'ECE':g['ece_15_bins']['mean']})
    with (dest/'核心分类指标.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pending=json.loads(Path('reports/ONE_HOUR_CLOSEOUT.json').read_text())
    explanation='Pi相对干净模型的提升稳定存在，但在当前测试条件下尚未证明随机尺度相对确定性适配有独立增益；区间包含零不等于等价性证明。'
    lines=['# Pi噪声研究：主要结果（精简版）','',f'更新：{stamp}。本阶段按一小时收尾要求整理。','',
           '**主要判断**：'+explanation,'',
           '灰＝干净，橙＝仅均值，蓝＝等容量适配器，玫红＝Pi，绿＝同一Pi的均值推理。正式测试集、适配种子0–4；区间为2,000次配对种子/共同图像bootstrap，条件于固定原型及已选模型，未校正多重比较。','',
           '[六组图表PDF](主要实验图表合集.pdf) · [详细结果与核验](详细结果/index.html) · [分类指标CSV](核心分类指标.csv)','']
    for section in sections:
        name=section['file'];lines.extend([f"## {section['title']}",'',f"**目的**：{section['purpose']}",'',
                       f"![{section['title']}]({name}.png)",'',f"**结论**：{section['conclusion']}",'',
                       f"**口径**：{section['note']}",''])
    lines.extend(['## 收尾范围与下一阶段','',
                  f"已形成{len(core)}组核心分类五种子对照、两个CUB概念库真实属性、完整GDINO定位、严格NEC、三数据集分支干预和{len(concepts)}个概念构建配置。背景/检索的最终完成范围以详细结果和收尾清单为准。",'',
                  '本阶段延期：剩余ViT数据集矩阵、Places365/ImageNet完整概念库及模型、ImageNet-R、SAM2完整配对定位。已保存模型和分片保留；未完成项没有填值，也不支持对这些数据集的结论。','',
                  '补充发现：原200轮Pi头重拟合改善CIFAR-10，却降低CIFAR-100；随机条件库对照存在初始化/几何混杂，不能解释成纯语义因果作用。更多图表及来源留在详细结果。','',
                  '**下一阶段建议**：以确定性均值/同容量适配器为强基线，优先研究真实概念约束和固定阈值在背景变化下的稳定性；需要主张随机分布收益时，再设计独立采样与匹配容量证据。',''])
    (dest/'主要实验结果.md').write_text('\n'.join(lines))
    cards=[]
    for s in sections:
        f=s['file'];cards.append(f'<article id="{f}"><h2>{html.escape(s["title"])}</h2><p><b>目的：</b>{html.escape(s["purpose"])}</p><img src="{f}.png" alt="{html.escape(s["title"])}"><p><b>结论：</b>{html.escape(s["conclusion"])}</p><p class="note">{html.escape(s["note"])}</p><p><a href="{f}.svg">SVG</a> · <a href="{f}.pdf">PDF</a></p></article>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pi噪声研究 · 主要结果</title><style>body{margin:0;background:#f4f6f8;color:#1b2937;font:16px/1.8 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:28px}header,article,footer{background:white;padding:24px 28px;border-radius:12px;margin-bottom:22px}img{width:100%;height:auto}h2{font-size:22px}a{color:#0072b2}.note{font-size:14px;color:#596678}@media(max-width:650px){main{padding:10px}article,header,footer{padding:16px}}</style><main>'
    page+=f'<header><h1>Pi噪声研究 · 六组主要结果</h1><p>{html.escape(explanation)}</p><p class="note">{stamp} · 完整五种子 · 正式测试集</p><p><a href="主要实验图表合集.pdf">图表PDF</a> · <a href="主要实验结果.md">简要说明</a> · <a href="详细结果/index.html">详细结果</a> · <a href="收尾清单.md">完成与延期清单</a></p></header>'+''.join(cards)+'<footer>未完成项不填值；现有证据不证明随机性无效或方法等价。详细统计口径、额外实验与来源核验保留在详细结果。</footer></main></html>'
    (dest/'index.html').write_text(page)
    (dest/'plot_data.json').write_text(json.dumps(plot_data,ensure_ascii=False,indent=2,allow_nan=False))
    record={'generated_at':stamp,'figures':sections,'source_directory':str(source.resolve()),
            'source_manifest_sha256':digest(manifest_path),'source_plot_data_sha256':digest(data_path),
            'source_delivery_verification_sha256':digest(audit_path),
            'source_evidence_counts':{k:v for k,v in audit.items() if k not in ['figure_sizes','delivery_file_sha256']},
            'plot_data_sha256':digest(dest/'plot_data.json'),'export_code_sha256':digest(__file__),
            'plot_helper_sha256':digest(Path(__file__).with_name('export_major_results.py')),
            'one_hour_closeout_sha256':digest('reports/ONE_HOUR_CLOSEOUT.json'),
            'limits':'Six main figures selected for presentation, with every source figure preserved in the full report. '
                     'Not completion of the postponed full five-dataset/three-backbone matrix; no missing values imputed. '
                     'Multiple figures/protocols reuse models and are not independent training repetitions.'}
    (dest/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps({'main_figures':len(sections),'destination':str(dest.resolve())},ensure_ascii=False))


if __name__=='__main__':main()
