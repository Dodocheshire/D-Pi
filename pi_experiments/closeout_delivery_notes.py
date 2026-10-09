"""Write concise, evidence-bound completion and deferral notes for this stage."""

import json
from pathlib import Path


def main():
    dest=Path('../Essay/D/report')
    audit=json.loads((dest/'详细结果/交付核验.json').read_text())
    data=json.loads((dest/'详细结果/plot_data.json').read_text())
    state=json.loads(Path('reports/ONE_HOUR_CLOSEOUT.json').read_text())
    cases=data['01_task_accuracy']
    background=data['19_background_attributes']
    assert len(background)==4 and all(r['verified_models']==25 for r in background.values())
    retrieval=json.loads(Path('reports/retrieval_evidence_verification_v1/RN50_SAM2.json').read_text())
    lines=['# 本阶段收尾清单','',
           '执行窗口：2026-10-05 11:23:25–12:23:25 UTC。按用户要求在一小时内收尾，不再自动继续旧实验队列。','',
           '## 完成并纳入报告','',
           f'- {len(cases)}组核心分类对照，每组五方法×五适配种子；区间在详细图表导出时重新复算。',
           '- CIFAR-10概念构建的全部8个预设配置已齐全，最后K=1024组25模型和五个配对区间独立核验通过。',
           '- CUB两个概念库×两个背景协议均齐全：100个模型引用、50个唯一检查点、500个条件缓存，100个属性BA区间与100个任务区间。不同协议复用模型，不能计成独立训练。',
           f'- 两库保存检索的原生标注核验完成：GDINO 41,175、SAM2 {retrieval["verified_query_selections"]:,}组前五选择；各25模型、完整5,794张测试图。仅核验标注与精度，不重新验证模型排序。',
           '- 既有真实属性、完整GDINO定位、严格NEC、三数据集均值/尺度干预及损坏压力测试保留。六张主图附目的、结论；完整19组图与来源核验放入详细结果。','']
    cub=next((r for r in cases if r['case']=='ViT-B/16 / CUB-SAM2'),None)
    if cub:
        r=cub['contrasts']['clean']
        lines.extend([f'本小时新增ViT-B/16 CUB：原24组验证NLL搜索与全部25个模型完成。干净→Pi准确率为{cub["groups"]["clean"]["accuracy"]["mean"]*100:.3f}%→{cub["groups"]["pi"]["accuracy"]["mean"]*100:.3f}%；Pi−干净{r["accuracy_delta"]*100:+.3f}个百分点，95%[{r["ci95"][0]*100:.3f},{r["ci95"][1]*100:.3f}]。确定性与同头对照见主图1。',''])
    else:lines.extend(['ViT-B/16 CUB未达到完整五种子及汇总要求，已完成文件保留，本版主图不纳入部分结果。',''])
    lines.extend(['## 延期与保存','',
                  '- 剩余ViT小数据集矩阵、Places365/ImageNet完整概念库及模型、ImageNet-R、SAM2完整配对定位延期；不填缺失结果。',
                  '- 大数据集全图特征虽已准备完成，仍不构成Pi性能结果；提案源图部分分片保留。',
                  '- SAM2已完成的部分定位与86个额外原子图像缓存保留，不代替五种子完整定位。',
                  '- 长任务与等待后续矩阵的跟随队列已按本阶段限时要求结束；原命令、进程身份及缓存恢复依据记录在D-Pi/reports/ONE_HOUR_CLOSEOUT.json。','',
                  '## 进入下一阶段的主要判断','',
                  '学习式适配的收益稳定，当前尚未证明随机尺度相对确定性适配有独立收益；包含零的区间也不证明等价。优先保留仅均值/同容量强基线，进一步研究真实概念约束和背景变化下固定阈值的稳定性。','',
                  '诊断线索：SAM2 Pi原图→灰背景的属性BA为61.148%→54.513%，AUROC为64.796%→65.254%，阳性召回为59.186%→26.178%。这些不同反应提示需同时检查排序和固定阈值判定，不能将全部退化归因为阈值；本阶段未用测试属性重新校准。','',
                  '[六张主要图表](主要实验图表合集.pdf) · [简要结果说明](主要实验结果.md) · [详细结果](详细结果/index.html)',''])
    assert state.get('compute_closeout_finished_at'), 'Finalize process state before writing completion notes'
    (dest/'收尾清单.md').write_text('\n'.join(lines))
    print(json.dumps({'core_cases':len(cases),'concept_configurations':audit['completed_concept_configurations'],
                      'background_groups':len(background),'notes':str(dest/'收尾清单.md')},ensure_ascii=False))


if __name__=='__main__':main()
