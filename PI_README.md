# DCBM 的 Pi 噪声研究入口

此目录由 [官方仓库](https://github.com/KathPra/DCBM) 克隆。原有代码保持原样，新增内容集中在 `pi_experiments/`。研究方案见 [Essay/D](../Essay/D/研究方案.md)，官方代码分析见 [实现解读](../Essay/D/官方实现解读.md)。

此目录现为主项目的独立 submodule，远端为 [Dodocheshire/D-Pi](https://github.com/Dodocheshire/D-Pi)。仓库保留 DCBM 的官方提交历史，`upstream` 指向官方仓库；原始提交与逐文件哈希见 `UPSTREAM.json`。查看原提交：`git show dfb3f7d964c318a81d1fc72d9083340475d87c2d --stat`。新增训练权重与大型实验缓存保留本地；上游历史中自带的示例特征保留来源记录。

## 从官方模型导出

在官方训练脚本或 notebook 中，完成 `cbm.preprocess_data(...)` 和 `cbm.train(...)` 后导出：

```python
from pi_experiments.model import export_cbm
export_cbm(cbm, "dcbm_pack.pt")
```

导出包含按原顺序排列的视觉原型、标准化参数、线性头，以及训练、验证、测试图像嵌入与标签。Pi 训练固定视觉原型与训练集标准化参数，按官方 `投影 → softmax → StandardScaler` 计算概念。

## 训练 Pi 噪声

从 `D-Pi/` 运行，使用含 PyTorch 的环境：

```bash
python -m pi_experiments.train --pack dcbm_pack.pt --output outputs/cub_pi \
  --mode mean_scale --site feature --conditioning concept \
  --epochs 20 --mc-samples 8 --seed 0 --device cuda
```

默认冻结分类头，仅更新噪声生成器。加 `--refit-epochs 200` 可在生成器训练后固定采样训练概念，并重拟合 L1 分类头。`--mode none --refit-epochs 200` 是同预算干净重拟合对照。

| 参数 | 实验组件 |
| --- | --- |
| `--mode none/fixed/mean_only/mean_scale/adapter` | 干净基线、固定高斯、仅均值、可学习高斯、同参数量确定性适配 |
| `--site feature/concept` | 图像嵌入注入、概念投影分数注入（softmax 前） |
| `--conditioning concept/random_concept/mlp` | 固定视觉原型、随机原型条件、仅图像条件 |
| `--initial-scale` | 初始相对扰动强度 |
| `--concept-weight` | 标准化概念保持损失 |
| `--energy-weight` | 期望噪声能量约束 |
| `--mc-samples` | 验证与测试的概率平均采样数 |
| `--l1` | 重拟合分类头的稀疏系数 |

输出 `metrics.json`（准确率、NLL、NEC、标准化概念漂移、验证历史和配置）及 `model.pt`。概念标注评估、定位解释、干预曲线和内部特征注入按研究方案开展。

## 验证代码

```bash
python -m pytest pi_experiments/test_model.py -q
python -m pi_experiments.smoke --output /tmp/dcbm_smoke.pt
python -m pi_experiments.train --pack /tmp/dcbm_smoke.pt \
  --output /tmp/dcbm_smoke_run --epochs 2 --refit-epochs 2
```

合成数据只用于验证训练、选模与导出链路。官方真实数据复现使用 `data/Segments/scripts/` 和 `dcbm_training/` 中的提案生成、嵌入和训练流程；先按 [实现解读](../Essay/D/官方实现解读.md) 处理数据嵌入入口。

## 真实数据研究

`pi_experiments.prepare` 已提供独立划分、完整图像特征缓存、SAM2 提案、官方聚类与原型选择、验证选模的分类头训练。`pi_experiments.suite` 管理验证超参数搜索、五种子对照及评估；各步骤记录进程与日志。

```bash
python -u -m pi_experiments.suite --datasets cifar10 cifar100 cub
python -m pi_experiments.summarize
```

验证搜索使用 `--validation-only`；其模型分析使用 `--split val`。测试报告和逐样本预测由最终实验导出。CUB 属性与干预评估使用 `cub_annotations`、`cub_metrics`。当前进展见 [实验日志](EXPERIMENT_LOG.md)。

## 随机性、定位与分布变化

同一检查点的均值/尺度关闭与倍率诊断不重新训练，可直接检验随机尺度的边际贡献：

```bash
python -m pi_experiments.noise_diagnostics --pack PACK --run RUN --split val
python -m pi_experiments.render_diagnostics --diagnostic RUN/noise_diagnostics_val.json \
  --output reports/validation_diagnostics/mean_scale
```

`localization` 使用 RN50 layer4 Grad-CAM，通过完整 Pi 路径计算，原图干净 top-5 概念与网格随机数固定。正式评估覆盖全部测试图，按图缓存；`--limit 8` 仅用于调试，输出单独的 `localization_debug.json`。CUB 须先建立属性映射。分别报告非零热图的条件均值，以及把零热图计为零的全部概念均值和覆盖率。

```bash
python -m pi_experiments.localization --pack PACK --run RUN --directory artifacts/datasets/cub
python -m pi_experiments.corruptions --stage prepare --directory artifacts/datasets/cifar10
python -m pi_experiments.corruptions --stage evaluate --directory artifacts/datasets/cifar10 \
  --pack PACK --run RUN
```

损坏测试在原始图像分辨率上生成五类、各三个固定参数的扰动后重新编码，不重拟合标准化或模型。这是自定义压力测试，不能作为 CIFAR-C 基准结果。零强度噪声已验证与原始缓存的特征完全一致。

`postprocess` 跟随 `suite` 的实际 PID，优先在 CPU 上评估完成模型的分支分解和已缓存损坏；待主队列成功结束，串行准备剩余损坏并完成 CUB 定位：

```bash
python -u -m pi_experiments.postprocess --wait-pid MAIN_MANAGER_PID
```

进程状态见 `artifacts/logs/study_v1/status.json` 与 `artifacts/logs/postprocess_v1/status.json`，正式汇总见 [阶段报告](reports/study_v1/EXPERIMENT_REPORT.md)。

## 内部特征与概念构建消融

```bash
python -m pi_experiments.internal_cache --directory artifacts/datasets/cifar10
python -u -m pi_experiments.internal_suite --datasets cifar10
python -m pi_experiments.concept_ablation --dataset cifar10 --stage baselines
```

内部缓存包含冻结 RN50 layer4 特征及训练通道 RMS；零扰动经过原注意力池化后必须还原原始 CLIP 嵌入。`internal_train --validation-only` 不打开测试缓存；`internal_suite` 完成验证选模后运行五种子对照。内部输出为 2,048 通道，训练参数数与全局版本不同，报告同时披露各位置的参数数和自身容量对照。

概念消融使用嵌套的 5/10/20/50 图子集和独立概念数变化，共享原始特征及提案缓存。GPU 显存较大的任务使用 `resources.gpu_lease` 串行协调，包括 SAM2、内部训练、GDINO、定位和额外基线；状态见 `artifacts/resources/gpu_owner.json`，活性以实际 PID 为准。

## CUB 的 GDINO 对照

```bash
python -u -m pi_experiments.gdino
python -u -m pi_experiments.gdino_study --wait-pid PROPOSAL_PID
```

模型使用固定修订和文件哈希，相同 CUB 划分及固定部位提示，正确转换检测器中心框到原图 xyxy。正式属性映射和评估采用 certainty≥3（原数据 2 表示猜测），另报告 ≥2/≥4 的敏感性结果。状态见 `artifacts/logs/gdino_v1/status.json`。

## 后续对照入口

```bash
python -m pi_experiments.concept_study --plan
python -u -m pi_experiments.concept_study
python -u -m pi_experiments.conditioning_study
python -u -m pi_experiments.head_sparsity
python -u -m pi_experiments.internal_report --wait-pid INTERNAL_MANAGER_PID
python -u -m pi_experiments.internal_suite --datasets cifar10 --grid expanded \
  --output-root outputs/internal_v2 --logs artifacts/logs/internal_v2 \
  --wait-pid INTERNAL_MANAGER_PID --upstream-status artifacts/logs/internal_v1/status.json
python -u -m pi_experiments.large_suite --wait-pid MAIN_MANAGER_PID
```

概念构建消融使用固定参考正则权重；随机条件追加对照保持参数初始化和训练随机数一致。L1 研究从现有冻结检查点拟合五个预设稀疏头，仅验证 NLL 决定轮次和 L1。内部扩展保留首轮结果。大数据集统一批量 512，ImageNet 明确标注为现有干净训练子集。执行记录及新增队列见 [实验日志](EXPERIMENT_LOG.md)。


## 背景、检索与跨骨干

```bash
python -m pi_experiments.natural_backgrounds --stage prepare
python -m pi_experiments.render_background_examples --natural
python -m pi_experiments.status_report
python -m pi_experiments.verify_completed
python -u -m pi_experiments.backbone_study --wait-weights-pid WEIGHTS_PID --proposal-pid MAIN_PID
python -u -m pi_experiments.backbone_study --wait-weights-pid WEIGHTS_PID --proposal-pid LARGE_PID \
  --datasets places365 imagenet --logs artifacts/logs/backbone_large_v1
python -u -m pi_experiments.rendition_backbones --wait-pid BACKBONE_LARGE_PID
```

自然背景使用独立 `artifacts/datasets/cub_background_natural/`，不覆盖最近像素填补的第一版。后者保留为合成压力测试；供体照片块也有缩放和边缘拼接的限制。`natural_background_study` 等待正式模型与背景缓存完成后用原属性阈值评估，分别写入 `outputs/natural_background_{sam2,gdino}_v1`。

`retrieval_regions` 建立独立 CUB 测试区域池；`retrieval_study` 等待对应正式模型，按真实属性、可见部位及 CLIP 可见区域计算标注代理 P@5，同时报告机会精度和覆盖率。`interpretability_report` 只读取完整正式定位文件生成能量—定位图。具体协议和当前可作出的结论见 [实验日志](EXPERIMENT_LOG.md)。

SAM2 准备默认 `--proposal-chunk-size 100`，独立子进程释放块间显存、完成记录自动复用，提案参数保持相同。活性以实际 PID 为准；状态快照是核验时间点的记录，使用上面的状态命令刷新。


第一版 Adam+L1 的头仍接近全密集，保留在 `outputs/head_sparsity_v1/`。近端版本单独启动，记录真正零权重及阈值 NEC：

```bash
python -u -m pi_experiments.proximal_sparsity
```

完整实现检查及结果核验分别见 `reports/IMPLEMENTATION_CHECKS.xml`、`reports/STUDY_VERIFICATION.json`；完整计划尚未全部完成，各正在运行的任务继续由管理器执行。


固定检查点采样预算按同一 16 次采样的前缀比较 1/4/8/16 次概率平均，不读取测试指标选择预算：

```bash
python -m pi_experiments.sampling_budget --pack PACK --run RUN
python -m pi_experiments.coverage_report
```

`sampling_study` 跟随正式管理器的 PID 和状态文件，在模型完成后自动评估；结果位于原模型目录及 `reports/sampling_budget_v1/`。覆盖清单仅检查文件存在，不能替代科学核验或判定完整目标完成。


条件库几何核验与保持库内几何的对齐消融：

```bash
python -m pi_experiments.conditioning_geometry
python -u -m pi_experiments.rotation_study
```

旋转对照使用统一带符号维度置换，保留分类原型并只改变生成器条件库；结果独立保存在 `outputs/rotation_conditioning_v1/`。随机条件实验与这一对照的区别见实验日志。


大图 SAM2 输入默认长边上限 1,024（Places365/ImageNet），使用与官方助手一致的 OpenCV 线性缩放。`--proposal-max-side 0` 明确选择不缩放；长边上限与 `--proposal-resolution` 的短边缩放不可同时启用。提案保存原图/提案画面尺寸，跨骨干裁剪由同一画面重建。

验证迭代上限触发的近端头扩展另存结果：

```bash
python -u -m pi_experiments.proximal_followup --wait-pid PROXIMAL_V1_PID
```

该入口等待首轮成功，并只根据验证选择文件决定延长上限，保存到 `outputs/proximal_sparsity_v2/`。

严格匹配每类非零 NEC 的评估等待延长拟合成功：

```bash
python -u -m pi_experiments.matched_nec --wait-pid PROXIMAL_V2_PID
python -m pi_experiments.sampling_report
```

`matched_nec` 的共同预算只由验证选定头的权重确定，逐类别核验实际非零数和 1e−5 阈值数均恰为 k；输出到 `outputs/matched_nec_v2/` 与 `reports/matched_nec_v2/`。近端拟合已产生零系数，已有 top-k 权重槽位曲线仅表示保留上限，不能直接当作 NEC=k。

`sampling_report` 在 CPU 汇总已完成的五种子统一前缀预测，输出十个预设准确率配对比较及均值推理对照区间图；部分完成组继续等待，不进行测试预算选择。保存于 `reports/sampling_budget_v1/paired_results.json` 和 `paired_accuracy_vs_mean.{svg,pdf,png}`。条件于固定模型与一次采样实现的描述性区间未作多重比较校正。

## 稀疏头容量对照与主要结果交付

```bash
python -m pi_experiments.sparse_capacity --wait-pid 591995
python -m pi_experiments.export_major_results
python -m pi_experiments.verify_major_results
```

第一条命令的管理器已成功完成，不要重复启动。它核验原严格 NEC 队列成功后，复用干净/Pi 选定头，追加仅均值与等容量头的相同近端拟合，再输出四方法共同可行的严格 NEC 曲线、配对区间和同 Pi 头的采样比较。状态见 `artifacts/logs/sparse_capacity_v1/status.json`。

`python -m pi_experiments.verify_sparse_capacity` 独立审核四方法共 20 个验证选定头：原检查点冻结不变量、源哈希、L1/迭代选择、共同预算、逐类严格非零数、保存预测和全部 28 个配对区间。七个共同预算为 1/5/10/20/40/80/87；原两方法的 91 预算曲线保留，不能混入四方法的共同预算。审核存于 `reports/sparse_capacity_v1/cifar10/exact_capacity_verification.json`。

`python -m pi_experiments.verify_attribute_evidence --bank RN50_SAM2`（或 `RN50_GDINO`）逐项核对完整属性预测的原生真值、certainty≥3 掩码、映射、验证阈值、源模型和测试索引，并重算每个属性的 BA/AUROC 与三个核心配对区间。两个库各 20 个核心缓存的审核已完成；分别评估 161/139 属性，不能直接比较不同属性交集的绝对宏平均。输出位于 `reports/attribute_evidence_verification_v1/`。

`python -m pi_experiments.verify_primary_matrix` 审核 RN50 三个小数据集的全部 12 方案×5种子，共 180 个模型、360 组均值/MC8 保存预测，并核验原 200 轮头重拟合的 30 对非头张量逐位一致；九个头重拟合配对区间另列。输出 `reports/primary_matrix_verification_v1/results.json`。原主矩阵已成功结束，后续解释和大数据集队列继续运行；这项核验不代表完整研究目标已完成。

导出入口在 CPU 核验完整五种子模型的冻结不变量和保存预测指标，并将主要结果绘成中文、固定方法颜色的 PNG/SVG/PDF，附目的、结论与口径。交付目录为 `../Essay/D/report/`；`index.html` 提供图表浏览，`主要实验图表合集.pdf` 提供合并文件，`plot_data.json` 与 `manifest.json` 记录数据和来源哈希。原报告快照另存 `artifacts/report_assets/major_results_source_snapshots/`，避免主交付目录混入大体积审计文件。绘图字体为本地缓存的 Noto Sans CJK SC。

核心分类图仅追加模型、预测、五方法五种子汇总及配对区间全部齐全的配置；每次导出独立复算全部核心配对区间。区间下界的浮点残差小于 1e−12 按贴近零边界解释，不将数值残差写成正增益。交付核验入口检查所有单图、合并 PDF 页数、文档/网页链接、目的/结论、来源 JSON 快照和数据/导出源码哈希，更新 `交付核验.json`；记录全体交付文件哈希。

定位区间使用 CPU 入口 `python -m pi_experiments.localization_uncertainty`，汇总已有完整五种子；跟随队列使用 `--wait-pid 597541`，已由管理器启动，勿重复运行。它重算逐图分子/概念覆盖的全测试指标，检查共同目标、网格及分母，再对共同图像和配对种子重采样。输出 `reports/localization_uncertainty_v1/`，不重跑 CAM，不改原定位缓存。主要图表增加真实属性—分类准确率、扰动能量—定位以及定位配对区间。

`python -m pi_experiments.manifest_prepare` 提前在 CPU 创建/核验 Places365 与干净 ImageNet 子集的最终清单，保存 `artifacts/datasets/{dataset}/manifest_preflight.json`。两个清单已完成；无需重复启动管理器。核验完整源池的索引覆盖、训练/验证分离、概念训练来源、类别对齐与每类数量。现有 ViT 大数据集队列可据此开始全图特征缓存，裁剪仍等待原 SAM2 提案，GPU 由原租约协调。ImageNet 明确不是全量训练，仍为 210,000 图源池再分 189,000/21,000，官方 50,000 验证图作为测试。

`python -m pi_experiments.large_feature_prewarm` 已由独立管理器执行原 RN50 的两个大数据集全图缓存；不改变原 `prepare --stage features` 的变换、权重、stage 锁或 GPU 租约。缓存完成后核验有限性、1024 维、清单索引和逐类数量，保存 `feature_prewarm_verification.json`。原大数据集研究管理器后续复用原子缓存，无需重新编码。ViT-B/16 的 Places365 全图缓存阶段已结束，独立核验记录见 `reports/preparation_verification_v1/ViT-B16_places365.json`；模型与区域库的正式结果仍在等待。

`python -m pi_experiments.backbone_feature_prewarm` 已由管理器执行剩余三组大数据集 ViT 全图缓存，依次为 B/16 ImageNet、L/14 ImageNet、L/14 Places365。复用原 `backbone_prepare` 入口、权重与 GPU 租约，完成后核验全部特征有限、清单索引和源标签逐图一致。记录存入 `reports/preparation_verification_v1/{backbone}_{dataset}.json`；区域阶段仍等待原 SAM2 几何。

`python -m pi_experiments.large_proposal_prewarm` 提前执行 Places365、ImageNet 的原 `prepare --stage proposals` 阶段。使用同一清单、SAM2 large 权重、64×64 采样、区域后处理和长边 1024 上限；原 stage 文件锁避免未来主队列重复写入，100 图子进程共享原 GPU 租约。各库齐全后逐图检查原生标签、原图/提案尺寸、框坐标、有限的 1024 维 RN50 区域特征和训练源覆盖，保存 `proposal_prewarm_verification.json`。显式记录空提案，不以全图补齐；该记录属于准备核验，不是 Pi 性能结果。

RN50 两个大数据集的完整全图缓存已完成。Places365 共 1,839,960 图，已与独立核验的 B/16 缓存逐项比较索引与标签；ImageNet 共 260,000 图，已通过 `python -m pi_experiments.rn50_source_audit --dataset imagenet` 逐项核对原生源标签、索引、维度和有限性。ImageNet 仍为原定训练子集；核验分别见 `reports/preparation_verification_v1/RN50_places365_source_alignment.json` 和 `RN50_imagenet_native_source_audit.json`。

`python -m pi_experiments.corruption_prewarm` 提前准备原 CIFAR-100 的 15 种原生分辨率损坏特征，既有 CIFAR-10 缓存只作复用核验。原 `corruptions.prepare` 变换、权重与 GPU 租约保持；每组核验原生标签、索引、有限性和元数据，记录 `corruptions/prewarm_verification.json`。原 CPU 后处理队列继续使用这些缓存评估模型。

`python -m pi_experiments.corruption_uncertainty --wait-pid <postprocess_pid>` 跟随实际 CPU 后处理进程，两个数据集各自五种子齐全后核验 25 个冻结模型与 375 组保存预测的指标，并给出 Pi 相对四个对照的全部 15 条件及等权汇总区间。等权汇总先在每个种子/源图内平均损坏条件，再重采样配对种子与共同源图，避免重复图像被当作独立观察。输出 `reports/corruption_uncertainty_v1/{dataset}.json`；这是自定义压力测试，不是 CIFAR-C，描述性区间未校正多重比较。

损坏预热成功结束后，CPU 模型评估也复用同一批次的确定性生成器缓存。五种模式的全部输出与随机数状态逐位一致检查已通过；原变换、编码器、MC8 和任务指标计算保持。旧进程按其已加载代码自然结束，后续子进程采用新入口；报告记录是否缓存与入口 SHA-256，旧/新代码均归档于 `artifacts/code_snapshots/`。预热的源哈希对应当时归档版本，不与后续评估版本混淆。

`python -m pi_experiments.noise_branch_evidence --wait-pid <main_study_pid>` 跟随三个小数据集的完整 Pi/重拟合 Pi 检查点，以 CPU 复算原定八个均值/尺度干预、固定 seed3010/MC16，保存逐图预测并汇总配对区间。每个未改变的输入批次只缓存确定性生成器输出；原 forward、随机数序列、概率平均与模型参数保持。输入原地修改时缓存失效，参数版本变化直接拒绝。输出位于 `outputs/noise_branch_evidence_v1/` 和 `reports/noise_branch_evidence_v1/`；旧诊断不覆盖，已有旧报告时复核其全部标量指标。单次均值与 16 次同值平均的浮点误差和类别变化单独记录，不要求逐位相等。

`python -m pi_experiments.selected_sparse_report` 在 CPU 汇总已齐全的五种子验证选定稀疏头，复算保存预测的分类指标和实际非零 NEC，再给出 Pi−完整对照的配对区间。输出 `reports/sparse_capacity_v1/cifar10/selected_head_comparisons.json`。各方法保留自身选定 NEC，与四方法严格 NEC 比较分开；不纳入未齐全的等容量种子。

原型命名支持的实际 CLIP 可见窗口核验与独立复核：

```bash
python -u -m pi_experiments.visible_semantics --follow
python -m pi_experiments.visible_support_report
```

该队列在 CPU 跟随原 CUB 检查点及解释评估，保守过滤被区域中心裁剪移出支持部位的命名。映射另存为各原型目录的 `attribute_mapping_visible_k2048.json`，属性与干预输出到 `outputs/visible_semantics_v1/`，背景另存到 `outputs/visible_{background,natural_background}_v1/`；原文件保留。已有检索与部位结果按相同过滤派生，网格、前景和模型响应不重算。报告覆盖变化，`visible_support_report` 使用相同属性交集比较两种映射，不能直接将不同属性覆盖下的宏平均差当作 Pi 效果。

可见支持版本的真实属性配对区间：

```bash
python -u -m pi_experiments.semantic_uncertainty --wait-pid VISIBLE_SEMANTICS_MANAGER_PID
```

在 CPU 复现并保存 `semantic_predictions.pt`，检查 balanced accuracy 和 AUROC 与既有报告一致；双方五种子齐全后，固定验证阈值和属性覆盖，按共同测试图及配对种子计算 macro balanced accuracy 的分层 bootstrap 区间。结果位于 `reports/semantic_uncertainty_v1/`；若某次重采样丢失任一属性的一个类别，披露该次未定义，而不改变属性集合。AUROC 的均值和标准差仍在原属性汇总中。

背景变化下的真实属性逐图证据由 `background_semantic_uncertainty` 在 CPU 跟随可见支持背景输出。已启动独立管理器（状态见 `artifacts/logs/background_semantic_uncertainty_v1/status.json`），请先确认进程实际终止再恢复，不重复启动存活任务。两个 RN50 概念库、原背景/自然照片块各五固定条件，沿用原模型、映射、原验证阈值与 seed=6090+适配种子/MC8；缓存核验所有原 CPU 概率逐位一致、逐属性 BA/AUROC 重现，再计算各条件五种对照的任务及真实属性配对区间。新缓存和汇总位于 `outputs/background_semantic_uncertainty_v1/`、`reports/background_semantic_uncertainty_v1/`，不覆盖原背景结果。单模型复核示例：`python -m pi_experiments.background_semantic_uncertainty --bank RN50_GDINO --version natural_background --variant pi --seed 0`；完整组汇总省略 `--variant/--seed`。同图多个背景条件不当作独立源图重复，身份背景条件的推理采样实现与原属性评估不同。

`background_semantic_prewarm --bank RN50_GDINO --version natural_background --variant <方法>` 可在独立 CPU 进程准备已有完整背景结果的属性缓存；其文件锁与跟随器共享，完成缓存自动复用。自然照片块四个辅助方法队列已完成，跟随器保留并继续其他组，不要重新准备已有缓存。

`python -m pi_experiments.verify_retrieval_evidence --bank RN50_GDINO` 对原与可见支持检索报告独立核验原生测试标签、5794份区域分片与合并池、实际裁剪窗口、全部保存前五区域的属性/部位真值及 P@5、不同图像 P@5、机会精度和宏平均。输出 `reports/retrieval_evidence_verification_v1/`。该审核不重新推理或验证模型分数排序，不把标注代理当人工语义判定；同一 GDINO 测试池的后端偏向仍需披露。SAM2 版本需先等完整25份原报告和派生报告齐全。


### 限时收尾后的报告入口（2026-10-05）

本阶段已按用户的一小时要求结束，科学管理器存活数为0，GPU租约已释放。上文“已启动/跟随”的进程描述属于历史调度；当前完成与延期范围见 `../Essay/D/report/收尾清单.md`，进程身份、原命令、保存分片及结束操作见 `reports/ONE_HOUR_CLOSEOUT.json`。完整五数据集三骨干矩阵仍未完成，后续阶段不自动重启旧队列。

主目录只展示六组主要图，详细19图位于 `../Essay/D/report/详细结果/`。复现报告按顺序使用下列入口；完整导出务必明确指定详细目录，保持主目录精简。

```bash
python -m pi_experiments.export_major_results --output ../Essay/D/report/详细结果
python -m pi_experiments.verify_major_results --output ../Essay/D/report/详细结果
python -m pi_experiments.closeout_delivery_notes
python -m pi_experiments.export_concise_results
python -m pi_experiments.verify_concise_results
```

代码版本在 `artifacts/code_snapshots/`；两层交付核验分别保存于对应报告目录的 `交付核验.json`。精简核验同时检查完整报告审核SHA、逐组源数据相等、40行CSV、图像与PDF页数以及本地链接。后续计算恢复前根据选择的阶段重新确定依赖及启动顺序，保留现有原子缓存，不复用已失效的进程身份。
