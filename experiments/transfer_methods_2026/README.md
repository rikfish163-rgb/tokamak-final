# J-TEXT → EAST 跨装置迁移方法调研与实验设计（Transformer backbone）

> **当前主线索引（2026-08-31）**：请先阅读
> [`MAINLINE_STATUS_20260831_CN.md`](./MAINLINE_STATUS_20260831_CN.md)。本文件前面较早的
> sweep 记录保留作历史背景，不能覆盖当前主线的标签、父模型、炮序和结果合同。
>
> 当前唯一研究主线由同一 Adv25 训练/标签/架构合同派生出两条严格隔离的下游
> profile：`jtext_exact → J-TEXT 独立在线主动分支`，以及
> `east_success → EAST support110 迁移 → balanced360 静态封存 → 重置后
> EAST V3C 炮内/炮间在线主动`。两者不是同一个 checkpoint SHA；J-TEXT 在线
> 状态绝不作为 EAST 迁移父模型。
>
> 正式实验产物只允许写入挂载盘
> `/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/`；
> `/home/hetaisheng/fuison_ext4_runs` 和 `/dev/shm` 只能被拒绝或作为临时运行位置，
> 不能作为最终结果目录。当前可复现入口为
> [`run_jtext_adv25_active_mainline.py`](./run_jtext_adv25_active_mainline.py) 和
> [`run_east_v3c_success_mainline.py`](./run_east_v3c_success_mainline.py)。后者是
> 已复现出 EAST 在线净增益的 `shot-first → endpoint` V3C 兼容谱系；
> [`run_east_v3c_classfirst_mainline.py`](./run_east_v3c_classfirst_mainline.py) 只保留为
> 采样器/父模型变化的诊断对照，不能与主线结果混用。

> 本次封存回执见 [`MAINLINE_RECEIPT_20260831_CN.md`](./MAINLINE_RECEIPT_20260831_CN.md)，
> 发布包中的机器可读结果见
> [`results/current_mainline_20260831/receipt/mainline_results.json`](../../results/current_mainline_20260831/receipt/mainline_results.json)，
> 中文结果表见
> [`results/current_mainline_20260831/receipt/MAINLINE_RESULTS_CN.md`](../../results/current_mainline_20260831/receipt/MAINLINE_RESULTS_CN.md)。
> 本公开仓库不包含原始 HDF5 和大体积逐窗口数组；外置盘运行根目录仅作为复现实验的
> 数据与结果来源，发布后的可携带证据统一位于 `results/current_mainline_20260831/`。
> 机器审计入口为 [`audit_current_mainline.py`](./audit_current_mainline.py)。发布包中的审计结果
> 位于 `results/current_mainline_20260831/receipt/mainline_audit_20260831_v2.json`，并明确区分
> J-TEXT 独立在线分支、EAST V3C 诊断主线和 `locked` 负对照。

> **EAST support 标签边界：** `config/splits/east_paper131_split_all_in.json`
> 只提供历史兼容 sampler 的炮号角色（声明 10D+100N）。主线运行会逐炮读取
> `data/EAST/unified_hdf5/*/*.hdf5` 的 `meta/IsDisrupt` 和 `meta/DownTime`；当前
> 这 110 个炮号的权威组成是 15D+95N，差异炮号会写入
> `east_support_label_audit`，不会被静默按 JSON 文字标签覆盖。

在线因果合同分为两层：

```text
炮内：当前信号前缀 → 端点预测/封存 → 无标签 fast-state 更新 → 下一端点
炮间：整炮预测/封存 → 不读结局的查询 → 仅被选炮 HDF5 揭示 → 下一炮前持久更新
```

炮内更新不能读取当前炮结局；未查询炮的结局在整个在线流完成前不可访问。
`causal_order_detail` 会随新运行写入 manifest，旧回执由审计器按兼容合同显式标记。

`shot_alarm.py` 是早期的单炮聚合诊断入口，不是当前 V3C 结果的生成器。它的默认
命令现在只允许用 support110 选阈值，测试阈值反选必须显式加
`--diagnostic-test-tuned`，并在输出中标记为泄漏风险诊断；因此不能把它输出的
`tpr0.8_test` 当作正式指标。

## 当前主线结果（2026-08-31）

以下数字来自挂载盘上的新鲜复现，而不是把不同历史目录拼接得到的汇总。
J-TEXT 使用 `jtext_exact` profile；EAST 使用 `east_success` profile 的静态
J-TEXT checkpoint，经
support110 适配后在 balanced360 上先做静态封存，再从同一父模型重置运行
在线主动阶段。

| 分支 | 静态 | 在线主动 | 解释 |
|---|---:|---:|---|
| J-TEXT TEST220（lead > 10 ms） | TPR 0.836364 / FPR 0.109091 | TPR **0.854545** / FPR **0.109091** | 查询 44/220；延迟反馈专家权重更新 |
| EAST seed131（lead ≥ 30 ms） | TPR 0.794444 / FPR 0.100000 | TPR **0.800000** / FPR **0.094444** | 查询 94/360；V3C 炮内/炮间路径 |
| EAST seed133（lead ≥ 30 ms） | TPR 0.761111 / FPR 0.116667 | TPR **0.761111** / FPR **0.111111** | 查询 99/360；TPR 不下降 |
| EAST seed137（lead ≥ 30 ms） | TPR 0.816667 / FPR 0.144444 | TPR **0.816667** / FPR **0.138889** | 查询 98/360；TPR 不下降 |

EAST 三 seed 算术均值为 static `TPR@30=0.790741, FPR=0.120370`，online
`TPR@30=0.792593, FPR=0.114815`，因此在本次开发复现中观察到
`ΔJ=(TPR@30−FPR)=+0.007407`。J-TEXT 20% 预算的
`ΔJ=+0.018182`。两组数据均为 `TEST_DRIVEN_EXPLORATION`：TEST220 和
balanced360 曾参与方法开发，不能写成独立未来确认结果。

EAST 这组数是完整 V3C 决策系统的开发诊断，不是单个神经模块的干净增益：
运行使用 `development-search`，同时包含炮内/炮间更新、endpoint memory、
episode gate 和告警保护，且三颗 seed 的查询数为 94/99/98。seed131 已补充同一
parent、阈值、告警器、炮序和 94 炮 query manifest 下的
`intra-only/inter-only/intra+inter/neural-only` 及禁用炮间神经更新对照；该筛查
仍是单 seed 开发诊断，多 seed 归因尚未完成，不能据此声称迁移或 Transformer
表示本身带来 `+0.007407`。明细见
`results/current_mainline_20260831/east_v3c/attribution_seed131/aggregate.json`。

为防止误读，使用 EAST profile 在 J-TEXT 上的统一 profile 诊断没有增益：
static/online 均为 `TPR=0.800000、FPR=0.109091、ΔJ=0`，详见
`results/current_mainline_20260831/jtext_active/source_profile_diagnostic/budget20/`。
因此当前结果不能写成一个单一 checkpoint 的端到端数字；如需该结论，必须
预注册同一 SHA 并重新完成两条分支。

新增的 J-TEXT 局部神经 episode 诊断见
[`JTEXT_EPISODIC_NEURAL_DIAGNOSTIC_20260831_CN.md`](./JTEXT_EPISODIC_NEURAL_DIAGNOSTIC_20260831_CN.md)。
它严格执行查询后拟合、过去端点路由和可回退更新，但 seed131 的端点集合
max-kernel 版本出现负迁移（FP 17→19），重复历史支持门虽抑制了伤害却没有
救回炮。因此该分支保持 `DIAGNOSTIC_ONLY`，不改写 canonical J-TEXT active-Hedge，
也不进入 EAST V3C 或 EAST 迁移父模型。

EAST 主线的可执行入口是
[`run_east_v3c_success_mainline.py`](./run_east_v3c_success_mainline.py)，结果
目录为外置盘上的 `fuison_ext4_runs/adv25_v3c_success_shotfirst_replay_20260831/`；
发布包中的精简证据位于 `results/current_mainline_20260831/east_v3c/`。
J-TEXT 入口为 [`run_jtext_adv25_active_mainline.py`](./run_jtext_adv25_active_mainline.py)，
结果目录为外置盘上的 `fuison_ext4_runs/jtext_adv25_active_canonical_replay_20260831/`；
发布包中的精简证据位于 `results/current_mainline_20260831/jtext_active/canonical_replay/budget20/`。

> **2026-08-16 当前合同：** EAST类别与DownTime只认HDF5元数据；`shot_list/TEST`只作候选和差异审计。当前独立测试清单是 `config/splits/east_hdf5_authoritative_balanced360_v1.json`，最终确定性五seed交付见 `final_delivery_hdf5_deterministic_v2/README.md`。旧清单、旧结果和旧checkpoint只作历史诊断。

生成时间：2026-08-15（本 session）
数据合约摘要：
- J-TEXT 源域：1733 炮（378 破裂 / 1355 安全），23 通道，1 kHz，长度 32–518 点（中位 347）。
- EAST 目标域：2953 炮（1198 破裂 / 1755 安全 HDF5 标签），23 通道，1 kHz，长度 374–68998 点（中位 ~5000）。
- 迁移训练集（paper131 split）：10 破裂 + 100 安全 = 110 炮。
- 测试集（paper131 360 平衡）：180 破裂 + 180 安全，一次性封存。

## 统一评估协议（本 sweep 固定）

- 输入：以 23 通道、128 ms 因果历史窗口为单位；正类 = 窗口终点距 DownTime ≤ 50 ms。
- 骨干：Conv-patch embedding（patch=8,stride=8）+ 2 层 TransformerEncoder（d=64, 4 heads, GELU, pre-norm）+ mean-pool + 二分类头。
- 源域预训练：J-TEXT train 1386 炮（302 破裂 / 1084 安全）。
- 迁移训练：只允许用 EAST 10D+100N；对齐方法额外使用源域窗口"无标签"参与特征对齐。
- 阈值：在 10D+100N 上按 "TPR≥0.90 时取最低 FPR" 选择；随后在 180D+180N 上报告 AUC / TPR / FPR。
- 归一化：所有方法统一使用 J-TEXT train 拟合的 robust（中位/IQR）缩放，避免各方法 scaling 不一致。

## 候选方法清单（按实现优先级）

1. **target_only（EAST-only 上限基线下界）**：随机初始化 Transformer 只在 10D+100N 上训练。
2. **linear_probe（线性探针）**：冻结源域 encoder，只换 EAST 分类头 —— 测"表示可迁移性"。
3. **fine_tune（全量微调）**：源域 pretrain 后小学习率全参数微调 —— 朴素迁移基线。
4. **Deep CORAL**：冻结 encoder，训练特征投影 + 头，并用源/目标二阶统计量对齐（COV 对齐）。
5. **DANN**：域对抗训练，梯度反转层 + 域判别器，混淆 J-TEXT/EAST 特征。
6. **Deep MMD**：多带宽 RBF-MMD 在隐空间对齐边际分布（DAN/DDC 路线）。
7. （后续批次）**CDAN**（条件域对抗）、**ADDA**、**AdaBN / domain-specific BN**、**Tent / 测试时归一化适配**、**源域掩码自监督预训练 + 微调**、**对比式类别原型对齐**。

## 文献依据（要点 + 来源）

- 跨托卡马克破裂预测证实"低层局部特征更易迁移、高层时序模型装置相关"，这是冻结低层、适配高层的依据：Zheng et al., *Transferable Cross-Tokamak Disruption Prediction with Deep Hybrid Neural Network Feature Extractor*, [Semantic Scholar](https://www.semanticscholar.org/paper/5168041cbc2c81daadc27bfe44946627775f0f3d)；Communications Physics 2023 (https://www.nature.com/articles/s42005-023-01296-9)。
- S-CORAL / DART 与 PGFE-U、FLS 的少样本跨装置流程与"特征同语义是前提"结论：paper131，[arXiv S-CORAL](https://arxiv.org/pdf/2309.05361)。
- PhyDANet（physics-guided domain adaptation network）为 DANN + supervised MMD + 物理约束 + 重建的组合，见 IAEA FEC 2025 材料 [poster](https://conferences.iaea.org/event/392/contributions/36206/attachments/20179/35844/poster_wei.zheng.pdf)。
- 跨机器故障诊断领域：DANN/CDAN/MMD/CORAL 的公开实现与评测（TL-Fault-Diagnosis-Library 等），可复用对齐损失的成熟范式：https://github.com/Feaxure-fresh/TL-Fault-Diagnosis-Library, https://github.com/licluo/TL-Bearing-Fault-Diagnosis。
- 时序 Transformer 微调 / 适配：LoRA、冻结浅层只调深层（deep transfer learning / survey of deep learning for time series, arXiv:2401.13912）。
- 负迁移控制（本任务关键）：源/目标类别分布差异、条件对齐比边缘对齐更稳（CDAN）；低样本下首选"冻结 encoder + 轻量适配头 + 分布对齐"而非全量微调。

## 本轮结论驱动逻辑

文献与既往实验共同指向：在 10D+100N 极小目标域下，方法有效性排序预期为
`条件/类别感知对齐 ≥ 轻量微调/线性探针 ≥ 全量微调 ≥ 无对齐`，
且任何方法都不可能超过"干净公共特征坐标系 + EAST-only 上限"。
因此本 sweep 先定位"谁更不差、谁负迁移"，后续再接特征血缘表重建与条件对齐。
