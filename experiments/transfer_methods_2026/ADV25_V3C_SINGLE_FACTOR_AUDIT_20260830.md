# Adv25 标签替换与旧 V3C 单因素审计

日期：2026-08-30  
运行环境：`rl_env`，RTX 5070 Laptop GPU，单 GPU  
状态：`DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION`

## 1. 审计问题

本轮只验证一个问题：把 J-TEXT 源模型的训练正窗口从历史固定
`0 <= TTD <= 125 ms` 改为

\[
H_i = \max(25\mathrm{\ ms},\;\mathrm{AdvancedTime}_i),
\qquad
y=1\ \Longleftrightarrow\ 0\leq\mathrm{TTD}\leq H_i,
\]

是否能够在不改变旧 EAST V3C 算法的前提下保持原有 EAST 在线增益。

除源模型训练标签外，以下内容保持旧 V3C 口径：

- `rebuilt17` 输入和 causal PatchTST/Transformer 结构；
- EAST support/adapt_high 训练流程；
- EAST support 阈值来源和 per-shot 风险聚合；
- 炮内 fast 更新、炮间 query/reveal 后监督更新；
- V3C 报警保护层、参数和 360 炮顺序；
- J-TEXT 在线分支不作为 EAST 的父模型。

HDF5 `/meta/{IsDisrupt,DownTime}` 是硬结局权威；
`AdvancedTime_J-TEXT.json` 只提供 J-TEXT 训练的逐炮弱时间边界。缺失
`AdvancedTime` 的破裂炮使用 25 ms，并在 checkpoint 元数据中记录。

## 2. 复现与完整性

历史固定125版本先用注册的 shot-first EAST support sampler 做了复现校验。
当前固定125重训的 encoder/head 张量与历史 source checkpoint 逐张量相同，
因此下游差异不是 source trainer 或随机初始化造成的。历史 V3C seed131
的最终静态/在线结果为 `0.8222/0.1556` 和 `0.8444/0.1333`。

Adv25 checkpoint 必须同时满足：

1. `training_label.policy == advanced_floor25`；
2. AdvancedTime 合同 schema 与当前文件 SHA 一致；
3. 不含可选灰带，`ambiguous_window_count == 0`；
4. `test_shards_read == 0`；
5. EAST 三个阶段记录相同 source checkpoint SHA。

新链路 runner 在运行前拒绝不满足上述条件的 checkpoint。三份完整
Adv25 四阶段 manifest 均通过 lineage 校验。

## 3. EAST 结果

指标为 balanced360 上的 per-shot V3C 口径：报警提前量至少 30 ms；
提前超过 3000 ms 仍计入 TPR，但另行记录。balanced360 已参与方法开发，
所以不能作为独立未来确认集。

| seed | 历史 V3C 静态 TPR/FPR | 历史 V3C 在线 TPR/FPR | Adv25 静态 TPR/FPR | Adv25 在线 TPR/FPR | Adv25 查询炮数 |
|---:|---:|---:|---:|---:|---:|
| 131 | 0.8222 / 0.1556 | 0.8444 / 0.1333 | 0.8667 / 0.1389 | 0.8667 / 0.1389 | 103 |
| 133 | 0.8389 / 0.1444 | 0.8389 / 0.1333 | 0.7556 / 0.1333 | 0.7611 / 0.1333 | 102 |
| 137 | 0.8611 / 0.2000 | 0.8611 / 0.1889 | 0.8333 / 0.1556 | 0.8333 / 0.1722 | 95 |
| **均值** | **0.8407 / 0.1667** | **0.8481 / 0.1519** | **0.8185 / 0.1426** | **0.8204 / 0.1481** | **100.0** |

以 \(J=\mathrm{TPR}_{30}-\mathrm{FPR}\) 计算：

- 历史 V3C：静态 `J=0.6741`，在线 `J=0.6963`，增益 `+0.0222`；
- Adv25：静态 `J=0.6759`，在线 `J=0.6722`，增益 `-0.0037`。

因此，严格的结论是：**旧 V3C 在线机制仍然执行，但 Adv25 标签替换后，
三 seed 平均的原有在线增益没有保持。** seed131/133 的在线 FPR 没有恶化
（133 还减少一个 false positive），seed137 产生了 3 个新增 false positive，
导致总体在线 J 下降。

为排除 epoch 选择造成的单一解释，另做了沿用历史 epoch=3 的诊断：

- seed131：静态 `0.7944/0.1000`，在线 `0.8000/0.0944`；
- seed133：静态 `0.7611/0.1167`，在线 `0.7611/0.1111`。

早期停止点能保留小幅 V3C 在线改善，但不能恢复历史三 seed 的 TPR 水平；
它不是可据 EAST 结果回选的正式模型。

## 4. J-TEXT 独立在线分支

J-TEXT TEST220 只作为独立的开发流，使用同一 Adv25 静态 checkpoint；
每个 seed 查询 22/220 炮，标签只在 `predict -> seal -> query -> reveal`
之后开放。炮内 fast adapter 每炮重置，炮间 adapter 只在已查询炮揭示后更新。

三 seed 平均：

| 分支 | timely TPR@10 ms | FPR | AUC | median lead |
|---|---:|---:|---:|---:|
| Adv25 静态 | 0.8758 | 0.1394 | 0.9711 | 65.32 ms |
| Adv25 在线 | 0.8758 | 0.1394 | 0.9711 | 65.32 ms |

当前 J-TEXT 在线更新有有限的参数变化，但没有炮级报警翻转，因此不能
宣称在线增益；它也没有进入 EAST 迁移。

## 5. 为什么“只改标签”会影响 EAST

这不是 V3C 报警器突然失效，而是源域监督分布发生了实质变化。训练炮上，
固定125的正窗口约有 21,912 个采样端点；Adv25 的逐炮窗口约为 5,557 个，
减少 16,355 个。188 个破裂训练炮的正边界发生改变；大多数有效边界为
25 ms，少数炮因 AdvancedTime 较大而扩展到约 166 ms。

因此 Adv25 训练会把原来 `25--125 ms` 的一部分端点从正类改成负类，
并使 Transformer 更集中于临近终端的表示。当前 seed131 的 Adv25
encoder 与历史 fixed125 encoder 的相对参数差异约为 0.1387。EAST
adapt_high 只开放上层目标侧参数，不能保证消除这个源表示变化；这解释了
为什么 EAST 结果会按 seed 改变，即使下游 V3C 代码完全相同。

## 6. 当前裁决

本轮没有证据支持“Adv25 标签替换后旧 V3C 增益自动保持”。可以保留的
工程结论是：

- 标签合同已经按用户指定的 `max(25 ms, AdvancedTime)` 实现并有 provenance；
- 旧 V3C 的 EAST 炮内/炮间反向传播和报警状态机没有被替换；
- J-TEXT 在线分支已与 EAST 父模型隔离；
- 数据访问顺序和 checkpoint 谱系通过自动检查。

如果必须满足“标签换成 Adv25 且 EAST 指标不低于历史 V3C”，下一轮必须
明确增加一个可审计的源表示保持机制（例如从历史 source warm-start 后的
受限 Adv25 微调，或显式旧/新标签多任务保持项），并把它作为新的单因素
消融；不能把 EAST 测试结果倒灌成 epoch/阈值选择，也不能暗中恢复固定125
标签而继续称为 Adv25。

## 7. 产物位置

- Adv25 三 seed 完整链路（历史产物的实际挂载盘位置）：`/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/v3c_adv25_samplerfix_20260830/`
- seed131 epoch3 诊断（历史产物的实际挂载盘位置）：`/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/v3c_adv25_epoch3_20260830/`
- Adv25 seed133 epoch3 诊断（历史产物的实际挂载盘位置）：`/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/v3c_adv25_epoch3_all_20260830/seed133/`
- 本报告：`experiments/transfer_methods_2026/ADV25_V3C_SINGLE_FACTOR_AUDIT_20260830.md`
