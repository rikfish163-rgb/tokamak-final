# J-TEXT → EAST 主线结果回执（2026-08-31）

## Material Passport

- **ID：** `JTEXT-EAST-MAINLINE-20260831`
- **类型：** sealed experiment-result receipt
- **状态：** `ANALYZED`；总体声明状态为 `DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION`
- **数据边界：** TEST220 与 balanced360 已参与方法或参数开发，不能写成独立未来确认集。
- **来源：** 本文由已封存 JSON 回执生成；每个输入文件的 SHA-256 均记录在同目录 `mainline_results.json`。

## 1. 主线与不可混用边界

```text
J-TEXT TRAIN1353 (Adv25 label)
  ├─ jtext_exact static source → J-TEXT TEST220 static / online-active (独立分支)
  └─ east_success static source → EAST support110 transfer → EAST balanced360 V3C online
```

J-TEXT 在线状态、查询记录和权重没有导出到 EAST。EAST 迁移只从登记的静态
J-TEXT checkpoint 开始；`jtext_exact` 与 `east_success` 是不同的 checkpoint profile，
不能把两条分支的指标拼成同一个 SHA 的端到端结果。

J-TEXT 破裂炮的正样本窗为 `H=max(25 ms, AdvancedTime_ms)`；缺失值取 25 ms。两台装置的结局均以 HDF5
`meta/IsDisrupt`、`meta/DownTime` 为准。

## 2. 三阶段主结果

### 2.1 J-TEXT 静态源模型（TEST220）

| seed | timely TPR (lead >10 ms) | FPR | AUC | median lead (ms) |
|---:|---:|---:|---:|---:|
| 131 | 0.900000 | 0.154545 | 0.979091 | 68.875 |
| 133 | 0.836364 | 0.100000 | 0.961736 | 63.415 |
| 137 | 0.863636 | 0.136364 | 0.962893 | 69.774 |
| **算术均值** | **0.866667** | **0.130303** | **0.967906** | **67.354** |

该表是三颗静态源 checkpoint 的 J-TEXT TEST220 结果；它不是 EAST 迁移后的
目标域结果。TEST220 已用于在线方法开发，因此这里只作当前源域基线记录。

### 2.2 J-TEXT 独立在线主动分支（决策层 canonical）

| 视图 | 查询炮数 | TPR (lead >10 ms) | FPR | AUC | J=TPR−FPR |
|---|---:|---:|---:|---:|---:|
| static | 0 | 0.836364 | 0.109091 | 0.974215 | 0.727273 |
| online active (20%) | 44/220 | 0.854545 | 0.109091 | 0.975620 | 0.745455 |
| online − static | 44 | 0.018182 | 0.000000 | 0.001405 | 0.018182 |

这一分支的已观测增益来自查询揭示后的专家权重/决策层反馈；回执明确
`representation_updated=False`，所以不能称为 Transformer 表示在线学习。
它不进入 EAST 迁移。

### 2.3 EAST 静态迁移 → V3C 在线主动（balanced360）

| seed | static TPR@30/FPR | online TPR@30/FPR | ΔTPR | ΔFPR | ΔJ | queries |
|---:|---:|---:|---:|---:|---:|---:|
| 131 | 0.794444/0.100000 | 0.800000/0.094444 | 0.005556 | -0.005556 | 0.011111 | 94 |
| 133 | 0.761111/0.116667 | 0.761111/0.111111 | 0.000000 | -0.005556 | 0.005556 | 99 |
| 137 | 0.816667/0.144444 | 0.816667/0.138889 | 0.000000 | -0.005556 | 0.005556 | 98 |
| **算术均值** | **0.790741/0.120370** | **0.792593/0.114815** | **0.001852** | **-0.005556** | **0.007407** | **97.00** |

EAST 的操作口径为 `lead >=30 ms`；提前量大于 3000 ms 仍计入 TPR，
但在原始回执中单独记录。balanced360 先完成静态封存，再从同一迁移父模型
重置后按炮号运行在线流；结果是完整 V3C 系统的开发诊断。

## 3. EAST seed131 同一 parent 归因筛查

该筛查固定同一 support110 parent、阈值、炮序和 94 炮 query manifest。
它回答的是在线变化来自哪一层，不是重新选择最佳参数。

| 臂 | 炮内神经 | 炮间神经 | 决策/记忆侧 | online TPR@30 | online FPR | ΔJ |
|---|---:|---:|---:|---:|---:|---:|
| intra_only | 是 | 否 | 否 | 0.794444 | 0.100000 | 0.000000 |
| inter_only | 否 | 是 | 是 | 0.805556 | 0.100000 | 0.011111 |
| intra_inter | 是 | 是 | 是 | 0.800000 | 0.094444 | 0.011111 |
| neural_only_fixed_compute | 否 | 是 | 否 | 0.794444 | 0.100000 | 0.000000 |
| side_only_no_neural_inter | 否 | 否 | 是 | 0.794444 | 0.094444 | 0.005556 |

seed131 的关键解释：炮内 fast-state 虽执行反向传播，但本轮没有改变炮级
报警边界；固定计算量的 pure-neural 对照也没有改变边界。禁用炮间神经更新、
保留查询后的报警/episode/endpoint-memory 侧更新，仍复现一个 FP 的下降。
因此目前最稳妥的结论是：旧 V3C 的可见 FPR 增益主要由决策/记忆侧路径贡献；
现有证据不足以声称 Transformer 已学习新的跨装置破裂物理。

## 4. J-TEXT 神经 episode 诊断

guarded local-neural episode：static TPR=0.900000, FPR=0.154545; online TPR=0.900000, FPR=0.154545; ΔJ=0.000000。
该适配器的表示路径可训练，但源 encoder/head 被锁定；本轮没有产生炮级边界
增益，故不进入 canonical 在线主结果，更不进入 EAST parent。

## 5. 论文可表述范围

可以表述：

- 在 Adv25 标签合同和 HDF5 结局权威下，J-TEXT 源模型、EAST 少样本迁移和
  EAST 按炮序 V3C 在线主动流均有可复现的封存回执。
- V3C 是包含炮内无标签适应、炮间监督更新和查询后决策保护的系统；在开发流
  上三 seed 算术均值的 TPR@30 从 0.790741 提升到 0.792593，FPR 从 0.120370 降到 0.114815。
- 同一 parent 的 seed131 归因表明，系统级增益不能单独归因于神经表示更新。

不能表述：

- 不能把 TEST220/balanced360 结果写成完全未见的未来确认；
- 不能把 J-TEXT 决策层专家权重增益写成 Transformer 表示学习；
- 不能声称模型已经识别所有破裂因果机理；
- 不能把不同 source profile、不同 support sampler 或不同报警器结果拼接。

## 6. 下一步唯一合理动作

锁定 source checkpoint SHA、support parent SHA、阈值、报警器、query manifest 和
代码版本，在完全未参与方法选择的 EAST chronological future stream 上一次性运行；
同时保留同一 parent 的 static、side-only、neural-only 和 full 配对臂。若没有独立
未来流，本文结果应继续标记为诊断，不升级为 `T0_ADMITTED`。
