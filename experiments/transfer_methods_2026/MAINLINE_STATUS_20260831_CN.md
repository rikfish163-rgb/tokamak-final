# J-TEXT → EAST 当前主线状态（2026-08-31）

本文是当前工作树的主线索引，优先级高于目录中较早的实验笔记。它把源域、
J-TEXT 在线分支、EAST 迁移和 EAST 在线分支分开记录，避免把不同谱系的指标
拼成一条不存在的结果。

## 1. 唯一主线

```text
J-TEXT TRAIN1353（Adv25 标签；同一训练/标签/架构合同）
        │
        ├── J-TEXT downstream VAL 选择的 jtext_exact 静态 checkpoint
        │       ├── J-TEXT TEST220 静态评估
        │       └── J-TEXT TEST220 炮序在线主动（独立分支）
        │
        └── EAST downstream VAL 登记的 east_success 早期 checkpoint
                ├── EAST support110 迁移适配
                ├── EAST balanced360 静态封存
                └── 从同一适配父模型重置后的 EAST 在线主动
```

这里的“同一”指同一 Adv25 标签、数据切分和模型架构合同，**不指同一个
checkpoint 文件**。J-TEXT 分支和 EAST 分支当前分别使用各自由 VAL160 登记的
`jtext_exact` 与 `east_success` checkpoint profile；两组 SHA 在机器回执中单独
锁定。J-TEXT 在线状态、查询记录和权重从不进入 EAST 迁移。

其中 J-TEXT 破裂炮的正样本时间窗为

\[
H_i=\max(25\ \mathrm{ms},\;\mathrm{AdvancedTime}_i),
\]

缺少 `AdvancedTime` 时使用 25 ms。破裂/非破裂结局和 `DownTime` 只从逐炮
HDF5 的 `meta/IsDisrupt`、`meta/DownTime` 读取；JSON 或文本炮表只提供候选
炮号和顺序，不能覆盖 HDF5 标签。

## 2. 存储和环境合同

最终产物根目录固定为挂载盘：

```text
/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/
```

`/home/hetaisheng/fuison_ext4_runs` 已不存在，并且当前主线入口会拒绝任何
位于 `/home`、`/dev` 或本机根文件系统上的最终产物。计算中的高频临时文件
可以放在 `/dev/shm`，但只在完成哈希校验后复制到挂载盘。统一运行环境为：

```text
/home/hetaisheng/miniconda3/envs/rl_env/bin/python
PyTorch 2.11.0+cu128
NVIDIA GeForce RTX 5070 Laptop GPU
```

路径约束实现于 [`mainline_paths.py`](./mainline_paths.py)。

## 3. J-TEXT 源模型

三颗 Adv25 静态源 checkpoint 位于：

```text
fuison_ext4_runs/v3c_adv25_exact_20260830/seed131/source/source_checkpoint.pt
fuison_ext4_runs/v3c_adv25_exact_20260830/seed133/source/source_checkpoint.pt
fuison_ext4_runs/v3c_adv25_exact_20260830/seed137/source/source_checkpoint.pt
```

它们使用 J-TEXT TRAIN1353 训练、VAL160 选择阈值和 epoch；TEST220 不参与
训练或阈值选择。seed131 的当前可核验 TEST220 记录在
`seed131/source/test220_metrics.json`：

| 指标 | 数值 |
|---|---:|
| timely TPR（lead > 10 ms） | 0.900000 |
| FPR | 0.154545 |
| AUC | 0.979091 |
| median lead | 68.875 ms |
| raw TPR | 0.981818 |

这里的 `timely TPR` 是 J-TEXT 的源域口径；它不能直接与 EAST 的 30 ms
指标混用。

EAST V3C 诊断主线使用的是同一 Adv25 训练合同下、按各 seed 的 VAL160 历史
选择并登记为 `east_success` 的早期 checkpoint，具体 SHA 和路径由 EAST 入口
白名单固定为：

```text
seed131: fuison_ext4_runs/v3c_adv25_epoch3_20260830/seed131/source/source_checkpoint.pt
seed133: fuison_ext4_runs/v3c_adv25_epoch3_all_20260830/seed133/source/source_checkpoint.pt
seed137: fuison_ext4_runs/v3c_adv25_epoch3_all_20260830/seed137/source/source_checkpoint.pt
```

这不是把 J-TEXT 在线模型用于 EAST；它是 EAST 诊断复现所登记的另一组静态
源 checkpoint。J-TEXT 在线分支和 EAST 迁移分别保存各自的 parent SHA，不能
交叉替换。当前证据不支持把两条分支描述成“同一 SHA 的端到端单链”；若要在
论文中声称单一共享 checkpoint，必须重新预注册并重跑两条下游实验。

## 4. J-TEXT 在线主动分支

入口：[`run_jtext_adv25_active_mainline.py`](./run_jtext_adv25_active_mainline.py)

该分支使用三颗**静态** Adv25 source experts，按 TEST220 炮号升序运行。每个
20 炮块先完成所有信号前缀预测和封存，再从主动价值排序中选择固定预算炮，
最后才打开被选炮的 HDF5 结局并更新跨炮专家权重。它不会把 J-TEXT 在线状态
传给 EAST。

当前复现的 20% 预算为：

| 视图 | 查询炮数 | timely TPR | FPR | J=TPR−FPR |
|---|---:|---:|---:|---:|
| static | 0 | 0.836364 | 0.109091 | 0.727273 |
| online active | 44 | **0.854545** | **0.109091** | **0.745455** |
| online − static | 44 | +0.018182 | 0.000000 | +0.018182 |

10%、30% 预算的当前诊断曲线为：

| 预算 | 查询炮数 | static TPR/FPR | online TPR/FPR | ΔJ |
|---:|---:|---:|---:|---:|
| 10% | 22 | 0.836364 / 0.109091 | 0.872727 / 0.145455 | 0.000000 |
| 20% | 44 | 0.836364 / 0.109091 | 0.854545 / 0.109091 | **+0.018182** |
| 30% | 66 | 0.836364 / 0.109091 | 0.845455 / 0.109091 | +0.009091 |

这组 J-TEXT 在线增益的准确归因是**延迟反馈的专家权重更新**，不是
Transformer 表示学习。对应的结果状态为 `TEST_DRIVEN_EXPLORATION`；
TEST220 已参与方法开发，不能写成独立未来确认。

为检查 source profile 是否会改变结论，曾用 EAST 分支的 `east_success` profile
在同一 J-TEXT TEST220 合同上做了显式一致性诊断。结果为 static
`TPR=0.800000/FPR=0.109091`，online `TPR=0.800000/FPR=0.109091`，
`ΔJ=0`；产物位于
`fuison_ext4_runs/jtext_adv25_active_east_success_replay_20260831/budget20/`。
它不是 J-TEXT canonical 主结果，但说明不能为了拼接指标而静默替换 source
profile。

结果目录：

```text
fuison_ext4_runs/jtext_adv25_active_mainline_20260831/budget20_verified/
fuison_ext4_runs/jtext_adv25_active_mainline_20260831/budget10_current/
fuison_ext4_runs/jtext_adv25_active_mainline_20260831/budget30_current/
```

## 5. EAST 迁移与在线主动分支

入口：[`run_east_v3c_success_mainline.py`](./run_east_v3c_success_mainline.py)

此入口运行已经复现出稳定在线净增益的 V3C 双时间尺度实现：

1. 使用 EAST 入口白名单登记的、静态且未经过 J-TEXT 在线更新的 Adv25
   `east_success` checkpoint；
2. 仅用 EAST support110（HDF5 实际 15D+95N）形成历史兼容的
   `shot-first → endpoint` support parent；
3. 先完整预测并封存 balanced360 的静态结果；
4. 重置到完全相同的迁移父模型，按炮号顺序执行在线炮内/炮间更新；
5. 查询决定和预测封存之后才读取被查询炮结局；更新只影响后续炮。

这里的 `10D+100N` 是历史 JSON 炮号角色表的声明组成，不是本次兼容回放实际
用于监督的 HDF5 组成。入口逐炮读取 `meta/IsDisrupt` 后得到 **15D+95N**，其中
`68928、71316、71393、92933、93888` 在角色表中列为安全炮、但 HDF5 标记为破裂炮。
因此当前合同明确规定：角色表只决定 legacy sampler 的炮号集合，HDF5
`meta/IsDisrupt` 才是 support 标签权威；该差异不会被静默改写成 10D+100N。
机器审计会保存声明组成、权威组成和差异炮号。

在线因果不是一个笼统的六词顺序，而是两个嵌套屏障：

| 层级 | 允许访问和更新的顺序 |
|---|---|
| 炮内端点 | 当前因果信号前缀 → 预测并封存端点分数 → 仅用无标签损失更新 fast state → 继续下一个端点 |
| 炮间持久状态 | 完成整炮预测并封存 → 不读当前结局地决定是否查询 → 仅对被查询炮揭示 HDF5 结局 → 在下一炮前执行持久监督更新 |

所以“`predict → seal → query → reveal → update`”只概括炮间反馈；它不意味着
炮内无标签更新可以读取结局，也不意味着未查询炮的 HDF5 元数据会提前进入
查询分数。新运行回执将同时保存 `causal_order_detail`。

这里的 `shot-first → endpoint` support sampler 不是无关实现细节。它决定了
迁移父模型的参数状态；将它与后来的 `class-first` 代码混用会得到另一条
模型谱系，并可能出现“在线无增益”。入口会校验每个 seed 的 source SHA 和
support parent SHA，哈希不符时直接失败。

当前 Adv25 + shot-first V3C 的 balanced360 诊断结果如下：

| seed | static TPR@30/FPR | online TPR@30/FPR | ΔTPR | ΔFPR | 查询炮数 |
|---:|---:|---:|---:|---:|---:|
| 131 | 0.794444 / 0.100000 | 0.800000 / 0.094444 | +0.005556 | −0.005556 | 94 |
| 133 | 0.761111 / 0.116667 | 0.761111 / 0.111111 | 0.000000 | −0.005556 | 99 |
| 137 | 0.816667 / 0.144444 | 0.816667 / 0.138889 | 0.000000 | −0.005556 | 98 |
| **三 seed 算术均值** | **0.790741 / 0.120370** | **0.792593 / 0.114815** | **+0.001852** | **−0.005556** | **97** |

因此在这次已登记的开发复现中，EAST 观察到正向系统级净变化：三 seed 平均
`J=TPR@30−FPR` 从 **0.670370** 增至 **0.677778**，净变化 **+0.007407**。
三个 seed 的 FPR 均下降，TPR@30 均未下降；seed133 的 raw TP 计数减少 1，
但该变化发生在 30 ms 以外的 raw/operational 口径差异中，不能把它表述成
三个 seed 的所有指标都同时提高。

这里不能把 `+0.007407` 直接归因于某一个神经模块或物理损失。当前运行是
`run_mode=development`，命令带有 `--development-search`，同时启用了炮内/炮间
更新、raw8/endpoint memory、episode gate 和告警状态机保护；查询数为
94/99/98，也不是预先固定的同一百分比 quota。由于 balanced360 参与过这些
选择，当前结果只能称为“完整 V3C 决策系统的开发诊断”。本轮已在 seed131
冻结同一 parent、阈值、炮序和 94 炮 query manifest，完成了
`intra-only/inter-only/intra+inter/neural-only` 以及“禁用炮间神经更新”的配对
归因筛查；多 seed 复现仍未完成，不能把 seed131 筛查当作最终显著性证据。

此外，当前回执没有给出“同一测试合同下不做 EAST support110 适配”的 source-only
对照，因此不能从这组表单独计算迁移本身的增益；表中 static 是迁移后的条件基线。

结果目录：

```text
fuison_ext4_runs/adv25_v3c_success_shotfirst_replay_20260831/seed131/
fuison_ext4_runs/adv25_v3c_success_shotfirst_replay_20260831/seed133/
fuison_ext4_runs/adv25_v3c_success_shotfirst_replay_20260831/seed137/
```

单颗结果在各目录的 `final_episode_v3/summary.json`，其中 `results` 下的
`static_unified` 与 `online_unified` 是唯一应读取的两组 EAST 指标。

## 6. 当前失败分支和禁止混用项

普通 J-TEXT 神经 residual、错误炮门控 residual 和阶段均衡采样臂已完成
诊断，但没有改变 seed131 的 TEST220 指标（仍为 0.900000/0.154545）。
这说明当前 residual 更新不能被写成“已经学会新的 J-TEXT 物理表示”。
诊断产物分别位于：

```text
fuison_ext4_runs/jtext_adv25_error_gated_20260831/seed131/
fuison_ext4_runs/jtext_adv25_stage_balanced_20260831/seed131/
```

以下结果不能与当前 Adv25 主线直接拼接：

- `unified_online_v8_multiseed_summary.json`：旧的 locked V8 源/父模型谱系；
- `adv25_locked_v8_replay_20260830`：Adv25 源但使用旧 V8 参数合同，非当前
  shot-first V3C 主线；
- `v3c_adv25_samplerfix_20260830`：另一套 source/support 组合，不能替代当前
  shot-first V3C 主线；
- `adv25_v3c_classfirst_replay_20260831`：class-first sampler 的诊断对照，
  不能替代当前 shot-first V3C 主线；
- 任何在 TEST220 或 balanced360 上重新选择阈值、查询规则或报警参数的结果。

它们可以作为历史对照，但不能用来替换当前主线的 parent 或宣称新标签下
同一方法已经达到旧指标。

当前 EAST 仍缺少多 seed 的完整正交归因回执，尤其是独立关闭告警保护、episode
和 endpoint-memory 的配对臂；但 seed131 的第一轮同 parent 归因已完成。该轮
结果表和逐臂机器回执见
[`east_v3c_timescale_ablation_20260831/seed131/aggregate.json`](../fuison_ext4_runs/east_v3c_timescale_ablation_20260831/seed131/aggregate.json)。
在多 seed 复现前，任何“在线增益来自 Transformer 学到新的跨装置物理规律”的
表述都必须停止在假设层面。

### 6.2 EAST 同 parent 炮内/炮间归因筛查（seed131）

本轮所有臂都从同一 support110 迁移 parent 开始，先完成 360 炮静态封存，
再按相同炮序重置；需要查询的臂使用同一份由历史无标签决策封存得到的 94 炮
manifest，不能因为关闭某个更新模块而改变查询身份。EAST 标签仍只在
`predict → seal → query → reveal` 之后读取，指标为固定 30 ms 操作口径。

| 臂 | 炮内神经 | 炮间神经 | 决策侧更新 | static TPR/FPR | online TPR/FPR | ΔJ |
|---|---:|---:|---:|---:|---:|---:|
| intra-only | 是 | 否 | 否 | 0.794444 / 0.100000 | 0.794444 / 0.100000 | 0.000000 |
| inter-only | 否 | 是 | 是 | 0.794444 / 0.100000 | 0.805556 / 0.100000 | +0.011111 |
| intra+inter | 是 | 是 | 是 | 0.794444 / 0.100000 | 0.800000 / 0.094444 | +0.011111 |
| neural-only（固定 16+16、2 step） | 否 | 是 | 否 | 0.794444 / 0.100000 | 0.794444 / 0.100000 | 0.000000 |
| side-only（禁用炮间神经） | 否 | 否 | 是 | 0.794444 / 0.100000 | 0.794444 / 0.094444 | +0.005556 |

这组结果给出两个需要分开的结论。第一，炮内 fast-state 的
`loss.backward()` 确实执行了（19,177 次尝试，约 12.6k 次非零更新），但在
本轮炮级边界上没有独立增益；因此不能把“发生了反向传播”写成“已学到新的
破裂物理”。第二，禁用 shared/private 炮间梯度后，保留查询后的报警、episode、
endpoint-memory 和 query-value 侧更新仍可减少一个 FP，说明完整 V3C 的可见 FPR
增益主要由决策/记忆侧路径贡献。纯神经固定计算对照也没有改变边界；联合臂
虽同时少一个 FP，但相对 inter-only 少恢复一个及时报警，表明炮内与炮间更新
存在非加性相互作用。

因此，当前最稳妥的论文表述是“V3C 是一个包含神经持续适配和决策保护的系统，
在开发流上产生小幅系统级增益”；不能表述为“Transformer 在线重新学习了
EAST 的破裂机理”。该筛查是单 seed、开发集证据，不能替代三 seed 复现或未来流
确认。

### 6.1 J-TEXT 局部神经 episode 诊断（2026-08-31）

为检验“查询后的错误炮能否在后续相似轨迹上形成真正神经纠错”，新增了独立的
`run_jtext_episodic_online.py`。该分支不改写 canonical active-Hedge，也不进入
EAST。它在同一 Adv25 source checkpoint、同一 44/220 查询清单和固定 VAL160
阈值下，比较普通局部 residual、原型条件 residual、端点集合 max-kernel，以及
至少两个严格过去同类错误才允许路由的 probation 门。

结果为 seed131：单 episode 结构要么几乎不改变炮级告警，要么在作用函数被修正后
出现负迁移（`FP 17→19`，`J 0.745455→0.727273`）；重复支持 probation
抑制了负迁移，但没有救回任何炮。该结果支持“相似表征不足以证明共享破裂机制”，
不支持将局部 episode 直接晋升为全局表示更新。完整公式、回执路径和复现命令见
[`JTEXT_EPISODIC_NEURAL_DIAGNOSTIC_20260831_CN.md`](./JTEXT_EPISODIC_NEURAL_DIAGNOSTIC_20260831_CN.md)。

这条诊断明确不改变当前主线数字：J-TEXT canonical 在线增益仍只归因于决策层
专家权重更新；EAST V3C 仍按其独立 parent 和报警器谱系报告。下一步若继续研究
神经在线学习，只能采用先 probation、再用后续已揭示炮验证、最后 delayed promotion
的预注册方案，不能在 TEST220 上继续搜索局部门控参数。

## 7. 一条命令复现

J-TEXT 主动 20%：

```bash
cd /media/hetaisheng/044A81D94A81C83E/Fuison
PYTHONPATH=experiments/transfer_methods_2026 \
/home/hetaisheng/miniconda3/envs/rl_env/bin/python \
experiments/transfer_methods_2026/run_jtext_adv25_active_mainline.py \
  --source-root fuison_ext4_runs/v3c_adv25_exact_20260830 \
  --data-root data \
  --output fuison_ext4_runs/jtext_adv25_active_mainline_20260831/budget20_replay \
  --budget 20
```

EAST seed131（复现正向在线增益主线）：

```bash
cd /media/hetaisheng/044A81D94A81C83E/Fuison
PYTHONPATH=experiments/transfer_methods_2026 \
/home/hetaisheng/miniconda3/envs/rl_env/bin/python \
experiments/transfer_methods_2026/run_east_v3c_success_mainline.py \
  --seed 131 \
  --mode development \
  --source-checkpoint fuison_ext4_runs/v3c_adv25_epoch3_20260830/seed131/source/source_checkpoint.pt \
  --data-root data \
  --output fuison_ext4_runs/adv25_v3c_success_shotfirst_replay_20260831/seed131_replay
```

两个入口都会拒绝非 `rl_env` CUDA、错误 source SHA、非挂载盘最终目录和
非空覆盖目标。

这里必须显式写出 `--mode development`：该命令复现当前已经观察到的 V3C
诊断增益，而 balanced360 已参与方法开发，所以结果只能标记为
`TEST_DRIVEN_EXPLORATION`。省略该参数会运行默认的 `locked` 冻结负对照；它
用于验证“固定参数、固定查询合同下不产生增益”这一反事实，不应被当作
V3C 增益主线。两种模式的父模型、查询预算和参数合同均由入口校验，不能混合。

## 8. 论文/汇报边界

当前可以严谨地说：

- Adv25 标签合同已经落到源模型训练和在线标签读取中；
- J-TEXT 有一个可复现的因果主动权重更新诊断增益；
- EAST 的 shot-first V3C 迁移父模型在三 seed balanced360 诊断中有正的
  平均在线净变化；
- 预测、封存、查询、揭示、更新的顺序以及冻结源参数哈希均有机器检查。

当前不能说：

- balanced360 或 TEST220 是完全未见的最终测试集；
- J-TEXT residual 已经改变了 Transformer 底层表示；
- EAST 的 raw TPR 在每个 seed 都提高；当前能严谨声称的是 TPR@30 不下降且
  FPR 下降，不能把它写成所有 raw/operational 指标都同时提高；
- 已经通过独立 future stream 或正式部署验收；
- 在线增益证明了模型学习到全部破裂因果机理。

下一步若要形成正式论文结果，必须先锁定这份主线的代码、配置、source
checkpoint SHA、support parent SHA、阈值和报警规则，再使用一批从未参与
方法开发的 EAST 顺序炮流做一次性确认。确认流不能反向改变任何上述项目。
