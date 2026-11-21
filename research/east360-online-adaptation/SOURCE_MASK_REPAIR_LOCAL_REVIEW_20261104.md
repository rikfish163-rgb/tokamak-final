# 静态补偿器炮内局部探针：v1 只读审查

## Material Passport

- 范围：源 VAL 单步反事实探针；检查标签/未来信号边界、真实视图、临时更新、对照及已保存结果。**不是 EAST 在线流、炮级报警收益或物理规律验证。**
- 结论：v1 的安全与及时正例 gate 均失败；当前代码链和封存记录未发现已证实的标签泄漏或假装发生更新。存在**完整观测参考的支持范围未验证**以及常量方向/梯度方向审查缺口，不能用其宣称已找到正确风险梯度。
- 只读依据：以 [v1 code.zip](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_mask_repair_local_val_20261104_v1/code.zip) 内 `mask_repair_20261104/intra.py`、`local_probe.py` 和 `SOURCE_MASK_REPAIR_INTRA_PLAN_20261104.md` 为准，不要求后来优化中的工作树源码继续等于 v1。辅以 [rows.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_mask_repair_local_val_20261104_v1/rows.json)、[summary.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_mask_repair_local_val_20261104_v1/summary.json)、[completed.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_mask_repair_local_val_20261104_v1/completed.json)。
- 我没有读原始炮数据、运行 GPU/模型或修改任何 producer；以 `jq` 对保存的派生记录做只读复算。主进程报告的 source TEST 静态结果不作为本审查独立复算结果，也不是 intra 收益。

## 1. 需要主进程注意的口径

**[高，条件性] “完整 MNM 参考”未按真实观测支持过滤。** v1 归档的 `local_probe.py:128–143` 中，`reference` 是未来窗口实际 mask 下、关闭补偿器的 native embedding；但 `supported = full.reshape(...).any(-1).any(-1)` 仅要求该 patch 有**任意通道**观测，并不要求 MNM/CIII 在有效点上真实可用。若这些通道缺失或只有部分可用，该 patch 仍会进入 `reference_delta` 和 gate，不能称为全通道完整观测参考。源码确有此口径；**缺失是否实际发生、其数值影响多大，本审查不读 raw，现有 rows 也未保存逐 patch 的 MNM/CIII 可用性，无法确认**。因此它是解释和可能的筛选风险，不是已证实的数据泄漏；本次 gate 已失败，不能借此把失败改成通过。

建议在另行固定的后续检查中，按预测时可用的逐 patch mask 报告真实 MNM/CIII 支持及缺失次数，并在真正具备参考支持的集合上配对重算；不可将无观测填成真值或只保留有利样本。v1 的 `reference_delta` 应照原定义保留，不得事后改写原产物。

**[中] 局部六臂没有独立常量更新方向。** 归档的 `local_probe.py:80–84` 读取拟合产物时丢弃静态 TRAIN 常量；`intra.py:74–87, 90–120` 的 hold 是把 CIII 时序替换为该窗口首个观测值，仍是随窗口变化的 teacher，不是固定 16 维方向或固定补偿的对照。六臂均按已固定计划执行，所以不是漏跑预注册的一臂；但静态常量修复与这次炮内常量更新不是同一对照，不能声称已排除“共同固定方向/掩码项驱动”的所有解释。

**[中] 没有保存实际梯度或临时状态向量。** rows 仅存梯度范数、状态范数、before/after logit 和误差，不能事后直接算 `cos(g_SSL,g_risk)`、证明各臂参数梯度同向，或判定发生 CIII 重复计入。下述风险方向是**真实 q 视图的有限一步输出变化**，不是已测得的梯度余弦或物理因果方向。留出 patch 改善也不等同真实视图更接近完整观测。

## 2. 真实 mask、标签与时序

在 v1 归档的 `local_probe.py:117–154`，更新窗口 `x,a` 从 `t` 的 128 点取出，并将 MNM 的可用性设为 false；`intra.step` 内仅对辅助 student 再隐藏 CIII。风险评估的 `q=t+26` 窗口只去 MNM，CIII 保持原始真实可用性和观测值；它没有把人为隐藏 CIII 的 student 视图冒充风险输出。若 q 原本缺 CIII，脚本也不伪造其可用性；由于 q 的分通道 mask 未随结果保存，不能将全部结果都描述成 CIII 完整可用的单缺失视图。

源模型前向用真实 availability 将缺失值替换为源中心，再嵌入；所以传入张量虽装有原值，被 mask 的 MNM/CIII 不作为该分支观测。冻结 teacher 是真实 no-MNM mask 下的 persistent `embedding_after`，而不是未补偿 no-MNM native teacher。代码防止把 MNM 补偿的消失设为直接训练目标，但**不保证更新后不会抵消补偿或重复计入真实 CIII**。

每炮的所有局部结果和 SHA 封存后才调用 `io23.read_outcome`；当前炮预测阶段显式禁止该读取，随后用 HDF5 结局和原 `max(25 ms, AdvancedTime)` 分类，及时正例使用严格 `>10 ms`。未来完整 MNM 仅进入同炮离线参考，不作为 `intra.step` 的输入、梯度、控制选择或 seed。注意这是离线实现：`read_shot` 在封存前读取整炮信号，文件 SHA 也先扫描整个 HDF5 文件的字节；后者作为不透明完整性摘要保存，并未发现被送入模型、seed 或分组，但不能声称运行过程没有物理读取未来文件字节。q 窗口在离线代码中先被读取并计算参考，更新函数的数据参数仍只有 t 的窗口和 mask；这属于对代码路径的核查，不等同真实流式 I/O 实测。

坐标网格事先利用整炮记录长度挑选，计划明确称之为离线诊断，**不是可部署在线调度**。160 炮中 11 炮没有完整起点及 q 被列为 omission；不以其余相关窗口充作完整 160 炮结果，也不臆测遗漏炮的结局类别。将来 EAST 合同仍要求严格 `>30 ms`、先锁定历史预测再更新、炮末清零和 52 点无支持过期；v1 不是完整逐炮调度，不能证明这些在线机制已实现。

## 3. 更新确实发生，但风险方向未通过

记录共 4,662 条，即 777 个 `(shot,t,q)` × 六臂。独立配对检查：每个坐标六臂齐全，`supported`、组别、q 时间和更新前 logit 均逐值一致。归档 `local_probe.py:138–140` 对每个坐标的每一臂重新调用 `initial_state`，故 v1 均从 `δ=0` 起步，而非炮内多步累积。

- no-update 的 777 条状态范数均为 0，**最大 `|after_logit−before_logit|=0`**；其梯度范数、候选范数及更新前训练误差与 real 逐坐标完全相同。保存风险差最大绝对值约 `1.11e-16`，是相同 logit 的 BCE 浮点运算次序残差，不是模型更新。
- 其他五臂每臂 777/777 记录标为 UPDATED，状态范数约 0.025；real 的 777 个 q logit 全部发生非零变化，训练 patch 误差 777/777 下降、留出真实 teacher 误差 759/777 下降。因此本次不是“根本没有动”，也不能把误差下降直接写成预警改善。
- 安全组 80 炮/497 窗口；及时正例 48 炮/72 窗口。下面风险代理为 `after−before`，**先炮内平均、再跨炮平均**；数值越负越好。本审查独立复算与 summary 一致至浮点舍入。

| 臂 | 安全组 | 及时正例组 |
|---|---:|---:|
| real | +0.000232686 | −0.001764898 |
| hold | +0.000295756 | −0.001338613 |
| reverse | +0.000162169 | −0.002023743 |
| random | +0.000215148 | −0.001946309 |
| spatial | −0.000020920 | −0.000848788 |
| no-update | 0 | 约 0 |

安全组 real 代理损失上升，且不及 reverse、random 和 spatial；及时组虽下降，却不及 reverse 和 random。两个预定 gate 均为 false，不能根据个别组或留出误差放宽。

按**窗口而非独立炮**统计，安全组 real 有 290/497 个 q logit 上升（负标签的 BCE 变坏）；及时正例有 31/72 个 q logit 下降（正标签的 BCE 变坏）。hold 分别为 285/497、33/72；reverse 为 296/497、28/72；random 为 294/497、29/72；spatial 为 263/497、35/72。这些是输出层有限步方向，不是各臂梯度向量或炮级误报变化。

相同坐标上，real 与 reverse 的 q-logit 变化同号分别为安全组 473/497、及时组 69/72；real 与 random 分别为 485/497、70/72。**机制推测**：额外隐藏时共同的 CIII mask/表示项可能主导了部分更新，导致真假时间排列产生相似输出；本次未分解这些贡献，也未保存梯度向量，不能据此认定重复计入 CIII，更不能把它解释为真实物理机制。

## 4. 版本与验证边界

- v1 `code.zip` SHA-256：`d979dda8af983ea31be5c08b4cfcd6c466621473f8406b0cc49ffb5435c86215`；归档内 intra 为 `9bd2c6666422b60f70d35de526854a8e1d9f0e952caedc8241e0ce639cd44bd3`、local_probe 为 `b60f1cc30ee8a78635e361b52f9540e5536413fc37179b918351a363abb55612`。rows、summary 和 ZIP 的实际 SHA 均与 completed 记录一致。主进程现在改工作树源码属于后续版本，不是 v1 producer 造假或复算失败。
- 主进程正在优化只取 Transformer 前 embedding 的路径；本审查**未审查或运行 v2**。要声明数值等价，应由主进程固定新 producer/计划后，配对核实五次 teacher/student 的 embedding、16 维梯度、实际候选状态和真实风险前向，并比较相同坐标、六臂、种子、支持及 gate；保留 v1/v2 各自 ZIP、哈希及容差。不能因 loss 相近就代替端到端风险输出核对。
- `status=UPDATED` 在通用 `intra.step` 中由梯度范数阈值决定，本身不保证未来持状态、半径投影或饱和情况下风险输出必然改变；v1 从零一步的状态与 logit 变化已如上独立核算。无实际 EAST 持状态运行、炮级 TP/FP、统计显著性或机制物理性结论。
