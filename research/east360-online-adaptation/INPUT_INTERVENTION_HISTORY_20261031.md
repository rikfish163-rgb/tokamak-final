# 输入干预历史审计：当前 cold20 / 360 父运行

日期：2026-10-31。范围是只读核对已有本地文档、代码、协议、映射及结果；唯一新写入物是本报告。未启动 GPU、推理、训练或修改共享实验代码。以下均为开发性对照，不是独立验证或物理单位认证。

## 结论与复用裁决

**在本次有界检索覆盖的记录中，未找到在当前同一 360 炮序、同一 cold20、同一冻结 cold 父状态上，分别执行 `n=1 amplitude` 中位数比值乘法、`CIII` 中位数比值增益、`dx_a` 或 `dy_a` 中位数差加法的完整炮级结果。** 这不是对未检索到的全部历史存档作穷尽性不存在证明；这四项各自的 TP/FP **目前未知**，不能移植其他实验的数值。

已有可直接复用的是**当前未干预 cold 基线及其前20炮封存预测**，不是上述四个干预结果。相同源文件并不自动意味着相同父预测器：当前 cold 包括前20炮结束后的监督准备状态，不能把仅冻结源网络的旧结果称为同父结果。若发现一个历史臂的输入参数、有效性合同、源与 cold 状态、炮序、封存边界、阈值和评价范围均完全一致，应先复用该臂封存结果，再考虑是否需要重跑。

目前检视到正在准备的 [输入因子测试](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_input_factors_20261031.py) 明确命名了 `cold`、`n1_gain`、`CIII_gain`、`dx_shift`、`dy_shift`；本次没有看到完整六臂事前合同，因此不能凭猜测为第六臂逐项断言新颖性。四个已说明变换与下面的多列匹配、选定端点替换及通道 mask 并不等价；`cold` 则是应复用的已有对照，而非新干预。

## 当前参照合同与可复用数字

参照 [lifecycle protocol](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028/protocol.json)、[input map](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028/input_map.json)、[completed](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028/completed.json) 与 [anchored 独立核算](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/ANCHORED_FULLSTREAM_AUDIT_RESULT_20261031.json)：

- 固定源 `gradient_aligned512_20260929/source.pt`，SHA256 `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`；固定阈值 `0.7258946895599365`，首次越阈，及时要求提前量严格 `>30 ms`。HDF5 标签口径 full360 为180破裂/180安全；这一开发集不是独立未来验证。
- 当前 cold20 顺序为 `71304, 71265, 71415, 74087, 71370, 70861, 69454, 69765, 69768, 69859, 69458, 69855, 69781, 69854, 70327, 70404, 70392, 70394, 69791, 70418`，按现行标签17破裂/3安全。前20炮是映射和 cold 状态产生前的原始封存预测；后340为163破裂/177安全。
- 当前 `density_gain` 的唯一非恒等项是 `ne_nG × 0.0003150853015380819`，其余23列中的其他列均恒等；掩码不变。映射之后还存在已训练好的 cold 状态，其文件在下方 hash 中单列。**不能把 aligned 源模型和 cold 监督准备状态混为一谈。**
- 原 source 全360为 **145 TP / 94 FP**；仅密度 aligned 为 **119 / 26**；加入当前 cold 监督准备为 **125 / 32**（TPR 0.6944、FPR 0.1778）。cold 前20为 **6/17 TP、1/3 FP**，后340为 **119/163 TP、31/177 FP**。这些是已封存基线，不是新的输入干预结果。
- [anchored 360 协议](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/anchored_full360_diagnostic_v1/protocol.json) 将当前 `input_map.json` 锁为 `a1516e9ce835dc8803797712003e80604e49aa69b4f726ea7045db49bca21e4b`，`models/cold.pt` 锁为 `25059ac3d491a16613beb23f90ee93950ceed3ab3346a865c9b286f90cbc2302`。该协议的十条曲线是 cold/identical 与 origin/mature 反馈对照，不是四个目标输入单列干预。[运行代码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/anchored_stream_20261031/run.py:60)复用前20封存曲线，后340载入同一映射及 cold slow 状态。

## 已检索到的最相近历史结果

| 历史证据与结果 | 与当前单因素合同的关键差别 |
|---|---|
| **prior45 单通道**：[报告45](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/astra_design_20261005/45_单通道尺度干预与零炮问题定位.md)；[原始结果](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/paper131_scale_policy30_foreground_20261009/result.json)；[独立核算](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/paper131_scale_policy30_foreground_20261009/verification_analysis.json)。同屏蔽源尺度在355炮为 **150/180 TP、105/175 FP**；`density_only` 为 **117/180、30/175**：少33 TP、少75 FP。原静态为145/180、94/175。 | 这个“单通道”是 `ne_nG` 的目标 median/IQR 尺度处理，不是另外四列的比值/平移；前30炮拟合，评价355炮且额外统一屏蔽，与当前 cold20/360 不同。 |
| **多列归一化**：[2026-10-22完整结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261022/前10与20炮输入对齐_完整结果与判断.md)；[completed](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/warmstart_alignment_test360_v2_20261022/completed.json)。旧20炮 `moments_all` 为 **154/180 TP、120/180 FP**；`robust_all` 为 **109/180、108/180**；旧20炮 `density_gain` 为 **118/180、27/180**。 | 虽也是完整360、相同源 SHA，它按炮号排序，cold20 是54157开头、2破裂/18安全；当前是71304开头、17/3。旧 density gain 为 `0.00025946043454283725`，不是当前 `0.0003150853015380819`。多列是除 MNM、`mode_number_n` 外可辨识连续列的同时仿射匹配，不是“四个各自只改一列”。 |
| **逐列掩码移除**：[startup 结果](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/startup_controls240_20261021_v2/completed.json)、[代码](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/startup_intervention_replay.py:27)。240炮静态 **82/101 TP、82/139 FP**；`mask_dx_a` **38/101、14/139**；`mask_ip` **40/101、10/139**；`mask_ne_nG` **59/101、28/139**。 | 这里把该列设不可用，改变 availability 输入，而非对有效值做加法且保留掩码；无当前 density/cold 状态，也不是当前360。不能用低FP隐去大量TP损失。 |
| **选定端点中心化**：[startup 窗口报告](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/EAST单炮适应_启动敏感性与结构调整_20261021.md)及 [2026-10-10峰值敏感性](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/astra_design_20261005/72_TPR下降的目标错位与误报配对诊断.md)。后者在旧120炮选定峰值，对 `dx_a` 的 asinh 坐标减窗口中位数后，30个原误报峰中27个跌破阈值，同时42个原及时破裂峰中21个跌破。 | 它是结局选出的单窗口诊断、在非线性 asinh 坐标逐窗口中心化；不是原始输入的固定全流加法，也不是27个整炮误报被消除。旧的将有效值替换为源中心同样删除该窗口该列的变化。 |
| **MNM恢复**：[四臂 completed](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/mnm_restore_test360_20261022/completed.json)、[audit](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/mnm_restore_test360_20261022/audit.json)。旧20炮密度映射下，屏蔽 MNM 为 **118/180 TP、27/180 FP**，恢复实际 MNM 为 **120/180、27/180**；后者移除67900误报、增加85599误报。 | 这是恢复真实 MNM 数值和 availability，不是 CIII 或 n=1 的增益，也不是 dx/dy 平移；使用旧炮号排序、旧 cold20 和纯静态源状态。相同FP总数不等于逐炮不变。 |

旧多列映射中确实改过四个目标列，例如 [robust_all_n20](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/warmstart_alignment_test360_v2_20261022/maps/robust_all_n20.json) 的 `n=1 amplitude` 是 gain `0.025048467802177`、offset `-16.16663088539347`；`CIII` 是 gain `21.112235090340903`、offset `0.05281527336923297`；`dx_a` 是 gain `0.4588258936647866`、offset `-8.0293086064217`；`dy_a` 是 gain `0.037945286753249555`、offset `0.5520626813545466`。它们连同其他列一起改变，参数也不等于下列当前 cold20 中位数计算，不能拆出四个单列因果效应。

## 当前拟合证据及重要风险

以下是从当前已保存 input map 的 `source_statistics` / `target_statistics` 读取的数值，按已有方法每个有观测炮等权、每炮每列至多128个可用点。比值和差值只是用这些存档中位数计算的**算术候选**，不是已运行结果，也不假定第六臂或任何额外截断规则：

| 列 | 源中位数 | 当前 cold20 中位数 | 源/目标比值 | 源−目标差 | cold20 有效炮数 |
|---|---:|---:|---:|---:|---:|
| `n=1 amplitude` | 3.8623672072 | 597.0863647461 | 0.0064686910 | −593.2239975389 | 20 |
| `CIII` | 1.0248369781 | 0.0153039554 | 66.9654968515 | 1.0095330226 | **1** |
| `dx_a` | 0.3881282048 | 4.3526518345 | 0.0891705148 | −3.9645236297 | 20 |
| `dy_a` | 0.4547564546 | 5.6809742451 | 0.0800490259 | −5.2262177905 | 20 |

对本次提出的原始输入操作，`n=1 amplitude` 和 `CIII` 可检查比值列，`dx_a` 和 `dy_a` 可检查差值列。当前存档 `CIII` 的128个抽样点**全来自同一炮**，不是20个独立炮；增益约67只说明这个估计器的数值结果，不代表物理单位校准或可推广性。现有合同将接近−10的 `CIII` 点屏蔽，必须先保留该规则，不能把哨兵乘成有效读数或创造观测。中位数比例/平移与全采样点加权的基础审计表不是同一种估计器。

如要保持可归因性，四臂各自在现有密度映射之上只改选定列的有效数值，保留其他值、availability、源归一化、cold slow、顺序、报警阈值及初始20炮预测。是否采用额外范围限制或组合臂必须在运行前明示；加限制、改变 cold 拟合或使用后340统计都不是与这里同一个参数合同。既有冷启动安全覆盖窄、CIII缺失、Bt排序、18炮标签冲突、跨装置语义与上游时钟/物理单位未认证等风险，均不能因干预成绩改善而视为解决。详见[基础数据审查](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/基础数据审查_尺度标签与时间轴_20261031.md)。

## 检索与核验边界

查阅 MAIN 的研究文档、`startup_intervention_replay.py`、`startup_sensitivity.py`、`mnm_sensitivity_20261022`、`warmstart_alignment_20261022`，并核对上述结果 JSON、旧与当前 mapping、当前 lifecycle/anchored protocol 与 WT 的相关测试。对 `experiments/risk_aligned_online/runs/*/protocol.json` 做了父运行及相关关键词检索；WT `probe_outputs` 的目录与 anchored 协议也已查看。旧文档中的计数已与对应 completed/result JSON 抽核；本审计没有重新运行历史模型或重新读取全部预测封存，亦不声称检查了所有其他 worktree、所有压缩快照或所有历史数据。
