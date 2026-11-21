# 炮内更新几何：历史查重与条件低秩候选审查

## Material Passport

- 范围：工作树及其链接到主树的 `experiments/risk_aligned_online`、指定研究文档及相关已保存结果；截至本次检索，不是全仓库和全部历史归档的穷尽证明。
- 验证状态：**ANALYZED**。读取实现、协议、结果；对已保存 JSON 做只读算术复核。没有重新训练、运行模型或 GPU、读取原始 HDF5、上传数据或复现历史实验。
- 用途：研究侧证据审查；所有新的实验建议均未执行，不是在线成绩。合格 J-TEXT 源模型固定，EAST 是唯一在线适应对象。

## 结论与即时查重

1. **未在本次有界检索中发现完全相同的已完成实现**：以同一个 358 维起点与到达反馈上下文，直接生成第一层 Q、K 各自 rank-2 的两个矩阵因子 `A(c), B(c)`，再用 `B(c)A(c)` 形成临时权重。检索覆盖两棵工作树的 `risk_aligned_online` Python 实现、相关报告和此工作树现有计划，关键词包括 hypernetwork、条件权重、LoRA、low-rank、矩阵形状与生成器。搜索不到不等于所有历史中不存在，当前主进程正在实现的新工作也不属于本审查的既有结果。
2. **但“起点 CLS 表示目标 versus 完整 Q/K”的事后可达性比较已经做过。** 在当前 cold20 的 176 个选定查询窗口上，两种标签可见有限搜索都修正了同样的 7/59 个危险误判和 1/3 个安全误判。优化器及步数不同、实际动作范数未逐项匹配，且是事后选点，故这既不是完整公平的几何对照，也不能表述为该子空间一定没有损失。[实现](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/diagnose.py:83)；[原始结果](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/context_vector_cold20_v1/query_capacity_diagnostic.json:2)。
3. **本项目也做过“雅可比行空间覆盖后续风险梯度”的 SVD 诊断**，但它针对旧事件状态反馈和末层 FFN，不是当前起点 CLS 与第一层 Q/K；方法可复用，数值不能移植。[代码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_feedback_geometry.py:30)；[保存的独立复核](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/event_dynamics240_20261021/feedback_geometry/independent_verification.json:1)。
4. 现有证据支持“更新方向与当前反馈—风险关系没有对齐”，也支持特定参数族存在表达限制；**尚不能确定这个限制是否是当前失败的主要原因**。不能把第一层 Q/K 整体说成无可达能力，也不能断言换成低秩生成器即可得到有用更新。

## 1. 当前规则的真实结构约束

基模的风险头是固定的 `Linear(64,1)`；冻结头并不冻结进入它的表示或风险分数。[模型定义](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple23.py:59)；[Q/K 实际前向及原风险读出](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_lowrank_20261029/model.py:31)。

旧表示目标控制器使用 `c=(h_CLS,h_last,五段到达变化和可用率)` 共 358 维，产生 64 维 `v(c)`。从零 Q/K 临时状态、同一次前向的 detached CLS 锚点求一步梯度，其未截断的动作是 `0.1 J_o^T v(c)`；投影到半径 0.1 后仍在 `J_o^T` 的像空间中。这里 `J_o` 是起点第一块 CLS 对第一层 8192 个 Q/K 坐标的导数，因此该单步像空间的秩至多 64；实际秩可能更低。[上下文和步进](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/learning.py:16)；[同前向锚点](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/prepare.py:32)；[全矩阵坐标](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_basis_20261029/learning.py:12)。

这是**旧规则的可表达动作集合受限**，不是“整个 Q/K 更新只能有 64 维”，也不是“冻结风险头必然使更新无效”。不同起点的 `J_o` 可以不同；后续窗口的风险梯度也不必处于这个空间。仅由 64 与 8192 的维数差无法推出后续风险梯度恰好大部分在空间外，还需要测量。

## 2. 已完成的可达性／位置实验

不同条目有不同父模型、样本和预算；不能把它们当成同一次严格对照。

| 实验、父模型及访问 | 具体更新位置和结果 | 可得与不可得结论 |
|---|---|---|
| **2026-10-05 内部位置**：合格源检查点 `c7270c1e…`，旧前30炮的7个事后选定错误点；未来结局决定目标方向；另用260个参考点，其中含受检炮。 | 128维 token gain/bias、23,616维 embedding、512维 Transformer norm、16,576维最后 FFN；32步、参数半径1。5个破裂点依次有 **0、5、5、5** 个越阈，2个安全点均可压回；首次越阈时较大的位置也可能改变许多参考点。[协议与结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/astra_design_20261005/04_internal_capacity_experiment.md:9)；[保存结果](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/internal_capacity_firstcross_20261005.json)。 | 位置和附带改变均重要；相同参数 L2 不等于相同功能预算；不是全局上界或线上救回。 |
| **2026-10-09 状态路由 FFN**：同合格源模型，旧密度映射前120炮；在事后27个错误端点使用结局标签。 | 8×64 偏移的5点搜索未越阈；将同位置改为 8×64×128 的状态路由权重后，32步半径1搜索使 **14/19** 破裂端点及 **8/8** 安全端点到正确侧；实际一次信号梯度只改正2/27且均为安全点。[位置与标签](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/astra_design_20261005/56_状态条件内部权重适应.md:7)；[结果](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/routed_weight_error_probe_20261009.json)。 | 扩大作用族可改变端点能力，却未自动得到无标签方向；旧120流纯炮内为20/6，相对其输入匹配静态20/8只减少2个FP，并非与当前cold20相同。 |
| **2026-10-29 当前 cold20，后340炮**：源 SHA `c7270c1e…`，固定映射、slow、阈值；398个 cold 峰值端点按全炮风险曲线和真实结局事后选择。 | 逐端点标签指导64步搜索，半径1：embedding后 token gain/bias 使44个漏/迟报炮中 **4** 炮至少一端点越阈、31个FP原峰中 **4** 被压低；第一 Transformer 后同形 gain/bias 分别 **1、1**。[端点选择](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/cold_capacity_20261029/probe.py:25)；[优化](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/cold_capacity_20261029/probe.py:50)；[保存结果](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/cold_capacity_v1/complete.json:5)；[另一位置](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/冷启动模型更新可达性_端点与位置配对诊断_20261029.md:47)。 | “点可越阈”不是该炮实际救回；压低旧最高点也不能保证整炮不误报。有限步、单起点、事后选点均不构成全局上界。 |
| **相同 cold20 和398端点的第一层低秩注意力**：沿用上一条的选点与标签。 | Q/K各rank2和 V rank4，各512个因子参数，同为64步、半径1。Q/K在 **40/44** 漏/迟报炮找到某个及时越阈点，其中 **38** 在30–125ms；原31个FP峰中 **29** 可压低。V分别 **44/44、44、31**。[实现](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_lowrank_20261029/capacity.py:23)；[保存结果](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/attention_lowrank_capacity_v1/complete.json:2)。 | 这是最强反例：不能说第一层 Q/K 没有端点作用能力。同半径的 factor、token、V 坐标不是同等函数扰动；本诊断不能证明 Q/K 或 V 的在线优越性。 |
| **同一个低秩架构的真实到达反馈**：当前 cold20、固定完整360、前20成本计入，当前炮结局只在预测封存后评分。 | 从信号预测损失反传，真实 Q/K **121 TP/32 FP**、倒序 Q/K **121/32**、V **120/33**，cold **125/32**；各更新臂实际执行10,774次更新。[协议和结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/注意力炮内更新_完整360失败与通道反馈方向_20261029.md:18)。 | 高 oracle 端点能力并没有转化为这个合法反馈规则的在线收益；不能把此差距唯一归因于参数族。 |
| **当前 cold20 的表示目标与全 Q/K oracle**：3折只隔离规则教学，父模型已见过全部20炮；未来选点、真实 query 输入和该炮标签均可见于诊断。 | 同105个安全窗口和71个及时危险窗口，完整 Q/K 8步归一化梯度上限0.1；表示目标在 `tanh` 约束下用30步 Adam、经实际 `J_o^T` 步进且半径上限0.1。二者都将安全正确数102→103、危险12→19，修正的错误窗口身份亦相同。[代码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/diagnose.py:106)；[摘要](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/context_vector_cold20_v1/query_capacity_diagnostic.json:6)。 | 本次重新读取176条已保存记录：按炮等权，完整搜索相对目标搜索的**有利方向 logit**差均值，安全约 **+0.10813**、危险约 **+0.05879**；分别为3/3、16/17炮均值为正。这个连续分数差不是受控几何因果效应，因为两种搜索算法、次数和实际范数不同，也不是炮级收益。 |
| **旧状态反馈行空间几何**：旧事件状态实验的7个丢失及时TP炮，每炮1个已接受更新；完整历史状态重放，事后使用后续查询输入和结局选例。 | 25个预测状态坐标对16,576维末层FFN求 Jacobian；下一窗口风险梯度投影范数比例 **0.2513–0.4166，中位0.3495**，7次中6次实际状态损失下降方向与风险上升方向余弦为负。三个SVD截断门槛结果一致。[原始逐项证据](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/event_dynamics240_20261021/feedback_geometry/completed.json:28)；[方法/解释](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/炮内适应研究推进_20261021.md:200)。 | 是先例，不是当前 CLS/QK 的测量；只覆盖有意选出的失败样本，是局部一阶量，不是可纠错比例或在线策略。 |

上述半径1与0.1不可直接比较；半径是参数坐标上的数，不是统一的风险或功能变化。尤其低秩因子的范数与最终 `BA` 的 Frobenius 范数不同。

## 3. 其他容易重复的结构和控制

- **旧低秩注意力不是直接生成器。** 旧 `AttentionAdapter` 用固定行正交 `A₀`、零 `B` 起步，以本炮信号预测损失梯度更新 `B(A₀+ΔA)`；`A、B` 是临时优化变量，不是由同一上下文前向同时生成的因子。[代码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_lowrank_20261029/model.py:16)。
- **历史学习的 rank-2 基也不是条件生成器。** 20炮学习一套共享行正交 `A`，在线用到达信号梯度 `G` 生成 `B∝GAᵀ`；其两版最终都选择零步，360五臂均125/32、曲线逐点相同。[公式和结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/历史反馈权重与注意力更新方向_配对实验_20261029.md:60)；[实现](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_basis_20261029/learning.py:21)。
- **输入条件权重、梯度映射与 Q/K 关系更新均有先例。** 旧 `185→32→23` 网络由输入决定通道损失权重，但其作用仍经信号损失反向传播至编码器，而非直接生成矩阵。[实现](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/contextual_feedback_20261023/learning.py:22)。当前 cold20 还拟合过从信号梯度到8192维风险梯度的共享岭映射，完整360主臂125/32，与 cold 相同。[报告](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/同坐标梯度映射_完整360与监督目标对照_20261030.md:55)。不要把“输入相关方向”“可学习映射”单独称为未尝试。
- **只改一个注意力头的诊断关系 loss 已完成。** cold20 的固定2707对中，仅第一层第一头Q/K行接受 `KL` 梯度，真实/倒序方向余弦中位0.795626，但真实臂安全3/3炮BCE变差、及时危险9/17炮变差；全部2707次辅助 KL 都下降，仍未通过效用门槛。[固定范围](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/RELATIONAL_ATTENTION_PLAN_20261030.md:19)；[结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/到达反馈失效定位与诊断关系更新_20261030.md:82)。这不是新候选的两个全 Q/K rank-2 矩阵，也不证明其有效。
- **归一化／输入适配已做过。** 旧可逆46维输入增益偏置在120炮为28/18，输入匹配静态20/8，新增8个TP也新增10个FP；仅保留窗口均值的23维增益为20/8。[自由仿射](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/astra_design_20261005/47_预测坐标回映与炮内更新实测.md:47)；[保留窗口水平](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/astra_design_20261005/48_保留工况水平的炮内更新否证.md:23)。这些均是旧父模型/流，不是当前 Q/K 位点的对照。
- **炮内递归状态不是空白。** 旧9×92诊断记忆沿炮写入、炮末清零，经92→64投影注入两个 Transformer 块之间，10炮筛查静态1/0、真实记忆1/1、倒序1/1。[代码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/delayed_recurrent_memory_20261025/model.py:41)；[结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/延迟诊断记忆_实测与顺序读取瓶颈_20261025.md:49)。后续源域教学记忆两种纯炮内完整360均124/32，cold125/32、倒序125/32。[结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/源域教学更新规则_两组完整360实测_20261026.md:41)。这些不是所有双时间尺度状态空间方案的否证。

## 4. 对主进程条件低秩诊断的具体审查

**与旧规则的差异。** 对固定已到达上下文，令 `ΔQ=B_Q(c)A_Q(c)`、`ΔK=B_K(c)A_K(c)`；在第一层原权重上临时使用它们，原基模、cold slow、原风险头、输入映射和报警合同保持不变。这条前向生成路径不要求 `Δ` 等于 `J_o^T v`，因此**有可能**作用到旧像空间外。它与旧的“损失梯度→固定基投影”以及“64维目标→CLS雅可比转置”机制不同。

**不应过度声称。** 每个生成的 `ΔQ、ΔK` 分别 rank≤2；旧 `J_o^T v` 输出的两个64×64矩阵不必各为 rank≤2。因此新候选和旧候选的可行集合一般**互不包含**；“绕过旧像空间”只表示结构上不再强制属于该空间，不保证实际学到的动作有空间外分量。两者还可能在控制器参数量、训练目标、优化和有限步幅上不同；效果不同不能单独归因于“位置更好”。实际是否越过旧子空间应测 `||(I-P_J)Δ||/||Δ||`，零动作记为未定义而非0余弦，且空间外比例本身不是风险收益。

**术语及标签。** 若炮内只是冻结生成器前向算 `A(c), B(c)` 并替换 Q/K 临时权重，这是“**已到达信号条件化的前向生成临时权重**”，可以说明在线不使用当前炮结局；它不是该炮上对模型/生成器执行无监督反向传播或梯度训练。若生成器用历史完成炮的真实风险标签离线训练，整体也不能写成完全无监督。只有实际计算了损失梯度并对临时参数或生成器执行更新时，才能分别记录相应的炮内反向传播；不能把离线训练反传、参数值改变、或后续评价梯度冒充在线更新。

**固定时钟与样本。** 按现有关系组件规格 `o=127+25k`，`m=o+25` 后才能使用完整到达25点，固定后续查询 `q=m+25`，每对从零临时状态构造，不把过去预测回写。已保存2707对包括安全527窗、及时危险63窗、早期破裂2095窗、末端22窗；这是20炮的相关窗口，安全独立单位只有3炮。已有 cold20 表示目标 oracle 的688个峰值选点**不是同一2707点**，不能直接拼表或冒称同一查询。[固定时钟](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/RELATIONAL_ATTENTION_PLAN_20261030.md:17)；[2707点分组](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/到达反馈失效定位与诊断关系更新_20261030.md:82)。

**建议的最小控制和停止点：**

1. 使用完全相同的起点、掩码、成熟时钟、后续 q 和 cold 分数。保留不更新、真实上下文、仅到达块倒序、仅到达块替换拟合折的炮等权均值；起点128维不随反馈消融改变。另保留固定基 `A` 的条件 `B` 对照，以检验让 `A` 依赖上下文是否必要。若比较旧表示目标，应在**相同2707点**产生其输出，明确它的既有训练数据和预算；不能把不同训练、不同选点的差异称为纯参数族对照。
2. 真实和各对照自身生成动作应单独报告；事后等范数版本可以作为条件性归因对照，但不能把借用真实动作范数或符号的对照称为完全独立策略。限制并记录 Q、K **最终矩阵增量联合 Frobenius 范数**，以及实际 origin/query logit 变化、零动作、有效支持、真/倒序方向余弦。匹配参数范数仍不等于匹配功能变化，也不能按未来结局选符号或幅度。
3. 预先固定并保留三折、优化预算和负结果：对各臂先炮内平均、再按炮等权；真实反馈需同时改善安全与及时危险的后续风险损失，并胜过均值、倒序及固定基对照；复用关系试验的安全至少2/3、危险至少一半改善作为最小开发门槛。任一支持不足或门槛失败即停止；3个安全炮仅能排错，不能宣称稳定泛化。只在全部预测封存后用留出炮结局分组；用于拟合的任何标签仅来自相应拟合折已结束炮。冷父已使用全部20炮，故这是条件组件检查，不是整条模型独立交叉验证。
4. **实现陷阱**：如果 `A` 和 `B` 的输出同时初始化为精确零，则 `BA` 对两分支的导数在零点都为零，可能无法启动教学。可用非零 `A`、零 `B` 达到零动作且保留 `B` 的初始梯度；此时 `A` 分支的初始梯度仍为零，需记录后续是否真正变化。应检查零动作与 cold 前向一致、两因子的数值有限性、只改变指定Q/K、炮末清零、无有效反馈的显式规则，以及修改 `m` 后数据不影响该时刻动作的因果性。因子有 `BA=(BR)(R⁻¹A)` 的重参数化自由度，不能用 A/B 自身大小直接解释有效风险作用。

## 5. 尚缺的最小几何判别：不是再做一次旧 oracle

本次未找到对**当前** `J_o = ∂h_CLS(o)/∂ΔQK` 与同一后续 `q` 风险梯度的已保存逐样本投影分析。它与第2节已做的冷20两种非线性搜索不同，也与旧事件状态／FFN的 SVD 先例不同。最小可证伪建议如下，只是建议，不在本侧任务执行：

1. 在预先固定的同一父模型、同一起点和同一后续查询上，从零 Q/K 状态取 `J_o` 和 `g_q=∂z_q/∂ΔQK`。用 SVD 得 `P_J`，报告数值秩、相对奇异值截断敏感性及 `ρ=||P_Jg_q||/||g_q||`；`g_q=0` 必须单列。复用现有 [SVD 函数](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_feedback_geometry.py:30)，但不能复用其旧数值。
2. 从同一零点比较固定一次、相同 Q/K 矩阵 L2/Frobenius 范数的 `±g_q/||g_q||` 与 `±P_Jg_q/||P_Jg_q||` 的真实有限前向；符号由事后安全/及时危险标签确定，单独报告零投影与不匹配。逐炮比较有利 logit、BCE及阈值距离，不只看跨阈数量。此投影臂是对整个线性像空间放宽后的局部理想对照：旧规则 `tanh(v)`、学习率和裁剪还会进一步限制它，不能直接把投影臂当成旧控制器可实现的动作。
3. 可在看新结果前固定一个**工程筛查**：例如采用与旧 cold20 诊断相同的唯一半径0.1；先每炮求 `ρ` 均值，再取各类别炮中位数。只有两类中位数均≤0.8，且完整方向减投影方向的有利 logit 变化按炮等权均≥0.05、至少2/3安全炮和9/17危险炮各自为正，才视为支持“该局部像空间的能力损失值得继续检查”；同时报告各炮离固定报警阈值的距离及实际跨越情况。若两类中位 `ρ` 均≥0.9，且两类按炮等权有限效应差的绝对值均<0.05，则本样本不支持该几何缺口为主要局部瓶颈；介于两者、支持不足或数值不稳定均为不确定。0.8、0.9和0.05是为了可否证而事先写死的示例工程界限，不是统计显著性、物理标准或工程部署有效性的保证。两类都要保留反例、逐炮差和样本量；点越阈仍不是整炮收益。这个诊断对后续 query 和结局使用了事后信息，**绝不可据此在线选方向、窗口、接受与否或模型**。
4. 现存 [cold20 梯度对](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/operational_transport_20261030/prepare.py:104) 保存 signal、point/operational 风险梯度及记录，但未保存完整起点 Jacobian；其688个峰值选点也不同于当前2707固定查询。现有 JSON/分数不能直接算出 `P_J`。如没有已授权、可复用的输入／导数缓存，应明确标记为不可测；本审查没有读取原始 HDF5 或为此重新计算，也不授权通过任何途径扩大数据访问。

该几何判别即使出现差距，也只识别所测局部可达性，不证明当前到达信息足以无标签选择有利方向。几何通过后，条件生成器仍须胜过反馈消融并通过同一安全、及时危险门槛；若未来另行批准完整 EAST 流，保留初始20炮成本、固定阈值、严格首次报警提前量>30ms、FPR≤0.20及>3000ms早报，完整报告救回与改坏。**不扫步长，不以辅助损失下降代替风险或告警提升。**

## 6. 本次核验与边界

- 读过用户指定的三份工作树历史/计划及主树“到达反馈失效定位”报告，并追溯上表具体实现和保存结果；搜索覆盖受限于本报告首节。
- 对 `query_capacity_diagnostic.json` 的176条保存结果按原固定阈值重算两种 oracle 修正身份，并计算先炮内、再炮间等权的有利 logit 差；对398点两套 `complete.json` 重新计数数组长度。没有重跑原优化、验证哈希或重新计算神经梯度。
- 本次仅创建本审查文件；未修改其它实验文件、训练、使用GPU、读取原始HDF5、对外上传、提交或清理。`academic-research-suite` 的实验规划约束和 `results-analysis` 的分组/负结果纪律用于区分局部诊断、历史结果和未来验收。
