# 源端事件缺测任务教学：标签用途、不可辨识性与实际未通过结论

## Material Passport（只读审查范围）

审查对象为已冻结的 `source_event_tasks_20261102/trial.py`、必要 helper，以及 `source_event_masked_tasks_v1` 的 completed、census、validation summary。只做代码推导与现成汇总解读，未执行拟合、测试、GPU、原始信号读取或记录级 audit；不替代主进程的 replay/CPU audit，也不判断其完成状态。本次只新建本文，两个 context_pair 文件及其他代码、输出均不改动。

结论：**JT 事件标签被实际利用，但尚未证明这种利用带来了有益的事件条件更新；本轮源端筛选明确失败，不能晋级为 EAST 收益证据。** `COMPLETED` 表示程序跑完，不等于科学目标完成。[计划](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/SOURCE_EVENT_TASKS_PLAN_20261102.md)、[完成记录](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_event_masked_tasks_v1/completed.json)报告耗时 333.353173 秒、源参数更新 0、EAST 读取 0、`source_unchanged=true`、`full_east_stream_tested=false`；这些是运行方声明，本审查不另行认证 checkpoint 或原始数据。

## 实际结果先限定解释

下表为归档 summary 的 ΔBCE，先在每炮内平均，再对炮平均；负数改善。主视图为屏蔽 MNM+CIII，balanced 是 safe/danger 两组均值的等权平均，不是告警 TPR/FPR。

| 视图／规则 | safe ΔBCE | danger ΔBCE | balanced |
|---|---:|---:|---:|
| MNM+CIII 缺测／shared_real | −0.039724762 | +0.041893084 | +0.001084161 |
| MNM+CIII 缺测／true_real | −0.030757144 | +0.065678523 | +0.017460690 |
| MNM+CIII 缺测／shuffled_real | −0.029275839 | +0.047494123 | +0.009109142 |
| MNM+CIII 缺测／true_reverse | −0.032433695 | +0.064365778 | +0.015966041 |
| MNM+CIII 缺测／true_hold | −0.033984416 | +0.069732193 | +0.017873888 |
| native／true_real | −0.012170694 | +0.021276078 | +0.004552692 |

真实事件规则只在上述 balanced 比较中胜过自身 hold，未胜过 shared、shuffled 或 reverse；danger 与 native 保护均失败，`passed=false`。主视图 danger 的 117 条查询中，96 条 BCE 变差、21 条改善。VAL 共 946 个局部 episode、145 炮，其中 safe 638 条/80 炮、danger 117 条/64 炮、early 191 条/51 炮；组间炮号可重叠，不能相加当独立样本。三视图、九种规则/反馈组合产生的 25542 条记录也不是独立试验数。[summary](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_event_masked_tasks_v1/validation_summary.json)、[census](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_event_masked_tasks_v1/census.json)

## 三个关键、可证伪的诊断

### 1. 标签进入了程序，但“事件语义有效”与“条件坐标有效”尚不可区分

`attach_event` 用绝对时钟附上 origin/query 类别；真正用于抽样的是 origin 类别，包括未知 −1。每步先 safe/danger 各取一条，再按有支持的类别、炮、记录均匀抽取，三臂共用同一 schedule。第二条路径是冻结的 true/shuffled 软分类器：只看 origin 的过去诊断变化，输出六维 q；不向当前输入灌真实类别，query 类别仅作元数据。配对 context 训练已用同初始化、同 minibatch，排除了旧版按类别采样造成的两头批次差异。[数据与抽样](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_event_tasks_20261102/data.py:133)、[context_pair](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_event_tasks_20261102/context_pair.py:41)

但 shared 也用了真实 family 分层，故 shared 对照检验的是“增加条件化是否有用”，不是“任何事件标签利用是否有用”。条件规则只经 `B_eff=Σq_k B_k` 起作用；所有 B_k 相同就退化为共享模板，q 与 B 同步置换也不改变输出。没有逐专家语义目标，不能把六个表解释为六种物理危险机制或有序阶段。[混合公式](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_conditional_feedback_20261102/core.py:9)

新 census 更限制了归因：TRAIN 全归档为 10044 条/1303 炮，并非本次实际选中的 435 个 episode；safe origin 未知占 8682/9293，VAL safe 为 618/638。TRAIN origin 类别 E 的 danger 仅 3 条/3 炮，D 为 104 条/76 炮；VAL safe 的 B、E 均无支持。六类 softmax 在未知区间仍给出六类权重，不是已认证的事件检测。A–F 来自 whole-record K6 开发教师；本轮不读 TEST 信号，并未消除上游可能的历史源 TEST 暴露。

**可证伪检查：**若主进程需要解释失败，可保持成品表及输入不变，仅以各头 TRAIN 均值 q 替换逐 episode q，按已有 known/unknown × safe/danger 单元比较原有收益。若几乎不变，变动事件条件不是必要因素；若真实 q 在有支持单元稳定胜过常量 q 与匹配打乱条件，才能排除这一退化解释。当前平均成绩不能作此证明；此检查不重训、不换模块，也不用于挽救已失败 gate。

### 2. 纯常数偏移被消去，但风险轴仍可把固定模板变成内部校准杠杆

辅助预测实际为 `F(W)+a(W)B_eff`，其中 `a=(logit−风险头bias)/风险头weight范数`；q 在 origin 发出并固定。设 C 为按每通道已观测时间点去均值，则内层使用 `SmoothL1(C辅助预测, C真值)`。对任意通道常量 c，`C(B+1cᵀ)=C(B)`：每个表跨 25 点的时间常数部分不被数据损失识别，也不能驱动内层更新；L2 只是选择较小的代表。因此不能笼统断言本轮“只学了常数 forecast 偏移”。[风险桥](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/risk_bridge_forecaster_20261102/core.py:12)、[中心化损失](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/gate_loss_audit_20261101/core.py:37)

仍允许的是固定的、有时间形状的条件模板。令 e 为中心化损失对辅助预测的导数，则其状态梯度含 `J_Fᵀe + ⟨B_eff,e⟩∇_W a`；后一项直接沿既有风险坐标传递。它可以利用诊断残差控制风险方向，而不等于学会新动力学。外层减去的未更新 BCE 是对参数的常数，不能自动约束“两组都改善”。safe 改善而 danger 恶化与风险下压相容，但不是每点 logit 下压的证明。[外层目标](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_event_tasks_20261102/trial.py:249)

**可证伪检查：**由现有归档逐炮核对 signed Δlogit，并对同 episode 比较 real−reverse/hold 的方向和损失，而非只看更新次数或合并均值。若 danger 的风险不是普遍被压低，且真实有序反馈独有改善，简单“共同下压”解释会被否定。当前 true_reverse 已优于 true_real 的 balanced，故没有有序新轨迹必要性的证据；这不证明所有方法都不需要轨迹。hold 在该损失中的目标时间形状归零，但 origin 输入与 q 仍来自观测；reverse 保留数值/分布，两者都不能叫“无信息”。W 改变的是实际 FFN 表示，也不应误称最终分数补丁。

### 3. JT 标签、EAST 标签与告警时限不是同一个可迁移对象

本轮后续点标签仍为 JT 的 `max(25ms, AdvancedTime)`，破裂 query 仅要求 lead>10ms；early 仅评价、不训练。每个 episode 从零 W 开始，origin≥127，成熟于 origin+25，查询在 origin+26…50。缺测视图同时屏蔽输入和内层隐藏通道目标，合法地检验“其余已到达诊断能否补偿缺测”，并没有隐藏真值重建监督，也没有连续炮内状态验证。[标签与时钟检查](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_event_tasks_20261102/data.py:42)、[每 episode 零状态](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_event_tasks_20261102/trial.py:234)

假如某 JT 炮有效窗为 25ms，同为 lead=50ms 的点属于源端 early/0，却在 EAST 的固定 125ms 点标签中属于正例。这不是 HDF 错误，而是任务定义不同；源端 lead>10ms 更不蕴含 EAST 的严格 lead>30ms。A–F 分类、局部 BCE 和缺测补偿都不能自行识别这个跨装置标签映射。当前 census 没有 lead 分箱，不能声称已覆盖或改善 EAST 合法提前窗。

**可证伪检查：**解释现有源端失败时，仅将归档查询按 `(10,30]、(30,125]、>125ms` 并列，保留原源标签、safe/danger/early 分组，不重标、不重训；检查伤害是否局限于 EAST 不合法时限或源标签不一致区间。若合法窗且源/EAST 标签一致的 danger 仍受损，“仅因时限不同”解释即被否定。即使该切片改善，也仍不是 EAST 成功证据。

## 冻结结论与交接

当前应按既定 gate 停止此成品晋级，不调参补救、不为它安排一轮 EAST360。上述三项只是可交给主进程的失败机制诊断，本文未执行。即使 replay/audit 通过，也只会增强执行可信度，不会把 `passed=false` 改成方法有效。继续保留合格源模型 c727…、权威 HDF 标签、固定阈值与既有最终风险路径，不把事件概率叠加到最终风险；任何未来重新授权的方向仍须先证明可归因的炮内收益，再讨论严格 >30ms 的实际告警效果。用户允许 FPR≤0.20 下 TP/FP 权衡，但本次没有 EAST 操作点，不能借此豁免源端 gate 或虚构 TP/FPR 收益。

材料锚点：census SHA256 `71b1dec2ec08f97d505dccc0fcc7c5d3dd75b62143118fe8e4ceff9896783bbb`；validation summary SHA256 `dec8e8c05363c84d44943270885a11e2cfb9986efbeae48dd67cb8c19a41a74c`。只锚定本文读取的汇总，不构成独立 records/神经重算认证。
