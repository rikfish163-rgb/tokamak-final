# Event 条件诊断反馈：有界复用与因果边界审查

## Material Passport

- 日期：2026-11-02；Origin Skill：academic-research-suite / experiment-agent。
- Verification Status：ANALYZED。只读代码与已完成 JSON；未重跑、未读原始 HDF、未启动 GPU。
- 已读新 `context_fit.py/context.py/core.py` 与既有五条 event 路线；新 pilot 的训练、持出成绩和部署调用链不在本次已验证范围。

## 结论先行

**高层想法已有明确先例，但当前参数化不是五条旧实现的逐字重做。** 115→6 源 event 读出、真实/打乱事件监督、event 条件化更新方向都不是新功能；最接近的遗漏先例是旧 `ContextFeedback.context/update_observed`，它已经把冻结的六类软概率送入更新控制器。

本候选最小差别是：在不改 c727 分类器及最终 score 的条件下，将**当前风险轴诊断预测系数**从共享 B 改为 `B(q)=Σq_e B_e`，让语义上下文影响成熟诊断误差对内部 W 的梯度。它不是首次 event 反馈、首次元学习或新增物理定律。源 q 的高分类率本身不能支持该差别有效。

已读新 core 没有必然的代数循环或同值自蒸馏零梯度，也未见 EAST 真 event/结局作为线上 q 输入。一个需主线明确的配对缺口是：**同 TRAIN 池、seed 和步数不等于同 minibatch 序列**，详见第 4 节。

## 1. 与既有实现逐项对应

| 既有实际函数 | 已实现的作用点 | 与新候选的最小区别及既有证据 |
|---|---|---|
| [EventMetricModel.prepare_matured_forecast_loss / source_step][M] | 冻结源 teacher 的五维坐标误差；学习一个全局 5×5 loss 矩阵，Frobenius 范数固定；源风险外目标经更新训练该矩阵。 | 已经做过“event 定义反馈几何并用风险训练”。新候选改为依 q 变化的诊断预测器 B，不是全局五维误差矩阵。VAL160：risk、identity、signal 均 72TP/6FP，static 73/6；源 gate 未过，[独立记录 PASS][MR]。不是 EAST360 结果。 |
| [EventProtectedFfnModel.project_intra_gradient][P] | 由六类源 logits 对比得到五维方向，把诊断梯度投到当前已到达窗口的局部 Jacobian 零空间。 | 是“保护 event 区分方向”，不是 q 选择预测器系数。240 炮固定阈值：static 61/31，event 58/29，random 59/29；不能据保护成功称风险方向正确，[记录 PASS][PR]。 |
| [measure_geometry / gradient_geometry][G] | 事后用未来查询和结局选出的失败案例检查 `g=Jᵀc` 与局部风险梯度覆盖。 | 是诊断，不是部署策略。七个选定案例覆盖范数比约 0.251–0.417，六个下降方向余弦为负；不能当总体收益或可救回上界，[数值核对 PASS][GR]。冻结 q 的 B(q) 仍主要改变既有 forecast/risk Jacobian 的组合系数，不凭六个专家就获得新物理信息。 |
| [event_teaching.loss_from_example / event_objective][T] | 源 event CE 与风险目标共同训练内部 memory，event head 在 EAST 移除；真实/整区间打乱/无监督三臂。 | 已做过“event 监督内部适配”，但不是线上冻结 q 路由 B。已完成配对360：三臂 intra 均129/35，intra_reverse也均129/35；这是炮级计数相同，不是曲线逐位相同，[配对审计][TR]。 |
| [event_pairing.pair_candidates / neighbour_score][K] | 同 event 内按源风险标签构造跨炮 triplet，学391→16表征；EAST20同历史留对 AUC 筛查。 | 已做过“同类活动不同危险演化”，不是成熟诊断 loss 更新 W。修正评价后 risk AUC≈0.632、shuffle≈0.543、raw≈0.453；仅4个独立破裂炮，64炮对不是64个独立样本；只获后续试验资格，[recheck][KR]。 |
| [ContextFeedback.context / update_observed][C] | 冻结推断六类 softmax 作为上下文，源风险外目标训练通道误差权重控制器，更新临时 embedding/first-block；无最终 score 分支。 | **功能层面最接近的重复先例。** 新候选区别只能落在“直接条件化风险轴诊断预测器、当前第一FFN低秩 W、成熟 H25 目标和现行更新规则”。旧 event 流150/107，相同 static145/94，不能换名后忽略，[完成记录][CR]。 |

这些行的队列、父适配状态和临时参数位置不同；不能横向拿 TP/FP 排名证明新候选优越。

## 2. 当前新实现实际确认了什么

- [context_fit.standardize_train / training_targets / fit_head][NFit]：TRAIN327炮/401事件/10583端点拟合；VAL83炮只诊断；mean/std 只由 TRAIN 事件等权拟合。TEST 成员字节会随归档解压，ID 用于角色隔离检查，但代码不对 TEST 做特征变换、拟合、预测或评分。不能表述成“完全没读取任何 TEST 字节”。
- [context.select_variation_features / at_origin][NCtx]：沿用旧 [causal_summary 的115维描述符][Raw]，末值与均值置零，仅留 std/half-change/availability；MNM 在源特征和线上辅助输入双方屏蔽，不改风险模型输入。`at_origin` 仅接收 values/available，取原点过去128点，`no_grad` 推断 q。
- [effective_table / prediction / update][NCore]：`context.detach()` 后混合六张表；shared 分支直接返回25×23表。预测仍调用原 [auxiliary/augment][Aux]，更新委托固定 metric 的 `full_path.core.advance`，不是 Armijo，也未把 q 加到最终风险。
- 新文件的源监督本质仍是旧 [event_prefix_probe.fit_probe/shuffle_event_targets][Raw] 的复用；旧 raw 源 VAL 条件家族识别率均值约87.0%只证明已标注源区间可区分，[旧缓存审计][RawR] 不证明 EAST event 检测或更新收益。

## 3. 循环、恒等梯度和标签边界

设固定原点上下文 q，辅助预测为 `P(W)=F(W)+z_norm(W)B(q)`。其 W 导数包含 `J_F+B(q)J_z`；风险输出仅在内部产生辅助梯度，之后由更新后的 W 影响下一查询。这是显式反馈计算，不是以 EAST 真标签解隐式固定点；它也不是 `CE(q,q.detach())` 那类同值伪目标。

冻结 q **不应**顺带 detach `F(W)` 或 `z_norm(W)`；后者会删掉拟检验的直接风险轴梯度。当前 `effective_table` 只 detach q、保留表的梯度，`auxiliary` 保留 W→forecast/logit 路径，未见这种恒等错误。

仍有三种可检验退化，而不是“接受更多就有效”：

1. 所有专家 B_e 相等或 EAST q 几乎恒定时，函数退化为一个共享 B；全零初始化的第一次数值路径也不区分 q，但不意味着其外层 B 梯度恒等为零。
2. 当前 [centered_loss][Center] 去掉每通道已观察 H25 均值：B 的时间常数分量在数据项中抵消；无目标通道同样无数据监督。应看中心化后的有效表和真实 W 梯度差别，不能只凭六张原始表数值不同声称条件化有效。
3. q 必须在原点签发并在同次成熟更新/候选评估中复用；看到成熟目标后重选 q，或让 q 依赖可训练 W 再回传，会改变当前宣称的固定上下文问题。已读 `at_origin`/`effective_table` 具备正确局部边界；新 pilot 的具体调用还须主线核对。

源 `outcome` 仅用于打乱分层，不进入115维模型输入；A–F仍是全记录离线产生的弱家族，不是风险序号、真实物理状态或在线起点。未知区间不得变正常类，也不能在 EAST 按真 event、事件结束/持续时间或结局选专家。前20已结束炮的结局可用于折内拟合 B 的风险外目标；持出炮标签只用于评价，不能参与其 B、q 或超参选择。

## 4. 具体配对缺口：标签相关抽样

`context_fit.training_targets` 用 `sampling = event_weight / support[y]`，而 shuffled 的 y 先被改写；`fit_head` 再用相同 seed、不同 sampling 调 `rng.choice`。各家族事件数不相等，故只要事件被换到不同频数家族，其抽样概率就会改变。**当前代码保证同一数据池和预算，未保证相同端点抽样序列。**

若主线合同要求“只改变 event 标签、连 minibatch 都完全配对”，这是一项待决控制混杂，不能称已满足；最小处理是共同固定抽样序列后只替换标签。若有意保持旧 probe 的各臂类别平衡配方，则应如实写明差别，不能把同 sampler seed 当作证明。此审查不改在跑拟合，也不据此宣判方法必然失败。

真实/打乱六表在名义架构和参数上同为3450；q 的路由分布与有效容量仍可能不同，需按现计划报告。共享 B 为575参数简约参照，不是同容量语义控制。

## 5. 只保留一个最小可检验差别

沿用已定的冷20四折留炮局部 pilot，不另开360：**源 event 语义是否经固定原点 q，使成熟诊断 loss 对同一内部 W 的更新方向在持出炮上更有用，而不是只识别家族或形成早期固定风格？** 主比较是同架构 true_real 对 shuffled_real；shared_real 是简约参考，true_reverse/true_hold 检查成熟目标顺序/到达轨迹是否必要，均不改报警器。

保留现计划的逐炮/安全-破裂平衡风险余量变化、反例和真实 W/梯度证据；没有持出方向收益，或仍不能超过 hold，就不足以支持新增语义反馈机制。cold父模型已经见过20炮，留出只针对新增辅助层；单步局部样本也不是整炮累积流，不能报告为新 TPR/FPR 或独立未来验证。

当前已完成基线也支持这种谨慎：full_real126/32与full_hold126/32相同，reverse126/31更高净收益；Armijo meta_real122/30，低于cold125/35的TP要求。两流记录/神经报告均保存PASS，但这不使当前规则通过归因门槛，[full完成记录][FullR]、[Armijo完成记录][ArmR]。

## 定位链接

[M]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_metric.py:41
[MR]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/runs/event_metric_source128_20261021/independent_verification.json
[P]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_gradient_projection.py:113
[PR]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/runs/event_projection240_20261021/independent_verification.json
[G]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_feedback_geometry.py:47
[GR]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/runs/event_dynamics240_20261021/feedback_geometry/independent_verification.json
[T]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_teaching_20261026/learning.py:81
[TR]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/runs/lifecycle_event_true360_service_20261028/paired_event_audit.json
[K]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_pairing_20261024/learning.py:40
[KR]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/runs/event_pairing_screen_20261024/same_history_recheck/completed.json
[C]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_feedback_20261022/model.py:32
[CR]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/runs/context_feedback_east_event_20261022/completed.json
[Raw]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_prefix_probe.py:94
[RawR]: /media/hetaisheng/044A81D94A81C83E/Fuison/outputs/event_prefix_probe_20261021/independent_verification.json
[NFit]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_conditional_feedback_20261102/context_fit.py:154
[NCtx]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_conditional_feedback_20261102/context.py:50
[NCore]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_conditional_feedback_20261102/core.py:9
[Aux]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/risk_bridge_forecaster_20261102/core.py:12
[Center]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/gate_loss_audit_20261101/core.py:37
[FullR]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/full_path_forecast_controls360_v1/completed.json
[ArmR]: /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/armijo_controls360_v1/completed.json
