# EAST 炮内 context operator：有界设计审查

## Material Passport

- 日期：2026-10-31。流程：academic-research-suite / experiment-agent 的设计审查，及 results-analysis 的比较和证据核对。状态：**ANALYZED / PLANNED**，不是新实验的结果或复现。
- 范围：指定工作树内当前代码、`current_order_context360_v2`、已结束 short_history 和既有审计。审查子代理唯一写入本文件，没有启动GPU、训练、模型推理、修改代码或搜索新文献。本审查由主控委派的gpt-6.1-terra/max子代理完成，未再启动子代理；接口未暴露服务档位，因此未设置或声称fast/priority。本路由说明由主控按实际工具记录校正。
- 已核对父 `cold.pt` SHA-256 `31ccf7a12567d3af4d4df93d41894339a81077e89d6d47aebcc24306673173a7`，map `7056d394be54463f6891c166e47de3918fb0d4c38a0585aad1341bb274725ff2`，order `ad44224b7f760eed43305420200c790ee893dc5511dd2f0c0bd280d1a8d5e5c9`；所读 operator helper 的 SHA-256 为 `0196f46ad654d23a93655167621051a8ad2b3aeb448737cfb29a1366053bbeda`。这些是本次文件指纹，不证明未来运行使用相同内容。

## 判定

**这是一个有针对性的“风险监督训练炮内更新算子”开发实验，但六臂本身不能完全排除“通过内部状态实现的分数修正”，更不能证明泛化或成功。** 在固定原 cold 的 source/base、`slow`、`memory.output`、map 和阈值下，若真实时间对齐反馈的 learned rule 相对 source rule、关反馈及指定反向控制有一致的逐炮信号，能够支持“经风险监督改变的炮内反馈机制在该开发流上产生了作用”。它排除的是直接修改分类头、阈值、慢状态或显式输出偏置；冻结读取头不阻止 operator 学会给中间 token 提供近似恒定的偏移，后续非线性仍可把它表现成分数平移、尺度变化或工况特征校正。

更严格地说，`learned_rule − source_rule` 识别的是**这次 cold20 风险优化后算子与源算子之间的整体差异**，其中可混有基函数、增益、静态工况依赖和反馈响应的改变。`learned_rule − reverse/negative/zero` 是固定 learned 参数下的**推理时干预**，不能单独证明训练阶段确由风险标签、而非一般参数扰动或等预算其他训练，产生了特定收益。若这些区分是最终科学主张的必要条件，需要以后另做预先定义的、同预算同数据的非风险或错配监督训练对照；不把额外训练混入当前六臂，也不根据 360 结果临时挑选对照。

## 提议的合同与已有证据

- 父状态是 [current_order_context360_v2](probe_outputs/current_order_context360_v2/completed.json) 的**原采样** cold，而不是 prefix16 cold。前 20 炮均先用原 source 预测并封存，后揭示；20 炮结束后沿用已拟合 map/slow/output，才在这 20 炮上训练 operator。前 20 炮不能用拟合后模型回算成主预测；后 340 炮不再持久更新，临时 8 维 context 每炮从零开始。
- 仅 `memory.key.weight`、`memory.key.bias`、`memory.kernel`、`memory.rate` 可训练，共 7,001 参数。两 epoch，每 epoch 对 18 条 D 按原序各一次，与 2 条 S 交替配对；每次 `0.5 L_D + 0.5 L_S`，Adam `0.001`、全局 clip 1，36 次 step，固定取最终而非最佳 epoch。记录第一对使用哪条 S 和 epoch 边界；若从时间顺序的 `83646` 开始交替，则每条 S 每 epoch 9 次，每条 D 1 次。两 epoch 每条 S 为 18 次、每条 D 为 2 次。
- 采用原 `>=127` sampler：有效点、D 还须 `round(lead_ms, 9)>30`，均匀／source 高分／正点去重；再由 `disruptive_positive_only` 在 D 中只保留正标签。既有 [history](EAST_OPERATOR_HISTORY_20261031.md) 根据封存 source 曲线重建 cold20：选中 1,210 窗口，实际 loss 保留 D 正 357、S 96；这些是相关窗口，不是 453 个独立炮。本次不是重新数样本。运行时应以真实 manifest 复核，不能拿候选模型的未来分数替换原封存 source 分数挑 hard 点。
- 即使 36 个 batch 的标量系数相同，两条 S 各占**总体目标系数 25%**，每条 D 各占 `1/36`，每条 S 是单条 D 的 9 倍；这是名义损失权重，不是经 Adam、顺序和裁剪后的参数贡献。`83646` 在 HDF5 为安全、TEST 清单为破裂；`74087`、`90874` 在 cold20 中有相反方向的争议，真值未裁定，见 [标签审计](TIME_LABEL_AUDIT_20261031.md:48)。主合同仍按 HDF5；不得把 `83646` 的争议当作已裁决，或把窗口重复当成更多独立安全工况。仅在评估时剔除 83646／其余争议炮，不能消除它们此前对 map、原 cold 和本次训练的影响，也不能在无可信破裂时刻时直接重算及时 TP。
- 前轮 [short_history 完成结果](probe_outputs/short_history_context360_v1/completed.json) 的 intra 为 **134 TP / 42 FP**，与其 cold 的 **134 / 42** 在该指标上相同，`intra_acceptance_met=false`；原 current-order 父结果 cold **134 / 42**、intra **133 / 42**，也未通过其 intra gate。short_history 是改变短端点覆盖并重新拟合慢状态／炮间路径的不同实验，不是本次固定 cold 的同一干预，也不证明新提议必败。[既有复核](SHORT_HISTORY_VERIFY_20261031.json) 标示完整但非独立确认，且没有重跑持久在线训练。其旧计划和早期审查描述的“尚无完整结果”已过时；precheck 写证据失败也不等于最终 360 试验失败。

## 六臂的最小定义与可识别内容

启动前锁定各臂参数来源与逐点 delta，不要仅凭名称推断。下表后四臂假定均使用**同一个最终 learned operator**；若实际实现不同，必须据实改变比较解释。

| 臂 | 21–360 的固定父与炮内输入 | 可回答／局限 |
|---|---|---|
| `cold` | 原 cold，显式零 delta | 无炮内修正的固定父。 |
| `source_rule` | 相同父，原 source context8 operator + 真实到达反馈 | 与 `learned_rule` 比较，测这次 operator 训练的整体增量；应重现原父的 intra。 |
| `learned_rule` | 相同父，最终 learned operator + 真实到达反馈 | 本提议的对象；仍可能在内部实现近似分数修正。 |
| `reverse_feedback` | 最终 learned operator，既有 reverse，reference norm 来自同一炮同一 learned 的真实轨迹 | 现有实现是到达目标在 horizon 维翻转、并做 mask 交集，且按 patch 对 **context 状态范数**匹配；不是把整条时间序列倒放，不保证 delta、输出 token 或 logit 范数匹配。早期只有一个 horizon 时翻转可无效；零方向及无法匹配须报告。它是一个特定破坏对照，不是所有错误反馈。 |
| `negative_shift` | 先算真实 learned delta，在进入冻结 `memory.output` 前取精确 `-delta` | 检查真实内部修正方向的反号；不是把反馈目标取反后重新递推。output 为无 bias 的线性层时插入 shift 精确反号，但下游非线性不保证 logit 反号。 |
| `zero_feedback` | 最终 learned operator，`trace(mode='off')` | 当前实现 context 从零开始且无 write，应给零 delta；同父条件下应与 `cold` 数值相同。这个冗余对照主要验证开关／接线，不是额外独立基线。 |

各臂第一 20 炮均复用原封存 source 曲线，不使用后来模型；在同数据、状态、dtype/backend 和代码合同下，`cold`、`source_rule` 全 360 应分别重现父的 `cold`、`intra`。若未逐点一致，先定位实现或数值差异，不能用本次六臂差异归因。对完整流先算原先定义的严格 `>30 ms` 及时 TP／整炮 FP，固定阈值 `0.7258946895599365`；再按同一炮配对列 TP 救回／丢失、FP 消除／新增和首次报警提前量。不要只报一个 J 或选最好的臂；前 20 在全部臂相同，后 340 的配对差异只是补充拆解，不能替代完整 360 的启动成本。

若 `learned_rule` 好于 `cold`、却不优于 `source_rule`，只能说该算子通路可能有作用，不能说本次训练有增益；若优于 source，但 reverse、negative 或简单整体分数移动也有类似收益，则反馈特异性仍有竞争解释。若差异均小于数值漂移或仅由少数越阈炮驱动，应如实报告；控制失败或方向相反同样保留。可在不改主六臂的事后描述中，报告每炮、每时段 `logit_learned − logit_source/cold` 的分布、中位偏移及去中心后的动态残差，并比较两类炮及阈值附近点；若要做常数 logit 偏置匹配，偏置只能在 cold20 预先估计／固定后作为**辅助诊断**用于后 340，不得使用后 340 调偏置或改正式阈值。此诊断能暴露简单偏移的竞争解释，但不能穷尽所有工况相关的隐式分数修正。

## 必需的最小验证清单（未来执行，不是本次已完成）

1. **参数与访问账本。** 比对 source、原 cold、map、order、threshold 和代码快照；初始化 operator 必须是原 source context8，非 prefix16 或其他目标训练状态。训练前后逐 tensor 比特核对冻结权重，列出仅四组 `requires_grad`／optimizer 参数及 7,001 总数；记录 20 条炮的封存、揭示、采样与模型版本顺序，确保任何训练标签均在该炮封存后出现，后 340 没有优化器 step 或跨炮 context 继承。列出 36 个 D/S ID、endpoint／标签／保留数、loss、裁剪前范数、裁剪发生与否、参数差异及最终状态 SHA；不得选择某个较优 epoch。
2. **真梯度链和缓存断点。** 必须是 `key/kernel/rate → 实际按到达顺序 write 的整条 context trace → endpoint_corrections → 固定 output/slow/第二 Transformer block/risk head → D/S BCE`。冻结权重仍需对输入求导；不能以 `torch.no_grad()` 包住后半段，也不能用 [lifecycle `memory_delta`](experiments/risk_aligned_online/lifecycle_stream_20261025/model.py:41) 或 [cache_example 的 detached delta](experiments/risk_aligned_online/lifecycle_stream_20261025/learning.py:14) 训练。不能在 patch 间 detach state、截断 BPTT，或跨 optimizer step 重用旧 trace/delta。当前 [operator helper](experiments/risk_aligned_online/east_context_operator_20261031/learning.py:13) 从 `trace` 直接进入 risk，`window_loss` 为目标；它没有规定调度、配对或完成运行。`forecast_loss` 在 helper 中仅作有限性检查／记录，不得偷偷加入优化目标或在线决策。
3. **真实炮数值与因果预检。** 在至少一条 D、一条 S 的真实冷启动数据上，先比较重新计算与缓存的 logits、loss 及四组梯度；只缓存 frozen first-block `h/valid` 和输入 patch means/complete，条件是 map、窗口、mask、dtype、eval/backend 完全相同，不缓存更新后的 token、logit 或 delta。查每组梯度有限；对非退化样本用小扰动有限差分或方向导数核对，记录被裁剪／饱和的情况，不能要求每炮每组都非零。测真实 `memory.output` 非零且冻结；它若为零，风险梯度可全部阻断。每次更新后 trace 必须重算。原 helper 根据完整 patch 数给 endpoint 上界，**不能推知最后不完整 patch 的真实炮长**，调用方仍须检查 `0 <= end < len(signal)`。
4. **前缀不泄漏。** 使用真实炮，选若干包含 patch 边界的端点，比较完整轨迹与在该端点截断／仅替换未来 values 和 availability 后的该端点 delta、logit、alarm；同时检查窗口左填充和 masks。只能用已完成 patch 的到达对；如果完整 trace 内预先计算诊断性 forecast loss，不得让其未来目标进入当前输出、采样或训练目标。固定同一父与 operator 时每炮 `0..15` 的所有臂应与 cold 相同，索引 16 只是**可能**首次起效，还需实际可用的两 patch 诊断；无可用反馈或零方向应单列。
5. **决策及对照核验。** 逐炮保存各臂首次越阈位置、及时性、完整曲线或其可核验 hash、实际 writes、context 范数、delta／内部 shift 范数及差异；reverse 另存 reference／实际范数、mask 交集、零方向与不匹配次数；negative 逐点核对 `delta_negative == -delta_real`；off 的 delta 为零且与 cold 同曲线。比较时确认 source/learned 同一输入同一固定父，不把 reverse 的 context 范数匹配误写成 logit 范数匹配。阈值边缘的微小 backend 漂移须与真正的报警翻转区分。
6. **标签敏感性与报告。** 主分析固定 HDF5 全 360、180D/180S；辅助报告 342 条两来源类别一致炮的相同配对指标和逐炮变化，明确其仍继承争议训练与父状态。分别列 `83646`、`87484` 在训练中的 loss、梯度范数或方向与权重；这是固定轨迹下的诊断，不等于去掉某炮后重新训练的因果效应。分开报告两条其他争议 cold D 的影响，不自行更改标签。

## 最重要的失败机制和解释边界

**训练 loss 下降不等于未来报警收益。** 首先，优化的是同 20 炮采样窗口的类别平衡点 BCE，D 仅保留破裂前正标签；正式指标是整炮首次越阈是否及时以及安全炮是否任一点越阈，两者目标不同。降低大量高置信窗口的 BCE 可以完全不改变一次越阈；也可能降低安全分数的同时损失破裂提前量。其次，18D/2S 的高相关窗口和每条 S 的重复暴露让 7,001 个参数记住两条安全轨迹、特定缺失模式或输入 map；36 个顺序相关 Adam step 及 clip 的结果对早期配对敏感。既然原 cold 的 slow/output/map 已由同一 cold20 拟合，这 20 炮不是独立验证，甚至其中留出一炮仅不训练 operator 也无法使父模型独立。两 epoch 不选最佳可避免一种挑选，却不能自动消除过拟合。

再者，sampler 仅直接监督 `>=127` 的端点；炮内每个受监督端点仍通过完整**此前**写入链反传，并非只训练一个时刻，但没有对 `16..126` 的预测直接施加 BCE。梯度可能经长链衰减、相互抵消、因状态范数截断／sigmoid 饱和而削弱；可用诊断缺失、output 的近零方向、trace 或缓存被 detach、后续非线性对 delta 不敏感，也可能使实际更新无效。另一个风险是训练让真实、reverse 和反号都以不同方式改变分数，但无真实时间配对专属性。

**前 16 个索引不可由该次炮内更新改变。** 既有已封存结果中原 cold 的 42 个 FP 有 7 个首次越阈 `<16`：`85320@15、60239@15、85316@15、87363@7、93983@5、85422@5、93306@15`；它们均在 cold20 之后。相同 source、map、slow、output 和阈值下，本次仅训练 operator、并在每炮从零启动，无法改变它们已经发生的首次报警，也不能用之后分数下降撤销整炮 FP。第一可能影响索引为 16，不保证实际有反馈。这七个 FP 对该固定合同不可挽回；不代表其余 35 个一定可消除，或其他实验中改变慢状态后仍不可变。索引是保存记录零基位置，不等同已证明的物理放电起点。

最后，即使完整流出现正向差异，它仍是反复用于开发的平衡 360 炮和单一训练轨迹；多窗口不是独立样本，单次配对差异不是独立前瞻确认，也不证明已学会物理规律、满足实时延迟或适合部署。已有源域风险训练、其他目标域记忆训练和 meta-learning 先例见 [operator history](EAST_OPERATOR_HISTORY_20261031.md)，所以此处不声称新颖。当前无本提议的训练或六臂结果，不能声称成功。
