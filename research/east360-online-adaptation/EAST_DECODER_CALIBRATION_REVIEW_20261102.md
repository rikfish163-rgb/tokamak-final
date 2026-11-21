# EAST 结束炮低秩 decoder 校准：限定设计审查

## 范围与已知结果

本次仅阅读相关代码、计划及完成汇总，不读 HDF/checkpoint，不执行拟合、测试、GPU 或记录级 audit。只新建本文，不改代码、旧报告和输出。下述 gates 是待主进程验证的条件，不是已通过的认证。

假设可作为有界开发诊断：保留 c727、cold20、inputmap、threshold，以 JT `forecast_only` 外部头为固定测量先验，只在已结束 fit15 炮训练残差 C。它不是继续纯 source-only 训练，而是 **JT 先验＋EAST 已结束炮监督校准更新规则**。不回写源/原 forecast_head、不作最终风险叠加，仍符合限定边界；不因此获得独立验证或连续炮内收益结论。

上一轮 JT 主缺测视图的 forecast-only 在 compatible_danger 上 ΔBCE=−0.009286331，但全部 danger 为 +0.000669485。已结束 EAST20、同100个固定 query 的真实反馈结果如下。这里是归档的**阈值余量损失变化**，字段名虽为 `mean_delta_bce`，并非直接与 JT 普通 BCE 比大小；均先炮内、再炮间平均。

| JT 成品头 | EAST D 全时限40点/8炮 | safe60点/12炮 | D30–125ms 24点/8炮 |
|---|---:|---:|---:|
| forecast_only | +0.006421167 | −0.000353963 | +0.002189975 |
| mean_meta | +0.059707637 | −0.020188446 | +0.042319897 |
| worst_meta | +0.033135690 | −0.012344435 | +0.018694340 |

旧 forecast-only 的 reverse balanced=+0.002822140，小于 real 的 +0.003033602；旧 gate 仍 passed=false，optimizer_steps=0、suffix_raw_reads=0、online_gain_measured=false。故选 forecast-only 作固定起点合理，但不能叫“已经验证有效的物理 prior”，更不能把后续校准归因为对源端失败的事后翻案。[JT summary](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/direct_decoder_source_v1/validation_summary.json)、[EAST20 summary](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/direct_decoder_ended20_v1/summary.json)、[完成记录](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/direct_decoder_ended20_v1/completed.json)

## 四个具体风险与可证伪 gates

### 1. 标签、operational 评价与成熟时钟必须分开

拟议 meta 训练只用 D 的 `30<actual_lead<=125` query 作为1、安全炮作为0，不能把 D 的150/250ms query 也作为125ms点标签正例。held 的 D全>30 可继续按已结束炮结局评价 operational 余量损失，但必须与 canonical125 点标签并列说明；它不是“125ms概率”。D30–125 子集则两种正标签一致。两类损失共用固定阈值也不意味着普通点 BCE 与 `margin_loss(z,threshold,y)` 相同，riskΔ采用哪一种须预先写明并三臂一致。[既有标签/时钟核对](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/direct_decoder_meta_20261102/east20.py:149)、[余量损失](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/critical_shot_meta_20261102/objectives.py:63)

**Gate：**每个训练样本均属于该折 fit15；D 正例满足实际严格时限而非名义35/60/100ms。固定 `origin=query−26`，目标仅为 origin+1…query−1，成熟后只影响 query；不得用 query 自身或更晚信号更新该次 W。安全0不取自“未知事件=正常”的补标。held 只用各自25个既定 query，四折合计100个、每炮恰好一次 held，不是每折都评价100个再择优；不重挑峰、不删除不利时限。任一越界即否定该校准比较的有效性，不能用收益抵消。

### 2. 固定 prior 不能固定掉输入梯度；两个 rank4 不可混淆

令 prior 权重 M∈R^(575×1088)，A∈R^(4×1088) 为其固定 top4 右奇异向量，C∈R^(575×4)。残差等价于 `ΔM=C A/2`，确为2300个持久参数、无新增 bias；这不是首 FFN 的 W∈R^(4×64)，也不是其4×128源基底。C 在结束炮间离线学习，W 在每次局部 query 重置、仅由已成熟诊断更新。

必须使用 **实际 W 下**的 final normalized/masked token flatten x(W)：`prior(x(W)) + [x(W)Aᵀ/2]Cᵀ`。冻结 prior/A 的参数，不可用 `no_grad` 包住整个 prior forward，也不可 detach x 或把投影预缓存成 x(0)。否则既有先验的 W Jacobian 或残差的混合导数会丢失，即使总 C 梯度仍非零。[现有无 detach 捕获](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/direct_decoder_meta_20261102/core.py:22)

**Gate：**C=0 时 wrapper 预测与固定 JT prior 一致，W=0 的风险与 cold 父模型一致；固定内层算法、接纳结果及 receipts 与原 prior 基线匹配。meta 训练须 `create_graph=True`，单独验证 `C→∂L_inner/∂W→W⁺→query risk` 的风险项梯度与同分支有限差分，并用实际 FFN hook 对照。拒绝/无目标时风险路径可以确实缺失，不能要求每样本非零，也不能仅拿绝对诊断损失梯度充当 meta 成功。旧 core 的随机 CPU 测试不自动认证新 wrapper。

### 3. 四折隔离的是 C，不是整个已暴露系统；三臂须真正配对

cold 父模型早已见过全部20炮标签，inputmap也绑定同20炮；query 选择使用已揭示结局，forecast-only prior 的选择亦已参考这批开发结果。四折不是全系统独立留出，而是固定已暴露父系统中的规则开发诊断。四折15/5模式本身已有[事件条件 pilot](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_conditional_feedback_20261102/pilot.py:322)与[分层折划分](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/risk_bridge_forecaster_20261102/prepare.py:29)先例；本轮新点是 JT prior 下只训练低秩 decoder C，不是新的独立验证原则。[旧 EAST20 协议](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/direct_decoder_ended20_v1/protocol.json)

**Gate：**固定各折 held2D3S、fit6D9S 的同一名单；每折三臂从相同 C=0 开始，共享128步、每步1D1S的实际 shot/query index，Adam .001、clip1及固定末步，不按 held 选 epoch/臂/子空间。forecast-only 的“.20绝对SmoothL1＋1e-4mean(C²)”与两种 meta 完全一致；meta 分别加平均/较大风险增量/.1，不能换成最大原始 BCE。单样本组 max 仍只是 minibatch worst-risk，不保证弱组泛化。

汇总 gate 应使用唯一 out-of-fold 100点、按炮聚合：预先指定候选须 safe、D全>30、D30–125 三组均取得超过数值容差的负 Δloss，并胜过固定 C=0 prior；真实反馈还须胜过同头 reverse/hold，meta 须说明是否胜过校准 forecast-only。逐折和 pooled 均保留，不能删坏折或把100点当100独立炮。失败则停止晋级；若只挑三臂中最优，应明确为探索性开发，不能另称确认性胜出。

### 4. top4 是测量解码方向，不是“危险物理子空间”或稳定性保证

M 的右奇异向量优化的是 decoder 在线性输入坐标下的增益，不保证覆盖 EAST 误差、风险梯度或跨装置动力学。该改动等价于固定子空间的低秩 head 微调，是既有直接 decoder 教学的受限校准，不宜宣称新优化原理。626175参数的完整 JT head 教学与2300参数 EAST校准同时改变了数据域、容量和训练样本数，旧/新成绩差不能单因归给低秩结构。

预测仍在固定源尺度/asinh 坐标，绝对 SmoothL1 约束测量水平，内层 centered loss 只约束时间形状，两者不可互称。`mean(C²)` 小不保证残差输出小，尚取决于 xAᵀ；W 的隐藏RMS上限也不保证 decoder 误差或风险一定改善。风险监督可能经诊断梯度实现内部校准，而非发现物理规律。

**Gate：**绑定具体 forecast-only prior、M/A/C形状、SVD截断/符号约定与 A 的哈希；A/prior为固定 buffer/冻结参数，优化器精确只含 C，原 source、cold、inputmap、threshold、原 forecast_head 均不变，340后缀读取为0。并报 held 绝对诊断损失、残差幅度、接纳及风险变化；若风险下降只伴随诊断语义恶化，应称规则校准而非物理先验迁移。若无收益，也只能否定该 prior/子空间/预算的当前假设，不能外推所有 decoder 或 meta 方法都失败。

## 冻结交接

本设计没有通过修改风险头、阈值、HDF标签或最终分数叠加来取巧的必要；但上述保护和梯度路径仍须由主进程实际实现、核验。本文不跟踪其新结果，不替代 replay/audit。即使四折局部 gates 全过，也只支持继续检验的资格，不是完整360 TPR/FPR、独立未来、实时性或新物理定律的成功结论。用户 FPR≤.20 的 TP/FP 权衡许可保留，当前100个局部 query 不产生该操作点成绩。先炮内可归因收益，再讨论完整运行效果。
