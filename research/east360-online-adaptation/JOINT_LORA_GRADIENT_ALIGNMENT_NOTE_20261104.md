# Joint LoRA：用结束炮风险梯度教诊断更新方向

## Material Passport

- 状态：**一个未实现、未训练的机制假设**，不是当前 joint LoRA 四因子
  实验的结果，也不是 EAST 收益或独立验证。当前普通 joint 方案正在由主进程
  运行，不能预设其成功或失败；本侧任务不修改它。
- 已读 [现有文献笔记](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/INTERNAL_FAST_MEMORY_LITERATURE_20261104.md)、
  [joint LoRA 计划](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/JOINT_LORA_PLAN_20261104.md)
  及其当前 `model.py`、`training.py`、`timeline.py`；原始论文限下述三篇。
- 目标：只在已结束 warm20 的有监督准备中学习一个小的**诊断损失权重**；
  炮内真正驱动 LoRA 的仍是已到达的真实诊断预测误差梯度，不在在线阶段
  计算风险标签梯度，不增加风险头。所有公式是可否证设计，不是安全保证。
- 本任务只读文献和本地源码，不读原始 HDF、不运行 GPU/训练。当前工具未
  暴露独立子代理派生接口；论文与反例检查由同一助手完成，非独立盲审。

## 1. 为什么普通联合训练还差一个条件

当前 joint 目标是 `BCE_risk + 0.1 × MSE_diagnostic`：它让两项损失共享
末层 Q/V rank-2 LoRA，但并不约束在**实际512维 LoRA 参数空间**内
`∇θ MSE_diagnostic` 与 `∇θ BCE_risk` 的方向。炮内只有诊断目标成熟，
不能假设离线两损失下降意味着这一步会改善风险。先前本项目已做过用结束的
EAST20 将诊断梯度映射到后续风险梯度的 Q/K 岭回归；其真实、反序等臂在
对应历史流均为 `125/32`，没有建立可归因收益。那是不同参数空间、父状态和
映射，不是本提议的结果，也不能把“诊断梯度对齐风险梯度”称为全新。详见
[既有历史审查](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/GATE_LOSS_HISTORY_REVIEW_20261101.md:45)。
近期固定 n=1 边缘缩放控制也已失败；本候选不采用其缩放或声称单位错误，见
[n1 结果复核](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/N1_ALIGNMENT_OUTCOME_REVIEW_20261104.md)。

这里有一个当前实现里的**零梯度起点**：诊断输出是 `W0·key(x,a) + Dψ(zθ)`；
`W0·key` 只依赖输入和冻结 source 坐标，残差头 `Dψ` 初始权重/偏置为零。
因此初始 `∂diagnostic/∂θ=0`，不能直接把一堆零向量当作已对齐。应先完成
普通 warm20 的既定联合准备，固定该结果，再确认可用位置实际有非零诊断和
风险梯度；如果没有，明确停止，而不是加任意梯度或声称已有更新。
该结论来自当前 [model.py](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/joint_lora_20261104/model.py:125)
的计算路径，不是对运行结果的观察。

## 2. 三篇原文能支持什么

1. **Yunshu Du, Wojciech M. Czarnecki, Siddhant M. Jayakumar, Mehrdad
   Farajtabar, Razvan Pascanu, Balaji Lakshminarayanan**, *Adapting Auxiliary
   Losses Using Gradient Similarity*, arXiv:1812.02224v2（2020；首版2018，
   这里按预印本引用）。亲读 [§3、Proposition 1、式(3)、§6](https://arxiv.org/html/1812.02224v2)。
   其更新包含**每步可得的主任务梯度**：
   `θ' = θ − α[g_main + max(0, cos(g_main,g_aux)) g_aux]`。
   原文特别说明梯度相似度是局部量；§6 也讨论噪声和 Adam 状态。
   **不能借其定理声称当前实时流有保障**：本问题没有当前炮的 `g_main`，
   本提议只在结束炮训练权重，在线仅计算诊断梯度，优化器和状态也不同。
2. **Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel Urtasun**, *Learning to
   Reweight Examples for Robust Deep Learning*, ICML 2018, PMLR 80,
   4334–4343；[正式记录](https://proceedings.mlr.press/v80/ren18a.html)，
   亲读[原文 §3.1 式(4)–(9)、§3.2 式(12)](https://arxiv.org/html/1803.09050v3)。
   它以带权训练样本的一步更新对干净验证损失求元梯度，式(12)联系了训练与
   验证梯度内积。可借鉴“用后果教权重”，但论文重加权的是样本且依赖其设定的
   清洁验证集；这里重加权诊断通道，warm20 并不是独立无偏验证集。
3. **Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, Moritz
   Hardt**, *Test-Time Training with Self-Supervision for Generalization under
   Distribution Shifts*, ICML 2020, PMLR 119, 9229–9248；亲读
   [§4 Theorem 1，式(10)–(11)](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf)
   和[正式记录](https://proceedings.mlr.press/v119/sun20b.html)。其一步改善结论
   依赖主损失可微、凸、`β`-smooth、有界梯度、同一点正内积和规定的 SGD
   步长。LoRA 非凸、Adam 加裁剪/投影、不同时间窗口不满足该结论的直接适用
   条件；**正余弦最多是局部代理，必须实测实际位移与风险损失**。

核查中明确拒绝三种扩张：Du 的定理不支持无实时主梯度的本部署；Ren 的
“干净验证集”不能套到已暴露的 warm20；Sun 的凸 SGD 结论不能直接套 Adam
或炮级 FPR。以上是对三个 claim/source 配对的同上下文人工反证，不声称
独立审稿、穷尽检索或论文作者报告过核聚变实验。

## 3. 最小候选：只学有界、过去条件化的诊断通道权重

### 固定模型及合法样本

先保留现有共同 warm20 结束时的 target-side `θ*`（512个 Q/V LoRA 参数）、
`W0`、诊断残差头 `Dψ*`、原风险头、source scaler 和原阈值；本候选的额外
准备阶段**只训练下面的权重器 `φ`**。这发生在正在运行的普通 joint 实验
之外，是否执行由主进程另行决定；要有额外计算预算相同的对照。

在20炮全部预测封存并逐炮揭示 HDF 结局后，按现有时钟构造离线教师样本：
`o=127+32k`，`m=o+8`；只有 `o+1…o+8` 真正存在且成熟才计算。对诊断，
沿用 `direct_target` 的 8×21 **真实未来增量**和真实 mask，排除 MNM 与
`mode_number_n`，CIII 只有实际可用时才参与；不把模型递归预测当真值。
对风险教师，使用同炮 `m` 时刻窗口及已经揭示的 HDF 标签，在与诊断梯度
**相同的更新前 LoRA 状态**计算 `g_r`。EAST 标签仍为125 ms 窗口、及时
正例严格 `lead>30 ms`；较早破裂负单列，不冒充安全。教师的标签只用于这20炮
结束后的离线拟合与审核，不输入任何后续未结束炮。

保留先前普通规则在 ended20 上按炮回放得到的 `θ_(m−)`、Adam 状态、origin
表示和 mask 作为固定教师状态；不把后续 candidate 轨迹称为这些教师轨迹。
可预先固定每炮每适用组最多8个等距合格 origin，并按**组→炮→窗口**平衡，
记录全部候选和不足情况；这只是离线抽样，线上仍执行计划的固定时钟。

### 权重器与损失

令 `z_o ∈ R^192` 为现有诊断头所用的过去窗口表示（CLS、最后 patch、
最后与前一 patch 之差）。令 `h_o = stopgrad(LN_nonparam(z_o))`，不把任何
风险分数、标签、event 类别、未来信号或总炮长送入权重器。给出一个具体的
低秩规格（`φ` 共 **447** 个数）：

```text
a = U V h_o + b       U:21×2, V:2×192, b:21
w_c = 1 + 0.25 [tanh(a_c) − mean_j tanh(a_j)]
```

所以 `0.5 < w_c < 1.5` 且 `mean_c w_c = 1`。令 `U=0, b=0`，`V` 按固定
种子取有限非零初值，起点严格为 `w=1`；否则两个因子同时为零会产生死梯度。
`h_o` 对 LoRA **detach**，但离线求 `φ` 梯度时不得 detach 权重器本身；线上
权重器冻结，当前窗口权重在 origin 固定、到 maturity 使用。

设 `e_hc` 为诊断预测减真实增量、`M_hc` 为真实可用 mask、`D=Σ_hc M_hc`：

```text
L_d(θ; w) = [Σ_c w_c Σ_h M_hc e_hc²] / D                 (D > 0)
g_c       = ∇θ [Σ_h M_hc e_hc² / D]                     (θ = θ_(m−))
g_d(w)    = Σ_c w_c g_c = ∇θ L_d(θ; w)
g_r       = stopgrad ∇θ BCE(logit_θ(x_m) − logit(τ), y_m)
```

分母**不按 `w` 改写**：`w=1` 时在数学上逐个 origin 退回当前单窗口未加权
masked MSE；分组累加可能产生浮点舍入差，需用预定容差测试。`D=0` 时不更新、
不编造目标。每个 `g_c`、`g_r` 都是同一顺序
拼接的512维 Q/V-LoRA 梯度；`g_r` 只在结束炮存在。由于输入表示对 `θ`
已 detach，等式不混入 `∂w/∂θ` 造成的额外伪方向。离线可在固定教师状态
缓存21个 `g_c` 与 `g_r`，训练时只组合向量，不必通过 Transformer 再做
二阶反传；对应直接加权 MSE 的一阶梯度必须通过测试。
21个全通道权重均值为1，**不保证实际可用点上的加权均值为1**；缺失模式
变化仍可能改变有效梯度尺度。应记录 `Σ_hc w_c M_hc / D`、梯度范数与
实际位移，比较同预算/边界控制，不能把尺度效应误称方向学成。

先对每个教师样本的 `G=[g_1 … g_21] ∈ R^(512×21)` 做数值秩及 SVD
投影审查：`||P_col(G) g_r|| / ||g_r||` 是任意非零 `G w` 与 `g_r` 的原始
余弦绝对值上界。若投影为零，任何通道权重都不能让**原始梯度**同向；若很
小，也无法达到强同向。反过来，投影大不代表当前正且有界的 `w` 一定可达，
也不界定 Adam 预条件后的实际位移。按炮及组记录秩、容差和上界；不要把
不可达样本偷偷从支持分母中去掉。

真实 intra 使用 Adam、`clip_grad_norm=1`、相对炮初 LoRA 的联合半径0.5
投影，所以只对齐 `g_d` 不够。记 `Δ_A(w)` 为在缓存的同一 `θ_(m−)`、
Adam moments/step 上，对 `g_d(w)` 按现有 Adam→裁剪/投影合同得到的
**实际参数位移**（裁剪在 Adam step 前，投影在 step 后），而不是把 Adam
近似为 `−ηg_d`。定义平滑余弦 `Cε(a,b)=a·b/[sqrt(||a||²+ε²)
sqrt(||b||²+ε²)]`，例如固定 `ε=1e−12`、数值累积用 float64：

```text
L_φ = mean_group→shot→origin [
         1 − Cε(g_d(w), g_r)            # 两个损失的梯度同向
       + 1 + Cε(Δ_A(w), g_r)            # 真正的位移指向风险下降侧
      ] + 0.01 mean_c[(w_c−1)²]
```

最小版本对 HDF-safe 与 timely-positive 两组各权重1/2，earlier-D 另报不混入
safe；`g_r` 及缓存的 `g_c`、教师状态都 detach，仅优化 `φ`。这是在单个
固定教师状态上的一阶目标，不保证多步 candidate 轨迹或实际风险下降。
`0.25`、`0.01` 和余弦数值量是**单个工程示例、不是最优值**，若执行须在
看对照结果前固定、不扫描。可以先定 `64` 个 gate 优化步、batch 16、
Adam `1e−3`、seed 131、只取末步；额外64步给所有对应 gate 控制相同预算。

在 `w=1` 下先固定支持集合：真实 `D>0`、`||g_r||`、`||g_d||` 和对应
`||Δ_A||` 均超过预声明容差。拟合后**不得按 candidate 的梯度再删困难样本**；
小/零梯度或投影零位移按平滑余弦计算并单列为无更新，覆盖不足即停止。初始
诊断梯度全零时不存在有意义的方向，也不得换目标伪造支持。

### 在线实际做什么

前20仍为原计划预测并封存；20炮结束后训练并冻结 `φ`。后续每炮复制对应
slow LoRA、重置 Adam，origin 用**当前及过去**窗口生成并保存 `w_o` 与诊断
预测；到 `m=o+8` 才使用已到达真实值，以 `L_d(θ;w_o)` 对 LoRA 更新。
按当前 joint 计划，成熟值到达后可更新、再发出时刻 `m` 的风险分数；更早
分数/已发布诊断预测不回写。炮末丢弃 fast，`φ` 不在后续炮用标签重训。
若比较 inter/both，仍须共用计划中的唯一 slow 更新和记录；本门控冻结，不能
用不同 gate 让两臂的历史监督清单不同。此处没有直接改风险头、加风险分数或
用模型预测生成真值；但 `φ` 是用结束炮风险标签训练的，**不能称整套方法为
纯无监督**。过去表示本身可能编码风险相关信息，权重器也可能间接形成风险
代理；不接最终风险分数并不证明它学到物理机制。基于 uniform 轨迹缓存的
教师状态也可能偏离 candidate 多步运行后的状态，必须在闭环配对中检验。

## 4. Source event 标签边界

现有 [source event 计划](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/SOURCE_EVENT_TASKS_PLAN_20261102.md)
只允许将 J-TEXT TRAIN 的事件 A–F/unknown 作为**源域离线**分层或辅助诊断；
本最小候选的在线权重器没有事件输入。一个具体辅助检查是：固定已训练的
`θ* / Dψ* / φ`，对既有 source TRAIN episode，按真实/打乱事件类别和真实
mask 分层比较实际诊断梯度、风险梯度及一次有限步风险变化；source 风险教师
仍按 `max(25 ms, AdvancedTime)`、及时标准严格 `>10 ms`，unknown 保留未知。
这不把事件类型变成 HDF 风险真值、不训练 source checkpoint，也不据此给
EAST 炮创造事件类别。该 source 检查不是独立验证；源事件条件此前已有负
结果，不能假设它迁移有益。若未来要把事件损失纳入拟合，须另行固定预算和
控制，而不是悄悄把它加入本方案。

## 5. 测试、比较及停止条件

1. **CPU 数学合同。** 合成 fixture 同一 `θ`/mask 上验证 `Σw_c g_c` 与
   对加权 MSE 一次自动求导一致；`w=1` 与现有单窗口诊断损失/梯度一致。
   覆盖 `D=0`、真实零值、CIII 部分缺失、全缺通道、零梯度、边界投影、
   梯度列空间与风险梯度正交、finite 和输入不变；无支持必须 no-op 且显式计数。
2. **优化器与时钟。** 函数式 `Δ_A` 对实际 `Adam` 在首步、非零 moments、
   clip 生效及半径边界分别数值对账；核对实际 `g_r·Δ_A` 和同一 query 的
   有限步 `ΔBCE = BCE_after−BCE_before`。实际位移不一致、方向预测与
   有限步经常矛盾、或梯度到不了 `φ` / Q/V 时停止，不以正余弦代替验证。
   修改 `m` 之后的信号或当前炮 EOS 标签，不得改变已发出的风险、已生成的
   权重及不晚于 `m` 的已到达更新；`o+1…o+8` 作为该 origin 的完整诊断
   监督不得使其参数更新早于 `m`。这些点逐个到达时仍可作为普通已观测输入
   影响其后的风险前向，不能错误地禁止这一合法影响。
3. **匹配控制。** 同一普通 warm20 父模型及计算预算比较：无更新、`w=1`
   原始 intra、按相同教师目标训练的非条件固定21权重、条件 `wφ`；并按
   现有合同比较真实反馈、hold（零增量）及 reverse（**真实增量取负，不是
   时间反转**）。若检验门控上下文，负对照只将同一当前过去窗口及其 mask
   时间逆序来生成 `w`，真实诊断目标与风险输入不逆序；它不是部署方案，且
   要报告 padding 混杂。各臂同目标支持、同起点、同 Adam/步数/边界，逐炮
   报告未加权真实诊断误差与保持基线；输入条件权重已有项目先例，不能仅凭
   更新存在就声称机制有效。
4. **小样本与 gate。** warm20 的 HDF 组成是8破裂/12安全。若事先固定留出
   最后2个破裂及最后3个安全炮**仅不用于 `φ` 拟合**，其余15炮教 gate，
   必须按炮而非按窗统计安全及及时正例的实际 `ΔBCE`、方向、更新率和
   覆盖；两组任一无支持，或任何一组相对无更新恶化、未优于对应 `w=1` /
   hold / reverse，判失败。这个留出只是**条件性 gate 检查**：普通 warm20
   模型、W0 和既有 density map 已见过这些炮，绝非独立留炮验证。真正的
   shot-held-out 检查需在折内重建所有会见到留出炮的拟合组件并单独声明 map
   暴露；不把两种检查混称。
5. **完整评价若另获执行授权。** 保持现有四因子、原 threshold、首20成本、
   其后唯一 slow 过程和真实 mask，先比 `intra−static`、`both−inter`，再报
   full360 的 TP/FPR/J、逐炮新增/消失错误及 `>3 s` 和 30–3000 ms 敏感性。
   没有同时满足预先固定的安全/及时组门槛、`FPR≤0.20` 和对假反馈的优势，就
   报失败；真实 MSE 或余弦改善单独不晋级，也不以调阈值或只报后340补救。

参考论文本身不担保本工程结果；当前测试和真实模型训练均**尚未执行**。
