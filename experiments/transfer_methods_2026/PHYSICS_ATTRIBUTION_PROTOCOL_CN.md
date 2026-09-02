# J-TEXT→EAST 在线适应：炮内/炮间与物理归因协议

**版本：** 2026-08-28-r2  
**状态：** 研究审计协议；当前 balanced360 结果仅为 `DIAGNOSTIC_ONLY`   
**适用谱系：** Adv25 源标签、EAST 两时间尺度神经路径，以及其上层 V3C 决策系统

## 1. 目的与结论边界

本协议解决的不是“怎样把在线曲线做得更好看”，而是回答三个可以被审稿人分别检验的问题：

1. 完整在线系统是否在固定报警合同下改善了炮级指标；
2. 改善是否来自炮内或炮间的神经参数更新；
3. 神经更新是否利用了可观测物理演化，而不是利用结局比例、炮号、炮长、报警器或标签泄漏。

三个命题不能互相替代。特别是，`endpoint memory`、报警状态机和专家权重能够减少误报，并不等于 Transformer 学到了新的物理表示。本项目目前采用以下审慎表述：

> 现有 Adv25 seed=131 的完整 V3C 在开发流上观察到系统级误报下降；干净的炮内/炮间神经消融尚未提供跨 seed 的神经增益证据，当前物理代理也尚未证明模型学习了物理定律。

balanced360 已经参与方法和参数开发，因此其全部结果只能用于诊断和设计选择。任何正式论文结论还需要在所有配置、阈值、查询策略和停止轮次锁定后运行一次未见的按时间排序流。

## 2. 谱系隔离

| 谱系 | 初始模型 | 允许的在线变化 | 可回答的问题 |
|---|---|---|---|
| J-TEXT Adv25 | J-TEXT TRAIN1353 源模型，正窗口为 `max(25 ms, AdvancedTime)` | TEST220 上的独立炮内/炮间研究 | 源域上的方法是否有因果、可回退的在线行为 |
| EAST clean two-scale | Adv25 源模型经 support110 适配后的同一 parent | EAST 目标 adapter、独立 residual/head、炮内 fast copy、炮间监督更新 | 炮内和炮间神经路径是否分别有效 |
| EAST V3C | 同一类迁移起点，但增加 endpoint memory、报警控制和 episode gate | 神经、物理、记忆和决策层同时变化 | 完整系统能否降低开发流的操作代价 |

J-TEXT 在线分支不进入 EAST 迁移；EAST 迁移只能从封存的 J-TEXT 静态源模型开始。不能用 V3C 的最终参数反向修改源模型，也不能把历史 V3C 的系统收益充当 clean neural ablation 的重复实验。

## 3. 数据、标签和时间访问合同

### 3.1 标签

J-TEXT 破裂炮的监督窗口为

\[
H_i^{\mathrm{JT}}=\max(25\,\mathrm{ms}, A_i),
\]

其中 (A_i) 是 shot list 中该炮的 AdvancedTime；缺失时按合同使用 25 ms。HDF5 中的炮结局是最终权威来源。EAST 的 operational 评价使用破裂时刻前的固定报警合同，当前审计的门槛为 lead≥30 ms；lead>3000 ms 仍计入 TPR，但必须单独报告秒级过早报警。

### 3.2 预测—封存—查询—揭示—更新

对第 (k) 炮只能执行：

\[
\text{observe}(x_{k,\le t})
\rightarrow \text{predict/seal}
\rightarrow \text{query decision}
\rightarrow \text{reveal if queried}
\rightarrow \text{select/update}
\rightarrow \text{next shot}.
\]

预测和查询决定完成前，当前炮的 `is_disrupt`、DownTime、AdvancedTime、onset 和未来窗口都不可达。未查询炮的结局在整条流结束前不可读。炮内更新只使用当前前缀及其过去；炮间更新只使用已揭示的早期炮。

### 3.3 固定合同

所有可比较的实验必须逐字共享：

- parent checkpoint 及其 SHA256；
- support-only scaler 和物理尺度；
- 固定阈值、报警状态机和 lead 定义；
- 查询 manifest、炮序、更新次数、回放容量和 optimizer steps；
- 随机种子及代码/配置/data SHA；
- 冻结源参数的位级 hash 和部署前后固定样本输出 hash。

如果任一项不同，结果只能进入另一条谱系，不能计算因果增益。

## 4. 当前模型与更新方程

### 4.1 风险路径

冻结源编码器输出隐藏状态 (h_t=f_{\theta_s}(x_{\le t}))。目标侧使用小型残差/adapter 得到

\[
z_t=z_t^{(0)}+r_{\phi}(h_t,x_t),\qquad
p_t=\sigma(z_t),
\]

其中 (	heta_s) 在 EAST 在线阶段保持冻结，(phi) 是目标侧可训练参数。部署分数来自独立 EAST risk head，不读取 J-TEXT risk head 的输出。

### 4.2 炮内 fast 更新

每炮从 persistent state 复制一个 fast state；炮结束后 fast state 丢弃。当前代码的无标签损失为

\[
L_{\mathrm{intra}}
=L_{\mathrm{next}}
 +\lambda_T L_{\mathrm{anchor}}(z^{\mathrm{fast}},z^{\mathrm{persistent}})
 +\lambda_P L_{\mathrm{pseudo}}(z,x),
\]

其中：

- (L_{\mathrm{next}})：下一状态/增量重建误差；
- (L_{\mathrm{anchor}})：风险输出与 persistent teacher 的 smooth-L1 约束；
- (L_{\mathrm{pseudo}})：只在强因果风险—物理矛盾时施加的无标签伪监督；
- 梯度只进入 fast projector、next head 和 residual head，源编码器不动。

实际执行的是 `loss.backward()` 后的梯度裁剪和 optimizer step；因此它确实是炮内反向传播。但当前实现把新观测端点的风险也直接 anchor 到 persistent teacher，并且有严格变化上限。在 Adv25 J-TEXT 审计中 8295 次炮内尝试全部回退，说明“执行了反向传播”不等于“产生了功能性在线适应”。

下一轮若改动，只允许先做一个预注册因素：teacher anchor 只保护已封存的更早前缀和低风险 guard，不约束当前新端点；其余损失、预算和报警器不变。这样才能检验当前 anchor 是否抵消了适应梯度。

### 4.3 炮间监督更新

炮后揭示标签 (y_k) 后，根据报警结果把炮分成 TP、TN、FP、FN、TARDY。当前 persistent residual 的监督目标包含：

\[
L_{\mathrm{inter}}
=L_{\mathrm{MIL}}
 +\lambda_R L_{\mathrm{replay\ distill}}
 +\lambda_O L_{\mathrm{opposite\ outcome}}
 +\lambda_P L_{\mathrm{physics\ consistency}}
 +\lambda_N L_{\mathrm{next\ state}}
 +\lambda_G L_{\mathrm{guard}}
 +\lambda_W\lVert\phi-\phi_0\rVert_2^2.
\]

更新只作用于目标残差路径及 EAST-private 分支；每次更新必须记录组别、学习率缩放、更新前后 hash、guard harm 数和是否回退。TP/TN 作为保护集，任何候选更新若使已确认正确炮翻转，必须拒绝。

这是一种局部、受保护的持续学习，不是无条件把一个错误标签写入全局 Transformer。要声称“学习了可泛化的新表示”，还必须证明同一更新改善了后续不同炮，而不是只改变被查询炮的回放损失。

## 5. 可观测物理代理：定义与限制

当前代码没有使用 (q_{95})、密度极限等独立测量的控制方程，而是从保留的 17 个通道构造因果创新。对通道 (c)，以固定滞后 (\ell=8) 计算

\[
d_{c,t}=|u_{c,t}-u_{c,t-\ell}|,
\]

对 (I_p) 还计算 (|\Delta\log|I_p||) 和按采样率换算的 (|dI_p/dt|)，并构造 (B_t/I_p)、(V_{loop}/I_p) 的滞后变化。通道按前缀分为 electrical、position、radiation、mhd 四类。

在 support cohort 上以每个创新量的 0.90 分位数 (q_c) 做 fold-local 归一化：

\[
s_{c,t}=\operatorname{clip}(d_{c,t}/q_c,0,1).
\]

对每一物理族取可用通道的平均能量 (e_{f,t})，令 (a_{f,t}) 为超过活动阈值 0.5 的通道比例，则总体支持度为

\[
S_t=\bar e_t\,\frac{\#\{f:e_{f,t}\ge0.5\}}{\#\{f\ \text{可用}\}},
\]

至少两族可用时才令状态 known，否则支持度为零并关闭物理门。该量表达的是“多族因果变化同时出现”的可观测代理，不是物理定律，也不保证变化方向对应破裂因果。

因此当前可以检验的是“物理代理是否提供增量可识别信息”，不能直接写成“网络发现了破裂机理”。要升级为物理归因，必须加入：

1. 真实物理代理保留其时间顺序；
2. 同炮时间反转；
3. 炮内时间打乱；
4. 物理族置换；
5. raw8 置零；
6. 标签错位；
7. 相同 optimizer/FLOPs/查询 manifest 的零物理基线。

真实物理臂必须在三个 seed 和未见未来流上稳定优于最强负对照。

## 6. 必须执行的归因消融

### 6.1 炮内/炮间 2×2

令 (M_{ab}) 表示炮内因子 (a\in\{0,1\}) 和炮间因子 (b\in\{0,1\})：

| 实验 | 炮内 (I) | 炮间 (R) | 解释 |
|---|---:|---:|---|
| M00 static | 0 | 0 | 不更新 |
| M10 intra-only | 1 | 0 | 只允许当前炮 fast 更新 |
| M01 inter-only | 0 | 1 | 只允许已揭示炮 persistent 更新 |
| M11 intra+inter | 1 | 1 | 联合神经路径 |

对任一指标 (m)：

\[
\Delta_I=m(M_{10})-m(M_{00}),
\quad
\Delta_R=m(M_{01})-m(M_{00}),
\]

\[
\Delta_{IR}=m(M_{11})-m(M_{10})-m(M_{01})+m(M_{00}).
\]

主报告使用 (J=TPR_{lead}-FPR)，同时列出 TPR、FPR、AUC、median lead、>3000 ms 数量和逐炮转移。不能只看联合曲线。

### 6.2 与决策层正交分解

在同一 parent、同一 query manifest 上增加：

| 因子 | 允许变化 |
|---|---|
| alarm-only | 只更新报警状态机参数，神经参数冻结 |
| weight-only | 只更新专家/组合权重，神经参数冻结 |
| neural-only | 只允许 M10/M01/M11 神经更新，固定报警器和权重 |
| full | 神经、记忆、报警层按完整协议运行 |

若 `full` 有增益而 `neural-only` 没有，结论应是决策层保护有效，而不是神经表示学习有效。若 `neural-only` 有增益但 physics-on 与负对照相同，只能说神经持续学习有效，不能说物理约束有效。

## 7. 查询器与预算归因

当前 shadow query score 的核心是

\[
Q=\max(Q_{FP},Q_{FN},Q_{uncertainty}),
\]

其中风险超过阈值的持续性和物理支持不足形成 (Q_{FP})，低风险但物理支持高形成 (Q_{FN})，距阈值越近形成不确定性项。统一价值模块另外使用重建惊讶度、teacher disagreement、physics mismatch、private innovation 和可用性/冗余度得到有界 optimizer scale。

这些量只能在当前炮预测封存前缀上计算。查询器“错误富集”不是“未来收益”：真正的查询价值定义为

\[
V(q_k)=\mathbb E\left[
J_{k+1:k+H}(\text{update after reveal})-J_{k+1:k+H}(\text{no update})
\mid\mathcal F_k\right]-\lambda\,\mathrm{cost}(q_k).
\]

开发阶段可以用已封存标签估计 (V)，但正式在线运行必须在 query decision 之后才揭示标签；查询选择不能使用本炮真实结局。必须在相同预算下比较 random、uncertainty-only、physics-only 和 expected-future-utility selector，并报告 queried error rate、后续 H 炮净收益、查询炮分布和每次更新的实际 FLOPs。

## 8. 统计与通过门槛

### 8.1 主要指标

- 炮级 TPR（分别报告 raw 与 lead-qualified）；
- FPR；
- (J=TPR_{lead}-FPR)；
- AUC；
- median lead；
- lead>3000 ms 的数量和比例；
- TP→FN、TN→FP、TARDY→TP 等配对转移；
- 更新尝试/接受/回退、参数范数与冻结 hash。

### 8.2 统计程序

每个 seed 先计算炮级配对差，再用 shot-level bootstrap 或 paired permutation 得到置信区间；不得把窗口数当作独立样本。多臂、多指标比较使用预注册主指标和 Holm 校正。单个炮在 180 炮上的最小 (J) 网格为 (1/180=0.005556)，所以一次炮的改善只能叫“观察到一个炮级变化”，不能叫稳定增益。

正式神经归因至少需要：

1. clean 2×2 在 seeds 131/133/137 完整重复；
2. 真实物理臂胜过所有强负对照；
3. 同一 parent/query/budget 下 `neural-only` 不低于 static 且 guard harm 受控；
4. 未见未来 chronological stream 一次性确认。

任何完整臂 TPR<0.60 或 FPR>0.40，立即停止并审计标签、报警索引、scaler、阈值、炮序和输出；不得用重新调 alignment 权重掩盖异常。

## 9. 当前审计结果（Adv25 seed=131）

### J-TEXT TEST220

静态、intra-only、inter-only、intra+inter 炮级结果均为：TP=108、FP=17、FN=2、TN=93，raw TPR=0.9818，TPR@10=0.9000，FPR=0.1545，AUC=0.9791，median lead=68.87 ms。炮内尝试 8295 次，全部因变化上限回退；炮间接受 5 次但未改变炮级混淆矩阵。

### EAST clean neural path / balanced360

静态为 TP=127、FP=10、FN=53、TN=170，TPR@30=0.4500，FPR=0.0556，AUC=0.9307。intra-only 混淆矩阵不变；inter-only 与 joint 只观察到一个 TARDY→timely 的炮级变化，TPR@30=0.4556，FPR=0.0556，故单 seed 观察到约 (+0.005556) 的 (J)，尚不能称为正式跨 seed 神经增益。

physics-on 与 physics-off 使用同一查询清单，joint 的 ΔJ=0，ΔAUC 约 (9.26\times10^{-5})。这轮没有物理增益证据。

### EAST 完整 Adv25 V3C

seed=131 的开发流中，static 为 TPR@30=0.7667、FPR=0.2333、J=0.5333；完整 online 为 TPR@30=0.7667、FPR=0.1944、J=0.5722，减少 7 个 FP，ΔJ=+0.0389。该运行同时启用了 fast/slow 神经更新、raw8、endpoint memory、报警参数和 episode gate，因此只能称为**完整决策系统诊断增益**。

### 旧 frozen v2.1 主线（必须单列）

`online_architecture_search/frozen_runs` 是此前真正完成三 seed 冻结复现的 EAST 主线，不能与本轮 Adv25 clean 两时间尺度结果混用。其固定配置为 20% 查询预算（72/360）、双阈值报警、64 个炮内事件、2 步炮间更新和 physics-support 0.80 保护门。三 seed 汇总为：static TPR@30=0.84074、FPR=0.16667；online TPR@30=0.84259、FPR=0.16667，平均 ΔJ=+0.00185。它证明了一个小的、受保护的系统级在线改善，尚不能证明物理表示学习。

另一个 `paper_grade_package/evidence/three_stage_results.json` 中的三 seed 数字（static TPR@30=0.84074、FPR=0.16667；online TPR@30=0.84815、FPR=0.15185；ΔJ=+0.02222）来自更早的 V3C/报警链和参数开发谱系。它可以作为历史诊断证据，但不是 frozen v2.1 的重复，也不是 Adv25 的结果。指标差异来自 parent、报警链、查询/决策层和开发合同不同；不能把两个差值相加或互相替代。

旧谱系三 seed 的系统级历史结果也只能单列为诊断，不能与本轮 Adv25 clean factorial 混合。父模型、标签/阈值和报警链不同，是指标差异的合理来源，不是可以直接平均的重复实验。

## 10. 下一轮执行顺序与回退

1. 不改变模型容量，先修正炮内 teacher anchor 的作用域，只保护已封存前缀；
2. 在同一 query manifest 上重跑 2×2，先看 inter-only 是否在三 seed 重复；
3. 加入完整物理负对照，保持相同预算；
4. 对查询器跑 equal-budget random/uncertainty/physics/future-utility；
5. 最后才把 neural-only 与 alarm-only/full 组合；
6. 所有方法锁定后使用独立未来流一次性确认。

若全部对齐和在线神经臂都不优于 target/static，应接受“当前表示缺少可识别跨装置支持”的结论，增加合法目标数据或共同物理观测，不继续堆叠损失项。任何回退都必须恢复上一个父 checkpoint、阈值、报警器和查询 manifest，不能覆盖原始证据。
