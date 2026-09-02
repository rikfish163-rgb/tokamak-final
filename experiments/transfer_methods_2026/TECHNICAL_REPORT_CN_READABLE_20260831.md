# J-TEXT → EAST：跨装置破裂预测与因果在线主动适应

## 面向答辩的技术报告（易读版）

**版本**：2026-08-31  
**当前证据状态**：<code>DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION</code>  
**运行环境**：<code>rl_env</code>，PyTorch 2.11.0+cu128，NVIDIA RTX 5070 Laptop GPU  
**结果原则**：本文只写入已经存在于封存 JSON 回执中的数值；没有独立未来流，因此不把当前结果写成最终部署确认。

---

## 0. 先看结论：这个项目到底完成了什么

项目研究的问题可以用一句话概括：

> 先用 J-TEXT 的大量破裂数据学到一个因果时序模型，再用少量 EAST 炮完成目标装置适配，最后让模型沿着 EAST 炮号顺序运行；每一炮先预测，只有被主动策略选中的炮才揭示结局，揭示以后才允许更新后续炮使用的模型。

当前有四个必须分开的实验对象：

~~~text
J-TEXT 静态源模型
    ├── J-TEXT TEST220 静态评估
    └── J-TEXT TEST220 独立在线附加分支

J-TEXT 静态源模型 ── EAST support110 适配 ── EAST 静态迁移模型
                                           └── EAST V3C 炮内/炮间在线主动
~~~

这四者不能混成一个模型或一张表：

1. **J-TEXT 在线模型不进入 EAST 迁移**。EAST 只能从登记的 J-TEXT 静态源 checkpoint 开始。
2. **J-TEXT TEST220 和 EAST balanced360 已参与方法开发**。所以当前结果是诊断性结果，不是完全未见数据的最终确认。
3. 当前 EAST V3C 确实产生了小幅系统级在线增益，但同一 parent 的消融显示，增益主要能在“查询后的决策/记忆侧”复现，现有证据还不足以说 Transformer 主干重新学到了新的跨装置物理规律。
4. 当前 J-TEXT 的神经 episode 更新确实执行了反向传播，但炮级指标没有改变；J-TEXT canonical 在线增益来自延迟反馈的专家权重更新，属于决策层在线适应。

### 0.1 当前最重要的数值

| 阶段 | 数据与口径 | TPR | FPR | AUC | median lead | 解释 |
|---|---|---:|---:|---:|---:|---|
| J-TEXT 源模型（三 seed 均值） | TEST220，lead > 10 ms | 0.866667 | 0.130303 | 0.967906 | 67.354 ms | 静态源域结果 |
| J-TEXT 在线主动 | TEST220，20% 查询 | 0.854545 | 0.109091 | 0.975620 | 68.466 ms | 相对同一 canonical static，J 增加 0.018182 |
| EAST 静态迁移（三 seed 均值） | balanced360，lead ≥ 30 ms | 0.790741 | 0.120370 | 未在回执报告 | 139.833 ms | support110 适配后的 static |
| EAST V3C 在线主动（三 seed 均值） | balanced360，lead ≥ 30 ms | 0.792593 | 0.114815 | 未在回执报告 | 144.500 ms | 相对 static，J 增加 0.007407 |

EAST 的三颗 seed 逐炮结果如下：

| seed | static TPR/FPR | online TPR/FPR | ΔTPR | ΔFPR | 查询炮数 |
|---:|---:|---:|---:|---:|---:|
| 131 | 0.794444 / 0.100000 | 0.800000 / 0.094444 | +0.005556 | −0.005556 | 94 |
| 133 | 0.761111 / 0.116667 | 0.761111 / 0.111111 | 0 | −0.005556 | 99 |
| 137 | 0.816667 / 0.144444 | 0.816667 / 0.138889 | 0 | −0.005556 | 98 |

这里的 <code>J</code> 定义为：

\[
J = \mathrm{TPR}-\mathrm{FPR}.
\]

### 0.2 答辩时必须主动说明的模型规格问题

早期计划中曾出现过 <code>256×8</code> 的模型描述，但**当前封存并产生上述结果的源模型实际规格是 <code>128×8</code>**：窗口长度 128 点、patch size 8、stride 8、<code>d_model=64</code>、2 层 Transformer、4 个注意力头。实际定义在 [<code>grid_search.py</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/grid_search.py) 中，版本名为 <code>rebuilt17-patch-transformer-128x8-d64-l2-h4-v1</code>。

因此答辩时应说：

> 本报告的结果严格对应封存代码中的 128×8 版本；256×8 是早期设计合同，不是当前结果使用的实际配置。若要比较 256 点窗口，必须重新训练、重新选阈值并重新封存，不能把两个版本的结果混用。

---

## 1. 先把几个最容易混淆的概念讲清楚

### 1.1 “炮”“时间窗”“端点”分别是什么

- **炮（shot）**：一次完整的等离子体放电实验。模型的最终报警评价按炮统计，而不是按每一个时间点简单统计。
- **时间窗（window）**：模型在某一时刻看到的一段历史信号，例如最近 128 个采样点。它只能包含当前时刻及更早的信息。
- **端点（endpoint）**：一个时间窗的右端时刻。模型在端点做一次风险预测。
- **炮内适应（intra-shot adaptation）**：同一炮还没有结局标签时，根据当前炮不断到来的无标签信号，对一个临时 fast state 做更新。该状态在炮结束后丢弃，不直接污染下一炮。
- **炮间适应（inter-shot adaptation）**：当前炮预测已经封存，主动策略决定查询，且 HDF5 真实结局被揭示后，才对持久模型做更新；更新只影响下一炮以及更晚的炮。

### 1.2 三个指标怎么看

设破裂炮数量为 \(N_D\)，安全炮数量为 \(N_N\)。

- **TPR（真正率）**：破裂炮中及时报警的比例。

\[
\mathrm{TPR}=
\frac{\#\{\text{及时报警的破裂炮}\}}{N_D}.
\]

- **FPR（误报率）**：安全炮中产生报警的比例。

\[
\mathrm{FPR}=
\frac{\#\{\text{报警的安全炮}\}}{N_N}.
\]

- **提前量 lead**：破裂时刻减去首次报警时刻，单位为毫秒。J-TEXT 要求 <code>lead &gt; 10 ms</code>；EAST 主口径要求 <code>lead ≥ 30 ms</code>。
- **AUC**：不固定一个阈值时，对炮级风险排序能力的总体评价。
- **J**：本项目用于同时看召回和误报的简化效用指标，定义为 <code>TPR−FPR</code>。

EAST 中 lead 大于 3000 ms 的报警仍计入 TPR，但另外记录为“秒级过早报警”，防止一个极早报警被指标掩盖。当前 EAST 三 seed 的秒级过早报警数也被写进回执。

### 1.3 “有报警”不等于“及时报警”

对一炮，模型在多个端点产生风险分数。只要有一个端点越过报警规则，就取**最早的报警端点**。对破裂炮：

- 有报警且 lead 达到规定门槛：TP；
- 有报警但 lead 不达门槛：TARDY；
- 没有报警：FN。

对安全炮：

- 有报警：FP；
- 没有报警：TN。

因此，降低阈值可能提高 raw TPR，但也可能增加 FPR；一个更早的报警如果没有达到最低提前量，也不能算 operational TPR。

---

## 2. 数据、标签和访问权限

### 2.1 J-TEXT 的数据分工

| 角色 | 炮数 | 能做什么 |
|---|---:|---|
| TRAIN1353 | 1353 | 训练源模型、拟合 scaler |
| VAL160 | 160 | 选择 epoch 和风险阈值 |
| TEST220 | 220 | 只在静态 checkpoint 和阈值封存后评估；同时用于当前在线方法开发 |

J-TEXT 破裂炮的正样本窗口不是简单固定 125 ms，而是当前主合同：

\[
H_{\mathrm{JTEXT},i}
=\max\left(25\ \mathrm{ms},\;\mathrm{AdvancedTime}_i\right).
\]

如果 shot list 中没有该炮的 <code>AdvancedTime</code>，取 25 ms。它的含义是：第 \(i\) 炮从破裂前 \(H_i\) 毫秒到破裂时刻的窗口，作为源域破裂监督区间。

必须区分两种标签来源：

1. **HDF5 是硬结局的唯一权威来源**：<code>meta/IsDisrupt</code> 判断这一炮是否破裂，<code>meta/DownTime</code> 给出破裂时刻。
2. **AdvancedTime 只是破裂前监督窗口的弱标签来源**：它决定 \(H_i\)，不能覆盖 HDF5 的炮结局，也不能替换 DownTime。

代码实现见 [<code>jtext_label_contract.py</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/jtext_label_contract.py) 和 [<code>train_jtext_source_transformer.py</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/train_jtext_source_transformer.py)。

训练标签的当前规则是：

\[
y(t)=
\begin{cases}
1,&0\leq \mathrm{TTD}(t)\leq H_i,\\
0,&H_i<\mathrm{TTD}(t)\leq 60000\ \mathrm{ms}.
\end{cases}
\]

当前 Adv25 主实验没有额外把 \(H_i\) 到 125 ms 之间做成“未知类”，因为那会改变“只改标签窗口”的单因素合同；未知区属于另一个已经登记的诊断方案，不能悄悄混入当前结果。

### 2.2 EAST 的数据分工

| 角色 | 炮数 | 用途 |
|---|---:|---|
| support110 | 110 | EAST 目标适配、EAST scaler、支持集训练 |
| balanced360 | 360（180 破裂、180 安全） | 先静态测试，再按炮序在线主动；已参与方法/参数开发 |

EAST 的操作口径为 <code>lead ≥ 30 ms</code>，并保留 <code>lead &gt; 3000 ms</code> 的独立统计。

需要特别说明一个数据审计结果：历史文件名或角色表曾把 support110 写成 <code>10D+100N</code>，但当前兼容实现逐炮读取 HDF5 后得到的权威组成是 **15D+95N**。因此本报告不把它写成干净的 10D+100N 正式小样本结论；它只能作为历史主线复现的 support 集，并在材料中同时报告这一标签冲突。

### 2.3 为什么没有数据泄漏

一炮的严格时序是：

~~~text
读取当前炮信号前缀
    ↓
模型预测每个端点
    ↓
封存本炮预测和报警轨迹
    ↓
根据不含结局的特征计算查询价值
    ↓
只有被选中的炮才 reveal HDF5 结局
    ↓
将 FP/FN/TARDY 等转成更新信号
    ↓
更新持久状态，供下一炮使用
~~~

因此以下信息在当前炮预测前不可用：

- 当前炮的 <code>IsDisrupt</code>；
- 当前炮的 <code>DownTime</code>；
- 当前炮的 AdvancedTime；
- 当前炮未来时间窗；
- 未查询炮的 HDF5 结局。

当前实现还要求 shot id 严格递增；静态阶段先完成整批 360 炮的预测封存，在线阶段重新从相同迁移 parent 开始。这样可以直接比较“静态模型”和“同一模型在在线流中的变化”，而不是把静态结果和已经被更新过的模型比较。

---

## 3. 模型到底看了哪些特征

### 3.1 原始 23 维与 rebuilt17

数据原始通道是 23 维。当前跨装置共享的 <code>rebuilt17</code> 为：

~~~text
Bt, ip_A, Bt_over_Ip, Ip_diff, v_loop, Z_proxy,
sxr_kurt, sxr_skew, sxr_var,
xuv_kurt, xuv_skew, xuv_var,
Mir_avg_fre, n=1 amplitude, mode_number_n, dx_a, dy_a
~~~

构造过程包含：

- J-TEXT 的 \(I_p\) 统一换算为 A；
- \(Bt/I_p\) 作为归一化磁场特征；
- <code>Ip_diff</code> 表示电流变化；
- <code>Z_proxy=v_loop/I_p</code> 作为一个可比的派生代理；
- 每个通道同时保存 availability mask；
- 不可用的 MHD 统计通道不会被假装成真实零值。

### 3.2 为什么还有 EAST-private raw8

J-TEXT 和 EAST 的 23 个原始通道并不天然同分布。当前做法不是强行把所有通道做全局对齐，而是：

- <code>rebuilt17</code>：认为跨装置具有较高可比性的共享路径；
- <code>raw8</code>：保留在跨装置共享路径中排除的 EAST 原始通道，形成 EAST 私有 sidecar。

当前 raw8 为：

~~~text
CIII, MNM, Mir_VV, Mir_avg_amp,
ne_nG, qa_proxy, sxr_mean, xuv_ratio
~~~

这些通道并不是“没有物理意义”，而是当前证据不足以把它们当作稳定的跨装置同构特征。因此它们被放入 EAST-private 分支，由 EAST 自己学习局部统计关系，而不是直接拿来与 J-TEXT 表示逐点对齐。

### 3.3 特征重构不是抹掉信息

共享 17 维负责“可以迁移的公共状态”；私有 8 维负责“装置特有的观测和局部关系”。这比两种极端方式更合理：

- 直接把 23 维全部做全局 DANN，可能把装置差异误当成域不变信息；
- 直接丢掉 8 维，可能损失 EAST 独有的破裂先兆。

当前私有分支对每个 raw8 通道计算 6 个因果统计量：末值、均值、标准差、全窗口变化、后半段变化、有效观测比例，再拼接共享模型的 base logit。它只看当前窗口及过去，不看未来。

---

## 4. 源域 Transformer 的实际结构

### 4.1 输入到风险分数

设一个窗口为 \(X\in\mathbb{R}^{128\times17}\)。代码先将它转成通道优先形式，用一维卷积提取 patch：

\[
P=\mathrm{Conv1D}_{17\rightarrow64}(X;\;k=8,\;s=8).
\]

随后加入正弦位置编码和一个可学习的 <code>[CLS]</code> token：

\[
Z_0=[\mathrm{CLS};P+E_{pos}].
\]

再经过 2 层、4 头的 causal-compatible Transformer encoder，取 <code>[CLS]</code> 的最终表示 \(z\in\mathbb{R}^{64}\)：

\[
z=\mathrm{LayerNorm}\left(
\mathrm{Transformer}_2(
\mathrm{Transformer}_1(Z_0))_{CLS}\right).
\]

风险头为：

\[
\ell= W_2\,\mathrm{GELU}(W_1z+b_1)+b_2,
\qquad p=\sigma(\ell).
\]

其中 \(\ell\) 是 logit，\(p\) 是破裂风险概率。

### 4.2 训练损失

当前源模型的主损失是二元交叉熵：

\[
\mathcal{L}_{src}
=-y\log\sigma(\ell)-(1-y)\log(1-\sigma(\ell)).
\]

采样器不是让长安全炮完全淹没破裂炮，而是在训练 batch 中对破裂/安全炮做平衡采样，并保证同一炮不会因为窗口更多而无限放大权重。

优化器为 AdamW，训练时做梯度裁剪。每个 epoch 都在 VAL160 上重新计算候选阈值，但 TEST220 只在 checkpoint 和阈值封存后读取。

### 4.3 阈值怎么来的

阈值不在 TEST220 上挑。对 VAL160 的所有候选分数 \(\tau\)，计算：

\[
\mathrm{TPR}_{val}(\tau)=
\frac{1}{N_D}\sum_{i\in D}
\mathbf{1}\left[
\max_{t:\,lead_i(t)>10ms}p_i(t)\ge\tau
\right],
\]

\[
\mathrm{FPR}_{val}(\tau)=
\frac{1}{N_N}\sum_{i\in N}
\mathbf{1}\left[
\max_t p_i(t)\ge\tau
\right].
\]

先只保留 <code>TPR≥0.90</code> 的阈值，再选择 FPR 最低者；若并列，优先较高阈值。这就是“在满足召回要求的前提下尽量压低误报”，而不是为了让测试结果好看而事后调阈值。

---

## 5. EAST 静态迁移具体做了什么

### 5.1 迁移起点

EAST 迁移的 parent 是 J-TEXT **静态**源模型，不使用 J-TEXT 在线模型的专家权重、查询历史或在线记忆。

迁移过程可写成：

\[
\text{J-TEXT static encoder/head}
\xrightarrow[\text{support110}]{\text{target adaptation}}
\text{EAST shared17 path + EAST-private raw8 path + EAST risk head}.
\]

源 encoder 和源 risk head 在 EAST 适配中冻结。目标侧只增加小型结构：

1. shared17 的 target adapter；
2. raw8 的 EAST-private residual branch；
3. 独立 EAST risk head/残差输出。

J-TEXT risk head 的分数不能直接作为 EAST 部署分数。EAST 必须使用自己的目标风险头。

### 5.2 shared17 目标 adapter

目标 adapter 接收冻结的共享特征、物理代理和 base logit，输出一个有界残差：

\[
\ell_{E,shared}
=\ell_{base}+\delta_{shared},
\]

\[
\delta_{shared}
=r_{max}\tanh(f_{shared}(x_{17},q,\ell_{base}))\cdot g(q).
\]

这里：

- \(q\) 是可用的物理代理向量；
- \(g(q)\) 是 availability gate，物理家庭数量不够时不允许盲目增大修正；
- \(r_{max}\) 限制 logit 残差幅度；
- 零初始化使迁移初始时尽可能接近 parent，不会一开始就把源模型完全破坏。

### 5.3 EAST-private raw8 分支

私有分支先把 raw8 的每个通道变成 6 个因果统计量，形成 \(u_{raw8}\)，再计算：

\[
\ell_{E,private}
=\ell_{base}+r_{private}(u_{raw8},\ell_{base}).
\]

私有 residual 的最后一层也采用零初始化，并将残差限制在约 ±0.50 logit 以内。它的作用是补偿 EAST 自有通道的局部关系，而不是重写 J-TEXT 的底层 Transformer。

### 5.4 support110 的损失

support 适配使用 HDF5 权威结局和 EAST 的操作窗口，基本目标是让目标风险头在 support 炮上分清：

- 破裂炮的合法预警区；
- 安全炮的低风险区；
- 破裂窗口之外的高风险误报。

EAST-private 支路还加入下一状态重建损失和小的残差正则：

\[
\mathcal{L}_{support}
=\mathcal{L}_{hazard}
+0.20\mathcal{L}_{next}
+0.02\mathcal{L}_{residual}.
\]

这一步是静态迁移，不是在线学习；它在 balanced360 之前完成，balanced360 不参与 support 训练。

---

## 6. EAST 在线主动学习：一炮如何走完

当前 EAST V3C 是“双时间尺度”结构：

~~~text
源模型冻结
    ↓
当前炮建立 fast state（从持久模型复制）
    ↓
炮内按时间窗预测
    ↓
后续无标签窗口到来后，fast state 做无监督更新
    ↓
当前炮预测轨迹封存
    ↓
查询价值函数决定是否花费一次标签查询
    ↓
被选炮 reveal HDF5 结局
    ↓
根据 FP/FN/TARDY/TP/TN 更新持久模型、记忆和报警状态机
    ↓
fast state 丢弃；下一炮从更新后的持久状态开始
~~~

### 6.1 炮内 fast state 为什么不会直接造成跨炮泄漏

进入一炮时：

\[
\theta^{fast}_k\leftarrow\theta^{persistent}_k.
\]

随后炮内可以更新 \(\theta^{fast}_k\)，但不会写回 \(\theta^{persistent}_k\)。炮结束时 fast state 被释放。只有炮间查询成功、真实标签揭示且更新通过 guard，才会改变持久参数。

这回答了老师常问的：“上一炮和下一炮没有物理连续性，为什么可以更新？”

答案是：不是假设两炮物理状态连续，而是把上一炮揭示出的**可重复错误模式**作为经验，在下一炮作为候选修正；同时用 replay 和 guard 防止把偶然错误写成全局规律。

### 6.2 炮内无监督损失

炮内不能使用 <code>IsDisrupt</code>、<code>DownTime</code>、AdvancedTime 或未来窗口。当前主要使用两类信号。

#### （1）下一状态重建

若当前窗口编码为 \(h_t\)，私有/目标分支预测下一状态：

\[
\hat q_{t+1}=f_{next}(h_t).
\]

只在之后真实观测到 \(q_{t+1}\) 时计算 Smooth-L1：

\[
\mathcal{L}_{next}
=\mathrm{SmoothL1}(\hat q_{t+1},q_{t+1}).
\]

“之后真实观测到”是严格的时间条件：当前端点 \(t\) 先封存，只有 \(t'>t\) 的观测到来后，才允许把 \(q_{t'}\) 当作无标签自监督目标。

#### （2）物理伪监督

从当前可观测物理代理计算一个有界的活动支持度：

\[
s_t=\mathrm{energy}_t\times\mathrm{concurrence}_t\in[0,1].
\]

代码中把“高风险但物理支持很低”的窗口看作 FP-like，把“低风险但物理支持很高”的窗口看作 FN-like：

\[
\mathcal{L}_{pseudo}(t)=
\begin{cases}
\mathrm{softplus}(m_t), & \ell_t\ge\tau,\ s_t\le0.10,\\
\mathrm{softplus}(-m_t), & \ell_t<\tau,\ s_t\ge0.65,\\
0, & \text{其他情况},
\end{cases}
\]

其中 \(m_t=\ell_t-\tau\)。这不是 HDF5 标签，也不是物理定律本身，而是将可观测的活动一致性作为弱约束。

#### （3）为什么还要 teacher anchor

如果只用伪监督，炮内更新可能把一次异常噪声当成大幅风险变化。因此当前 fast state 还保留一个 parent anchor：

\[
\mathcal{L}_{anchor}
=\mathrm{SmoothL1}(\ell^{fast}_t,\ell^{persistent}_t).
\]

综合形式为：

\[
\mathcal{L}_{intra}
=\lambda_{next}\mathcal{L}_{next}
+\lambda_{anchor}\mathcal{L}_{anchor}
+\lambda_{phys}\mathcal{L}_{pseudo}
+\lambda_{temp}\mathcal{L}_{temporal}.
\]

当前典型权重为 \(\lambda_{next}=1\)、teacher 权重约 0.50、物理权重 0.10，temporal risk 权重在 canonical V3C 中为 0。它的设计目的不是让当前风险永远不变，而是只允许无标签适应在有可解释的下一状态和物理支持时发生有限变化。

### 6.3 炮间揭示后怎么把错误变成更新

当前炮被查询后，HDF5 结局揭示，按已封存的报警和 DownTime 把它分组：

| 组别 | 含义 | 训练方向 |
|---|---|---|
| TP | 破裂且 lead 达到 30 ms | 保护已有正确边界 |
| TN | 安全且无报警 | 保护低风险区域 |
| FP | 安全但报警 | 压低相似高风险窗口 |
| FN | 破裂且无报警 | 提高相似窗口风险 |
| TARDY | 报警了但 lead < 30 ms | 把报警边界向更早的合法区调整 |

炮间神经更新不是把整套 Transformer 重新训练，而是优先更新目标侧 residual/private 分支。核心监督项与“封存前的预测”绑定：

\[
e_i=y_i-\sigma(\ell^{sealed}_i),
\]

\[
\Delta_i=\mathrm{clip}(\gamma e_i,-\Delta_{max},\Delta_{max}),
\]

\[
\ell^{target}_i=\ell^{sealed}_i+\Delta_i.
\]

这样做有两个好处：

1. 更新方向来自“当时模型究竟错在哪里”，不是事后把所有窗口都改成正类或负类；
2. 修正量有界，不允许单个查询炮把模型 logit 推到极端。

炮间损失还包括：

\[
\mathcal{L}_{inter}
=\mathcal{L}_{sealed\ residual}
+0.25\mathcal{L}_{replay}
+\lambda_{guard}\mathcal{L}_{guard}
+0.10\mathcal{L}_{next}
+0.02\mathcal{L}_{residual}.
\]

其中：

- \(\mathcal{L}_{replay}\)：在早先已揭示的同类错误炮上回放，防止只适应最后一炮；
- \(\mathcal{L}_{guard}\)：在已知正确的保护集上检查，不能把 TP/TN 变成错误；
- \(\mathcal{L}_{residual}\)：限制目标残差过大。

当前固定计算量对照使用每次查询 16 个当前窗口、16 个历史 replay 窗口、2 个 optimizer step；更新后如果 guard 出现非允许伤害，恢复更新前的参数和优化器状态。

### 6.4 记忆为什么是“局部可回退”的

查询后的错误不会立刻成为全局规则，而是先存进 typed endpoint memory。记忆键包含：

- 归一化 latent endpoint；
- 当前 source/target logit 的有界信息；
- risk trend；
- elapsed time；
- 物理支持度和可用性。

只有满足相似度、最少支持炮数、物理 gate 和保护比例条件时，记忆才对后续端点产生有限 logit shift。正向修正和负向修正分开存储，FP 的“压低风险”不能直接覆盖 FN 的“抬高风险”。

这就是“以前遇到过一个错，之后同类轨迹不再重复犯错”的工程实现，但它不是无条件记忆，而是有：

- 相似度门槛；
- 至少多个历史支持炮；
- 正负错误类型分离；
- TP/TN 保护集；
- 残差上限；
- 可回滚状态。

---

## 7. 查询价值函数：哪一炮值得花标签预算

查询不是随机取炮，也不是看结局后倒推。查询时只能使用当前炮的 outcome-blind trace，包括：

- 风险峰值、均值、尾部均值、最后风险；
- 风险变化速度和突变程度；
- 风险轨迹恢复程度；
- shared/private 分支的 disagreement；
- 物理支持度和可用性；
- 与已有错误记忆的距离；
- 预计梯度可行动性；
- 预计改变报警决策的可能性；
- 探索价值和历史覆盖度。

### 7.1 基础查询价值

简单的 endpoint 价值可写为：

\[
V_{update}
=a\,V_{recon}
+b\,V_{teacher}
+c\,V_{physics}
+d\,V_{private}
\times(1-0.5\,redundancy).
\]

再乘以当前有效观测比例。极低价值的普通稳定窗口进入 dead zone，极端冲突窗口的更新比例饱和，防止每一个小噪声都触发更新。

### 7.2 炮级查询价值

当前 V3C 开发路径使用的炮级价值模型包含五类专家项：

\[
V_{query}=\mathrm{clip}\left(
0.05+\sum_j w_j E_j-P_{protect},0,1\right).
\]

五类 \(E_j\) 为：

1. **error correction**：预计是错误 × 能否改变决策 × 预计梯度有效性；
2. **information gain**：不确定性 × 新颖性 × 覆盖价值；
3. **physics conflict**：物理冲突 × 物理信号可靠性；
4. **gradient actionability**：预计梯度 × 私有分支创新性 × 稳定性保护；
5. **coverage exploration**：覆盖新轨迹的价值。

保护项会惩罚“模型很稳定、很可能正确、且更新也不容易改变正确决策”的炮。

查询分数可以用下面的直观解释：

> 一炮越接近决策边界、物理信号越矛盾、不同分支越不一致、且历史上没有重复覆盖，它越值得查询；一炮如果已经稳定且与已知模式高度重复，就不应消耗标签预算。

### 7.3 查询预算和实际查询数

当前 EAST 锁定主线登记了 block size 20、block quota 4 的配置，但各封存回执的实际选中炮数是 94、99、98。报告必须使用回执中的实际数值，不能把它们硬写成某个整齐的百分比。

J-TEXT canonical 20% 预算实际查询 44/220 炮；当前 10% 和 30% 预算诊断分别查询 22 和 66 炮。预算改变时，必须保持相同的炮序、阈值来源和评估方法，才能比较预算曲线。

---

## 8. 报警状态机：为什么不只是一个固定阈值

EAST V3C 不是单纯比较 \(p_t\ge0.5\)，而是把风险 margin、物理支持和风险趋势合成两条报警路线。

定义：

\[
m_t=\ell_t-\tau_h,
\]

\[
trend_t=\max\left(\mathrm{MA}_5(m_t)-\mathrm{MA}_{100}(m_t),0\right),
\]

\[
early_t=m_t+w_p s_t+w_t trend_t.
\]

报警条件为：

\[
alarm_t=
\mathbf{1}[m_t\ge h_t]
\lor
\mathbf{1}[early_t\ge e_t\ \land\ s_t\ \text{满足早期路线条件}].
\]

其中：

- <code>high_threshold</code> 初始为 0.0；
- <code>physics_weight</code> 初始为 8.0；
- <code>trend_weight</code> 初始为 12.0；
- <code>early_threshold</code> 初始为 7.9；
- 快慢移动平均窗口固定为 5 和 100 个端点。

当前允许炮间调整的连续参数及范围是：

| 参数 | 初始值 | 允许范围 |
|---|---:|---:|
| high_threshold | 0.0 | [−0.10, 0.10] |
| physics_weight | 8.0 | [6.0, 10.0] |
| trend_weight | 12.0 | [12.0, 15.0] |
| early_threshold | 7.9 | [7.1, 8.7] |

快慢窗口宽度是离散历史算子，当前不在线改变；如果实时改变 5/100，会导致之前报警含义不稳定，也无法用一次局部梯度解释。

### 8.1 查询后如何改报警参数

- 对 FP：提高 early/high 阈值，降低物理和趋势对早期报警的推动；
- 对 FN/TARDY：降低早期阈值，提高物理/趋势权重，让相似轨迹更早进入报警区；
- 每次步长为 <code>alarm_lr × update_value</code>，其中 <code>alarm_lr=0.02</code>；
- 所有参数投影回合法区间；
- 如果已保护的 TP/TN 被破坏，整次更新拒绝。

这解释了为什么系统能减少误报，又不简单地“把所有阈值抬高”：阈值调整是有方向的，而且必须经过保护集 guard。

---

## 8.2 代码里每个关键动作对应什么

为了让复核者可以从公式回到实现，当前主线的关键对应关系如下：

| 研究动作 | 代码对象 | 实际职责 |
|---|---|---|
| 读取 J-TEXT 炮和硬结局 | <code>load_jtext_shots</code>、<code>jtext_label_contract.py</code> | 读取信号；HDF5 提供 IsDisrupt/DownTime；AdvancedTime 只生成 \(H_i\) |
| 因果窗口 | <code>causal_window</code>、<code>jtext_online_io.causal_prefix_window</code> | 截取当前及更早历史，左侧补零并记录 valid length |
| 源模型前向 | <code>grid_search.Enc.forward</code>、<code>grid_search.Head.forward</code> | patch embedding、位置编码、Transformer、风险 logit |
| 阈值选择 | <code>select_validation_threshold</code> | 只在 VAL160 满足 TPR 门槛后压低 FPR |
| EAST 目标适配 | <code>DynamicHazardAdapter</code>、<code>EastPrivateDynamicsResidual</code> | shared17 有界 residual，raw8 EAST-private residual |
| 炮内状态 | <code>begin_shot</code>、<code>intra_update</code>、<code>intra_physics_update</code> | 从持久状态复制 fast state；只用无标签重建和物理伪监督 |
| 炮间监督 | <code>reveal_and_inter_update</code> | reveal 后用封存预测、真实标签、replay 和 guard 更新 |
| 错误记忆 | <code>TypedEndpointResidualMemory</code>、<code>SameErrorReplayBuffer</code> | 分离 FP/FN，按相似度和保护条件做局部修正 |
| 查询价值 | <code>CausalDecisionQueryValueModel</code>、<code>CausalUpdateValue</code> | 在 reveal 前计算价值，在更新时缩放学习率 |
| 报警状态机 | <code>IncrementalDualThresholdAlarm</code>、<code>ProjectedAlarmController</code> | 因果 MA5/MA100、high/early 双路线和有界投影 |
| 结果封存 | <code>build_mainline_receipt.py</code> | 汇总封存 JSON，拒绝缺字段、非有限指标和错误状态 |

因此，老师如果问“公式在代码哪里”，可以先说类/函数名称，再打开相应文件；不要只展示最终指标。

---

## 9. 消融实验：增益到底来自哪里

### 9.1 正确的因素分解

定义四个因素：

\[
I=\text{炮内神经适应},\quad
R=\text{炮间持久神经适应},
\]

\[
A=\text{报警/记忆侧更新},\quad
W=\text{专家权重更新}.
\]

至少要区分：

- static：\(I=R=A=W=0\)；
- intra-only：只有炮内 fast 更新；
- inter-only：只有炮间神经和/或揭示后更新；
- intra+inter：两种神经更新都打开；
- side-only：关闭持久神经更新，只保留报警、episode、endpoint memory 等侧路径；
- neural-only：保持报警器和记忆侧不动，只允许神经更新。

所有消融必须使用同一 parent、同一查询 manifest、同一阈值、同一报警器和同一炮序。否则差异可能来自 parent 或 query 变化，不来自模块。

### 9.2 EAST seed131 同 parent 配对结果

这是当前最能回答“神经更新是否真的改变炮级预测”的回执：

| 实验臂 | 炮内神经 | 炮间神经 | 决策/记忆侧 | online TPR/FPR | ΔJ | 结论 |
|---|---:|---:|---:|---:|---:|---|
| static reference | 否 | 否 | 否 | 0.794444 / 0.100000 | 0 | 共同起点 |
| intra-only | 是 | 否 | 否 | 0.794444 / 0.100000 | 0 | 反向传播发生，但没有改变炮级边界 |
| inter-only | 否 | 是 | 是 | 0.805556 / 0.100000 | +0.011111 | 恢复一个 timely lead；神经和侧路径同时开 |
| intra+inter | 是 | 是 | 是 | 0.800000 / 0.094444 | +0.011111 | 少一个 FP，但不等于神经独立贡献 |
| neural-only fixed compute | 否 | 是 | 否 | 0.794444 / 0.100000 | 0 | 纯神经固定预算未改变边界 |
| side-only | 否 | 否 | 是 | 0.794444 / 0.094444 | +0.005556 | 仅侧路径仍减少一个 FP |

这个表不能被包装成“神经模块没用”。准确说法是：

> 代码中确实有炮内和炮间反向传播；但是在当前 seed131、固定 parent、固定报警器和固定查询清单下，纯神经路径没有带来可见的炮级边界变化，而侧路径可以单独复现一个 FP 减少。因此当前系统级 EAST 增益不能独立归因于 Transformer 表示学习。

### 9.3 J-TEXT 的神经在线诊断

J-TEXT 另有一个真正更新局部 representation path 的 guarded episode 实验：

- query 44 炮；
- source parameters 保持冻结；
- 局部 representation path 允许训练；
- static 与 online 的 TPR、FPR、AUC、median lead 完全相同。

它证明“执行反向传播”本身不等于“炮级指标一定改变”。当前 J-TEXT canonical 的可见增益来自专家权重/决策层延迟反馈，不应写成 Transformer 底层表示已经学会了新规律。

### 9.4 为什么当前报告不宣称“学到了全部物理机理”

模型使用了物理代理、活动支持、下一状态重建和 guard，但这些是对可观测量的一致性约束，不是对磁流体方程完整求解。因此论文或答辩中应说：

> 方法在工程上利用了可观测物理代理来约束在线适应方向，并通过时间一致性、跨分支一致性和错误记忆提高可解释性；现有消融不足以证明模型识别了完整的破裂因果机制。

不要说：

> 模型已经发现了全部破裂物理规律。

---

## 10. 为什么 J-TEXT 和 EAST 的数值不能直接横向拼接

### 10.1 两个不同的 operational lead

J-TEXT 统计 <code>lead &gt; 10 ms</code>，EAST 统计 <code>lead ≥ 30 ms</code>。同一个报警在 J-TEXT 可能算及时，在 EAST 可能算 TARDY。因此不能把两个 TPR 当成同一个任务的准确率。

### 10.2 J-TEXT 源模型三 seed 均值与 canonical online 不是同一个统计对象

J-TEXT 源模型三 seed 结果是三个独立静态 checkpoint 的算术平均：

~~~text
TPR 0.866667, FPR 0.130303, AUC 0.967906
~~~

canonical J-TEXT 在线表使用的是三个静态 expert 的固定组合，因此它的 static 是：

~~~text
TPR 0.836364, FPR 0.109091, AUC 0.974215
~~~

它们不是同一个模型对象，不能拿来算一条“在线前后”的伪差值。正确的在线差值必须使用 canonical 表中的 static 和 online：

~~~text
static  TPR 0.836364 / FPR 0.109091
online  TPR 0.854545 / FPR 0.109091
ΔJ = +0.018182
~~~

### 10.3 EAST 的“迁移成功”与“神经表示增益”不是同一个命题

EAST 静态迁移后的系统可以达到约 0.79 的 TPR@30；V3C 在线后约 0.793，且 FPR 从约 0.120 降到约 0.115。这个结果说明：

- 源特征和 EAST 目标适配并非完全失效；
- 在线系统有小幅系统级收益；
- 但收益的独立神经归因还没有被同 parent 多 seed 消融证明。

这三个结论必须分开写。

---

## 11. 老师可能追问的 18 个问题与建议回答

### Q1：J-TEXT 的破裂/非破裂标签到底从哪里来？

**答**：硬结局只来自 HDF5 的 <code>meta/IsDisrupt</code> 和 <code>meta/DownTime</code>。<code>AdvancedTime</code> 只决定破裂前监督窗口。J-TEXT 当前窗口是 \(H_i=\max(25\text{ ms},AdvancedTime_i)\)，缺失时取 25 ms。JSON 炮表不是硬结局来源。

### Q2：为什么不能直接用固定 125 ms？

**答**：固定 125 ms 会把不同炮的有效先兆时长强行设成同一长度。当前 Adv25 合同允许每个破裂炮使用其 AdvancedTime，但保留 25 ms 下限。它更接近炮级标注信息；不过它仍是弱标签，不能说是真实因果起点。

### Q3：EAST 的标签窗口和 J-TEXT 一样吗？

**答**：不一样。当前 EAST operational 评价要求 lead≥30 ms，迁移训练使用 EAST 自己的目标适配窗口；J-TEXT 源模型使用 Adv25。两装置的报警口径不能混用。

### Q4：为什么 HDF5 和 shot list 的破裂数量有冲突？

**答**：shot list 是角色/顺序登记，HDF5 <code>IsDisrupt</code> 才是硬标签。support110 的历史角色表写过 10D+100N，但逐炮 HDF5 审计得到 15D+95N，所以报告保留冲突并把这组结果标为历史复现诊断，不能伪装成严格 10D+100N。

### Q5：模型为什么使用 rebuilt17，不直接使用 23 维？

**答**：23 维中有一部分通道在两台装置上的定义、可用性或统计意义不完全一致。当前用 rebuilt17 做共享跨装置路径，另外 8 个通道保留给 EAST-private raw8 分支。这样不是把 8 维说成废特征，而是避免把装置特异差异强行当作可迁移表示。

### Q6：当前源模型是 256 点还是 128 点？

**答**：当前封存结果严格是 128 点窗口、patch 8、stride 8。256 点只是早期计划规格，没有用于这份结果。若后续做 256 点，必须视为新实验。

### Q7：迁移时到底冻结了什么？

**答**：J-TEXT encoder、source risk head、onset/horizon 等源侧参数冻结；EAST 只训练 target adapter、EAST-private raw8 branch 和独立 EAST risk head。部署分数来自 EAST head，不直接使用 J-TEXT risk score。

### Q8：EAST 迁移用了 J-TEXT 在线模型吗？

**答**：没有。EAST 迁移 parent 是 J-TEXT 静态 checkpoint。J-TEXT 在线主动是独立附加分支，查询记录和专家权重不会流入 EAST。

### Q9：炮内更新为什么不泄漏标签？

**答**：炮内只使用当前及更早信号；下一状态重建只有在后续观测真正到来后才计算。代码明确拒绝 label、DownTime、IsDisrupt、sealed outcome 等字段。fast state 每炮重置，不把当前炮的无标签状态直接写回跨炮持久模型。

### Q10：炮内更新和重新训练有什么区别？

**答**：重新训练通常可重复使用一个训练集的标签和任意历史样本；炮内更新只能使用当前炮不断到来的无标签前缀，更新的是临时 fast adapter，时间结束后丢弃。它的目标是补偿当前炮的观测/时间尺度偏移，不是重新学习完整数据集。

### Q11：炮间更新是否假设相邻两炮物理状态连续？

**答**：不假设。炮间更新只把已揭示的错误轨迹作为一个候选可重复模式，并通过 endpoint memory 的相似度、支持炮数、正负类型和 guard 判断是否泛化。相邻炮没有关系时，更新会被 replay/guard 限制，或保持不变。

### Q12：查询为什么不是随机选？

**答**：因为标签查询是昂贵操作。价值函数优先选择可能错误、物理冲突明显、不同分支不一致、且更新后有机会改变决策的炮；稳定正确且与已有样本重复的炮优先级低。随机策略是对照，不是主方法。

### Q13：查询分数用了真实标签吗？

**答**：没有。查询分数只使用当前炮 outcome-blind 的风险轨迹、物理代理、可用性、分支不一致和历史记忆距离。只有 query 决定后才 reveal HDF5 标签。

### Q14：FP 和 FN 的更新方向是否一样？

**答**：不一样。FP 要降低相似轨迹风险或提高相应报警门槛；FN/TARDY 要提高相似轨迹风险或让早期路线更敏感。记忆按错误类型分开，不能用一个正负号混合所有错误。

### Q15：为什么查询正确的 TP/TN 也要记录？

**答**：它们不是为了强行训练，而是用作 guard/protection。系统需要知道哪些相似轨迹原来是正确的，防止一次 FP 修正把原来正确的 TN 变成 FP，或一次 FN 修正制造新的过早报警。

### Q16：当前在线增益是不是神经网络学到了新的物理规律？

**答**：不能这样下结论。EAST 系统级 V3C 结果有小幅增益，但 seed131 的 side-only 仍能减少一个 FP，neural-only fixed-compute 没有改变炮级边界；所以当前证据更支持“决策/记忆侧贡献明显”，神经表示的独立贡献仍需多 seed 正交消融。

### Q17：这是不是强化学习？

**答**：严格说不是完整强化学习。它有“查询价值”和“未来收益”的思想，但当前更新使用的是 reveal 后的监督误差、重建损失和 guard，属于延迟反馈的在线监督/持续学习。若称为强化学习，应进一步定义环境状态、动作、奖励、策略更新和离线/在线一致性。

### Q18：当前结果可以直接写成 Nature 或 ICLR 的最终结果吗？

**答**：不能直接这样写。当前 balanced360 和 TEST220 已用于方法开发，且 EAST support110 存在角色表与 HDF5 标签组成差异；因此当前结果是 <code>DIAGNOSTIC_ONLY</code>。正式投稿还需要锁定所有方法和超参数后，在完全未参与开发的未来 EAST 顺序流上一次性确认，并完成多 seed 的同 parent 神经/决策归因。

---

## 12. 当前结果的科研表述边界

### 可以说

- J-TEXT Adv25 源模型在 TEST220 上具有较高的炮级区分能力；
- 少量 EAST support 炮可以通过共享 rebuilt17 + EAST-private raw8 结构得到可运行的静态迁移模型；
- EAST V3C 按 <code>predict → seal → query → reveal → update</code> 执行，未查询结局不会提前进入预测阶段；
- EAST 三 seed 开发回放中，在线系统平均 TPR@30 略有提升、FPR 略有下降；
- J-TEXT canonical 20% 查询预算下，决策级在线分支的 J 有可见提升；
- 消融证明：炮内和炮间更新在代码层面确实执行了反向传播，但当前炮级增益不能单独归因于 Transformer 表示学习；
- 物理代理和状态机为在线更新提供了可解释的限制条件。

### 不能说

- 不能说 balanced360 是完全未见测试集；
- 不能说模型已经证明学到了完整破裂因果机理；
- 不能把 J-TEXT 在线模型作为 EAST 迁移 parent；
- 不能把 J-TEXT 的 10 ms 指标和 EAST 的 30 ms 指标直接比较；
- 不能把 side-only 的增益写成神经网络底层特征学习增益；
- 不能把 <code>10D+100N</code> 的历史文字标签当作 support110 的 HDF5 权威组成；
- 不能把当前 128 点模型写成 256 点模型；
- 不能把三颗 seed 的算术均值写成统计显著性检验。

---

## 13. 复现实验的文件和命令

### 13.1 统一环境

~~~bash
source /home/hetaisheng/miniconda3/bin/activate rl_env
export FUSION_PROJECT_ROOT=/media/hetaisheng/044A81D94A81C83E/Fuison
export FUSION_DATA_ROOT=/media/hetaisheng/044A81D94A81C83E/Fuison/data
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
~~~

### 13.2 生成当前主线机器回执

~~~bash
cd /media/hetaisheng/044A81D94A81C83E/Fuison
python experiments/transfer_methods_2026/build_mainline_receipt.py
~~~

主要结果文件：

- [<code>mainline_results.json</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/mainline_receipt_20260831/mainline_results.json)
- [<code>MAINLINE_RESULTS_CN.md</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/mainline_receipt_20260831/MAINLINE_RESULTS_CN.md)
- [<code>mainline_audit_20260831_v2.json</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/mainline_audit_20260831_v2.json)

### 13.3 代码入口

- J-TEXT 源模型：[<code>train_jtext_source_transformer.py</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/train_jtext_source_transformer.py)
- J-TEXT 标签合同：[<code>jtext_label_contract.py</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/jtext_label_contract.py)
- J-TEXT canonical 在线：[<code>run_jtext_adv25_active_mainline.py</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/run_jtext_adv25_active_mainline.py)
- EAST V3C 兼容入口：[<code>run_unified_online_active.py</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/v3c_success_compat/run_unified_online_active.py)
- EAST 炮内/炮间控制器：[<code>two_timescale_online_adapter.py</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/v3c_success_compat/two_timescale_online_adapter.py)
- EAST 私有 raw8 分支：[<code>east_private_continual.py</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/v3c_success_compat/east_private_continual.py)
- 价值函数与报警状态机：[<code>unified_online_value.py</code>](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/v3c_success_compat/unified_online_value.py)

### 13.4 回归测试

当前已执行：

~~~bash
cd /media/hetaisheng/044A81D94A81C83E/Fuison
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m pytest -q tests/experiments
~~~

结果：<code>297 passed, 6 warnings</code>。

这些是工程回归测试，不等于统计显著性证明，也不等于独立未来流验收。

---

## 14. 下一步怎样才算真正严谨

如果目标是论文或正式部署，下一轮应只做以下几件事，不要再把不同历史分支拼接：

1. 固定一个明确的源 checkpoint profile，并重新确认 J-TEXT 与 EAST 是否必须使用同一 SHA；如果必须同一 SHA，就用同一 SHA 完整重跑两条下游分支。
2. 固定 support110 的 HDF5 标签名单，解决 <code>10D+100N</code> 文字表与 <code>15D+95N</code> 权威标签的冲突。
3. 固定报警参数、查询 manifest、更新预算和代码 SHA。
4. 在同一 parent、同一查询清单下，用三个 seed 完成 <code>static / intra-only / inter-only / intra+inter / side-only / neural-only</code> 正交消融。
5. 对真正的神经表示变化报告参数哈希、logit 变化、latent 距离变化和炮级边界变化，不能只报告最终 TPR/FPR。
6. 设计完全未参与方法开发的 EAST future stream；在 seal 之前锁定阈值和所有候选方法，只运行一次。
7. 对秒级过早报警、炮长相关性、炮号相关性和异常状态机更新单独报告。

在这些条件完成以前，最准确的项目结论是：

> J-TEXT → EAST 的静态迁移链路和 EAST V3C 在线主动链路已经工程打通，并在开发流上观察到小幅系统级增益；J-TEXT 也有独立的决策级在线增益。但是，当前证据还不能把 EAST 的增益全部归因于神经表示学习，也不能把开发流结果写成独立未来部署确认。

这不是对项目的否定，而是把“已经完成的工程事实”“已有的实验支持”和“还没有证明的科学命题”分开，避免答辩时被一个标签、一个 checkpoint 或一个消融问题击穿。

---

## 附录 A：一页答辩速记

~~~text
标签：HDF5 IsDisrupt/DownTime 是硬结局；J-TEXT 窗口 H=max(25ms, AdvancedTime)
源模型：J-TEXT TRAIN1353 训练，VAL160 选 epoch/阈值，TEST220 后读
实际结构：rebuilt17，128 点窗口，patch8，stride8，d64，2层4头
迁移：只用 J-TEXT 静态 checkpoint；EAST support110 训练目标 adapter/private raw8/head
静态 EAST：先在 balanced360 完整预测并封存
在线：预测 → seal → query → reveal → update；未查询炮不读结局
炮内：fast state、无标签下一状态重建/物理伪监督、每炮重置
炮间：只在查询后使用真实标签，FP/FN/TARDY 分型，residual+replay+guard
查询：不确定性、物理冲突、分支不一致、梯度可行动性、轨迹新颖性
报警：high margin 路线 + physics/trend early 路线；参数有界投影
结果边界：EAST 在线平均 J +0.007407，但当前主要是系统级开发诊断
最诚实结论：工程链路通了，神经表示独立增益尚未被充分证明，未来流确认尚未完成
~~~
