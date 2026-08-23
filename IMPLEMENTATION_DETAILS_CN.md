# J-TEXT → EAST 主线：代码级实现说明

本文解释当前冻结的 paper v3c 主线到底做了什么。内容以代码和
[`primary_chain_v3c.json`](experiments/transfer_methods_2026/paper_grade_package/configs/primary_chain_v3c.json)
为准，不把历史实验、未启用分支或仅作 shadow 记录的模块写成主结果。

如果只需要一句话概括：系统先在 J-TEXT 的 17 个跨装置可比特征上训练
Transformer，再用 EAST support110 微调高层表示并增加 EAST 私有动态残差；
进入 360 炮顺序流后，每炮内部使用一个“用完即丢”的 fast copy 做无标签适配，
炮后则只对价值函数选中并揭示 HDF5 标签的炮更新 persistent 模型。所有当前炮
预测都先封存，当前炮标签只能影响后续炮。

## 1. 先区分四个容易混淆的量

### 1.1 风险概率、风险 logit 与报警阈值

模型输出的是风险 logit \(z_t\in\mathbb{R}\)，风险概率为

$$
p_t=\sigma(z_t)=\frac{1}{1+e^{-z_t}}.
$$

EAST support110 选择概率阈值 \(\theta\)，报警状态机在 logit 空间使用

$$
m_t=z_t-\operatorname{logit}(\theta)
$$

作为“离报警边界还有多远”的 margin。\(m_t=0\) 表示刚好越过基础阈值。

### 1.2 “偏移”不是一个单独参数

代码里至少有四类偏移，汇报时应明确说的是哪一种：

1. **特征域偏移**：J-TEXT 的 `ip` 从 kA 转为 A，并重新计算
   \(B_t/I_p\)、\(\Delta I_p\) 和 \(V_{loop}/I_p\)，避免把单位差异当成装置差异。
2. **风险 logit 残差**：共享动态支路、EAST 私有支路和历史端点记忆分别给
   基础 logit 加一个有界修正。
3. **揭示标签后的目标位移**：把封存 logit 朝 HDF5 标签方向最多移动 0.5，
   用于构造稳定的监督目标，不允许一步把概率推到 0 或 1。
4. **报警边界位移**：查询到 FP、FN 或 TARDY 后，小幅更新报警状态机的四个
   连续参数，并投影回预注册范围。

### 1.3 炮内 fast 与炮间 persistent

- `persistent` 是跨炮保存的慢模型；只有查询后揭示的标签能够改变它。
- 每一炮开始时执行 `fast = deepcopy(persistent)`。
- 炮内无标签梯度只进入 fast；本炮结束后 fast 被丢弃，不直接传给下一炮。
- 下一炮重新从更新后的 persistent 复制 fast。因此“上一炮和下一炮有关”只可能
  来自被查询后的炮间更新、报警参数、查询权重或因果记忆，绝不是把上一炮的
  fast 权重直接带到下一炮。

### 1.4 静态分支与在线分支

balanced360 先由迁移后父模型完整静态预测并封存。在线实验随后重新载入同一个
迁移后父状态，再按炮号顺序从第 1 炮走到第 360 炮。静态分支不是在线分支走完后
再回测，因此二者不会共享在线更新后的权重。

## 2. 数据、标签和时间口径

### 2.1 权威标签

结局和破裂时刻只读 HDF5：

- `/meta/IsDisrupt`：是否破裂；
- `/meta/DownTime`：破裂时刻。

CSV/JSON 只负责炮表、顺序和审计，不得覆盖 HDF5 标签。在线流遵守：

```text
当前炮因果预测 → 封存风险/报警/query 决策 → 若选中才打开 HDF5 标签
→ 更新慢模型 → 下一炮
```

### 2.2 时间窗标签

J-TEXT 和 EAST 的训练风险标签都使用 HDF5 破裂时刻构造：

$$
y_t=\mathbf 1\{0\leq \mathrm{TTD}_t\leq125\ \mathrm{ms}\}.
$$

安全炮的窗口全部是负类；破裂炮中 TTD 大于 125 ms 的合法窗口是负类。训练时
先等概率选正/负类，再在该类内等概率选炮，避免长炮或窗口多的炮支配梯度。

正式评估不是普通窗口分类：

- J-TEXT：破裂炮首次报警提前量必须 `>10 ms`；
- EAST：破裂炮首次报警提前量必须 `≥30 ms`；
- 当前冻结口径是任一因果点满足报警状态机即可，不要求连续 10 点；
- 提前超过 3 s 的报警仍计入正式 TPR，但会另列审计，检查 campaign/炮长捷径。

## 3. 跨装置特征如何处理

实现入口：
[`rebuilt_harness.py`](experiments/transfer_methods_2026/rebuilt_harness.py) 的
`rebuild_features()`，以及
[`east_private23_sidecar.py`](experiments/transfer_methods_2026/east_private23_sidecar.py)。

### 3.1 共享 rebuilt17

23 个原始通道中，下列 17 个进入 J-TEXT/EAST 共享主干：

```text
Bt, ip_A, Bt_over_Ip, Ip_diff, v_loop, Z_proxy,
sxr_kurt, sxr_skew, sxr_var,
xuv_kurt, xuv_skew, xuv_var,
Mir_avg_fre, n=1 amplitude, mode_number_n, dx_a, dy_a
```

关键变换是：

$$
I_p^{A}=1000I_p^{kA}\quad(\text{仅 J-TEXT，且只变换一次}),
$$

$$
B_t/I_p=\frac{B_t}{I_p},\qquad
\Delta I_p(t)=I_p(t)-I_p(t-1),\qquad
Z_{proxy}=\frac{V_{loop}}{I_p}.
$$

无穷、NaN 和不可用通道由 availability mask 管理；`Mir_avg_fre` 和
`mode_number_n` 若整炮全零，按“通道不可用”处理，不把缺测零解释成真实物理零。
共享17维使用 **J-TEXT TRAIN1353** 拟合的逐通道 median/IQR scaler：

$$
\tilde x_c=\operatorname{clip}\left(
\frac{x_c-\operatorname{median}_{J}(x_c)}
{\max(\operatorname{IQR}_{J}(x_c),10^{-6})},-10,10\right).
$$

EAST 不重新拟合这 17 维 scaler，否则会把目标域统计泄漏进源表示合同。

### 3.2 EAST 私有 raw8

不适合强行跨装置同义对齐的 8 个通道没有被丢掉，而是进入 EAST 私有支路：

```text
CIII, MNM, Mir_VV, Mir_avg_amp,
ne_nG, qa_proxy, sxr_mean, xuv_ratio
```

每个通道只使用当前因果窗，计算 6 个摘要：末值、均值、标准差、全窗差、近期差、
可用比例。8 个通道得到 48 维摘要，再与共享基础 logit 合并。这样做的含义是：
共享主干只承载跨装置可比机制，EAST 专属表象通过小支路补充，而不是要求每个
原始通道都与 J-TEXT 一一对齐。

## 4. J-TEXT 源模型怎么训练

代码入口：
[`train_jtext_source_transformer.py`](experiments/transfer_methods_2026/train_jtext_source_transformer.py)
和 [`grid_search.py`](experiments/transfer_methods_2026/grid_search.py)。

### 4.1 网络

- 输入：128 ms × 17 通道因果窗；
- patch：宽 8、步长 8，共 16 个时间 token；
- embedding：`Conv1d(17, 64, kernel=8, stride=8)`；
- 一个可学习 CLS token，加正弦位置编码；
- 两层 Transformer encoder，4 头，`d_model=64`，FFN=256；
- LayerNorm 后取 CLS；
- 风险头：`64 → 64 → GELU → Dropout(0.1) → 1`。

短历史窗只在左侧补零，并把真实长度传给 Transformer mask；绝不读取 endpoint
之后的数据。

### 4.2 源域损失

源模型当前实际生效的目标就是二元风险交叉熵：

$$
\mathcal L_{src}
=-\frac1B\sum_{i=1}^{B}
\left[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\right].
$$

优化器为 AdamW，学习率 \(3\times10^{-4}\)，weight decay \(10^{-4}\)，
batch 128，梯度范数裁剪 1.0。冻结配置最多 30 epoch，每 epoch 128 step；至少
训练 12 epoch，VAL160 连续 7 epoch 不改进才早停。

### 4.3 epoch 与阈值怎么选

每个 epoch 只在 VAL160 上找满足 timely TPR≥0.90 的阈值，再选 FPR 最低者。
epoch 排序优先低 FPR，其次高 timely TPR，再其次高 AUC。模型和阈值锁定前
`test_shards_read=0`；TEST220 只做一次结果计算。

## 5. J-TEXT → EAST 具体怎么迁移

最终主线不是 DANN、MMD 或 CORAL。迁移由三个连续阶段组成。

### 5.1 高层小样本微调：shared17 backbone

代码：[`grid_search.py`](experiments/transfer_methods_2026/grid_search.py) 的
`train_adapt()`，由
[`shot_alarm.py`](experiments/transfer_methods_2026/shot_alarm.py) 的
`train_support_only()` 调用。

步骤如下：

1. 载入同 seed 的 J-TEXT encoder、风险头和源 scaler；
2. 冻结卷积 embedding、CLS 和第一层 Transformer；
3. 只解冻最后一层 Transformer 与最终 LayerNorm；
4. **丢弃 J-TEXT 风险头，随机初始化独立 EAST 风险头**；
5. 在 support110 的 HDF5 标签上训练 12 epoch，每 epoch 32 batch，batch 128；
6. 仍使用正/负 50:50 且炮均衡采样；AdamW 学习率 \(10^{-4}\)。

损失仍为 BCE：

$$
\mathcal L_{adapt}=\operatorname{BCEWithLogits}(z^{EAST},y^{HDF5}).
$$

这一步的设计假设是：低层局部模式比高层时间尺度更容易跨装置共享，因此保留
低层参数，只让最高时间语义层和 EAST head 适应目标装置。

support110 同时选择 EAST 阈值：先要求 support timely TPR≥0.80，再在可行阈值中
最小化 FPR。当前 HDF5 实际构成为 15 炮破裂、95 炮安全；因此这是小样本目标
拟合/校准集，不是独立验证集。

### 5.2 共享动态 hazard adapter

代码：
[`dynamic_hazard_adapter.py`](experiments/transfer_methods_2026/dynamic_hazard_adapter.py)
和
[`run_dynamic_hazard_adapter.py`](experiments/transfer_methods_2026/run_dynamic_hazard_adapter.py)。

冻结的 Transformer 为相邻因果窗提取 CLS、最后 token 和隐藏增量，adapter 再
拼接物理族向量及 availability。共享动态残差为

$$
\delta^{shared}_t=
g_t\tanh\!\left(f_{shared}(h_t,q_t,a_t,z_t^0)\right),
\qquad |\delta^{shared}_t|\le 1,
$$

其中至少两个物理族可用时 \(g_t=1\)，否则为 0。先用 4 epoch 学下一时刻物理量：

$$
\mathcal L_{next}=\operatorname{SmoothL1}(\hat q_{t+8},q_{t+8}),
$$

再用 8 epoch 联合优化：

$$
\mathcal L_{DHA}=\mathcal L_{BCE}
+0.20\mathcal L_{next}
+1.00\mathcal L_{anchor}
+0.05\,\mathbb E[(\delta^{shared})^2].
$$

\(\mathcal L_{anchor}\) 只保护原模型已经判对的 support 窗口：若 adapter 让这些
窗口离正确边界更近而不是更远，就产生 ReLU 惩罚。

### 5.3 EAST 私有 raw8 动态残差

私有支路给共享 logit 再加一个有界修正：

$$
\delta^{private}_t=0.5\tanh(f_{private}(r_t,z_t^{shared})),
\qquad |\delta^{private}_t|\le0.5.
$$

support110 上的私有训练损失为

$$
\mathcal L_{private-support}
=\mathcal L_{sealed-residual}
+0.20\mathcal L_{next-raw8}
+0.02\,\mathbb E[(\delta^{private})^2].
$$

其中不是直接把封存预测替换成 0/1 logit，而是先构造有限位移：

$$
e_i=y_i-\sigma(z_i^{seal}),\qquad
z_i^{target}=z_i^{seal}+\operatorname{clip}(e_i,-0.5,0.5),
$$

$$
\mathcal L_{sealed-residual}
=\operatorname{SmoothL1}(z_i,z_i^{target})
+0.25\operatorname{BCEWithLogits}(z_i,y_i).
$$

最终静态风险近似为

$$
z_t=z_t^0+\delta_t^{shared}+\delta_t^{private}.
$$

在线时若因果端点记忆有足够的同类历史错误，还可临时加入
\(\delta_t^{memory}\)，见第 11 节。

## 6. EAST 物理支持度怎么得到

代码：
[`online_active_audit.py`](experiments/transfer_methods_2026/online_active_audit.py)
的 `causal_innovations()`、`fit_innovation_scales()` 和
`score_physics_families()`。

系统以 8 个采样点为 lag，只计算“当前减过去”的因果创新，包括：

- electrical：\(\Delta\log |I_p|\)、\(dI_p/dt\)、\(\Delta(B_t/I_p)\)、
  \(\Delta V_{loop}\)、\(\Delta Z_{proxy}\)；
- position：\(\Delta d_x,\Delta d_y\)；
- radiation：SXR/XUV 高阶统计量变化；
- MHD：Mir 频率、n=1 幅值和模数变化。

每个创新除以 support110 上拟合的 90% 分位尺度并裁剪到 [0,1]。族内取均值，
再以“能量 × 多族并发度”组合：

$$
s_t=E_t\cdot C_t,\qquad 0\le s_t\le1.
$$

至少两个物理族可用时 `physics_known=True`；否则支持度按 0 处理并禁止把缺测
当作反证。所有量只用当前点和过去点，不对整炮做双向平滑。

## 7. 报警不是单阈值：实际状态机

代码：
[`dual_threshold_alarm.py`](experiments/transfer_methods_2026/dual_threshold_alarm.py)
和 `ProjectedAlarmController`。

令 \(m_t=z_t-\operatorname{logit}(\theta)\)，因果趋势为

$$
r_t=\max\{\operatorname{MA}_5(m)_t-
\operatorname{MA}_{100}(m)_t,0\}.
$$

初始状态机有两条报警路线：

$$
\text{high route}:\quad m_t\ge h,\qquad h=0,
$$

$$
\text{early route}:\quad
m_t+w_ps_t+w_rr_t\ge e,
$$

其中初始 \(w_p=8,w_r=12,e=7.9\)。任何路线第一次成立就产生候选报警。在线
融合还有一个独立支持约束：候选点必须同时得到“在线模型配原报警参数”或
“静态模型配在线报警参数”之一支持，因此适配分支不能凭一个孤立自由度凭空
创造更早报警；episode gate 也只能延迟或否决，不能提前或新建报警。

正式混淆矩阵按 HDF5 结局判定：

```text
安全 + 报警             → FP
安全 + 不报警           → TN
破裂 + 不报警           → FN
破裂 + 报警且 lead≥30ms → TP
破裂 + 报警但 lead<30ms → TARDY（正式口径仍是未及时命中）
```

## 8. 炮内更新：什么时候更新、用什么损失

### 8.1 调度器先决定这个窗口值不值得算梯度

严格窗口模式每 8 个因果 endpoint 才产生一个候选。每个候选增加 0.25 token，
token 上限 4；一次梯度消耗 1 token。历史不足 8 个候选时 cutoff=0.40，之后取
最近 64 个候选值的 80% 分位；两次更新至少相隔 8 个 endpoint，每炮最多 64 次。

便宜预筛值为

$$
V_{cheap}=\max\left[
e^{-|m|/1.5},
\sigma\!\left(\frac{m+8s-7.9}{1.5}\right),
s,
8|p_t-p_{t-8}|
\right],
$$

并裁剪到 [0,1]。它只负责决定是否值得计算完整自监督误差。

### 8.2 完整炮内价值函数

选中候选后，使用 endpoint \(t-8\) 的窗口预测已经观测到的 \(t\)，计算：

- \(R\)：共享物理和私有 raw8 下一状态重建误差，经 support q80 归一化；
- \(T=2\max(|p^{fast}_{shared}-p^{slow}_{shared}|,
  |p^{fast}_{private}-p^{slow}_{private}|)\)：fast/slow 分歧；
- \(P=\max[p(1-s),(1-p)s]\)：模型风险与物理支持的矛盾；
- \(I\)：EAST 私有 raw8 重建惊讶；
- \(A\)：当前通道可用比例；
- \(D\)：与最近 16 个状态的最大余弦相似度映射到 [0,1]，越重复越大。

基础价值为

$$
V_0=A(1-0.5D)(0.40R+0.20T+0.25P+0.15I).
$$

再用报警邻近和强物理机制门抑制与告警无关的更新：

$$
G_1=\sigma\!\left(\frac{m+8s-7.9}{1.5}\right),
$$

$$
G=\max\{G_1,\mathbf1[s\ge0.65\land R\ge0.75]\},
$$

$$
V=\operatorname{clip}[V_0(0.25+0.75G),0,1].
$$

学习率不是简单乘一个很小的 loss weight，而是直接缩放 optimizer 的实际步长：

$$
\alpha(V)=
\begin{cases}
0,&V\le0.03,\\
\operatorname{clip}\left(\dfrac{V-0.03}{0.15-0.03},0,1\right),&V>0.03.
\end{cases}
$$

最终炮内学习率为 \(10^{-4}\alpha(V)\)。0.03 是硬死区，0.15 已饱和。

### 8.3 共享 fast 支路的实际损失

冻结 v3c 当前生效的是：

$$
\mathcal L^{shared}_{intra}
=\mathcal L_{next-physics}
+0.25\operatorname{SmoothL1}(z^{fast},z^{persistent})
+0.10\mathcal L_{physics-pseudo}.
$$

物理伪监督只处理强矛盾：

- 基础模型已越阈值但 \(s\le0.10\)：用 `softplus(margin)` 向下推；
- 基础模型未越阈值但 \(s\ge0.65\)：用 `softplus(-margin)` 向上推；
- 其他区域不给方向性伪标签。

这不是拿模型自己的类别预测当真值反复自训练。下一状态是真实已经到达的物理
观测，persistent 只提供防漂移教师，物理项只在强冲突区提供方向。

### 8.4 EAST 私有 fast 支路的实际损失

$$
\mathcal L^{private}_{intra}
=\mathcal L_{next-raw8}
+0.50\operatorname{SmoothL1}(z^{fast},z^{persistent}).
$$

每次只做 1 个 optimizer step，梯度裁剪 0.5。共享 fast 在 legacy 范围更新
`projector.3`、`next_head` 和 `residual_head`；私有 fast 更新 `trunk`、
`next_head` 和 `residual_head`。persistent 哈希在炮内必须不变。

### 8.5 当前没有生效的炮内项

代码已经实现“用稍后观测到的风险变化构造 temporal pseudo-target”和
`observed_physics_loss`，但冻结 v3c 未在 JSON 中打开它们，默认分别为：

```text
intra_temporal_risk_weight = 0.0
intra_observed_physics_weight = 0.0
intra_parameter_scope = legacy
```

所以不能在当前主结果中声称这两个损失带来了增益。

## 9. 查询价值：哪一炮打开标签

### 9.1 当前真正生效的两项价值

一炮预测封存后，先提取：

$$
H=\sigma[\operatorname{logit}(p_{max})-
\operatorname{logit}(\theta)]
$$

表示整炮峰值越过阈值的程度；

$$
P_{run}=\min\left(\frac{\text{阈值以上最长连续点数}}{128},1\right)
$$

表示高风险是否持久；\(S_r\) 是高风险区的物理支持中位数，\(S_{max}\) 是
整炮最大已知物理支持。由此得到：

$$
C_{FP}=H\,P_{run}(1-S_r),
$$

$$
C_{FN}=(1-H)S_{max},
$$

$$
C=\max(C_{FP},C_{FN}),
$$

$$
U=\exp\left(-\left|\operatorname{logit}(p_{max})-
\operatorname{logit}(\theta)\right|\right).
$$

当前 operational query value 为

$$
Q=wU+(1-w)C,
$$

初始 \(w=0.55\)，且始终投影到 [0.55,0.75]。直观地说：

- 峰值贴近阈值，标签有助于解决边界不确定性；
- 模型高风险但物理长期不支持，像潜在 FP；
- 模型低风险但物理强异常，像潜在 FN；
- 模型很确定且物理一致，则标签价值低。

### 9.2 查询预算不是固定每 20 炮硬选 6 炮

冻结配置 `query_quota=6, query_block_size=20` 被换算成每炮增加
\(6/20=0.3\) token，token 容量 12。只有 token≥1 才能查询。历史不足 20 炮时
cutoff=0.35；之后 cutoff 是最近 160 炮 Q 的 70% 分位。若 40 炮没有查询，允许
一次探索查询。查询条件是：

$$
\text{query}_j=
\mathbf1[\text{token}_j\ge1]\cdot
\mathbf1[Q_j\ge q_{0.70}(Q_{past})\ \text{或触发探索}].
$$

未使用的预算可以滚存，所以不是每个 20 炮块恰好查询 6 炮。三个 seed 实际查询
`102/90/99` 炮。冻结主配置中 `query_cost=0.0`，即预算通过 token 限制，未额外
从 Q 中减一个价格项。

### 9.3 标签揭示后，查询权重怎么学

把 FP/FN/TARDY 记为查询真正找到错误（\(e=1\)），TP/TN 记为 \(e=0\)。
两特征线性模型的梯度为

$$
g_w=(Q-e)(U-C),
$$

$$
w\leftarrow\operatorname{clip}(w-0.05g_w,0.55,0.75).
$$

因此如果“不确定性高而物理矛盾低”的炮经常确实是错误，w 会增加；反之物理
矛盾更能命中错误时，w 会下降。

### 9.4 shadow 分支不能写成实际查询器

episode v3 还计算了 error correction、information gain、physics conflict、
gradient actionability 和 coverage exploration 五专家价值，但主配置是：

```text
episode_query_mode = shadow
episode_gradient_mode = legacy
episode_isolate_learning_feedback = true
error_memory_weight = 0.0
```

所以五专家值只进入审计账本，不决定主结果的 query；错误表型记忆虽然存储历史，
其权重为 0，也不会改变 Q。主结果只能归因于上面的 \(U+C\) 两项 operational
query value、滚动 cutoff 和 token 预算。

## 10. 查询后，真实标签如何决定更新幅度

只对已选炮打开 HDF5 标签。代码先计算封存预测与真实标签的窗口 BCE：

$$
R_{label}=\operatorname{clip}\left(
\frac{\operatorname{BCE}(p,y)}{\log 2},0,1\right).
$$

错误类别权重为

$$
O=
\begin{cases}
1,&FN,\\
0.6,&TARDY,\\
0.2,&FP,\\
0,&TP/TN,
\end{cases}
$$

提前量缺口为

$$
L=
\begin{cases}
1,&FN,\\
\operatorname{clip}((30-\mathrm{lead})/30,0,1),&TARDY,\\
0,&\text{其他}.
\end{cases}
$$

最终监督价值：

$$
V_{sup}=\operatorname{clip}(0.50R_{label}+0.30O+0.20L,0,1).
$$

它直接缩放炮间 optimizer 学习率。只有 FP、FN、TARDY 触发模型梯度；TP/TN
不改模型，而是进入 guard/protection，防止后续更新破坏已判对区域。

## 11. 炮间 persistent 模型怎么更新

### 11.1 监督窗口选择

从当前查询炮中最多选 16 个对封存报警边界最有影响的窗口。FN/TARDY 优先限制
在 30–125 ms 合法及时区间；安全误报则围绕最强错误报警证据。再加入 16 个历史
replay/support 窗口以及必要的相反结局/相似错误样本。每个分支做 2 个 optimizer
step。

### 11.2 部署一致的 MIL 报警损失

报警是“整炮任一点命中”，不能把一炮每个窗口都当成独立正类。代码对最强 4 个
证据做温度 0.25 的 smooth-top-k：

$$
\operatorname{STopK}(a)=
\tau\left[\log\sum_{i\in topK}e^{a_i/\tau}-\log K\right],
\qquad K=4,\ \tau=0.25.
$$

对破裂炮，只要求及时区间中存在强证据：

$$
\mathcal L_{op}^{D}=\operatorname{softplus}
[-\operatorname{STopK}(z-b)_{timely}],
$$

对安全炮，要求最强证据回到边界以下：

$$
\mathcal L_{op}^{N}=\operatorname{softplus}
[\operatorname{STopK}(z-b)].
$$

其中 \(b_t\) 是 **标签揭示前已经封存** 的双路线报警边界，不能根据真实结局事后
重算。

### 11.3 共享 persistent 损失

共享慢模型只允许 `residual_head` 更新，基础 Transformer 和 source head 不动：

$$
\mathcal L^{shared}_{inter}
=\mathcal L_{op}
+\mathcal L_{replay-distill}
+0.10\mathcal L_{physics}
+0.10\mathcal L_{next}
+0.25\mathcal L_{guard}
+0.02\mathcal L_{residual}.
$$

基础学习率 \(5\times10^{-4}V_{sup}\)，梯度裁剪 0.5。

### 11.4 私有 persistent 损失

EAST 私有支路允许其小网络全部更新：

$$
\mathcal L^{private}_{inter}
=\mathcal L_{supervised}
+0.25\mathcal L_{replay-anchor}
+0.25\mathcal L_{guard-anchor}
+0.10\mathcal L_{next-raw8}
+0.02\mathcal L_{residual}.
$$

\(\mathcal L_{supervised}\) 是 sealed temporal residual 与 operational boundary
loss 的组合。基础学习率 \(2\times10^{-4}V_{sup}\)，同样只做 2 step。

### 11.5 零伤害 guard 与回滚

更新前保存模型和 optimizer 状态，并找出 support guard 中当前判对的窗口。更新后
若出现任意一个原本判对的 guard 翻转，或结果出现非有限值，则：

```text
accepted = finite and guard_harms <= 0
```

否则同时恢复模型参数和 optimizer 状态。换言之，查询到了错误不代表梯度一定会
提交；它必须先通过“不破坏已知正确样本”的零伤害约束。

## 12. 报警参数、端点记忆与 FP episode gate

### 12.1 报警参数在线更新

参数投影范围为：

```text
high_threshold h ∈ [-0.10, 0.10]
physics_weight wp ∈ [6, 10]
trend_weight wr ∈ [12, 15]
early_threshold e ∈ [7.1, 8.7]
```

令 \(\eta_a=0.02V_{sup}\)：

- FP：\(e+=\eta_a,w_p-=\eta_a,w_r-=0.5\eta_a\)；若由 high route 触发，
  \(h+=0.25\eta_a\)；
- FN/TARDY：方向全部相反；
- TP/TN：不改参数，只加入 guard。

候选参数若破坏任一已有 TP/TN guard 就回滚；并强制 `e≥7.9` 时 `wp≥8`，防止
通过抬高 early threshold 同时削弱物理支持而改变原始语义。

### 12.2 类型化端点残差记忆

被查询炮的错误端点按因果表型存储。新端点只检索过去、距离≤0.35 的表型，并按
来源炮聚合；至少需要 2 个不同历史错误炮才生效：

- 过去 FN/TARDY 可加最多 +0.35 logit，但当前物理支持必须≥0.5，且不能被历史
  TN 保护压倒；
- 过去 FP 可减最多 −0.25 logit，但不能被历史 TP 保护压倒；
- TP/TN 是反例保护，不产生任意方向修正。

这是一种非参数、可审计的历史纠错，不给 backbone 反向传播。

### 12.3 episode FP gate

gate 使用 20 个因果基础特征，再拼接短/长均值差和 32 点波动，形成 60 维序列
表型。在线 logistic 只用已查询历史训练；至少 12 条历史且安全/破裂各至少 3 条
才可能启用。只有过去安全 episode 的 FP 分数与过去破裂 episode 分离，才建立
保护边界；重叠时 fail open，保留原报警。

gate 只能延迟/否决候选报警，强独立物理证据（physics≥0.85）可以逃逸。当前
`episode_isolate_learning_feedback=true`：部署结果使用 gate 后报警，但模型更新、
查询权重和报警参数仍使用 gate 前 legacy 报警，目的是隔离“决策过滤”与“学习
反馈”的因果作用。

## 13. 一炮完整执行顺序

下面的伪代码对应
[`run_unified_online_active.py`](experiments/transfer_methods_2026/run_unified_online_active.py)：

```python
for shot in shots_sorted_by_id:
    fast_shared  = deepcopy(persistent_shared)
    fast_private = deepcopy(persistent_private)

    for endpoint in causal_prefix_order:
        # 1. 先预测并封存当前 endpoint
        z = base_transformer(endpoint)
        z += fast_shared_residual(endpoint)
        z += fast_private_residual(endpoint)
        z += past_endpoint_memory(endpoint)   # 只能读过去已揭示炮
        seal_endpoint_prediction(z)
        alarm_state.observe(z, physics_support)

        # 2. 调度器只在已封存后决定是否做无标签 fast update
        if window_scheduler_selects(endpoint):
            V = causal_intra_value(past_window, observed_current_window)
            update_fast_shared(V)             # persistent 不变
            update_fast_private(V)            # persistent 不变

    sealed_alarm = seal_shot_alarm()
    Q = operational_query_value(sealed_trace, physics_trace)
    selected = rolling_token_selector(Q)
    seal_query_decision(selected)

    if not selected:
        discard_all_pending_label_slots()
        continue

    outcome = reveal_hdf5_label_for_this_shot_only()
    group = compare(sealed_alarm, outcome, minimum_lead_ms=30)
    Vsup = supervision_value(group, sealed_trace, outcome)

    update_or_rollback_persistent_shared(group, Vsup)
    update_or_rollback_persistent_private(group, Vsup)
    update_or_rollback_alarm_parameters(group, Vsup)
    update_query_weight(group)
    update_past_endpoint_memory(group)
    update_episode_gate(group)
```

最重要的防泄漏点是：`seal_endpoint_prediction()` 先于炮内梯度，
`seal_query_decision()` 先于 HDF5 reveal，而且未查询炮的标签槽被直接丢弃。

## 14. 冻结 v3c 的实际生效表

| 模块 | 代码是否存在 | v3c 是否影响主结果 | 说明 |
|---|---:|---:|---|
| shared17 Transformer + EAST 高层微调 | 是 | 是 | 迁移主干 |
| EAST shared dynamic residual | 是 | 是 | 有界 ±1 logit |
| EAST private raw8 residual | 是 | 是 | 有界 ±0.5 logit |
| 炮内 next-state + teacher + physics pseudo | 是 | 是 | 只更新每炮 fast copy |
| temporal-risk 炮内伪目标 | 是 | 否 | 权重 0 |
| observed-physics 炮内附加项 | 是 | 否 | 权重 0 |
| operational \(U+C\) 查询价值 | 是 | 是 | 当前真正的 query score |
| 五专家 episode query value | 是 | 仅 shadow | 不决定 query |
| error phenotype memory 对 query 的修正 | 是 | 否 | `error_memory_weight=0` |
| typed endpoint residual memory | 是 | 是 | 过去错误的非参数 logit 修正 |
| episode protected FP gate | 是 | 是 | 只延迟/否决，学习反馈隔离 |
| DANN/MMD/CORAL/OT/flow | 有历史代码或实验 | 否 | 不属于最终 v3c 主线 |

## 15. 代码与配置对应表

| 问题 | 代码入口 |
|---|---|
| 17+8 特征怎么构造 | `rebuilt_harness.py`, `east_private23_sidecar.py` |
| J-TEXT Transformer 与源损失 | `train_jtext_source_transformer.py`, `grid_search.py` |
| support110 高层迁移与阈值 | `shot_alarm.py`, `run_online_active_a1.py` |
| 共享动态残差 | `dynamic_hazard_adapter.py`, `run_dynamic_hazard_adapter.py` |
| 物理创新与 query 分量 | `online_active_audit.py` |
| 炮内/炮间共享更新 | `two_timescale_online_adapter.py` |
| 炮内/炮间私有更新 | `east_private_continual.py` |
| 价值函数、token、报警参数、端点记忆 | `unified_online_value.py` |
| 双路线报警 | `dual_threshold_alarm.py` |
| episode FP gate / shadow query | `online_episode_value_v3.py` |
| 完整因果顺序与账本 | `run_unified_online_active.py` |
| 冻结主线命令生成 | `paper_grade_package/scripts/reproduce_primary_chain.py` |
| 数据审计与一键回放 | `reproducibility/cli.py` |

主线代码和冻结产物的机器清单见
[`config/mainline_code_manifest.json`](config/mainline_code_manifest.json)。仓库测试会验证
清单中的每个文件存在且已被 Git 跟踪。

## 16. 结果应当怎样表述

当前三 seed 均值是：J-TEXT TEST220 `TPR=0.8515, FPR=0.2121,
AUC=0.9219`；EAST static360 `TPR@30=0.8407, FPR=0.1667`；EAST online360
`TPR@30=0.8481, FPR=0.1519`。这说明在当前开发流上，完整在线机制相对同一迁移
父模型降低了 FPR 并小幅提高 TPR。

但 balanced360 参与过架构开发和参数筛选，support110 又同时承担拟合和阈值
选择，因此这些数字属于 `DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION`。
可以汇报“代码级因果回放已打通、开发流有增益”，不能汇报成装置侧部署已通过，
也不能把 shadow/零权重分支写成已证实贡献。最终确认必须在新的、结局揭示前封存
的时间顺序 EAST 流上，冻结代码、目标验证阈值和所有对照后一次完成。
