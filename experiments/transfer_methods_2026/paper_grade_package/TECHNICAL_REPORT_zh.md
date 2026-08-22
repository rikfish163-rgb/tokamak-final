# J-TEXT → EAST 小样本迁移 → EAST 在线主动学习：Operational-v2 论文级技术报告

> 版本状态：`DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION`  
> 主结果状态：`ONLINE_DIAGNOSTIC / TEST_DRIVEN_EXPLORATION`  
> 标签权威：逐炮 HDF5 `/meta/IsDisrupt` 与 `/meta/DownTime`  
> 执行环境：`rl_env`、RTX 5070、单 GPU  
> 主种子：131 / 133 / 137
> 正式指标：J-TEXT `TPR@lead>10ms`；EAST `TPR@lead≥30ms`；raw alarm occurrence 仅作机器诊断

## 1. 最终结论

这项工作已经把用户要求的三段链路在同一模型血缘上完整打通：

```text
J-TEXT TRAIN1353 训练 causal Patch Transformer
→ VAL160 选择 epoch 与源域阈值
→ 在同次运行内锁定后评估 count-aligned TEST220 比较集
→ 同 seed checkpoint 在 EAST 固定110炮上做小样本迁移
→ 完整静态预测并封存 EAST balanced360
→ 重置到相同迁移父模型
→ 按炮号走完360炮在线主动流
→ predict → seal → query → selected HDF5 reveal → later-shot update
```

当前最佳、同一血缘的三组核心数据如下：

| 阶段 | 正式评估口径 | operational TPR | FPR | AUC | operational median lead |
|---|---|---:|---:|---:|---:|
| J-TEXT 源模型 | TEST220，lead > 10 ms | **0.8515** | **0.2121** | **0.9219** | **116.6 ms** |
| J-TEXT→EAST 静态迁移 | balanced360，lead ≥ 30 ms | **0.8407** | **0.1667** | 未统一保存 | **242.0 ms** |
| EAST 在线主动 | 同一 balanced360 顺序流，lead ≥ 30 ms | **0.8481** | **0.1519** | 未统一保存 | **229.5 ms** |

同预算口径的严格 EAST-only 基线只读取 support110，随机初始化完整17维Transformer并使用EAST本地scaler，三seed均值为 `TPR@30=0.7963、FPR=0.1444、J=0.6519、median lead=165.6 ms`。迁移静态相对它提高 `0.0444` TPR、增加 `0.0222` FPR，净提高 `0.0222` 的 `J=TPR30−FPR`，提前量增加约76.5 ms。只复用J-TEXT scaler但不继承权重的随机模型降到 `TPR@30=0.7315、FPR=0.1056`，说明跨装置归一化本身不能直接共用。target-only更新完整encoder，而迁移模型只更新受限目标路径，二者不是严格等参数量或等FLOPs；这些数值同样只是balanced360开发证据。

在线相对静态的三 seed 配对圆形 moving-block 重采样（block=20，5,000 次）得到：

- FPR 变化中位数 `-0.01389`，95% 描述区间 `[-0.03521, -0.00193]`；
- operational TPR@30 ms 变化 `+0.00717`，95% 描述区间 `[+0.00171, +0.01500]`；
- operational median lead 变化 `-11.66 ms`，95% 描述区间 `[-69.01, +4.00] ms`。

block size 10/20/30/45 的描述方向相同，但这不能当作显著性结论：重采样没有在新历史上重新执行在线学习，而且 balanced360 参与过方法逆向分析和 972 个 episode 候选筛选。它只能描述已选策略在这条开发账本上的局部稳定性，不能重新命名为“未见测试集”或未来流置信区间。operational lead 没有稳定改善。

此外，论文级审计发现了一个不能掩盖的源模型缺陷：当前成功迁移血缘的 J-TEXT 表示可以高精度识别窗口有效历史长度，padding/context probe AUC 为 `0.955/0.929/0.937`，显著超过门槛 `0.65`。PADFIX 能把它降至 `0.624/0.645/0.649`，但会降低源域及时灵敏度，而且把原在线策略直接接到 PADFIX 父模型后，三个 seed 的 FPR 全部上升。这说明当前系统已形成有效工程链路，但还不能声称“完全学到所有破裂机理”或“已经部署级完美”。

## 2. 数据合同和标签修复

### 2.1 J-TEXT

HDF5 总计 1,733 炮：378 破裂、1,355 安全。正式源域划分为：

| 角色 | 总数 | 破裂 | 安全 | 用途 |
|---|---:|---:|---:|---|
| TRAIN1353 | 1,353 | 188 | 1,165 | 源模型与 scaler 拟合 |
| VAL160 | 160 | 80 | 80 | epoch 和阈值选择 |
| TEST220 | 220 | 110 | 110 | 同次执行内锁定后比较评估 |

三者炮号零重叠，但不是严格时间块。每个文件只是为稳定序列化按炮号排序；TRAIN/VAL/TEST 的炮号范围相互穿插。它们是固定 `paper131-count-aligned` 比较队列，VAL160/TEST220 又是结果平衡集，不模拟真实源域患病率或未来部署流。三个 roster SHA256 分别为：

```text
TRAIN1353 ad403f388694e2dc2b8855dae9d084c319243bd02bdf465abe743919a1b4f9d4
VAL160   bf726a031953c6d0aceba04e10b02b713ecca70a32e27077cca372e45ee9d2be
TEST220  b963c2973c198a833e44017f73e0a11749d50d7144f5758b0ec9bfcd7763868e
```

### 2.2 EAST

EAST HDF5 总计 2,953 炮：1,198 破裂、1,755 安全。

- 迁移支持集固定为 110 炮；历史 JSON 标题分组是 10D/100N，但 HDF5 真标签重算后为 **15D/95N**。
- balanced360 为 **180D/180N**，按炮号顺序执行，与 support110 零重叠。
- balanced360 manifest SHA256：

```text
657cab542bd987f9c97c3cb88867a9cc828373b32e6ee58d450451461c6ec825
```

历史清单只负责候选炮号来源；任何实验、指标和时刻标签都只读取 HDF5。证据生成器现在同时保存“历史分组 10/100”和“权威结果 15/95”，防止后续再次混用。

## 3. 为什么之前迁移长期失败

失败不是“深度学习完全不能迁移”，也不只是一个损失权重没调好，而是三种结构性混杂叠加：

1. **23 维名义同名不等于物理同构。** CIII、MNM、Mirnov 幅值、`ne_nG`、`qa_proxy`、SXR 均值、XUV ratio 等通道在单位、可用性、探针实现或构造方法上不稳定。直接 DANN/MMD/CORAL 会把装置差异、缺失模式和类别比例一起“对齐”。
2. **上层时间动力学并不共享。** J-TEXT 炮短，EAST 炮长；两者破裂演化速度和采样上下文分布不同。强制共享完整上层时间读出会造成负迁移。
3. **源域 padding 成为捷径。** J-TEXT 短炮和破裂窗口的左填充比例不同；即使模型整体指标不差，它也可能利用有效历史长度而不是仅靠物理演化。

真正带来突破的改动是结构重分解：共享可验证的 17 维语义子空间，把剩余 8 维留给 EAST 私有分支；共享低层表达，但允许 EAST 私有时间动力学和风险残差；在线阶段把查询、梯度、告警和误报 veto 的价值分开。

## 4. 特征结构

### 4.1 跨装置 shared rebuilt17

正式共享特征为：

```text
Bt, ip_A, Bt_over_Ip, Ip_diff, v_loop, Z_proxy,
sxr_kurt, sxr_skew, sxr_var,
xuv_kurt, xuv_skew, xuv_var,
Mir_avg_fre, n=1 amplitude, mode_number_n, dx_a, dy_a
```

关键处理：

- J-TEXT `ip` 只做一次 kA→A；
- `Bt/Ip`、`Ip_diff`、`Z_proxy=v_loop/ip` 在单位统一后重算；
- 全零 MHD 频率/模数炮标记为 unavailable，而不是物理零；
- scaler 由合法训练域拟合，缺失先保留 mask，再以缩放后的零作为 sentinel。

### 4.2 EAST-private raw8

以下 8 个通道不再强行跨域对齐：

```text
CIII, MNM, Mir_VV, Mir_avg_amp,
ne_nG, qa_proxy, sxr_mean, xuv_ratio
```

它们仍有价值，但只在 EAST 域内进入 1,089 参数的 sidecar。每通道提取最后值、均值、标准差、全窗增量、近期增量和可用率，最后层零初始化，logit 残差有界。raw8 scaler 只由 support110 拟合。

## 5. J-TEXT 源模型

### 5.1 实际架构

实际 checkpoint 合同是：

```text
rebuilt17-patch-transformer-128x8-d64-l2-h4-v1
```

即 128 ms 因果窗口、patch=8、stride=8、`d_model=64`、2 层、4 头。更早计划中的 256 点没有被这一血缘实际执行，论文中已经显式记录这个偏差。

破裂窗口在 TTD `[0,125] ms` 为正；更早窗口和安全炮窗口为负。训练按炮和类别平衡采样，VAL160 在 operational TPR@>10ms≥0.90 的候选中选择最低 FPR 阈值，TEST220 在 checkpoint 和阈值锁定后读取。这里的“锁定”只描述本次执行顺序；由于划分本身是非时序、结果平衡的 count-aligned 比较队列，不能把 TEST220 升格为真实前瞻独立集。

源域逐炮账本已补齐首次越阈的 `alarm_endpoint` 与 `alarm_time_s`。证据构建器不再直接相信归档的 `lead_ms`，而是用 `HDF5 /meta/DownTime - alarm_time_s` 独立重算，并以 `1e-6 ms` 容差核对；缺失 endpoint 时直接失败。三个冻结 checkpoint 在注册的确定性 PyTorch 设置下重放后，原有 score、报警决策和 lead 全部一致。

### 5.2 三 seed 结果

| seed | TP>10/FP/FN>10/TN | operational TPR >10ms | FPR | AUC | operational median lead |
|---:|---:|---:|---:|---:|---:|
| 131 | 97/25/13/85 | 0.8818 | 0.2273 | 0.9307 | 107.2 ms |
| 133 | 90/21/20/89 | 0.8182 | 0.1909 | 0.9253 | 120.7 ms |
| 137 | 94/24/16/86 | 0.8545 | 0.2182 | 0.9098 | 122.1 ms |

这组数据是后续最佳 EAST 迁移链路实际继承的源 checkpoint，不能换成另一个源模型的指标冒充同一血缘。

## 6. EAST 小样本迁移

支持集只使用 110 炮 HDF5 真标签。全部 110 炮同时用于目标组件拟合和 support-derived 阈值（名义 support TPR=0.8）选择，没有独立 EAST validation；其中只有15个破裂结局，因此任何 target 泛化解释都必须降级为开发证据。有效路线不是全局无条件域对齐，而是：

```text
J-TEXT rebuilt17 Transformer
→ EAST adapt_high 高层时间/风险适配
→ EAST DynamicHazardAdapter
→ EAST-private raw8 sidecar
→ 支持集阈值与双分支因果告警
```

DynamicHazardAdapter 使用冻结 CLS、末 token、lag-8 增量及可用物理族摘要，参数量 3,893，风险残差限制在 `[-1,1]`。正残差需通过 support 正例分位数门。源 checkpoint 和冻结基座在运行前后做 SHA 比较。

三 seed 静态 balanced360：

| seed | TP30/FP/FN30/TN | operational TPR | FPR | operational median lead | >3s |
|---:|---:|---:|---:|---:|---:|
| 131 | 148/28/32/152 | 0.8222 | 0.1556 | 263.0 ms | 22 |
| 133 | 151/26/29/154 | 0.8389 | 0.1444 | 205.1 ms | 20 |
| 137 | 155/36/25/144 | 0.8611 | 0.2000 | 258.0 ms | 21 |

用户规定大于 3,000 ms 的报警仍计入 TPR，因此主指标遵守该约定；但论文把数量单列，避免把极早报警直接解释为可用破裂前兆。

## 7. EAST 在线主动学习

### 7.1 严格因果顺序

每炮执行：

```text
只看当前及过去窗口做无标签炮内更新
→ 输出并封存当前炮风险、告警、模型哈希和查询分
→ 不读标签决定是否查询
→ 仅被查询炮读取 HDF5 结局和 DownTime
→ 按 FP/FN/TARDY/TP/TN 形成炮后监督
→ guard + replay + accept/rollback
→ 更新只影响下一炮
```

静态阶段先一次性跑完并关闭文件；在线阶段从完全相同的迁移父模型重置。这样“静态模型看过全 360 标签后再在线”的泄漏不存在。

A1 辅助入口也已改为严格 feature-only 流：训练只读取 support110；预测当前360炮时只读取信号与采样元数据，不打开 `IsDisrupt/DownTime`。惰性 oracle 仅在当前炮预测封存且查询决策为真后读取该炮结局；未查询炮直到整条流结束后的最终评估才读取。

### 7.2 四类价值函数

当前架构不再让一个标量同时控制所有动作。

1. **炮内无标签价值**：下一物理状态重建、教师分歧、物理冲突、raw8 innovation、availability、非冗余和告警接近度共同缩放 fast update；只用当前及过去窗口，下一炮前重置。
2. **查询价值**：执行分数由 uncertainty 与 physics contradiction 组成；过去分数的滚动分位数按 20 炮块冻结，rolling token bucket 控制长期平均查询率。配置中的 `query_quota=6` 在 shadow/legacy 路径中是每 20 炮的 token 补充率，不是硬性逐块上限；实际单块可短时突发到 11 炮，但总查询数仍为 102/90/99。五专家复杂价值只做 shadow 审计，因为 v3a 让它直接接管查询后 operational TPR 下降。
3. **炮后梯度价值**：HDF5 揭示后比较封存预测与真实时序状态；FN/TARDY 优先选择 30–125 ms 内最接近跨过告警边界的窗口，而不是选择最难学习的最低风险点；固定两步更新，学习率由价值缩放。
4. **报警 episode 价值**：用 `delta60` 因果片段表示和过去已查询标签学习 FP 概率，只允许延迟或 veto，不允许创造或提前告警；独立物理支持可以 escape。

### 7.2.1 Q30 与主在线查询预算不是同一实验

项目中“固定30炮”和“主动选择30炮”指向同一个 Q30 实验的两个先后阶段，不矛盾：

| 对象 | 标签数 | 如何产生 | 后续如何使用 | 评估集合 |
|---|---:|---|---|---|
| seed131 低预算 Q30 | 30 | 按 balanced360 时间顺序执行；每炮先预测并封存，再由只读当前/过去特征的查询价值与滚动预算即时决定是否查询 | 选择完成后冻结为同一 manifest，供 `inter_only`、`intra+inter` 和30炮前置迁移复用 | 全360的时序消融；涉及前置训练时只评估未训练的330炮 |
| 当前主在线 v3c | 102/90/99 | 三个 seed 各自在流中自适应查询；不是固定30炮 | 形成论文主在线结果及相应的炮后更新 | 各自完整 chronological balanced360 开发流 |
| 同标签预算前置迁移 v2 | 102/90/99 | 不重新选炮；读取每个 seed 已封存的主在线查询清单 | 将查询标签一次性前置加入目标域训练，作为标签访问量对照 | 分别排除训练炮后的258/270/261炮 |
| 因果随机查询对照 | 与主在线逐块标签数相同 | 标签不可见的块匹配随机日程，每 seed 三次重复 | 使用相同父模型、runner 和配置更新预算 | balanced360 开发流 |

因此，Q30 不是事先人工指定的固定 cohort，也不能表述为看完360炮后取得的“全局最高价值30炮”。更准确的定义是：在每一炮到达时，按当时可见信息、过去分数形成的动态阈值、滚动 token 预算和探索规则，因果地买下标签；已经买下的早期标签不会被未来更高分样本替换。冻结 Q30 只用于控制消融中的标签身份，使不同更新路径比较同一批标签。

当前包已保存 Q30 的30个炮号、流位置、`label_oracle_used=false` 和来源哈希；但 2026-08-21 tmpfs 重启后原始360行 `query_decisions.jsonl` 未恢复，只保留其 SHA256。因此可以审计“冻结清单是什么”和“没有声明标签 oracle”，却不能仅凭现存包逐炮重算原始 cutoff、价值分数与 exploration/exploitation 路由。Q30 结论必须保留为 seed131 的开发性诊断，不能替代主在线三 seed 结果，也不能作为独立确认。

### 7.3 多参数报警函数和状态机

当前开发流中执行的报警器不是单一 max threshold；它尚未经过装置侧部署验收。每个因果时刻 (t) 使用两条路径：

```text
high route:
  margin_t >= high_threshold

early route:
  margin_t
  + physics_weight * physics_support_t
  + trend_weight * max(MA5(margin)_t - MA100(margin)_t, 0)
  >= early_threshold
```

四个连续参数受投影边界约束并允许在被查询炮揭示后更新；快慢窗口 `5/100` 固定，防止在线改变历史算子的语义。FP 使阈值更严格，FN/TARDY 使阈值更早并加强物理/趋势证据，TP/TN 则进入保护 guard。候选更新只要破坏任一已保护正确轨迹就回滚。episode gate 位于双路径之后，只能延迟或否决，不能创建或提前报警。

冻结日志证明参数确实在线变化，而不是只在配置中存在：

| seed | 查询/更新事件 | 实际改变参数 | 初始 `(h, wp, wt, e)` | 最终 `(h, wp, wt, e)` |
|---:|---:|---:|---|---|
| 131 | 102 | 38 | `(0, 8, 12, 7.9)` | `(-0.02138, 8.03083, 12.01541, 7.86988)` |
| 133 | 90 | 30 | `(0, 8, 12, 7.9)` | `(-0.02180, 8.07657, 12.03828, 7.83735)` |
| 137 | 99 | 40 | `(0, 8, 12, 7.9)` | `(-0.01883, 8.03904, 12.01952, 7.86219)` |

在当前炮预测时，系统不知道未来 DownTime；炮后揭示才把结果分成 TP、TN、FP、FN 或 TARDY。TARDY 表示“报警但 lead<30ms”，在正式评价中与 FN 一样是 operational error，并推动后续炮更早报警。

### 7.4 动态误报保护

跨 seed 固定 `p(FP)>0.85` 曾失败。最终阈值只从过去已查询历史构造：

```text
tau_FP = max(0.50, past disruptive-class FP-score q0.90 + 0.06)
```

只有它低于 past safe-class q0.25 且历史样本、类别和置信度充分时才启用；否则 fail-open。`physics_support >= 0.85` 可以放行。操作告警和反事实学习告警相互隔离，避免“本炮被救成 TN 后反过来改变同炮学习组”造成后续轨迹泄漏。

### 7.5 在线结果

| seed | TP30/FP/FN30/TN | operational TPR | FPR | operational median lead | 查询数 | >3s |
|---:|---:|---:|---:|---:|---:|---:|
| 131 | 152/24/28/156 | 0.8444 | 0.1333 | 228.5 ms | 102 | 23 |
| 133 | 151/24/29/156 | 0.8389 | 0.1333 | 214.0 ms | 90 | 20 |
| 137 | 155/34/25/146 | 0.8611 | 0.1889 | 246.0 ms | 99 | 22 |

三个 seed 合计相对各自静态模型减少 8 个 seed-shot FP，并增加 4 个 lead≥30ms 的 operational TP。在线改为 TN 的安全炮为：

- seed131：73790、74928、76861、80801；
- seed133：71060、74562；
- seed137：74939、75135。

## 8. 查询是否有针对性

| seed | 查询率 | 全流 operational error | 查询炮 error | 富集倍数 |
|---:|---:|---:|---:|---:|
| 131 | 28.3% | 14.4% | 34.3% | 2.38× |
| 133 | 25.0% | 14.7% | 32.2% | 2.19× |
| 137 | 27.5% | 16.4% | 41.4% | 2.53× |

说明在 hindsight 标签下，被查询炮的 operational error 比例更高。现已补做 causal random-query 在线重跑，但它只能支持开发流内的方向性比较：查询分数和策略在同一 balanced360 上开发过，三个 seed 也不足以构成确认性显著性证据。

### 8.1 同查询标签预算前置训练消融

为排除“在线主动只是因为多看了 EAST 标签”这一替代解释，针对每个 seed 读取其真实查询炮身份，把这些炮的 HDF5 标签在流开始前直接加入普通迁移重训。所有被加入训练的查询炮随后从该 seed 的评分集合剔除；静态模型、前置重训模型和因果在线模型只在完全相同的标签不相交补集上比较。

正式比较前先做父模型复现门。support110-only 重训的 encoder/head、shared adapter、private adapter SHA256、support 阈值和静态预测均与历史最佳父模型完全一致。早期 v1 因后来采样器从“先抽炮、再在炮内抽窗口”漂移成“先抽类别、再抽炮”而未通过复现门，已标记 `INVALID_PARENT_FIDELITY`，不进入结论。

| seed | 查询 D/N | 评分 D/N | 静态 TPR30/FPR/J | 查询标签前置 TPR30/FPR/J | 在线 TPR30/FPR/J |
|---:|---:|---:|---:|---:|---:|
| 131 | 61/41 | 119/139 | 0.8824/0.0432/0.8392 | 0.9328/0.1727/0.7601 | 0.8992/0.0360/0.8632 |
| 133 | 49/41 | 131/139 | 0.8702/0.0576/0.8127 | 0.9466/0.2446/0.7020 | 0.8702/0.0504/0.8199 |
| 137 | 43/56 | 137/124 | 0.9124/0.0565/0.8560 | 0.9781/0.2419/0.7362 | 0.9124/0.0484/0.8640 |
| 等 seed 均值 | — | — | 0.8883/0.0524/0.8359 | 0.9525/0.2197/0.7327 | 0.8939/0.0449/0.8490 |

这里 `J = TPR@30ms − FPR`，只是显式的平衡操作点摘要。查询标签前置重训相对静态的变化为：

```text
ΔTPR@30 = +0.0641
ΔFPR    = +0.1673
ΔJ      = -0.1032
```

三个 seed 的方向完全一致：查询标签前置后召回、AUC 和提前量上升，但大量安全炮同时越过报警边界；在线模型相对该前置模型的 `ΔJ=+0.1163`。即使用 support110 重新选阈值，前置模型 J 仍降到 `0.7252`，不能靠阈值重标挽救。

另做每 20 炮查询数严格匹配、完全不读取标签的 random front-load，每 seed 三个重复，并先在 seed 内平均。common-mask 的静态、selected front-load、random front-load、在线 J 分别为 `0.8387/0.7322/0.7033/0.8510`。因此可以排除“标签数量本身足以解释在线操作点”；结果更符合延迟的错误类型更新、受限步长、support replay、rollback 与报警状态保护共同作用。

该前置消融仍不是最终因果证明：balanced360 参与了方法开发；前置重训只做到 label-access matched，历史在线日志没有保存精确窗口索引、optimizer/RNG 状态和 FLOPs，无法做 exact equal-gradient replay；只有三个模型 seed，三次同向的单侧精确 sign-test 概率为 0.125，不能冒充显著性结论。

### 8.2 因果随机查询在线对照

另从每个真实主动查询账本构造三条不读取标签的随机查询轨：每20炮查询数、总查询数、父模型、静态预测、runner和配置的optimizer-step上限均与主动臂一致。随机身份在预测前固定，每炮仍严格执行 `predict → seal → query → reveal → update`。

| seed | 主动 J | 随机 J（3重复均值） | 主动−随机 |
|---:|---:|---:|---:|
| 131 | 0.7111 | 0.6907 | +0.0204 |
| 133 | 0.7056 | 0.6963 | +0.0093 |
| 137 | 0.6722 | 0.6537 | +0.0185 |
| 等 seed 均值 | 0.6963 | 0.6802 | +0.0160 |

三个 seed 方向一致，但不能写成“同计算量主动选择已被显著证明”：主动臂平均发生36.67次共享炮间更新尝试，随机臂为16.56次，比值2.21。错误触发更多更新既是主动机制的一部分，也是实现计算量的中介变量；精确三符号单侧概率仍为0.125。该对照关闭了“根本没有causal random在线基线”的缺口，但没有关闭严格等FLOPs隔离和未来流确认缺口。

### 8.3 固定窗口、batch形状与optimizer-attempt的查询名单对照

为隔离上一节的实现计算量中介，另建立一个**不同于主在线机制**的固定同化算子。它从同一迁移父模型和同一静态360账本出发，只比较“归档主动查询名单”与“标签不可见、每20炮查询数匹配的随机名单”。炮内更新、报警参数学习、episode gate、query-value learner、error memory 和 typed memory 全部冻结。每个被查询炮无论事后属于 TP/TN/FP/FN/TARDY，均强制执行：

```text
16个当前炮监督窗口 + 16个固定support110回放窗口
→ shared adapter恰好2个optimizer attempts
→ private adapter恰好2个optimizer attempts
→ 当前炮预测已封存；参数只能影响下一炮及以后
```

逐炮回执记录窗口索引、batch shape、可训练参数scope hash、尝试/提交/回滚步数和首次可影响的下一炮。三个 seed 的实际预算分别为：

| seed | 查询数 | 当前/回放窗口总数 | shared/private attempts | 主动 J | 随机 J（3重复均值） | 主动−随机 |
|---:|---:|---:|---:|---:|---:|---:|
| 131 | 102 | 1632 / 1632 | 204 / 204 | 0.6611 | 0.6630 | -0.00185 |
| 133 | 90 | 1440 / 1440 | 180 / 180 | 0.7000 | 0.7000 | 0 |
| 137 | 99 | 1584 / 1584 | 198 / 198 | 0.6667 | 0.6722 | -0.00556 |

事后开发阶段明确写入报告的初筛规则是三个 seed 上主动名单都优于随机均值；它没有独立时间戳预注册凭据。结果为 `0/3` seed 通过，等 seed 平均差为 `-0.00247`。因此必须撤回“查询身份本身已经被证明优于随机”的论点。可支持的较窄结论是：在这个新固定同化算子下，没有观察到归档主动名单相对三条随机名单的稳定优势；该实验不估计原自适应选择策略的政策效应。主在线结果仍只能作为查询身份、错误触发更新强度、同错回放、rollback、报警状态保护和 episode veto 的**耦合处理效应**汇报。

这个负结果不等于证明主动与随机等价，也不等于所有可能同化算子下查询价值无效。原因是只有三个随机日程、balanced360 已参与开发，而且归档主动身份是一个固定 roster treatment，并非在新流上重新执行自适应策略。它只使“这个固定算子下的归档名单优越性”论点失败，不能否定查询身份在原耦合机制中的贡献，也不能给出普遍零效应或确认性统计结论。完整证据见 `evidence/equal_compute_selector_control_v1/report.json`。

## 9. 负对照与失败证据

| 对照 | 三 seed 总 FP 减少 | operational TP 损失 | 裁决 |
|---|---:|---:|---|
| 不揭示标签 | 0 | 0 | 无法形成保护边界 |
| 特征错位一炮 | 0 | 0 | 表象相似不足以替代当前因果片段 |
| 标签错位一炮 | 1 | 0 | 真实增益基本消失 |
| 标签翻转 | 2 | 23 | 错标签会明显伤害合格报警 |
| 时间演化特征置零 | 3 | 0 | 只保留部分增益 |
| 名为 `raw8_zero` 的归档episode/保护校准screen | 3 | 1 | 只说明该screen保留部分收益；未封存模型级sidecar干预和完整轨迹重跑，不能解释为八个私有通道的独立消融 |
| 真实 v3c | 8 | 0 | 最大净收益且不损失合格报警 |

这组对照支持“正确标签 × 炮内时间演化 × EAST 私有特征”共同作用，但不能证明网络学到了所有破裂因果机制。

已明确否决的路线包括：raw23 无条件全局对齐、普通 DANN/MMD/CORAL 反复加权、高维炮后 residual 盲目增大学习率、复杂价值直接接管查询、跨 seed 固定 FP 概率门、未隔离反馈的 veto、40% 更大查询预算、把 PADFIX 父模型直接接到旧在线策略。

## 10. 时间漂移与科研漏洞

把 balanced360 分成连续四个 90 炮块后，第二块 FPR 很高：

| seed | 在线 worst TPR@30 | 在线 worst FPR |
|---:|---:|---:|
| 131 | 0.6667 | 0.2727 |
| 133 | 0.7907 | 0.3333 |
| 137 | 0.7333 | 0.3636 |

safe score 与炮号的 Spearman 相关为 `0.220/0.091/0.311`，seed137 的 campaign/shot-state 风险明显。当前结果因此不能只拿整体均值就声称稳定通过。

另一个核心漏洞是源域 context shortcut。原 checkpoint 在同一 endpoint、只删除更早历史的 32/64/96/128 视图上：

| seed | padding probe AUC | mean |Δrisk| | alarm flip |
|---:|---:|---:|---:|
| 131 | 0.955 | 0.061 | 5.70% |
| 133 | 0.929 | 0.064 | 8.55% |
| 137 | 0.937 | 0.071 | 7.02% |

均未通过 `AUC≤0.65、|Δrisk|≤0.05、flip≤5%`。

## 11. PADFIX：修复捷径但出现负迁移

PADFIX 为同一 endpoint 构造 `{32,64,96,128}` 合法短上下文，只删过去、不看未来，并加入风险一致性、表示一致性及按结局条件化的 padding 桶对抗项。三个 seed 的 VAL160 audit 全部过门：

| seed | probe AUC | mean |Δrisk| | flip |
|---:|---:|---:|---:|
| 131 | 0.624 | 0.011 | 1.54% |
| 133 | 0.645 | 0.009 | 2.63% |
| 137 | 0.649 | 0.008 | 2.19% |

但锁定后 TEST220 的平均 operational TPR@>10ms 变为 `0.8273`，FPR `0.1515`，AUC `0.9325`。把旧 target/online policy 原样接上时：

| 阶段 | operational TPR@30 | FPR |
|---|---:|---:|
| PADFIX → EAST static | 0.8537 | 0.1630 |
| PADFIX → EAST online | 0.8537 | 0.1704 |

三个 seed 的在线 FPR 都上升。正确解释不是“PADFIX 没用”，而是旧在线策略已适配原源模型的风险几何，不能无代价迁移到新父模型。为了避免继续过拟合 balanced360，本轮停止重调。下一项真正有论文价值的实验应在新的目标验证资源上，为 PADFIX 单独选择目标策略，再一次性跑新未来流。

## 12. 报警参数鲁棒性

前一版双阈值报警对 high threshold、physics weight、trend weight、early threshold、快慢均值窗做 37 个局部/联合扰动，31 个满足：

```text
operational TPR@30 >= 0.75
FPR <= 0.20
```

注册点为 operational TPR@30 `0.8000`、FPR `0.1222`、median lead `139 ms`。但 bootstrap 的 TPR@30 区间是 `[0.7389,0.8556]`，因此不能说“高置信超过0.8”。该鲁棒性属于前置 alarm stage，不冒充最终 v3c 的完整鲁棒性。

## 13. 可视化交付

全部定量图由机器证据生成，使用色盲友好配色、183 mm 双栏宽、PDF/SVG 矢量和 600 dpi PNG；未使用生成式图片制造曲线或数据点。

1. `fig0_end_to_end_model_architecture`：源域监督、共享迁移、EAST-private 支路与在线更新边界的总架构；
2. `fig1_protocol_and_causal_contract`：数据角色、迁移链路和在线因果顺序；
3. `fig2_same_lineage_metrics_and_block_resampling`：J-TEXT 源域指标与 EAST 静态→在线配对变化分开呈现，禁止跨装置折线暗示，并给出 moving-block 描述区间；
4. `fig3_source_shortcut_and_padfix_tradeoff`：padding shortcut 与 PADFIX 代价；
5. `fig4_chronological_blocks_and_active_query_value`：时间块漂移、错误富集和累计报警变化；
6. `fig5_adaptive_alarm_state_machine`：四个报警参数沿360炮顺序流的真实变化轨迹；
7. `fig6_causal_false_alarm_rescue_shot73790`：安全炮 73790 的风险、物理支持和 episode veto；
8. `fig7_negative_controls_and_alarm_robustness`：负对照与前置报警鲁棒性；
9. `fig8_equal_label_budget_frontload_ablation`：同查询标签预算前置重训、随机前置与因果在线的配对消融；
10. `fig9_active_vs_causal_random_query`：完整主动机制与因果块匹配随机查询的效用差异及实际更新尝试数；
11. `fig10_target_only_vs_transfer`：严格 EAST-only、source-scaler target-only 与 J-TEXT→EAST 静态迁移的三 seed 比较；
12. `fig11_equal_compute_selector_screen`：固定16+16窗口、2+2 optimizer attempts（guard提交数可不同）算子下的查询名单负结果。

`figure_manifest.json` 保存所有输入和输出 SHA256。图 4 所需 NPZ 已复制进本论文包，不再依赖 `/home` 的个人工作目录。

## 14. 当前能写进论文的主张

可以写：

- 在当前开发链及历史失败对照中，17 维语义共享 + 8 维 EAST 私有分支比继续强行无条件对齐 raw23 更可行；
- 固定110炮能把 J-TEXT 表示重组为具有较高 EAST balanced360 TPR 的目标模型；
- 在线执行在因果信息边界内完成，标签只影响后续炮；
- 主动查询对 operational error 有约 2.2–2.5 倍内部富集；
- 同查询标签直接前置重训会提高召回却显著扩大误报，不能复现在线的平衡操作点；
- 最终 episode gate 在三个 seed 上同方向降低误报，且负对照不能保留完整增益；
- 源模型 padding shortcut、PADFIX 负迁移和 campaign block 漂移是明确、可复现的失败发现。

不能写：

- “模型学到了所有破裂机理”；
- “balanced360 是完全未见独立测试集”；
- “已经部署通过”；
- “三个 seed 等于1080个独立样本”；
- “bootstrap 已证明未来 campaign 泛化”；
- “PADFIX 已兼容当前 EAST 在线策略”；
- “>3秒报警必然是有效物理前兆”。
- “J-TEXT TEST220 是严格时序或真实患病率下的独立测试”；
- “support110 提供了独立目标验证”；
- “主动查询已在严格等FLOPs、独立未来流上显著优于随机查询”（当前只完成开发流内同标签数、同配置预算的 causal random-query 对照）。

## 15. 下一步唯一科学正确的路线

在新 EAST 顺序流到来前，冻结：

- git SHA、环境、三 seed 源 checkpoint；
- support110 roster/scaler/target parents；
- alarm 公式、阈值、参数投影范围；
- query 策略、预算、更新步数和 FLOPs；
- v3c 配置与全部负对照；
- PADFIX 是否作为独立候选及其目标策略选择规则。

新流逐炮只做一次：静态分支先完整封存；在线分支从相同父模型重置；每炮 predict/seal/query/reveal/update；所有炮必须报告，不能在失败后换阈值或删炮。只有三个 seed 在整体和最差时间块均通过，在线不退化，且随机/no-label/time-shift 等对照不能复现增益，才能把状态升级为独立确认。

## 16. Nature Portfolio 投稿就绪裁决

当前材料已经达到“可审计开发稿”的报告完整度：标签、口径、因果顺序、逐炮账本、哈希、负结果和图表均可复查。但它**尚未达到可提交为确认性 Nature 子刊主张的证据等级**。四个不可通过文案修复的阻断项是：

1. J-TEXT 主划分不是严格时序，VAL/TEST 是 count-aligned 平衡比较集；
2. EAST support110 同时用于拟合和阈值选择，缺少目标验证；
3. balanced360 用于972候选筛选，且没有完全未见未来流；
4. 严格等窗口/等optimizer-attempt查询名单对照已补齐，但主动名单未显示稳定优势；完整在线机制仍没有在独立未来流上做等计算确认，同时缺少可公开或受控访问的数据/代码 DOI。

作者姓名、CRediT、经费、利益冲突和数据访问责任人也必须在投稿前补齐，不能由自动化工具推断。机器可读/逐项裁决见 `NATURE_PORTFOLIO_READINESS.md`。

## 17. 产物入口

- 英文主文：`manuscript/main.tex` 与编译后的 `manuscript/main.pdf`；
- 英文补充材料：`manuscript/supplement.tex` 与 `manuscript/supplement.pdf`；
- 机器证据：`evidence/three_stage_results.json`；
- 论文图：`figures/`；
- 图哈希：`figures/figure_manifest.json`；
- 表哈希：`manuscript/tables/table_manifest.json`；
- 证据生成器：`scripts/build_evidence.py`；
- 图表生成器：`scripts/generate_figures.py`、`scripts/generate_latex_tables.py`；
- 最终验证：`scripts/validate_paper_package.py`；
- 本报告源文件：`TECHNICAL_REPORT_zh.md`；
- 本报告排版版：`TECHNICAL_REPORT_zh.pdf`（A4，26页）。

本报告的价值不在于把结果包装成“没有漏洞”，而在于把已知漏洞变成可复现的实验对象、把不能支持的结论明确排除，并给出下一次真正能改变证据等级的确认协议。

---

## 18. 数学符号、时间轴与因果信息集

设装置域为 $d\in\{S,T\}$，其中 $S$ 表示 J-TEXT，$T$ 表示 EAST。第 $s$ 炮的多变量观测为

$$
X_s=\{x_{s,0},x_{s,1},\ldots,x_{s,N_s-1}\},\qquad x_{s,k}\in\mathbb R^C,
$$

$N_s$ 是实际采样点数；共享输入 $C=17$，EAST 另有 $C_p=8$ 个私有通道。采样时刻由 HDF5 元数据决定：$t_{s,k}=t_{s,0}+k/f_s$。模型在因果端点 $k$ 只能使用

$$
\mathcal F_{s,k}=\sigma\!\left(x_{s,0},\ldots,x_{s,k};\theta_s^-;\mathcal H_{<s}^{q}\right),
$$

其中 $\theta_s^-$ 是进入该炮前的持久模型状态，$\mathcal H_{<s}^{q}$ 是此前**被查询**炮已经揭示的历史。当前炮结局、端点之后的数据和未查询炮标签均不属于 $\mathcal F_{s,k}$。

长度 $L=128$ 的因果窗口为

$$
W_{s,k}=[x_{s,k-L+1},\ldots,x_{s,k}]\in\mathbb R^{L\times17}.
$$

当 $k<L-1$ 时只允许左填充，不能从右侧借用未来值。真实有效长度和填充率为

$$
\nu_{s,k}=\min(k+1,L),\qquad \pi_{s,k}=1-\frac{\nu_{s,k}}{L}.
$$

shortcut audit 检验隐藏表示能否预测 $\pi_{s,k}$ 分桶；这比只查输入是否显式含长度更严格，因为卷积边界和零填充分布也可隐式编码长度。

### 18.1 HDF5 权威结局与窗口标签

逐炮结局严格定义为 $D_s=\operatorname{bool}(\texttt{/meta/IsDisrupt}_s)$。若 $D_s=1$，破裂时刻 $T_s^d=\texttt{/meta/DownTime}_s$，端点到破裂的剩余时间为

$$
\operatorname{TTD}_{s,k}=10^3(T_s^d-t_{s,k})\ \mathrm{ms}.
$$

风险监督使用125 ms hazard horizon：

$$
y_{s,k}=\mathbf1[D_s=1\land0\le\operatorname{TTD}_{s,k}\le125].
$$

$y_{s,k}$ 表示“在125 ms操作危险窗内”，不等于物理 onset A--F 阶段。安全炮全部窗口为零；破裂炮在125 ms以前的窗口同样为零。

### 18.2 报警、提前量与 operational 指标

设接受报警事件为 $A_{s,k}$，首次报警索引 $k_s^a=\inf\{k:A_{s,k}=1\}$。破裂炮提前量

$$
\ell_s=10^3(T_s^d-t_{s,k_s^a})\ \mathrm{ms}.
$$

J-TEXT 与 EAST 的及时命中分别是

$$
g_s^J=\mathbf1[k_s^a\text{存在且}\ell_s>10],\qquad
g_s^E=\mathbf1[k_s^a\text{存在且}\ell_s\ge30].
$$

所以 EAST 破裂炮即使报警，只要 $\ell_s<30$ ms，仍按 operational FN/TARDY 处理。安全炮任意接受报警均为 FP。正式指标为

$$
\operatorname{TPR}_{op}(L)=\frac1{|\mathcal D|}\sum_{s\in\mathcal D}g_s(L),
$$

$$
\operatorname{FPR}=\frac1{|\mathcal N|}\sum_{s\in\mathcal N}\mathbf1[k_s^a\text{存在}],
$$

$$
\widetilde\ell=\operatorname{median}\{\ell_s:s\in\mathcal D,g_s(L)=1\}.
$$

平衡摘要 $J=\operatorname{TPR}_{30}-\operatorname{FPR}$ 只是二者等权差值。若实际漏报/误报代价为 $c_{FN},c_{FP}$，一般风险应写成

$$
R=c_{FN}(1-\operatorname{TPR}_{30})+c_{FP}\operatorname{FPR}.
$$

因此 $J$ 不能冒充真实控制系统效用。

## 19. 数据处理与 17+8 特征分解

### 19.1 训练域稳健缩放

共享特征 $c$ 只用 J-TEXT TRAIN1353 拟合

$$
\mu_c=\operatorname{median}_{(s,k)\in\mathcal T_S}x_{s,k,c},\qquad
q_c=Q_{0.75}(x_{\cdot,c})-Q_{0.25}(x_{\cdot,c}).
$$

缩放为

$$
\widetilde x_{s,k,c}=\operatorname{clip}\left(\frac{x_{s,k,c}-\mu_c}{\max(q_c,\varepsilon)},-10,10\right).
$$

缺失值先生成 $m_{s,k,c}\in\{0,1\}$，然后才置零：$\bar x_{s,k,c}=m_{s,k,c}\widetilde x_{s,k,c}$。因此零是缩放数值/缺失 sentinel；物理约束必须同时读取 mask。raw8 使用同样形式，但中心/IQR 只在 support110 拟合，并限制每炮贡献。

### 19.2 派生代理量

电流统一后计算

$$
I_p^A=10^3I_p^{kA},\quad r_{BI}=\frac{B_t}{I_p^A+\varepsilon},\quad
\Delta I_{p,k}=I_{p,k}^A-I_{p,k-1}^A,
$$

$$
Z_{proxy,k}=\frac{V_{loop,k}}{I_{p,k}^A+\varepsilon}.
$$

这些是可观测代理，不是完整电阻、稳定性边界或第一性原理公式；$Z_{proxy}$ 忽略几何、电感和非稳态项。

### 19.3 为什么不能直接对齐 raw23

抽象地写成 $x_S=g_S(z,\eta_S),x_T=g_T(z,\eta_T)$，其中 $z$ 是潜在共同状态，$\eta_d$ 是装置、探针、单位和缺失机制。直接最小化 $D(P(x_S),P(x_T))$ 时，若 $g_S\ne g_T$ 或类别先验不同，优化器可能通过抹掉结局信息来缩小域距离。因此正式分解为

$$
z_S=f_{sh}(x_S^{17}),\qquad
z_T=f_{sh}(x_T^{17})+a_T(x_T^{17})+b_T(x_T^8).
$$

$f_{sh}$ 承载共享低层表示，$a_T$ 是 EAST 高层时间/风险适配，$b_T$ 是不参与跨域对齐的私有残差。这不宣称17维完全装置不变，只是明确隔离不可强制共享的通道。

### 19.4 raw8 六类因果统计与参数量

对私有通道 $c$ 的窗口构造

$$
u_c=[x_c^{last},\bar x_c,\operatorname{std}(x_c),x_c^{last}-x_c^{first},x_c^{last}-x_c^{mid},|\Omega_c|/L].
$$

八通道拼为 $u\in\mathbb R^{48}$，再与基础 logit 拼接：

$$
r_p=0.5\tanh(\operatorname{MLP}_{49\to16\to16\to1}([u,z_b])),\qquad z_p=z_b+r_p.
$$

最后一层零初始化使初始 $r_p=0$。参数量为 $(49\times16+16)+(16\times16+16)+(16+1)=1089$。

## 20. 源域 Patch Transformer 完整推导

kernel=stride=8 的卷积把 $W\in\mathbb R^{128\times17}$ 映射为16个64维 token：

$$
E=\operatorname{Conv1D}_{8,8}(W)\in\mathbb R^{16\times64},\qquad
H^{(0)}=[e_{cls};E+P]\in\mathbb R^{17\times64}.
$$

“因果”来自窗口端点不读未来；窗口内部 encoder 是全可见 self-attention，并非自回归三角 mask。第 $l$ 层第 $h$ 个头为

$$
Q_h=\widehat H^{(l)}W_h^Q,\quad K_h=\widehat H^{(l)}W_h^K,\quad V_h=\widehat H^{(l)}W_h^V,
$$

$$
\operatorname{Attn}_h=\operatorname{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{16}}+M_{pad}\right)V_h.
$$

四头拼接后做预归一化残差 FFN：

$$
U^{(l)}=H^{(l)}+\operatorname{Dropout}(\operatorname{MHA}(\operatorname{LN}(H^{(l)}))),
$$

$$
H^{(l+1)}=U^{(l)}+\operatorname{Dropout}(W_2\operatorname{GELU}(W_1\operatorname{LN}(U^{(l)}))).
$$

共两层，FFN宽256。最终 $h_s=\operatorname{LN}(H_{cls}^{(2)})$，经 $64\to64\to1$ risk head 得到 $z_s$。

窗口 BCE 为

$$
\ell_{BCE}(z,y)=-y\log\sigma(z)-(1-y)\log(1-\sigma(z))
=\operatorname{softplus}(z)-yz.
$$

代码使用 logits 形式避免数值溢出。训练是平衡采样分布 $Q_S$ 下的估计 $\widehat{\mathcal L}_{src}=128^{-1}\sum_i\ell_{BCE}(z_i,y_i)$。$Q_S$ 与真实部署分布不同，因此输出不能自动解释为校准概率。

VAL160 阈值选择为

$$
\tau_S^*=\arg\min_{\tau\in\mathcal T}FPR_{val}(\tau)
\quad\text{s.t.}\quad TPR_{val,>10}(\tau)\ge0.90.
$$

checkpoint 排名为 $(-FPR,TPR_{>10},AUC)$。TEST220 在 checkpoint 与阈值锁定后读取，但其非时序平衡设计仍不能被称为未来 campaign 验证。

## 21. EAST 小样本迁移结构与损失

迁移采用较弱假设：局部表示可能共享，但长时间动力学不同：

$$
P_S(h_{local}\mid z_{mech})\approx P_T(h_{local}\mid z_{mech}),\qquad
P_S(Y\mid h_{1:k})\ne P_T(Y\mid h_{1:k}).
$$

DynamicHazardAdapter 输入冻结表示 $f$、物理代理 $v$ 和可用掩码 $a$：

$$
q=\operatorname{MLP}([f,v,a])\in\mathbb R^{32},\quad
\widehat v_{t+1}=W_nq+b_n,
$$

$$
\delta_h=\mathbf1[\sum_j a_j\ge2]\tanh(\operatorname{MLP}([q,z_b])),\qquad z_h=z_b+\delta_h.
$$

共同物理族不足两个时残差严格为零，最大幅度为1 logit。support 训练组合为

$$
\mathcal L_{support}=\mathcal L_{hazard}+0.20\operatorname{SmoothL1}(\widehat v_{t+1},v_{t+1})+0.02\mathbb E[\delta_h^2].
$$

目标阈值由同一 support110 选择：

$$
\tau_T^*=\arg\min_\tau FPR_{110}(\tau)\quad\text{s.t.}\quad TPR_{110}(\tau)\ge0.8.
$$

这不是独立验证；HDF5 真标签只有15个破裂炮，故目标泛化解释必须降级为开发证据。

## 22. 双通路报警状态机

基础概率和阈值先转为 logit margin：

$$
m_t=\operatorname{logit}(p_t)-\operatorname{logit}(\tau_T^*),\qquad \operatorname{logit}(p)=\log\frac p{1-p}.
$$

对可用物理代理 $v_{t,j}\in[0,1]$，定义

$$
e_t^{phy}=\frac{\sum_j a_{t,j}v_{t,j}}{\max(\sum_j a_{t,j},1)},\quad
c_t^{phy}=\frac{\sum_j a_{t,j}\mathbf1(v_{t,j}\ge0.5)}{\max(\sum_j a_{t,j},1)},\quad P_t=e_t^{phy}c_t^{phy}.
$$

风险趋势 $R_t=[MA_5(m)_t-MA_{100}(m)_t]_+$。两条通路为

$$
A_t^{high}=\mathbf1[m_t\ge h_t],
$$

$$
S_t^{early}=m_t+w_{p,t}P_t+w_{r,t}R_t,\quad A_t^{early}=\mathbf1[S_t^{early}\ge e_t],
$$

$$
A_t^{raw}=A_t^{high}\lor A_t^{early}.
$$

初始参数 $(h,w_p,w_r,e)=(0,8,12,7.9)$；投影范围 $[-0.1,0.1]\times[6,10]\times[12,15]\times[7.1,8.7]$。炮后更新统一写为

$$
\phi_{s+1}=\Pi_\Phi(\phi_s-\eta_av_sg_{G_s}),
$$

其中 $G_s$ 是 FP/FN/TARDY/TP/TN 类型。FP 使报警更严格，FN/TARDY 使报警更早；正确轨迹进入保护集合。最终参数来自逐炮账本，不是事后静态网格搜索。

## 23. 炮内无标签适配：训练对象与因果边界

### 23.1 fast/persistent 双时间尺度

进入第 $s$ 炮时复制

$$
\theta_{s,0}^{fast}\leftarrow\theta_s^{persistent}.
$$

炮内被选端点只更新 fast copy：

$$
\theta_{s,k+1}^{fast}=\theta_{s,k}^{fast}-\eta_{intra}v_{s,k}\nabla\mathcal L_{intra}.
$$

当前炮结束后 fast copy 被丢弃，不直接污染下一炮；只有炮后查询且通过 guard 的 inter update 才写入 persistent。故“炮内个性化”只服务当前炮，“炮间知识”只来自被查询且审计通过的历史错误。

### 23.2 下一状态、教师和风险锚定

动态表示 $q_{s,k}$ 预测下一个已到达的物理摘要：

$$
\mathcal L_{next}=\operatorname{SmoothL1}(\widehat v_{s,k+1},v_{s,k+1}).
$$

训练在 $k+1$ 到达后发生，只影响后续端点，绝不回改已封存预测。父模型/EMA 教师提供锚点

$$
\mathcal L_{anchor}=\operatorname{SmoothL1}(z^{fast},z^{anchor}),\qquad
\mathcal L_{teacher}=\operatorname{SmoothL1}(q^{student},q^{EMA}),
$$

$$
\theta^{EMA}\leftarrow\beta\theta^{EMA}+(1-\beta)\theta^{student}.
$$

这些项限制单炮噪声造成的漂移，不把模型自身预测当成真实结局。

### 23.3 物理矛盾伪监督

令 $b_t$ 为更新前 base logit，$\tau_l=\operatorname{logit}(\tau_T^*)$。只在两类强矛盾上施加方向性梯度：

$$
\mathcal I_{FP}=\mathbf1[b_t\ge\tau_l,P_t\le0.10,n_t\ge2],
$$

$$
\mathcal I_{FN}=\mathbf1[b_t<\tau_l,P_t\ge0.65,n_t\ge2].
$$

损失为

$$
\mathcal L_{phy}^{u}=\frac1{|\mathcal I|}\sum_t
\left[\mathcal I_{FP}\operatorname{softplus}(z_t-\tau_l)
+\mathcal I_{FN}\operatorname{softplus}(\tau_l-z_t)\right].
$$

没有激活端点时该损失严格为零。因此物理代理不是硬标签，只在风险和多个可用物理族强烈矛盾时提供伪监督。

### 23.4 炮内总损失和更新价值

共享 fast 路径可概括为

$$
\mathcal L_{intra}=\mathcal L_{next}+\lambda_a\mathcal L_{anchor}
+0.10\mathcal L_{phy}^{u}+\lambda_t\mathcal L_{temporal}
+\lambda_o\mathcal L_{observed-phy}.
$$

梯度范数裁剪为0.5，基础学习率 $10^{-4}$，每次1步。更新价值 $v_{s,k}\in[0,1]$ 只缩放学习率：$\eta_{eff}=v_{s,k}\eta_{intra}$。低于 dead zone 0.03 不更新，0.15以上饱和。窗口还有 stride=8、最小间隔8、历史64、$Q_{0.8}$ cutoff和容量4 token 约束，避免高度相关窗口重复训练。

## 24. 主动查询价值与因果预算

### 24.1 查询特征与执行分数

风险峰值的不确定性近似为

$$
U_s=4p_s^{max}(1-p_s^{max}),
$$

在 $p=0.5$ 取1，在 $p\to0/1$ 时趋0。物理冲突 $C_s=\max(C_s^{FP},C_s^{FN})$。重建惊讶、教师分歧和私有创新使用炮内端点的 $Q_{0.90}$，避免单尖峰支配。执行的 legacy 分数为

$$
Q_s=w_sU_s+(1-w_s)C_s,\qquad w_s\in[0.55,0.75].
$$

五专家复杂价值还包含错误修正、信息增益、物理冲突、预期梯度和覆盖探索；揭示后可用指数权重更新专家。但最终 v3c 只以 shadow 模式记录，因为 v3a 让其直接控制查询后 TPR 下降。

### 24.2 Rolling token bucket

令 token 数为 $B_s$，每炮补充率 $r=0.30$，容量12：

$$
B_s^-=\min(12,B_{s-1}^++0.30).
$$

历史 cutoff 只来自过去分数：

$$
c_s=0.35\quad(|\mathcal H_s|<20),\qquad
c_s=Q_{0.70}(\mathcal H_s)\quad(|\mathcal H_s|\ge20).
$$

历史最多160炮，并保留一个与分数无关的探索位置。需要特别区分“注册参数”与“最终执行语义”：独立的 `StratifiedRollingTokenBudgetSelector` 实现具有每20炮最多6次的硬上限；但最终 v3c 使用 shadow/legacy 组合路径，配置中的 `query_quota=6` 参与长期补充/分层控制，并未形成逐块不可突破的硬上限，归档中确有单块11次查询。因此最终执行约束应以 token 可用、过去 cutoff 和实际账本为准，而不能把配置字段错误解释成每块严格6炮。利用查询的基本必要条件仍为

$$
Q_s\ge\max(c_s,c_{label}),\quad B_s^-\ge1,\quad quota\ available.
$$

查询后 $B_s^+=B_s^--1$。早期买下的标签不能在看到未来后撤销。主结果实际查询102/90/99炮；Q30是另一条低预算 seed131 消融，二者不能混称。

## 25. 炮后监督、错误类型学习与回滚

### 25.1 从结局到监督价值

只有查询后才读取 HDF5 并定义 $G_s\in\{FP,FN,TARDY,TP,TN\}$。整条 risk trace 与时序目标的 BCE 为

$$
E_s=-\frac1{N_s}\sum_k[y_{s,k}\log p_{s,k}+(1-y_{s,k})\log(1-p_{s,k})].
$$

归一化 $r_s=\operatorname{clip}(E_s/\log2,0,1)$。监督价值

$$
v_s=\operatorname{clip}[0.50r_s+0.30o(G_s)+0.20g(G_s,\ell_s),0,1],
$$

其中 $o(FN)=1,o(TARDY)=0.6,o(FP)=0.2$；$g(FN)=1$，$g(TARDY)=\operatorname{clip}((30-\ell_s)/30,0,1)$。对 FP 还使用

$$
v_s^{FP}\leftarrow\max(v_s,0.30+0.20c_s^{episode}),
$$

保证昂贵误报标签不会因平均 BCE 偏小而失去学习作用。

### 25.2 sealed residual target

私有分支绑定到揭示前封存 logit：

$$
e_s=y_s-\sigma(z_s^{seal}),\quad
\Delta_s=\operatorname{clip}(\gamma e_s,-0.5,0.5),\quad
z_s^{target}=z_s^{seal}+\Delta_s.
$$

损失为

$$
\mathcal L_{sealed}=\operatorname{SmoothL1}(z_s,z_s^{target})+0.25\ell_{BCE}(z_s,y_s).
$$

它明确学习“相对本炮实际犯错修正多少”，不是把查询炮当普通离线样本无限重训。

### 25.3 persistent inter-shot 目标

炮间更新固定2步，共享学习率 $5\times10^{-4}$，私有学习率 $2\times10^{-4}$：

$$
\mathcal L_{inter}=\mathcal L_{hazard}^{current}
+\mathcal L_{distill}^{replay}+\lambda_p\mathcal L_{physics}
+0.10\mathcal L_{next}+\lambda_g\mathcal L_{guard-anchor}
+0.02\mathcal L_{residual}.
$$

回放蒸馏 $\mathcal L_{distill}^{replay}=\operatorname{SmoothL1}(z_{replay},z_{replay}^{before})$。same-error memory 只把 FP 与历史 FP、FN 与历史 FN 配对，容量64炮；这学习相似错误的局部修正，不是把所有历史标签混成不断扩大的离线训练集。

### 25.4 守卫与原子回滚

更新前在 support guard 上形成保护集合

$$
\mathcal G^+=\{i:\mathbf1(z_i^{before}\ge\tau_l)=y_i\}.
$$

候选更新后的伤害数

$$
H=\sum_{i\in\mathcal G^+}\mathbf1[\mathbf1(z_i^{after}\ge\tau_l)\ne y_i].
$$

正式配置要求 $H=0$ 且输出有限；否则参数和 optimizer state 一起恢复。它只保护 guard 覆盖到的已知正确区域，不保证所有未知未来正确炮都不受损。

## 26. 因果误报 episode gate

双通路先产生 $A_s^{raw}$。episode 模型用报警附近20维表示、快慢差分和波动组成 $d_s\in\mathbb R^{60}$，输出 $p_s^{FP}=\sigma(w^\top d_s+b)$。最终

$$
A_s^{final}=A_s^{raw}\land\neg V_s,
$$

所以 $A_s^{final}\le A_s^{raw}$：它只能延迟/veto，不能创造或提前报警。

历史至少12例、每类至少3例时，候选阈值

$$
\tau_s^{FP}=\max[0.50,Q_{0.90}(p^{FP}\mid\text{past disruptive})+0.06].
$$

还必须与过去安全类 $Q_{0.25}$ 分离且置信度至少0.75；否则 fail-open。若 $P_s\ge0.85$ 则 physics escape。部署 gate 与反事实 raw alarm 学习轨迹隔离，避免本炮被 veto 成 TN 后反向改变同炮更新类型。

## 27. PADFIX：padding shortcut 修复推导

同一端点构造 full view 和只删除更早历史的短视图 $v\in\{32,64,96,128\}$，两者不读未来。记输出 $(z_f,h_f),(z_v,h_v)$，训练目标

$$
\mathcal L_{padfix}=\mathcal L_{task}+0.5\mathcal L_{view}
+0.2\|z_f-z_v\|_2^2+0.5\|\widehat h_f-\widehat h_v\|_2^2
+0.005\mathcal L_{pad-adv}.
$$

adversary 按结局条件化并通过梯度反转让主干难以恢复长度桶，避免仅从类别比例预测 padding。probe AUC 降至 `0.624/0.645/0.649`，说明已知 shortcut 受抑制；但完整准入要求 shortcut、源域性能和目标兼容同时通过。PADFIX 未满足全部门，且直接复用旧 v3c 后三 seed FPR 均升高，所以不能继续用 balanced360 调参“救回”。

## 28. 消融如何排除替代解释

### 28.1 同标签访问量前置训练

对 seed $r$，主在线查询集合为 $Q_r$：

$$
\mathcal T_r^{front}=support110\cup Q_r,
\qquad \mathcal E_r=balanced360\setminus Q_r.
$$

静态、front-load、online 均在同一 $\mathcal E_r$ 评分，排除模型在训练炮上得分。front-load 的 TPR 上升但 FPR 增幅更大，$J$ 从0.8359降到0.7327；online 为0.8490。它否定“标签数量本身足以复现在线操作点”，但不声称严格等 FLOPs。

### 28.2 随机查询和等 optimizer-attempt screen

因果随机日程在每个20炮块匹配标签数，控制标签数量和时间密度，但错误类型与 guard 接受数仍可不同。进一步固定每次16个当前窗口、16个 support 窗口、2个 shared 与2个 private optimizer attempts，并关闭其他机制；该 screen 中 archived-active roster 没有稳定优于随机（0/3 better，一次 tie）。因此不能声称 selector 本身已被证明最优，当前证据只支持完整耦合系统的开发性效果。

### 28.3 EAST-only 基线

严格 EAST-only 从随机初始化17维 encoder/head，scaler 只拟合 support110；source-scaler diagnostic 不继承 J-TEXT 权重。迁移相对严格 EAST-only 的 $J$ 提高0.0222，但可训练参数/FLOPs不完全相等。因此它支持“源权重有增量信息”，不证明全部增益来自单一共同物理机制。

## 29. 时间相关统计与 moving-block 重采样

360炮有时间相关性，不能把1080个 seed-shot 当独立样本。长度 $n=360$、块长 $b=20$；随机抽圆形连续块

$$
B(a_j)=\{a_j,a_j+1,\ldots,a_j+b-1\}\pmod n
$$

并拼至长度 $n$。相同索引用于三 seed 和 static/online：

$$
\Delta_r=\frac13\sum_{q\in\{131,133,137\}}[M_{q,r}^{online}-M_{q,r}^{static}].
$$

5,000次的2.5%与97.5%分位给出描述区间。它没有在重采样历史上重新执行 $\theta_{s+1}=F(\theta_s,x_s,q_s,y_s)$，又未校正972候选选择，因此不是严格总体置信区间或确认性显著性检验。

## 30. 完整因果算法伪代码

```text
1  用 support110 拟合 EAST 私有 scaler、目标 adapter 和独立 risk head
2  用 support110 选 tau_T；冻结迁移父模型与全部 SHA256
3  静态分支对360炮只预测并封存，然后关闭
4  在线分支重新加载与步骤2完全相同的父模型
5  for s = 1,...,360（严格炮号顺序）:
6      persistent -> fast；当前炮标签 oracle 保持关闭
7      for 每个已到达因果端点 k:
8          仅用 x[s,<=k] 计算风险、物理支持、重建误差、更新价值
9          若窗口预算允许，对 fast 做1步无标签更新
10         输出端点风险和双通路报警
11     封存预测、首次报警、查询特征、fast/persistent hash
12     用过去 cutoff + token bucket 决定 query_s
13     if 未查询：丢弃 fast；不读标签；进入下一炮
14     if 查询：此时才读取 IsDisrupt/DownTime
15         生成时序目标和 FP/FN/TARDY/TP/TN 类型
16         候选更新 persistent、私有支路与报警参数
17         guard 若受损或非有限则原子回滚
18         仅用此已查询标签更新 episode/query 历史
19     丢弃 fast；新的 persistent 仅供 s+1 使用
20  流结束后才读取全部结局做最终评估
```

第11步早于查询/揭示是无泄漏核心；第13步保证未查询炮不进入监督学习；第17步限制一次昂贵标签破坏已有正确区域；第19步保证炮后学习只影响未来炮。

## 31. 符号表与代码—公式—证据映射

### 31.1 核心符号表

| 符号 | 含义 | 是否预测时可见 |
|---|---|---|
| $s,k$ | 炮序号、炮内因果端点 | 是 |
| $D_s,T_s^d$ | 结局、HDF5 DownTime | 当前炮预测时否；查询后是 |
| $W_{s,k}$ | 截止端点 $k$ 的128点窗口 | 是 |
| $y_{s,k}$ | 125 ms temporal hazard target | 当前炮预测时否 |
| $p_{s,k},z_{s,k}$ | 风险概率和 logit | 是，由模型产生 |
| $\tau_S^*,\tau_T^*$ | 源/目标冻结阈值 | 是，流开始前冻结 |
| $m_t$ | 相对目标阈值的 logit margin | 是 |
| $P_t,R_t$ | 物理支持、快慢风险趋势 | 是，仅当前/过去 |
| $h,w_p,w_r,e$ | 双通路报警状态参数 | 是，只由过去查询更新 |
| $Q_s$ | 不读标签的查询价值 | 是 |
| $G_s$ | FP/FN/TARDY/TP/TN组 | 查询揭示后才可见 |
| $\theta^{fast}$ | 当前炮临时个性化状态 | 是；下一炮前重置 |
| $\theta^{persistent}$ | 跨炮持久目标状态 | 是；只由过去查询更新 |

### 31.2 代码—公式—证据对应表

| 技术对象 | 数学位置 | 实现入口 | 主要证据 |
|---|---|---|---|
| HDF5标签与lead | §18 | `train_jtext_source_transformer.py`、`online_active_audit.py` | `three_stage_results.json`、逐炮 predictions |
| rebuilt17 | §19.1–19.3 | `rebuilt_harness.py` | roster/content manifest |
| raw8 sidecar | §19.4 | `east_private23_sidecar.py` | private adapter SHA、support scaler |
| Patch Transformer | §20 | `grid_search.py`、`train_jtext_source_transformer.py` | source checkpoints/history |
| dynamic hazard | §21 | `dynamic_hazard_adapter.py` | 参数审计、adapter SHA |
| 双通路报警 | §22 | `dual_threshold_alarm.py` | alarm ledger、鲁棒性 summary |
| 炮内 fast update | §23 | `two_timescale_online_adapter.py` | intra receipts、状态 SHA |
| 查询预算 | §24 | `online_episode_value_v3.py` | query ledgers、102/90/99 counts |
| 炮后更新与回滚 | §25 | `two_timescale_online_adapter.py`、`east_private23_sidecar.py` | update receipts、guard harms |
| episode veto | §26 | `online_episode_value_v3.py` | shot73790 case、v3c screen |
| PADFIX | §27 | `train_jtext_source_padfix.py` | shortcut audit、PADFIX transfer summary |
| 前置/随机/等attempt消融 | §28 | `evidence/` 对应子目录 | comparison/report JSON |
| moving-block | §29 | `scripts/build_evidence.py` | 描述区间 JSON/LaTeX 表 |

任何结论都应沿“公式定义 → 实现函数 → 不可变账本/哈希”追溯。若三者不一致，以实际 HDF5、冻结执行配置和逐炮账本为准，并把偏差标成 lineage deviation，不能修改定义迎合结果。

## 32. 复现实验的最小完整步骤

### 32.1 CUDA 环境预检

正式训练、迁移和在线执行统一使用用户指定的 `rl_env`：

```bash
conda run -n rl_env python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
PY
```

CPU CI 只验证包结构、单元测试和无数据失败关闭，不能替代 RTX 5070 的正式数值执行。

### 32.2 数据挂载和内容验证

仓库不复制 HDF5。外部数据按 README 指定路径挂载后，先核对 `required_hdf5_content_manifest.json`：炮号、结局、DownTime、信号集合和内容 SHA 任一不一致都必须停止。shot list 只提供候选炮号，不能覆盖 HDF5 标签。

### 32.3 固定执行顺序

```text
数据/标签审计
→ 三 seed J-TEXT source training + VAL160 threshold lock
→ TEST220 replay
→ support110 EAST adaptation + target threshold lock
→ balanced360 static prediction seal
→ reset identical parent
→ balanced360 chronological online replay
→ evidence/table/figure build
→ package validator
```

正式命令从仓库 `README_CN.md` 和 package README 复制，所有 Python 命令加 `conda run -n rl_env`。checkpoint 恢复必须同时绑定代码 SHA、配置 SHA、数据清单 SHA、父 checkpoint SHA 和已完成炮序；任一项变化都禁止续跑。

### 32.4 可重复性与可复现性的区别

同一代码/数据/硬件/seed 得到相同逐炮分数称为计算可重复性；换 campaign 或新 EAST 流仍保持性能才属于经验可复现性。当前包通过冻结哈希、确定性 CUDA 和逐炮账本支持前者；由于缺少全新未来流，尚未完成后者。

## 33. 结果解释的逐层边界

### 33.1 当前数据支持的结论

1. 同一 checkpoint 血缘完成了 J-TEXT 源训练、110炮 EAST 适配和360炮因果在线执行。
2. shared17/private8 分解在当前开发证据中比 raw23 无条件对齐更稳定。
3. 在线 v3c 在三个 seed 上均降低 FPR，总体 operational TPR 小幅增加。
4. 查询炮对后验 operational error 有2.19–2.53倍富集。
5. 同标签直接前置训练不能复现在线的 TPR/FPR 平衡，原因主要是 FPR 明显增加。
6. padding shortcut 确实存在；PADFIX 可降低其可探测性，但当前 target policy 与其不兼容。

### 33.2 当前数据不支持的结论

1. 不能说模型识别了所有破裂因果机理；只能说表示对若干时间/物理代理敏感，并排除了部分已知捷径。
2. 不能说 active selector 本身优于随机；固定形状等 optimizer-attempt screen 没有稳定优势。
3. 不能说360炮是独立最终测试，因为它参与方法逆向分析和972候选筛选。
4. 不能说 support110 提供无偏阈值，它同时参与训练和阈值选择。
5. 不能把 moving-block 描述区间称作选择后校正的显著性检验。
6. 不能把超过3秒的报警自动解释为破裂前兆；主指标按约定计入，但必须单列 shortcut/campaign 审计。

### 33.3 真正确认所需设计

获得新 EAST 顺序流后，必须在首炮到达前冻结：三 seed checkpoint、阈值、查询率、token bucket、报警投影、更新步数、episode 边界和停止条件。新流不能用于选候选；最好另设 target validation cohort，使 support fitting、threshold selection 和 future confirmation 三者分离。只有这样才能从 `TEST_DRIVEN_EXPLORATION` 升级为 prospective confirmation。

## 34. 技术总结合科研裁决

这套系统的核心不是某个单独迁移损失，而是三个层次的结构化拆分：共享特征只承担有证据的跨装置语义；EAST 私有分支吸收不可比诊断和时间动力学；在线层通过炮内无标签个性化、昂贵标签查询、错误类型监督、保护回滚和单调误报 veto，使一次查询只在因果允许的方向影响未来炮。

当前三阶段数值证明工程链已经贯通，并且所有主结果都能追溯到 HDF5、冻结配置、逐炮预测和哈希。但 balanced360 的方法开发复用、support110 的训练/阈值双重用途、固定形状 selector screen 的负结果以及源域 padding shortcut，共同决定当前成果仍是“可审计的论文开发证据”，不是已经完成的独立确认研究。最严谨的论文表述应同时呈现有效结果和这些边界；只有新未来流才能真正提高证据等级。
