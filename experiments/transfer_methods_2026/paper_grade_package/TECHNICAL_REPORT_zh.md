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
- 本报告：`TECHNICAL_REPORT_zh.md`。

本报告的价值不在于把结果包装成“没有漏洞”，而在于把已知漏洞变成可复现的实验对象、把不能支持的结论明确排除，并给出下一次真正能改变证据等级的确认协议。
