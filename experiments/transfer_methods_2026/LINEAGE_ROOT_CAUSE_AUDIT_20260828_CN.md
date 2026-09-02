# V3C EAST 主线与后续实验指标差异审计

日期：2026-08-28  
结论级别：主线纠偏记录，不是新的性能结果

## 1. 结论

此前 V3C EAST 主线的增益是真实记录，但它被错误地与后续的 J-TEXT
独立在线实验混在了一起。后续实验指标下降并不说明 V3C 失效，而是说明
实验谱系发生了替换：parent、在线可训练模块、报警状态机和标签/评估入口
均不再相同。

因此，后续工作必须恢复如下边界：

```text
J-TEXT 静态源模型
├── J-TEXT 在线主动学习（独立附加分支，不进入 EAST）
└── V3C EAST 静态迁移
    └── V3C EAST 在线主动学习
```

J-TEXT 的新实验不能用来改写或替代 V3C EAST 主线；EAST 的新实验也不能
把普通 residual/BCE 结果称为 V3C 结果。

## 2. V3C 的可核对结果

证据文件：

```text
paper_grade_package/evidence/three_stage_results.json
paper_grade_package/configs/primary_chain_v3c.json
```

balanced360 为已经参与方法开发的 180D/180N 流，因此下列结果只能标记为
`TEST_DRIVEN_EXPLORATION`，不能当作独立未来确认。

| seed | EAST 静态 TPR@30 | 静态 FPR | EAST V3C 在线 TPR@30 | 在线 FPR | ΔTPR | ΔFPR |
|---:|---:|---:|---:|---:|---:|---:|
| 131 | 0.822222 | 0.155556 | 0.844444 | 0.133333 | +0.022222 | −0.022222 |
| 133 | 0.838889 | 0.144444 | 0.838889 | 0.133333 | 0.000000 | −0.011111 |
| 137 | 0.861111 | 0.200000 | 0.861111 | 0.188889 | 0.000000 | −0.011111 |
| 均值 | 0.840741 | 0.166667 | 0.848148 | 0.151852 | +0.007407 | −0.014815 |

对应的均值目标函数为：

\[
J=\mathrm{TPR}_{30}-\mathrm{FPR},
\]

所以：

\[
J_{\mathrm{static}}=0.674074,
\qquad
J_{\mathrm{V3C}}=0.696296,
\qquad
\Delta J=+0.022222.
\]

V3C 的主要增益是减少 FP，同时没有在三个 seed 上损失 operational TPR。
这正是其设计目标，而不是重新训练底层 Transformer 的结果。

## 3. V3C 实际做了什么

V3C 的固定配置包括：

- J-TEXT/EAST `rebuilt17` 源表示和固定迁移 parent；
- EAST 的独立目标侧路径；
- `delta60` 因果报警端点表示；
- 查询后更新的报警状态机参数：
  `high_threshold`、`physics_weight`、`trend_weight`、`early_threshold`；
- fast/slow 因果窗口为 5/100；
- 状态机只能延迟或 veto 已有报警，不能凭空创造或提前报警；
- FN/TARDY、FP、TP/TN 使用不同的方向更新；
- 所有更新只影响后续炮；
- 三个 seed 采用相同的 V3C 合同和独立的 seed parent。

其核心报警形式为：

\[
 a_t =
 \bigl(m_t\ge h\bigr)
 \,\lor\,
 \left[
 m_t+w_p p_t+w_r\max\{\operatorname{MA}_5(m_t)-
 \operatorname{MA}_{100}(m_t),0\}
 \ge e
 \right].
\]

其中 (m_t) 为基础报警 margin，(p_t) 为因果物理支持，(h,w_p,w_r,e)
为受限在线参数。V3C 的 episode gate 只对已有候选报警做保护，不能生成
一个基础模型没有的物理信号。

## 4. 后续下降实验为什么不能与 V3C 比较

最近运行的 `full_active_advanced` 是 J-TEXT 专用 runner，主要差异如下：

| 项目 | V3C EAST 主线 | 最近运行的实验 |
|---|---|---|
| 域 | EAST | J-TEXT |
| parent | V3C 固定 EAST 迁移 parent | `corrected_mainline_screen` J-TEXT source checkpoint |
| 在线模块 | V3C `delta60` 保护状态机及其查询后更新 | J-TEXT residual + BCE 错误定向权重 |
| 作用对象 | EAST 后续报警决策和受限参数 | J-TEXT residual 风险 logit |
| 状态机 | V3C 多尺度保护状态机 | J-TEXT 基础阈值报警 |
| 标签口径 | EAST `max(125 ms, AdvancedTime)` | J-TEXT `max(25 ms, AdvancedTime)` |
| 评价流 | balanced360 | TEST220 |
| 能否代表 V3C | 可以 | 不可以 |

因此，近期出现的较低 TPR 或较低 J 不能归因于“V3C 越做越差”。它只能
说明 J-TEXT 新在线更新臂本身尚未证明有效。

## 5. 纠偏后的实验制度

### EAST 主线

1. 固定 V3C 三个 seed 的迁移 parent、scaler、阈值和报警器；
2. 先复现 V3C static 与 V3C online 的逐炮 ledger；
3. 只在同一 parent 上做 `intra-only / inter-only / intra+inter`；
4. 任何新神经更新必须与 V3C 状态机单独做正交消融；
5. 不允许用 J-TEXT 在线 checkpoint 初始化 EAST；
6. balanced360 仍标记为开发诊断，最终结论必须留给未来流。

### J-TEXT 分支

1. 使用 J-TEXT `max(25 ms, AdvancedTime)` 标签；
2. 独立做 static、intra-only、inter-only、intra+inter；
3. 查询后错误监督只作用于 J-TEXT residual 或明确注册的局部 adapter；
4. 不把 J-TEXT 结果写入 EAST V3C 结果表；
5. J-TEXT 在线增益需要单独证明，不能借用 EAST V3C 的增益。

## 6. 当前最重要的修复

当前应该停止的行为：

- 用普通 J-TEXT residual 实验替代 V3C EAST 主线；
- 用不同 parent 的结果比较“在线增益”；
- 把 V3C 的 FP 保护收益解释成底层 Transformer 学到新物理规律；
- 在 balanced360 上反复调参后把结果写成独立确认。

当前应该恢复的行为：

- V3C 作为 EAST 主线基线；
- J-TEXT 在线作为独立附加分支；
- 神经更新、报警状态机、专家权重分别做消融；
- 对所有结果保存 parent hash、报警器 hash、标签合同和逐炮 ledger。

## 7. 论文表述边界

可以表述：

> 在已锁定的 balanced360 开发流中，V3C 的查询驱动多尺度报警保护在三个
> seed 上平均降低 FPR，并未降低 lead-qualified TPR；该结果支持其作为一种
> 受约束的在线报警保护机制。

不能表述：

> V3C 已经在独立未来 EAST 数据上证明了模型学到了新的破裂物理规律。

原因是当前 balanced360 参与了参数和结构开发，而且 V3C 的主要变化发生在
报警决策保护层，而非被冻结的 Transformer 主干。
