# EAST V3C 与 J-TEXT 在线分支谱系状态

更新时间：2026-08-29

本文档只做谱系登记，不产生新的性能结果。目的是防止把历史 EAST
V3C 主线、后续冻结复现、Adv25 兼容实验和 J-TEXT 独立在线分支混成一条线。

## 1. 当前结论

历史 EAST V3C 主线确实出现过同时改善 TPR 和 FPR 的结果；它的证据位于
`paper_grade_package/evidence/primary_online_artifacts/`。J-TEXT Adv25
在线分支目前已经执行了严格的因果炮内/炮间更新，但首轮修复实验尚未产生
可归因的指标增益。因此，不能用 J-TEXT 的失败改写 EAST V3C，也不能借用
EAST V3C 的增益宣称 J-TEXT 在线学习已经成功。

## 2. 已锁定的 EAST V3C 历史主线

数据流是 `EAST 静态迁移 -> balanced360 静态封存 -> 同一 parent 的
V3C 在线流`。balanced360 已参与方法开发，所以状态仍是
`TEST_DRIVEN_EXPLORATION / ONLINE_DIAGNOSTIC`，不是独立未来确认。

| seed | 静态 TPR@30 | 静态 FPR | V3C 在线 TPR@30 | V3C 在线 FPR | ΔTPR | ΔFPR |
|---:|---:|---:|---:|---:|---:|---:|
| 131 | 0.822222 | 0.155556 | 0.844444 | 0.133333 | +0.022222 | −0.022222 |
| 133 | 0.838889 | 0.144444 | 0.838889 | 0.133333 | 0.000000 | −0.011111 |
| 137 | 0.861111 | 0.200000 | 0.861111 | 0.188889 | 0.000000 | −0.011111 |
| 均值 | 0.840741 | 0.166667 | 0.848148 | 0.151852 | +0.007407 | −0.014815 |

这条线的在线变化包含 V3C 的多尺度报警保护、查询后参数更新、episode
保护和历史端点记忆；它不是纯 Transformer 表示更新的消融。论文中应将
它称为“受约束的在线报警/决策适应系统”，不能把全部收益归因于底层物理
表示学习。

## 3. 不能互换的相邻实验

| 目录/实验 | parent 与在线结构 | 当前用途 | 能否替代历史 V3C |
|---|---|---|---|
| `primary_online_artifacts` | 历史 V3C parent，动态查询与 episode 路径 | EAST 历史主线诊断证据 | 是主线证据 |
| `online_architecture_search/frozen_runs` | 冻结 v2.1 配置，72 次查询，较小状态机 | 冻结复现/敏感性诊断 | 否 |
| `full_v3c_replay`（ext4） | 旧兼容 replay，72 次查询 | 兼容性检查 | 否 |
| `jtext_adv25_v3c_online_20260829` | J-TEXT Adv25 source，独立低秩 adapter | J-TEXT 炮内/炮间消融 | 否，且不进入 EAST |

## 4. J-TEXT 分支的当前可核对状态

J-TEXT 使用新源标签

\[
H_i=\max(25\,\mathrm{ms},\mathrm{AdvancedTime}_i),
\]

源模型与 EAST 迁移 parent 均来自静态 checkpoint；J-TEXT 在线状态不会被
`reproduce_advanced25_chain.py` 传给 EAST。在线 runner 遵循：

```text
当前信号前缀 → 预测并封存 → 用当前观测更新下一 endpoint 的 fast state
→ 决定是否查询 → 仅对已查询炮 reveal → 炮间监督更新
```

首轮 seed=131、20% 查询预算的修复实验已经证明：炮内更新不再因为身份初始
参数的范数定义而全部回滚，确实有 8,264 次应用的 fast update；但逐炮
TPR/FPR 尚未改变，不能称为在线增益。主要原因是当前 44 个查询没有覆盖
两个低风险 FN，且平均风险 residual 对局部错误轨迹的作用过弱。

下一轮只允许增加可审计的“查询后错误轨迹记忆/局部监督 adapter”实验，
并且必须与 `memory-only`、`neural-only`、`memory+neural` 分开报告；不允许
改动 EAST V3C 历史主线文件来制造表面增益。

## 5. 结果表的书写边界

可以写：

> 在已用于方法开发的 balanced360 上，历史 V3C 线路在三个 seed 的均值中
> 提高了 lead-qualified TPR，并降低了 FPR；该结果是开发流上的受约束在线
> 决策适应证据。

不能写：

> EAST V3C 已在独立未来流证明模型学习了新的破裂物理规律。

同样，J-TEXT 在线首轮只能写为“因果更新链路已执行、尚未观察到可归因增益”，
直到有固定查询合同、多 seed 和未参与开发的时间流支持更强结论。
