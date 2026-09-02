# Adv25 标签替换根因审计与迭代裁决

审计时间：2026-08-30T15:02:06.006398+08:00  
状态：`DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION`  
运行环境：`rl_env`（审计脚本不训练模型）

## 结论先行

本次复核没有发现 AdvancedTime 的秒/毫秒换算错误、J-TEXT 炮表交叉、Adv25 checkpoint 越权读取 TEST220，或 EAST V3C 在线代码被替换的问题。专项时间审计也确认 AdvancedTime 与信号末端间隔并不要求相等；它是独立的炮级弱 lead-time 标注，不是采样末端时间戳。Adv25 结果下降的直接原因是监督分布发生了实质变化：在当前硬标签解释下，固定 125 ms 正窗中相当一部分端点被改成了硬负类。这个结论不能被表述为‘迁移算法失败’或‘J-TEXT 与 EAST完全不相通’。

旧 V3C 的 EAST 在线机制仍可复现；但把源模型标签改为 Adv25 后，不能自动保证旧在线增益保持。下一次允许的迭代应是预注册的‘源表示保持/标签边界消融’，而不是用 balanced360 继续反向调报警参数。

## 1. 标签合同与实际影响

主合同固定为：

$$H_i=\max(25\,\mathrm{ms},\mathrm{AdvancedTime}_i),\quad y=1\Leftrightarrow 0\leq\mathrm{TTD}\leq H_i.$$

这里还明确采用了互补硬负类 `H_i < TTD <= 60000 ms`。如果研究意图是‘AdvancedTime 只是最早可靠边界’，则该互补负类不是同一个实验，必须另行注册灰区或软标签方案。

| 角色 | 炮数 | 破裂炮 | 缺失 AdvancedTime | fixed125 正端点 | Adv25 正端点 | 被改为硬负类 |
|---|---:|---:|---:|---:|---:|---:|
| train | 1353 | 188 | 1 | 21912 | 5557 | 16369 |
| val | 160 | 80 | 1 | 9651 | 3002 | 6767 |
| test | 220 | 110 | 1 | 12755 | 3213 | 9542 |
| **合计** | **1733** | **378** | **3** | **44318** | **11772** | **32678** |

正端点从 44318 减少到 11772，减少 32546（73.4%）。

AdvancedTime 与最后一个信号采样点的差值统计（仅用于语义审计，不用于改标签）：中位 8.000 ms，最大绝对值 165.000 ms。该差异说明 AdvancedTime 是弱标签 lead-time，不能被当作绝对时间戳。

## 2. EAST 配对结果

下表的主对照是已经注册的锁定 V3C/v8 下游实现与 Adv25；balanced360 已参与方法开发，因此两者均为诊断性结果。

| seed | 锁定V3C静态 TPR/FPR | 锁定V3C在线 TPR/FPR | Adv25静态 TPR/FPR | Adv25在线 TPR/FPR | Adv25查询炮 |
|---:|---:|---:|---:|---:|---:|
| 131 | 0.8111/0.1444 | 0.8222/0.1444 | 0.8667/0.1389 | 0.8667/0.1389 | 103 |
| 133 | 0.7667/0.1278 | 0.7722/0.1278 | 0.7556/0.1333 | 0.7611/0.1333 | 102 |
| 137 | 0.8167/0.1389 | 0.8222/0.1389 | 0.8333/0.1556 | 0.8333/0.1722 | 95 |
| **均值** | 0.7981/0.1370 | 0.8056/0.1370 | 0.8185/0.1426 | 0.8204/0.1481 | 100.0 |

锁定 V3C 在线 ΔJ=+0.0074；Adv25 在线 ΔJ=-0.0037。因此旧在线收益没有在 Adv25 标签下稳定复现。该结论只适用于锁定 V3C 配对；不能把另一条 episode-v3 开发链的结果混入。

另存的 development-v3c 历史证据均值：静态 0.8407/0.1667，在线 0.8481/0.1519，ΔJ=+0.0222；它仅作诊断，不能替代锁定主对照.

## 3. 谱系与泄漏检查

- 三个 Adv25 source checkpoint 均声明 `advanced_floor25`，AdvancedTime 文件 SHA 一致，且 `test_shards_read=0`。
- J-TEXT 在线分支的 checkpoint 没有作为 EAST 的 parent；EAST 阶段使用静态 Adv25 source checkpoint。
- 固定 125 ms 的当前 `shot_first` V3C seed131 静态/在线指标与静态预测哈希复核：`PASS`。
- 旧 A1 reference JSON 没有记录当前 sampler/code lineage；因此当前代码对它的逐分数校验可能产生误导性的失败。该旧 reference 不应继续作为 Adv25 判据。
- 固定 25 ms 的 full V3C 临时运行没有产生完整 summary，且曾因 ENOSPC 中断；它不进入任何结果表。

## 4. 下一步唯一允许改变的因素

1. 保持 EAST 适配、报警器、炮序和阈值来源不变；
2. 只注册一组源模型表示保持方案（例如 Adv25 主损失 + fixed125 表示蒸馏/局部多任务保持），并与纯 Adv25、fixed125 做同 parent 的配对消融；
3. 先在 J-TEXT VAL160 选择 epoch/阈值，不能用 balanced360 选择；
4. 重新跑三个 seed，报告源域和 EAST 的 static/online 变化；
5. 在新方案锁定前，balanced360 只保留诊断，不作为独立未来测试。

## 5. 产物

机器可读审计：`/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/adv25_root_cause_audit_20260830/audit.json`  
本报告：`/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/transfer_methods_2026/adv25_root_cause_audit_20260830/REPORT.md`  
AdvancedTime SHA256：`559da4f719373b0c13b96d34e27a2b46b80eba7f0ef185c3b6ac58b3761e0199`
