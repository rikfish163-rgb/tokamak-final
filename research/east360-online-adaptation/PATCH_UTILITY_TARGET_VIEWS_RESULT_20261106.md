# 观测缺失能增加有用的炮内教学实例，但还不是在线增益

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run and independent recount
- Origin Date: 2026-11-06
- Verification Status: complete for masked TRAIN capacity, HDF5, and raw model witnesses
- Version Label: patch_utility_target_views_v1

**决定：可开展跨视图留炮训练；不推进 VAL 或 EAST 实流。**
同一 J-TEXT TRAIN 的三个观测视图按原炮号去重，单写入有益炮
从原生 10 个扩大到 39 个。两种缺诊断视图共有 4 个独立迟报
存在**内部更新能救、所有同锚点恒偏移都救不了**的单次动作，
覆盖原固定五折中的 3 折，满足[事前支持门槛](PATCH_UTILITY_TARGET_VIEWS_PROTOCOL_20261106.md)。
其中只有 3 个相对原生动作库新增；相同 1052388 不重复计作新增。
这些动作由事后真值认定，不代表炮内已经知道该何时、往哪更新。

## 1. 实验回答的问题

主实验的 150,840 个非零动作中，30 个有利单动作只来自 10 炮。
这一轮只改变波形的观测掩码：原生、缺 MNM、缺 MNM 和 CIII。
其余保持同一原始 HDF5、相同采样时刻、J-TEXT 源模型与阈值、
每 32 点的释放时间、.10 logit 历史保护与 .35 状态范数。
两个新视图各自重新生成真实内部动作，但不访问结局；两者都
封存后才用 HDF5 结局与时钟评分。每个视图对自己不更新的基线
评分，另以同锚点恒偏移作为完整的非神经动作库。

在现行 EAST 数据合同中，MNM 被标为不可用，CIII 在部分炮
完全不可用。这一对照只模拟这两种**观测可用性**，不改变
诊断单位、不声称已复现 EAST 的跨装置漂移。有关 TTA 在
大分布偏移下不保证改善，以及 Transformer 可能对缺输入
敏感的证据分别见 [Liu 等，NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html)
和 [Ma 等，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html)；
它们说明为什么应单独验证输入条件，不支持本项目的物理或部署结论。

## 2. 事后容量，不是选择器成绩

| 视图（188 破裂／1,165 安全） | 静态及时／误报 | 有利／有害单动作 | 有利／有害独立炮 | 可救迟报：单次／多次 | 可消误报：单次／多次 |
|---|---:|---:|---:|---:|---:|
| 原生 | 182 / 117 | 30 / 92 | 10 / 28 | 2 / 2 | 8 / 11 |
| 缺 MNM | 178 / 113 | 31 / 115 | 10 / 36 | 1 / 1 | 9 / 9 |
| 缺 MNM 和 CIII | 179 / 200 | 68 / 197 | 22 / 53 | 3 / 3 | 19 / 21 |

两缺失视图的全部同锚点恒偏移动作均不能救回迟报，单次只消除
一个双缺失误报（1053101）。也不能把观察到的新错误全部说成
可修正：双缺失静态误报变成 200 炮（17.17%），即使使用整个
真实动作库的事后最佳多写入，也只覆盖其中 21 炮。与受益一样，
潜在有害动作亦增多；后续策略必须同时限制新增误报。

| 原始炮号 | 视图 | 独立复跑：首报提前量（ms） |
|---|---|---:|
| 1053940 | 缺 MNM | 6.56 → 107.56 |
| 1052388 | 缺 MNM 和 CIII | 6.36 → 35.36 |
| 1053042 | 缺 MNM 和 CIII | 7.56 → 34.56 |
| 1055239 | 缺 MNM 和 CIII | 8.87 → 19.87 |

及时性沿用 J-TEXT **严格 >10 ms** 的操作口径；较早首报不等于
证实了真实物理先兆，也不自动说明告警出现在规范标签窗内。
四例均是在同一缺失视图内对比真实状态写入和零写入，不把
单纯屏蔽输入造成的风险差异记作适应增益。

## 3. 运行和下一步

GPU session15789 200.65 s、退出 0，两个视图分别 14 个分片、
1,353 炮、12,570 释放块，结局读取均为零；每视图三个截断
见证的最大绝对数值差合计 2.43e−6，动作可用性完全相同。
CPU session53920 7.99 s、退出 0，封存后按原 HDF5 评分。
独立核验 session6807 8.76 s、退出 0，重算 326,820 个候选
硬标签与 2,706 条可达首报集合；原网络 CPU 复跑上述四炮，
未来曲线最大误差 8.82e−6 logit，越阈位一致，已到达三点
保护点的最大实际绝对变化 .0703 logit。相关回归
157 项测试及 14 个子测试通过；未读新 VAL、TEST 或 EAST 数据。

按[下一阶段计划](PATCH_UTILITY_MULTIVIEW_POLICY_PROTOCOL_20261106.md)
比较原生、单视图和配对视图训练。所有视图共享原 shot ID
划分，严格隔离训练、校准、留出，真实、倒序和同锚点分数
对照使用相同动作库。只有实际因果决策能在留出炮上救回
首报并控制误报，才进入固定源 VAL 和后续 EAST 炮内／
炮间消融；目前不将事后容量写成在线增益。

[双视图封存](probe_outputs/patch_utility_target_views_20261106_v1/completed.json) ·
[容量和逐炮清单](probe_outputs/patch_utility_target_views_20261106_v1/capacity.json) ·
[独立原模型核验](probe_outputs/patch_utility_target_views_20261106_v1/verification.json)
