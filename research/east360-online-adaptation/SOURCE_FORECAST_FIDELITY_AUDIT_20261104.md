# Source forecast fidelity：独立产物复核

日期：2026-11-04。只读已完成产物，CPU 重算，不读取原始信号或运行模型。**结论：完成清单、预测封存、坐标映射和按炮分组汇总内部一致；在已评分的源域 VAL 窗口中，future125 头的 zero-fast 后续诊断预测损失高于原生头和保持基线。**这不是已验证的 adaptive forecast、在线告警收益或物理机理结论。

## 封存与身份

- [completed.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_update_utility_fidelity_20261104_v1/completed.json) 的 6 个制品哈希全部独立重算匹配；completed 自身 SHA-256 为 `6ea2f00f4d18c9f4dbca4cd68e357d2892440440f9d39f31230080c4bda8594f`。其中 [forecasts.npz](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_update_utility_fidelity_20261104_v1/forecasts.npz) 为 `ca27a2d32f81ef0a27e8fc0dc53843f4055b99a3495f997d40d4788d4a3f2fe1`，与 [prediction_seal.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_update_utility_fidelity_20261104_v1/prediction_seal.json) 相同；seal 本身哈希亦匹配完成清单。
- NPZ 使用 `allow_pickle=False` 核查：indices 为 764 个 int64；forecasts 形状 `(2,4,764,25,23)`、float32；hold 形状 `(2,764,25,23)`、float64；所有元素有限。764 个索引按顺序与 rows 中 eligible 的原始 index、连续 prediction_index 完全一致。视图和头顺序在 seal 与 protocol 中一致。
- [protocol.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_update_utility_fidelity_20261104_v1/protocol.json) 绑定的 fit 与 design 完成文件哈希均匹配实际文件；逐行核对 design 的 946 条 VAL 记录，shot、origin、query、HDF 结果字段及分组与 rows 的顺序完全一致。此处核查的是已保存的 HDF 衍生记录，未重新打开 HDF。protocol 记录的源检查点身份为 `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`。
- completed 的 `state_sha256_before` 与 `state_sha256_after` 四个条目逐项完全相等：native_reference `99affd49f71f41129b87b03624201ef94ef6dc44e652eaef411b2924f8186dc8`；point_contract `b13203b7fb1253dd7f90c49c7675e931ffab881e353a649c3b75b583c84b69a7`；anticipation125 `03c12d3b909baa87f6b04fbdfc936568907b2d583fb99f9101ffca95ccf86b83`；time_shuffled125 `a40e973d53c7e03230a88e524874c50e84d4e965c76a508dabc1cc3c32e41fdb`。制作者同时记录 source/head 更新均为零；由于这里没有独立恢复两份运行时状态，这证明**记录的前后摘要一致**，不是独立内存监控。

## 覆盖及选择边界

946 个 shot/query 坐标互不重复；764 个完整、182 个未来点不足。完整表示 25 个未来**采样索引**齐全，并非全部 23 个通道均参与评分；不足记录没有封存预测或评分。按已存 rows 独立重算：

| HDF 分组 | 全部窗口 / 炮 | 完整窗口 / 炮 | 不足窗口 |
|---|---:|---:|---:|
| 总计 | 946 / 145 | 764 / 133 | 182 |
| 安全 | 638 / 80 | 537 / 80 | 101 |
| 破裂前 >30–125 ms | 155 / 53 | 155 / 53 | 0 |
| 破裂前 >125 ms | 62 / 20 | 62 / 20 | 0 |
| 破裂前 0–30 ms | 91 / 65 | 10 / 10 | 81 |

同一破裂炮可能出现在多个时距组，不能把各组炮数相加。尤其 0–30 ms 中 81/91 窗口不足，完整窗口只覆盖 10/65 炮；不能把完整子集的分组表现外推为该组全部窗口表现。保留下来的相邻窗口也有相关性，**统计单位不能按 764 个独立样本处理**。

## 从 rows 独立复算 summary

计分按 protocol 为 source asinh 坐标下 float64 SmoothL1：先在每个窗口对观测标量求均值，再对同一炮窗口求均值，最后对该组炮等权平均。对 2 视图 × 4 头 × 5 组的 **40 个单元**，重新计算 mean_loss、delta_vs_native、hold_loss、delta_vs_hold，以及炮数、窗口数、观测数；所有结果与 [summary.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_update_utility_fidelity_20261104_v1/summary.json) 一致，四项浮点字段最大绝对差 `2.220446049250313e-16`。coverage 各组、缺失点数直方图和两视图可评分数也全部一致。

| 视图：133 炮 / 764 完整窗口 | 原生头 | 原点标签头 | future125 头 | 时间打乱头 | 保持基线 |
|---|---:|---:|---:|---:|---:|
| 无 MNM/CIII | 0.119408106 | 0.324474954 | 0.534311694 | 0.439455325 | 0.056258397 |
| native | 0.091421096 | 0.345495962 | 0.515729182 | 0.491322775 | 0.053994268 |

数值越低表示此指标的预测损失越小，不是百分比或原始物理量误差。所有完整窗口的 native 视图每窗口评分 550 个标量，无 MNM/CIII 视图为 500 个；两视图评价通道集合不同，跨视图绝对值不应直接解释为单纯的缺失通道效应。分组炮数和损失是描述性汇总，本审查未对相关窗口计算独立样本置信区间或显著性。

## 可复核边界与未完成事项

- protocol 将预测定义为在 query 的零临时 fast offset 上直接预测 q+1 至 q+25，保持基线取 query 末值并使用同一目标掩码；**没有内层更新**。protocol 还记录预测 NPZ 的哈希封存后才读取未来真值进行评分。哈希可证明现存文件与 seal/清单相符，但单凭现存文件不能独立证明历史访问顺序、模型真实执行路径或清单自身的外部真实性。
- NPZ 只包含 indices、forecasts 和 hold，不含逐点未来真值及完整评分掩码。本次能独立复算的是**已保存 rows 到 summary**，不能从 NPZ 独立重新计算每条窗口损失，也未重跑任何预测。这一区别不影响上述摘要和封存字节的一致性结论。
- 该源域验证窗口集用于开发，且完整性筛选并非随机抽样；不代表独立未来测试集。结果支持“在此范围内，学习的辅助头 zero-fast 诊断预测保真度较差”，不支持“适应后预测已验证”“基模全部能力崩溃”“物理机理崩溃”或 EAST 完整流改善。原风险 gate 未改。

本次没有执行全套测试、GPU、模型推理、训练、原始信号或其他文件编辑；使用 results-analysis 工作流将记录事实、独立复算和不能独立验证的运行过程分开。前一份方法审查未修改；本审查完成后冻结。
