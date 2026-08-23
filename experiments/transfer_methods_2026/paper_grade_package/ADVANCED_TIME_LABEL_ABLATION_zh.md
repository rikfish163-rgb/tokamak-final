# AdvancedTime 正样本上界对照实验（seed 131）

## 结论

本实验严格只改变破裂炮正样本窗口的上界，其余模型、数据、训练预算、损失权重、随机种子、阈值规则和测试顺序保持不变。结果不支持用 `max(25 ms, AdvancedTime)` 替换 J-TEXT 当前的固定 125 ms 正样本区间。该规则应保留为诊断或辅助监督候选，不应成为当前迁移主标签。

EAST 的 `max(125 ms, AdvancedTime)` 在 support110 上只改变两个 HDF5 破裂炮，新增 34 个正端点，因此作用很小，也没有产生稳定增益。

## 标签合同

破裂/非破裂结局只读取 HDF5 `meta/IsDisrupt`，破裂时刻只读取 HDF5 `meta/DownTime`。AdvancedTime JSON 只为已经由 HDF5 判定为破裂的炮提供正样本上界，不能改变炮级结局：

\[
L_i^{\mathrm{J}}=\max(25\ \mathrm{ms}, A_i),\qquad
L_i^{\mathrm{E}}=\max(125\ \mathrm{ms}, A_i).
\]

若炮号不在 JSON 中，则分别回退到 25 ms 和 125 ms。JSON 数值本身就是毫秒，不做乘 1000 的转换。原工程 JSON 与 GitHub 仓库镜像逐项解析后完全相等；两个文件的字节哈希不同仅因为仓库镜像补了末尾换行。

## 2×2 因子实验

四组实验分别为：固定源/固定 EAST、动态源/固定 EAST、固定源/动态 EAST、动态源/动态 EAST。固定源臂复现了现有 PADFIX 父模型的 encoder 位级哈希，说明对照链没有改变模型训练本身。

| J-TEXT 标签 | EAST 标签 | 正端点数量（J / E） | support110 阈值下 raw TPR | TPR@lead≥30 ms | FPR | AUC |
|---|---|---:|---:|---:|---:|---:|
| 固定 125 ms | 固定 125 ms | 21,912 / 1,867 | 0.7889 | 0.5278 | 0.0611 | 0.9265 |
| 动态上界 | 固定 125 ms | 5,557 / 1,867 | 0.7167 | 0.5056 | 0.0778 | 0.8989 |
| 固定 125 ms | 动态上界 | 21,912 / 1,901 | 0.7722 | 0.5111 | 0.0667 | 0.9249 |
| 动态上界 | 动态上界 | 5,557 / 1,901 | 0.7222 | 0.5056 | 0.0778 | 0.8967 |

上述阈值全部只由 EAST support110 选择，balanced360 没有参与训练或阈值选择。由于 support110 阈值按炮级最大分数满足 TPR=0.8，它并不保证 30 ms 提前量，因此四组在 operational TPR 上都触发了既定的低于 0.60 异常审计。这是现有 per-shot-max 阈值合同的共同限制，不是动态标签独有的问题。

作为另一种诊断口径，若在 360 炮预测全部封存后，直接选择满足 `TPR@lead≥30 ms=0.8` 的阈值，固定/固定组 FPR 为 0.1944；动态 J-TEXT 组的 FPR 上升到 0.3000–0.3056。该口径使用了 balanced360 结局选阈值，只能用于机制比较，不能作为部署结果。

## 机制解释

J-TEXT TRAIN1353 中有 188 个 HDF5 破裂炮，187 个能在 JSON 中找到 AdvancedTime，但只有 37 个值大于 25 ms。动态规则把正端点从 21,912 减到 5,557，减少 74.6%。它提升了 TEST220 的终端分类 AUC和 raw TPR，却把中位提前量从 117.0 ms 压缩到 44.1 ms，并将 `TPR@lead>10 ms` 从 0.8545 降到 0.8182。这表明模型更擅长识别破裂末端，却丢失了更早的演化监督。

这个损失在迁移后进一步表现为 AUC 下降约 0.028、raw TPR 下降 0.05–0.072，以及相同 30 ms operational TPR 下明显更高的 FPR。seed 131 的配对结果中，固定源独有的正确破裂报警为 16 炮，动态源独有为 3 炮，精确二项检验 `p=0.0044`。该统计仍是单 seed、开发流上的诊断证据，不能替代多 seed 独立确认。

## 裁决与后续

当前主线继续保留固定 125 ms 正样本上界。AdvancedTime 若继续使用，建议作为 horizon/onset 辅助头或分阶段权重，而不是删除 25–125 ms 的前兆正监督。这样可以利用其逐炮时间信息，同时避免把源模型收缩成“只识别临破末端”的分类器。

可复现入口为 `experiments/transfer_methods_2026/run_advanced_time_label_ablation.py`；固定合同见 `paper_grade_package/configs/advanced_time_label_ablation_v1.json`，机器可读结果见 `paper_grade_package/evidence/advanced_time_label_ablation_seed131.json`。
