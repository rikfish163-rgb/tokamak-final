# 持续运动探针：预检停止，尚无拟合或评估结果

## Material Passport

本次是限定在原 warm20 与已封存流索引 21–100 的 CPU 离线诊断。预先固定的方案见 `probe_outputs/joint_lora_persistent_motion_probe_20261104_v1/protocol.json`。进程在第一个 warm 炮的输入一致性检查处退出；**没有拟合任何读出，也没有评估后 80 炮，因此不能报告 BCE、AUC 或运动特征是否有增量。**

已先核对首炮原 seal、曲线 SHA、原始 HDF SHA，以及 v5 的 HDF 结局、采样时钟、大小与修改时间。进程加载了 v5 元数据清单并对既有案例文件计算哈希，但没有读取后 80 炮的 HDF 信号或结局，也没有使用其标签拟合。既有子空间产物未修改。

## 已实测的不一致

原 producer 的 `prepare_actual` 对流索引 1–20 不应用 density map，对索引 21 起应用该 map。本探针原计划为了让 warm 拟合和后 warm 评估使用同一视图，对两者均调用索引 21 的映射；但预检错误地要求这个映射视图的 native 分数复现原始未映射 seal 曲线。

对原始首炮 **71304**、共同成熟点 `t=135`，同一冻结 native 源模型的实际 CPU 结果：

| 输入视图 | native 概率 | 与旧封存曲线的绝对差 |
|---|---:|---:|
| 原 warm 索引 1，不应用 map | 0.894742966 | 0.000000060 |
| 索引 21，应用固定 map | 0.936777174 | 0.042034149 |
| 旧 seal 的 native 曲线 | 0.894743025 | — |

这证实首点失败来自**两个不同输入视图被当成同一曲线比较**，并不证明真实或置乱运动特征有效或无效。该数值只实测了首炮首点，不能泛化为其余炮的差异大小。

## 建议的修正后重跑方案

1. 保留原 seal 和 HDF/v5 真值、时钟、raw SHA 检查。额外使用该炮原 producer 的历史输入视图核对 native 曲线：warm20 为未映射，21–100 为映射。
2. 单独构造分析视图：warm20 与后80均使用同一固定 map，在该视图上计算冻结 native logit、时刻水平、8/32 点运动及全部匹配置乱控制。明确不把映射 warm 分数描述为旧未映射曲线的重现。
3. 先写入修正后的固定 protocol 和回归测试，再执行 warm20 拟合、冻结读出，最后读取并评估后80。保留此次失败的 protocol 和本报告，不覆盖历史输出。

按研究执行技能的失败处理规则，本报告停在这里，等待用户决定是否按该修正方案重跑。

## 文献动机，不是本尺度的证据

- [Zheng et al., *Communications Physics* 6, 181 (2023)](https://www.nature.com/articles/s42005-023-01296-9)讨论 J-TEXT/EAST 迁移及时序尺度差异；原文提及“different temporal features with different time scales”。
- [Montes et al., *Nuclear Fusion* 61, 026022 (2021)](https://doi.org/10.1088/1741-4326/abcdb9)的[作者单位公开稿](https://dspace.mit.edu/entities/publication/e800cf5e-03cf-4caf-a394-af232f5bce5b)描述从放电中“detecting distinct events”。

两篇只为检查时序和事件演化提供动机，均不证明这里的 8/32 点尺度或特征有用。来源文字由当前会话核对；当前没有独立引文质疑子代理，不能声称独立复核。
