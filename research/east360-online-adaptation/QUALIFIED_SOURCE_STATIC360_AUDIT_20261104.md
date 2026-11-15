# Qualified source static360：独立封存曲线与 HDF 结局复核

日期：2026-11-04。**结论：360 个封存文件、1,080 条曲线与原 HDF 结局重算的逐炮及三臂汇总一致。**此结果是固定合格源模型的离线完整流静态诊断，不是 intra 更新收益或独立未来验证。

## 身份、封存与数据边界

- [completed.json](probe_outputs/qualified_source_static360_20261104_v1/completed.json) 的四个制品哈希全部匹配；completed 自身 SHA-256 为 `ef41bc2db11c19d12f7fd2f6b2fea24a5208d0daa4cf5b818c996bf80d66d78f`。逐炮的 360 个 NPZ 均与对应 seal 及 completed 内相同 seal 记录的 SHA-256 一致；360 seal 均记录当前结局未用于预测。逐点验证 map_after20：前 20 炮与 unmapped 完全相等，后 340 炮与 retrospective_map_all 完全相等；每臂共 1,509,354 个封存时刻。
- [code.zip](EXTERNAL_ASSETS.md) SHA-256 为 `149145d79afa1e72eab89c1432f504458ac3e9352e667b4564218751a5639c94`；ZIP 内 117 个代码文件及计划的字节哈希全部匹配 [protocol.json](probe_outputs/qualified_source_static360_20261104_v1/protocol.json)。本制品实际归档的入口为 `experiments/risk_aligned_online/source_anticipation_20261104/static_stream.py`，SHA-256 `030a4b1af040ae742df9e35d51e7b634c10743568995fa1990993cc7dfebced8`；不要求后来当前代码与历史版本相同。
- 归档入口第 82–102、151–203、217–224 行及归档 `redesign_20261004/prepare_prior.py:22–23,54–73` 显示其加载哈希锁定的 source、静态调用、封存后读取当前结局并检查状态；protocol 和 completed 分别记录未载 cold checkpoint、零神经更新，source state 摘要为 `99affd49f71f41129b87b03624201ef94ef6dc44e652eaef411b2924f8186dc8`。这些是历史代码和运行记录证据，不是本审查对当时进程的独立监控；本次未重新加载或哈希源检查点。
- 只按 protocol 的 360 个唯一 shot ID 各打开一次 `data/EAST/unified_hdf5/{floor(shot/100)*100}/{shot}.hdf5`，只读取 `meta/IsDisrupt` 和 `meta/DownTime`：当前为破裂 180、安全 180。三臂保存的 shot 身份／破裂标志与新读 HDF **0 处冲突**；基于 HDF DownTime 和曲线重算 1,080 条逐炮记录，**0 处差异**。安全炮 DownTime 不作为破裂时间。这里未读取旧清单或 CSV，故不声称这些外部来源也无冲突；无报警炮也不能从已存逐炮 lead 单独验证历史 DownTime。为遵守只读元数据范围，未重新计算需要读取整个 HDF 的原始文件哈希。

## 独立重算：首次报警和分段

采用固定概率阈值 `0.7258946895599365`；对破裂炮只考虑 `(DownTime−time)×1000 ≥ −1e−6 ms` 的时刻，对安全炮考虑全部封存时刻；取首个 `score ≥ threshold`。破裂炮首次报警提前量以毫秒四舍五入到 9 位小数后，**严格 >30 ms** 才是 TP；有有效报警但不及时为 tardy，无有效报警为 miss。安全炮有报警为 FP。保留的全部时钟严格递增、曲线有限且在 [0,1]；未在原 HDF 信号属性中重新核验时钟。三臂全部封存分数没有与阈值完全相等的点。

| 分段 | 破裂 / 安全 | 臂 | TP | FP | tardy | miss |
|---|---:|---|---:|---:|---:|---:|
| 全 360 | 180 / 180 | unmapped | 145 | 94 | 21 | 14 |
| 全 360 | 180 / 180 | map_after20 | 116 | 29 | 38 | 26 |
| 全 360 | 180 / 180 | retrospective_map_all | 116 | 23 | 38 | 26 |
| 前 20 | 8 / 12 | unmapped | 7 | 6 | 1 | 0 |
| 前 20 | 8 / 12 | map_after20 | 7 | 6 | 1 | 0 |
| 前 20 | 8 / 12 | retrospective_map_all | 7 | 0 | 1 | 0 |
| 后 340 | 172 / 168 | unmapped | 138 | 88 | 20 | 14 |
| 后 340 | 172 / 168 | map_after20 | 109 | 23 | 37 | 26 |
| 后 340 | 172 / 168 | retrospective_map_all | 109 | 23 | 37 | 26 |

全部 9 个分段的计数、TPR/FPR/J、超过 3 秒提前量计数及记录的理想提前量敏感性，均与 [summary.json](probe_outputs/qualified_source_static360_20261104_v1/summary.json) 逐字段完全一致。以全 360 为例，map_after20 相比 unmapped 少 65 FP，同时少 29 个及时 TP、tardy 多 17、miss 多 12；不能只报告单一指标。一个实际边界例是 shot 71013：两条映射臂的记录 lead 恰为 30.0 ms，因此是 tardy，不是 TP，见 [per_shot.json](probe_outputs/qualified_source_static360_20261104_v1/per_shot.json)。

归档 `simple_online.py:61–95` 与 `simple_data.py:41–48` 表明评分从第一个采样端点开始，不足 128 点的窗口左侧填充并保留可用性掩码。本次从封存曲线独立计数：首个有效报警落在索引 0–126 的破裂／安全炮，unmapped 为 50／44，map_after20 为 16／8，retrospective_map_all 为 15／4。这是评分覆盖的事实，不单凭此归因于填充或推断物理过程。

## 解读与限制

- map 是已准备的数值输入映射，不是经认证的物理标定；本轮未重新拟合前 20 炮的 map，也未独立读取其外部文件复核拟合过程。after20 只核实了既存曲线的应用边界；map_all 从第一炮使用后验准备的 map，前 20 的结果是回溯诊断，不能冒充冷启动可部署成绩。原 360 炮及顺序已参与开发，后 340 也不是独立未来集。
- 归档计划第 17 行的 oracle 提前 5–50 ms 使用真实结局，是非因果的时间敏感性检查，不是部署策略、方法收益或严格全局可救上限。首个报警、严格 30 ms 和 preterminal 容差定义均不能替换为报警窗口内任一点或其他阈值规则。
- 用户提及旧 `125 TP / 35 FP` 属于带 slow 64×64 持久状态的旧父谱系，不能称作本轮纯源基线；本审查未打开旧产物独立复核其 FP。归档计划第 19 行另记 `125 TP / 35 tardy / 20 miss`，其中 tardy 与 FP 是不同量，不能混写。
- 哈希证明现存产物的内部一致性，不单独证明历史执行顺序或未改写的外部来源。没有信号重放、GPU、训练、其他 HDF 字段、全盘重扫或老 cold checkpoint；不作在线效果或物理机制结论。采用 results-analysis 区分独立数值复核、历史运行记录及未验证假设。本文件完成即冻结，先前 fidelity 审查未修改。
