# 全部已接受更新的炮内贡献检查

## Material Passport

- Origin Skill: results-analysis / tdd-workflow
- Origin Mode: run / validate
- Origin Date: 2026-10-31
- Verification Status: PLANNED_ORIGINAL_HISTORY_CENSUS
- Version Label: paired_gap_census_v1

前一轮不是只有65067的总损失数值：原完整流及680次更新已精确重放并保存每炮前状态。本轮对原both轨迹中全部57次已接受更新，用其当时的历史8炮和固定源事件回放，分别比较开启/关闭炮内的组损失。不使用任何未来病例或未来标签构造检查量。

原运行：stratified_cold20_context360_v1；状态来源：slow_fast_attribution_v1，completed SHA bbbea164486e5f515942e4e078052a504016d67244db4f3e1e15bb0504700f79。所有数据已有开发暴露。65067已知病例用于核对重复计算，不重新选择容差。

三层嵌套检查（固定容差1e-7，沿用原数值容差，不扫描）：

1. original：各组开启炮内损失不增加。
2. dual_absolute：在1基础上，各组关闭炮内损失也不增加。
3. paired_gap：在2基础上，各组“开启−关闭”的差值也不增加。

组包括原历史安全、历史破裂（存在时）及source_event，缺失组不虚构为0。原先拒绝的283次没有保留候选矩阵，新增条件不可能在同一原状态下把它们转为接受；本轮只对57个已接受候选检查。新轨迹会改变候选和历史，不能把此次满足数当新策略实际接受数或性能。

训练窗口仍是当时既有采样，使用历史原sealed预测形成hard windows；不重新按新模型风险选窗口。已接受更新前/后的开启损失须逐项等于原guard_before/after。前20数据只在已完成冷启动准备的历史中使用；后续各更新只用其前8炮。最后一次若已接受，用原最终检查点作after；它没有后续评分作用，需明确标记。

输出完整57条身份、历史、各项前后损失、差值及检查结果，全部聚合计数，不只报65067。先判明各条件是否几乎阻断所有学习，再决定是否做独立前向运行；不将候选条件直接晋升主线。未改变预测、原始数据、J-TEXT基模、阈值或原学习历史。

本机rl_env CUDA，外部timeout900秒，隔离输出paired_gap_census_v1；先运行纯数值规则测试，再核对真实原保护损失及65067的已存诊断。
