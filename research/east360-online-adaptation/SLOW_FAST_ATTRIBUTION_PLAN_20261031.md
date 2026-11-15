# 炮间学习后，临时更新为何新增误报：固定历史的部件对照

## Material Passport

- Origin Skill: results-analysis / tdd-workflow
- Origin Mode: run / validate
- Origin Date: 2026-10-31
- Verification Status: PLANNED_DIAGNOSTIC_REPLAY
- Version Label: slow_fast_attribution_v1

## 问题

stratified_cold20_context360_v1中，64562、83376、94402三个安全炮在both_shadow正确、both误报；初始cold+intra没有这些误报。固定当前序列、父源模型、输入映射、阈值及原学习历史，分离慢矩阵S和临时状态读出O。

同一个已观测首层token h和因果临时预测delta，经 h_last + S h_last + O delta 进入冻结第二层、norm和风险头。O不是最后风险分数补丁，而是92维临时预测到64维内部token的线性读出。原模型非线性，因此不能预设S与O的影响相加。

## 方法与八条曲线

C是cold20状态，B是该炮开始前的原both状态，I是原inter状态：

1. cold_off：S_C，关闭delta；应逐点等于原cold。
2. cold_on：S_C、O_C；应逐点等于原intra。
3. both_off：S_B，关闭delta；应等于原both_shadow。
4. both_on：S_B、O_B；应等于原both。
5. both_slow_cold_readout：S_B、O_C，单独把临时读出换回cold。
6. cold_slow_both_readout：S_C、O_B，单独保留已学习读出。
7. inter_off：S_I，关闭delta；应等于原inter。
8. inter_slow_cold_readout：S_I、O_C，检查未用fast教学的持久分支上启用原临时作用。

所有新组合只作诊断，不反馈到训练。完整重放原始20步cold准备及后340逐炮两个更新；每一步损失、梯度范数、接受/回滚记录须与原回执精确一致，最后状态与原检查点逐张量一致。原模型曲线也须精确一致。**这不是用混合曲线重新学习得到的新可部署算法**，不得称新策略在线收益。

保存每炮开始前的持久状态、全部曲线、更新回执、模型/原始数据/代码哈希。先保存该炮曲线再揭示结局并执行原更新。当前结局不控制部件选择。前20全部沿用原源预测并计入全流；cold状态只在第20炮结束后产生。完整360身份与原HDF5标签保持不变。

## 分析与判断

对全部360计算相同严格lead>30ms、首次报警、FPR及J；不只报告三个错误炮。对prior三个误报及60239，报告各组合首次报警、峰值、同一参考端点的logit和内部修正范数。非线性交互按四种S_C/S_B与O_C/O_B的logit差分计算，仅为代码层诊断，不作物理因果解释。

如果更换O可消除误报但更换S不能，支持读出漂移在该父状态下的重要作用；若只有共同变化出错，则需要耦合解释。即使整流诊断组合较好，仍需另行运行独立学习历史的前向对照。原阈值、标签、更新参数不改；不允许事后挑最佳组合替换原主线。

单个已开发流、单源seed，无独立未来验证；不声称物理规律学习。若重放不一致则停下检查实现/版本，不放宽容差使其通过。外部timeout1800秒，本机rl_env CUDA，主进程独占实验调度。现有模型/结果不覆盖。
