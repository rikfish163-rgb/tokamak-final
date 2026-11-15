# 固定预期 + 炮内偏差：源接口配对（B阶段）

A阶段固定epoch10的诊断预测组件已通过预先约定的预测检查，不代表风险判断改善。
本阶段固定其参数，比较F=0和F=已学习预期；两臂同一初始化、源分类器、阈值、样本顺序。
同一46维条件key、8维临时状态、unit-ball更新、内部token注入位置，不增加输出分数修正。

每个已到达片段j：从此前封存的起点j-h取F预测，目标为实际变化减该F预测。
仅残差预测B(current)c作为内部神经修正；完整诊断预测为F(current)+B(current)c。
F固定且无梯度，c每炮从0开始；跨炮训练只教学key、kernel、步长、内部读出。
源分类器、slow模块冻结；本阶段没有EAST在线或J-TEXT在线部署。

训练固定2轮，seed131，Adam .001，8炮梯度累积，clip1；完整TRAIN1353参与风险教学，
仅有完整未来4片段时才有预测目标。沿用bag+.5 local+.5 detached-direction+1 forecast。
两臂相同名义/可训练参数量，F在零预期臂不参与计算。固定epoch2，不选择最优轮次。

VAL160报告逐炮memory/off/reverse/hold风险损失；都只是采样端点诊断，不是TPR/FPR。
reverse打乱已到达目标的时距对应关系，保持原时距的F和basis，并匹配状态范数。
hold将实际变化设为0，因此F非零时误差为-F；它不再是零状态对照，off才是。
单独报告真实反馈是否改善F-alone未来预测，不能把F自己的收益算成fast收益。

组件检查：memory的bag/local均优于off、reverse、F=0臂；D/S的两项均不高于off；
fast预测总体优于F-alone，并至少81/160、41/80安全炮改善。这是保守机制检查，
不是最终用户验收；未通过不等同于EAST不可能成功，不靠修改阈值掩盖失败。

所有代码/计划/输入检查点hash记入protocol，新结果expected_residual_pair_v1。
不覆盖旧结果，不复制大缓存，不提交或推送。
