# 事件辅助更新失败后的目标分解实验

## Material Passport

- Mode: ARS experiment-agent / run，TDD。
- Status: 执行前固定计划，开发诊断，非独立确认。
- Previous turn: progress；source_event_masked_tasks_v1完整运行并定位到源TRAIN危险损害。

## 只检验两个原因，不改变架构

保持同一c727源模型、零slow、第一FFN rank4临时W、真实事件软上下文、输入与目标
缺测处理、内层metric步及接纳方式。基模不重训，没有J-TEXT在线分支。
固定复用source_event_masked_tasks_v1的435个TRAIN episode、512步实际日程和946个
VAL episode，不重新抽样、不选轮次。所有新臂同形状6×25×23辅助表，零初始化。

1. 对照old_mean_aux：复用上一轮真实事件条件成品及保存输出，目标是平均风险变化
   /.1 + .20诊断误差 + 1e-4表平方。它不是重新拟合，也不是新的独立重复。
2. mean_risk：仅优化两条样本的平均风险变化/.1 + 1e-4表平方。
3. worst_risk：仅优化两条样本中较大的风险变化/.1 + 1e-4表平方。

mean_risk与old_mean_aux对比分离直接诊断维护项的影响；worst_risk与mean_risk比较
分离两类收益相互抵消的影响。两样本max不是总体group-DRO复现，也不保证每个
未来炮不退化。仍使用Adam .001、clip1、seed131、固定512步及原生任务循环。
内层依旧只能用已到达诊断。外层JT历史标签仅用于离线规则教学。

删去外层诊断维护项是病因消融，不是宣称无需诊断预测精度；要并报内层诊断损失。
不得把辅助表系数称为物理定律，或把元学习/分组损失本身声称为项目首创。

## 因果与语义边界

所有源点标签保持max(25ms,AdvancedTime)，严格lead>10ms；EAST规范125ms、严格
lead>30ms未变。VAL危险117点中85点位于(10,30]，30点位于(30,125]，2点>125ms。
后两种时限/标签不等价。额外预定compatible_danger为原danger且30<lead<=125，
保留原标签、不重新挑点；它只能诊断时限一致子集，不是EAST结果。

## 固定评价与晋级

两个新成品在三种视图（native、no_MNM、no_MNM_CIII）、三种反馈（真实、反序、常值）
上评价全部946条VAL episode；共17028条配对记录，不当作独立样本数。
报告安全、危险、early、compatible_danger按炮均值，以及损失/分数变化。
另在全部435条训练episode的主缺测视图只读重放，区分优化失败与泛化失败。

预先指定worst_risk为候选，mean_risk为对照。不根据VAL改选获胜方法。
主视图要求安全、危险均改善、compatible_danger改善，native两类不退化；
真实反馈均衡收益须胜过反序、常值、mean_risk及old_mean_aux。
未通过不晋级EAST360；通过也仍须完整EAST流的独立炮内增益验证。
源端局部BCE、事件F1、更新次数均不替代TPR/FPR。

## 文献定位

- MT3，AISTATS2022：https://proceedings.mlr.press/v151/bartler22a/bartler22a.pdf
  官方摘要及相关方法段已读；支持“训练自监督适应方向”的思路，不保证跨装置可用。
- Sagawa等，ICLR2020：https://arxiv.org/abs/1911.08731
  官方摘要已读；组平均可以掩盖弱组，worst-group仍可能泛化差。本轮只是两样本
  风险变化max诊断，不冒用论文的全组目标或收敛保证。
- PCGrad，NeurIPS2020：https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf
  官方§1–2.1已读。梯度冲突有条件性，本轮不实现或声称复现PCGrad。

## 复现与保护

rl_env CUDA，本机单GPU；原始HDF及已有模型不写。仅读取实际需要的规范TRAIN/VAL
原始炮，逐文件对照上轮绑定哈希；历史事件家族重新用绝对时钟核对。
基模与包装参数前后不变，新臂同日程，配对损失须保留每组梯度并detach参考模型。
保存源代码、协议、成品、每步风险分组量、验证记录及只读训练诊断。
