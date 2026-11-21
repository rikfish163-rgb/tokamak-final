# 炮内内部更新：本轮定向文献核对

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / supporting source check
- Origin Date: 2026-11-02
- Verification Status: ANALYZED
- Version Label: lora_stream_primary_evidence_v1

这是本轮两项定向证据核对，不是新的100篇系统综述，也不声称通读了所有检索结果。它不改变已经启动的实验参数。

## 1. 辅助损失下降不等于主任务改善

Sun等，ICML2020，Test-Time Training with Self-Supervision for Generalization under Distribution Shifts。

实际读取：作者在PMLR的正式PDF，第2节方法和第4节Theorem1及其邻近讨论。该定理在主任务损失可微、凸、光滑和梯度有界等条件下，以辅助梯度与主任务梯度的正内积给出一步收益的充分条件。不能将它当作本项目非凸Transformer、逐窗口限幅和有限步更新的性能保证。

项目可借鉴的检查是：在已完成放电的事后分析中，固定同一参数状态，比较信号预测梯度与后续监督风险梯度；以及直接比较更新前后的后续风险损失。这是诊断，不允许在当前炮选择更新时读取其真实标签。不同反馈臂在不同参数状态的梯度余弦，不等同于同状态因果比较。

来源：[PMLR正式PDF](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf)。

## 2. 数据时间上的提前量不等于实际部署的及时性

Sreeram、Kwon、Mascolo，ICML2026，Tempora: Characterising the Time-Contingent Utility of Online Test-Time Adaptation。

实际读取：PMLR官方摘要与书目信息，未将其完整方法或所有实验标为已读。摘要明确研究TTA的准确率与延迟权衡，并针对硬截止等部署约束设计评价。本轮只借此确认：真实部署中应另外评价计算延迟，不能只报告离线流的预测精度。

项目当前360炮属于数据访问因果的离线回放。32端点批处理是一种计算实现，并未证明每个1ms采样点都能即时得到预测；炮内反向传播也有耗时。本轮总运行时间不能替代逐端点延迟、队列积压或截止时间测试。因此，结果即使达到TPR/FPR门槛，也不能直接写成“满足工业实时部署要求”。

来源：[PMLR官方页面与摘要](https://proceedings.mlr.press/v306/sreeram26a.html)。

## 未纳入正式方法依据的检索材料

OpenReview上DISTA相关TMLR2025 PDF的索引显示了附录A.2的未来标签诊断；直接打开正式PDF遇到站点验证，attachment也未能读取。因此不将该论文全文或主方法标为已核验，也不据索引片段移植算法、宣称保证或认定新颖性。本轮没有绕过站点验证。

低秩内部适配、固定SVD基、去均值预测误差和梯度归一化均不能仅凭本工程组合就声称首创；真正需要证明的是：目标装置已到达信号的更新，在原先正确炮不被大量改坏的前提下，改善后续预警。
