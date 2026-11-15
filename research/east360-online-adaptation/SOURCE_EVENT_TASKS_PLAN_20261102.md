# J-TEXT事件片段用于EAST更新规则：固定源端诊断试验

## 问题与边界

上一轮 `event_conditional_pilot_v1` 已证明辅助规则的梯度能到达后续风险输出，
但真实事件条件没有胜过打乱标签与无事件条件，且安全片段及30–125ms片段退化。
本轮检验：仅用J-TEXT TRAIN的历史事件片段，教会辅助规则在部分诊断缺失后，
利用已到达的其余诊断改变内部表示。不是重新训练基模，不是J-TEXT在线分支。

源模型SHA保持c7270c1e…，风险头、阈值、归一化和Transformer全部冻结。
源端包装器的持久slow参数为零，不载入EAST冷启动模型。零临时状态须复现源输出。
所有实验属于开发诊断，不构成独立验证或物理规律发现。

## 补充标签的两种实际用途

1. TRAIN事件片段的A–F类别仅作为离线任务分层。按safe/danger平衡，再按已支持
   类别（包括未知-1）、炮、片段均匀抽样，避免长正常片段淹没稀少事件。
2. 训练一个仅看过去128点诊断变化的软类别读出，再条件化辅助诊断损失。
   当前预测永远不输入真实事件类别。未知区间不补正常类，A–F不排序。
   whole-record K6标签上游可能有历史源测试暴露，只能作为开发教师。

## 固定数据与训练

- 复用已审计源端episode索引；不按新方法或损坏视图重新挑风险峰。
- J-TEXT TRAIN1353拟合，VAL160只做固定末步组件评价；不读TEST/EAST信号。
- 标签仍为max(25ms, AdvancedTime)，破裂查询必须lead>10ms，early只诊断不训练。
- 每条episode为origin>=127，maturity=origin+25，query=origin+26…50。
- 三种视图：native、屏蔽MNM、屏蔽MNM及CIII。仅改可用性，不改物理值或标签。
  隐藏通道也从内层未来诊断目标屏蔽，不能用EAST没有的真值指导部署更新。
  这是已知诊断缺测的一种源端教学任务，不假定它覆盖全部装置差异。
- 源事件读出使用 `source_event_context_paired_v1`：两组同初始化、相同批次，
  真实类别/整事件打乱类别，TRAIN拟合1024步；VAL不选超参数。
- 三个更新规则：shared(575参数)、true/shuffled(各3450参数)。后两者容量相同。
  每步所有规则使用相同episode、相同视图；视图按三种顺序循环。
- 固定512步，每步safe/danger各1条，Adam lr=.001，梯度裁剪1，seed131；只取末步。
- 外层目标为 `(后续点BCE-未更新BCE)/.1 + .20*诊断误差 + 1e-4*规则平方均值`。
  外层使用JT历史真实点标签；内层只用已到达25点诊断。
- 内层保持既有第一FFN rank4临时矩阵、隐藏RMS步长上限.05/总上限.25和严格损失下降接纳。
  每条局部episode从零临时状态开始。此处不能宣称完成连续炮内部署。

## 固定评价与晋级

在VAL全部归档episode、三种缺测视图上，分别评价真实/反序/常值反馈。
按炮聚合safe、danger、early的BCE变化；early不并入主收益。
主筛选视图为屏蔽MNM+CIII。真实事件规则需：safe与danger均降低BCE、平衡均值
胜过shared、打乱标签、反序反馈及常值反馈；native两组均不得退化超过1e-6。
若不通过，停止该规则，不消耗一轮360流；通过也只能晋级EAST真实流检验，
不能将源端BCE、事件分类F1或更新次数写成EAST TPR/FPR提升。

## 文献启发而非已有保证

- CoRL/PMLR 2025，Simple Masked Training Strategies Yield Control Policies That Are Robust to Sensor Failure：
  https://proceedings.mlr.press/v270/skand25a.html （已读官方摘要，传感器缺失训练的依据，非核聚变结论。）
- ICLR2025，Test-Time Adaptation for Combating Missing Modalities in Egocentric Videos：
  https://proceedings.iclr.cc/paper_files/paper/2025/file/cdd0640218a27e9e2c0e52e324e25db0-Paper-Conference.pdf
  （已读§3–4。MiDl在完整模态样本上更新；EAST长期缺MNM不满足其直接使用条件。
  本轮借鉴缺测应独立验证这一问题设定，不冒称复现MiDl。）
