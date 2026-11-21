# 成熟诊断误差与表示空间步长：固定开发实验

## 问题及范围

critical_shot_meta360_v1 已完成：cold125TP/35FP，operational125/34，reverse125/33。
该方法不合格。本实验不回写旧结果、不改变已达标J-TEXT权重、不调整报警。
只检验EAST炮内适应；不是炮间持续学习结果，不是独立未来确认。

## 固定合同

- HDF5 IsDisrupt/DownTime为唯一结局依据；EAST125ms点标签不变，及时报警严格lead>30ms。
- 相同source/cold/input_map/360顺序/前20炮成本/阈值0.7258946895599365。
- 前20炮结束后拟合诊断辅助系数B(25×23)，后340炮冻结B。
- 源网络、风险头冻结；仅本炮第一FFN低秩W(4×64)变化，炮后清零。
- 相同159+128k成熟时钟、128点因果输入、25点一次性诊断预测；先封存再读已成熟目标，第一次影响160。

## 两个可检验变化

1. 真正预测误差：centeredSmoothL1(auxiliary_prediction, matured_truth) +0.01 mean(W²)。
   不再减去含同一可训练预测值的hold损失；数据项在准确预测时梯度为零。
2. 参数单位步长对照：旧的单位梯度方向×0.1，保持不变。
   表示度量步长：C=phiᵀphi/(有效token数×64)，
   G=C+(0.01 trace(C)/4+1e-8)I；dW=-G^{-1}grad（η=1固定），
   在当前已观察窗口将dW诱导的隐藏RMS限制在0.05以内。
   原有总隐藏修正RMS上限0.25保留。不是输入尺度CORAL，也不宣称完整坐标不变性。
   同样只在本次成熟数据目标下降时接受；不证明未来风险下降。

三种辅助头均从B=0、Adam0.001、128步、clip1、seed131训练，使用相同每步1D+1S样本：
unit_operational、metric_operational、metric_forecast。
完整炮累积回放；关键点选择与旧实验相同；最后一次更新TBPTT1。
operational外目标=(固定报警余量BCE_after-BCE_before)/0.1+0.20诊断误差+1e-4mean(B²)。
forecast对照只拟合诊断误差+同样B正则。
训练预算及参数个数相同，不以360结果挑epoch或学习率。

## 完整360实验臂

cold；旧contrast_unit（原operational辅助头，原规则）；
innovation_unit；innovation_metric；metric_forecast；
metric_zero（B=0）；metric_reverse（同metric头，已成熟25点逆序）；
metric_hold（同metric头，当前观测保持不变的对照目标）。
所有目标共用原有实测/逆序可用性交集掩码。
每个臂完整跑340后续炮；前20炮保留原封存结果，计入360总体。

## 验收与失败处理

必须FPR≤0.20，TP不少于cold且FP不更多，J严格提高；
metric必须严格优于reverse、hold、forecast和zero对照，才允许称本阶段可归因的炮内收益。
本轮不能凭单炮改善称显著或物理规律。未过不升级，不调报警补救。
输出每炮完整曲线、临时状态/真实梯度、接受记录与封存hash。
代码测试先验证准确预测零数据梯度、误差趋零时度量步长趋零、RMS约束、
无目标不更新、因果时钟/前缀不回写，再做真实GPU预检与完整试验。
固定四炮74087/73403/63656/62573进行真实hook重放核对，不按改善选样。
