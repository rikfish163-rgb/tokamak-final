# 原始输入正倍率：参数范围×训练目标的完整流对照

## Material Passport

- Origin Skill: academic-research-suite / results-analysis / tdd-workflow
- Origin Mode: run / validate
- Origin Date: 2026-11-01
- Verification Status: PLANNED_DEVELOPMENT_EXPERIMENT
- Version Label: operational_input_gain_v1

## 已有失败与新问题

本轮不是第一次提出整炮峰值损失。旧shot_objective_alignment_20261023曾在外部20炮训练encoder并比较mean/safe_peak/peak；单密度监督拟合也曾使误报由27升到88。最新正常参考统计对齐4臂全失败，FPR0.43–0.50。本轮具体改变参数空间及准备样本合同，检验在源网络完全固定时，受限正倍率与操作性训练目标是否相容，不将有监督本身当作答案。

## 完全固定的合同

源SHA c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02；源网络、归一化缓冲区、风险头、报警阈值0.7258946895599365不改。EAST同一stratified_cold20_context360_v1完整360身份和顺序，180D180S；前20为8D12S，自身7TP6FP计入，不提前享有新映射。前20结束、封存、揭示后才准备，第21炮起固定使用；不读后340标签作训练或选择。此顺序为离线初始电流代理排序模拟，不是自然历史未来验证。18个争议标签仍按HDF5主合同，不筛掉。

## 2×2实验

保留source直推和原density映射两基线。四候选为density_mean、density_peak、structured_mean、structured_peak。

- density：只拟合密度倍率的1个对数调整量。
- structured：拟合9个对数调整量，作用于10列：密度、CIII、Mir_VV、Mir_avg_amp、n=1幅值、dx_a、dy_a、SXR响应和XUV响应。SXR响应使sxr_mean乘g、sxr_var乘g²；XUV响应只使xuv_var乘g²，其他XUV无量纲形状统计不变。这是共同乘性响应的明确假设，不是认证过的仪器标定。不同Mirnov特征不强行共用倍率。
- Ip、Bt、qa_proxy、v_loop、Z_proxy、频率、模数、偏度、峰度和比值不拟合。50个源炮和前20个目标炮的小范围检查并未支持Z_proxy严格等于v_loop/Ip，故不以该等式构造约束。
- 所有倍率严格为正、无加性偏移、不混合通道，保持零值和符号，避免上轮负方差及asinh空间超线性还原。初值仅密度用既有20炮经验倍率，其余为1；theta初值0。

在同一端点集合上比较mean与peak。安全炮集合为所有保存的有效时间点，包括短前缀；破裂炮为30ms<lead≤125ms的原正标签时间点。没有合格端点就显式失败，不删除炮或用其他炮替换。设固定报警logit门槛为b：

- 安全mean：平均softplus(z−b)；安全peak：softplus(max(z)−b)。
- 破裂mean：平均softplus(b−z)；破裂peak：softplus(b−max(z))。

每步同一随机安全炮与破裂炮，各权重0.5。四臂共用256个已预定配对，随机种子2131。只对输入theta执行Adam，lr0.03，梯度总范数裁剪1，固定0.01×sum(theta²)正则；系数是开发选择，不扫描。最后一步固定导出，不按后340结果或训练最佳点挑检查点。峰值扫描与重算使用相同32窗口批形状，必须逐位复现选中logit；平均版本对完整集合累积梯度，峰值版本不保留整炮计算图。不同目标计算成本如实记录，不声称算力完全相等。

所有源参数requires_grad=False且state逐位不变；所学输入参数只用于EAST。新候选没有炮内、炮间神经更新，不作为核心炮内目标完成。若基础结果有益，下一阶段仍须从同一模型分离真实炮内贡献。

## 证据与执行

现有9项核心CPU测试先缺模块失败后通过，包括正值/零/缺失保留、排除列不动、SXR矩关系、mean/peak分块损失梯度与全批一致。完整执行前做2步真实GPU预检，仅检查数值、梯度、原基模不变及时间开销，不看后340表现、不改参数。

完整运行外部timeout3600秒，本机rl_env CUDA；各20炮当前预测封存后读结局。记录全部4×256步参数、损失、峰值端点、更新前后状态；保存若干优化器状态供逐步复核。输出6条完整流曲线，重复核算所有炮级指标并重放选定真实曲线。发生崩溃保留原目录和日志，不能把部分流成绩当完整结果。停止条件不是“调到好看”，而是按固定协议执行并审查是否满足FPR≤0.20且保留/增加及时检出。

阈值校准、源重训、额外EAST准备炮、未来标签、最终风险附加修正均不在本轮。所有数据已经参与开发；报告单种子开发对照，不作独立未来或新物理规律声明。
