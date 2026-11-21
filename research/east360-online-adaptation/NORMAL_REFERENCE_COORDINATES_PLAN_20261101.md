# 同一对齐映射的直接模型坐标接入

## Material Passport

- Origin Skill: academic-research-suite / tdd-workflow / results-analysis
- Origin Mode: run / validate
- Origin Date: 2026-11-01
- Verification Status: PLANNED_DEVELOPMENT_NUMERICAL_REPAIR
- Version Label: normal_reference_coordinates_v1

原normal_reference_alignment_v1保留，不重写。四候选原先在source-asinh坐标拟合，随后sinh还原raw float32，再让同一个模型asinh。CIII在73403/63656被映射到约93/103，raw还原超出float32。这包含接口中不必要的逆变换问题，不能单据它判定统计映射本身不成立。

此次保留原统计、20炮样本、四张映射、源模型权重、阈值、掩码、所有超大坐标；不裁剪、不换参考样本、不重新调ridge。仅将映射后的坐标直接送入原输入投影层之前的对应数值槽。通过有作用域的embedding pre-hook操作，不替换Transformer、风险头或预测函数；调用后清除hook，异常也清除。MNM与mode以及不可估通道仍取原前向生成的数值，缺失和填充保持原样。

必须验证：原坐标直接接入与原模型logit/forecast逐位一致；可逆范围内与旧raw路线在浮点误差范围内一致；大坐标不经sinh依然有有限输出；掩码、未选择通道不受影响；异常清理。合成测试不代替真实完整流比较。直接接入与旧路线可能有float32舍入差异，完整记录差异和报警翻转，不冒称所有候选bitwise等价。

再跑同一完整360、前20成本仍计入，不改变四映射的可用时点：第20炮封存后才加载其统计映射，第21炮起使用。source/density两基线逐点与旧记录一致。所有6臂完整计分，不因过大坐标删除炮。前版本失败臂的已有前缀仅用于数值等价审查，不作为完整流成绩。

本机rl_env CUDA，timeout1800秒；新目录normal_reference_coordinates_v1。无神经训练，无源权重改变；这仍是基础输入处理诊断，不是炮内增益。所有数据参与过开发，结论不能作独立确认。统计变换可能破坏物理正值或多通道关系，绕开数值溢出并不能消除这些问题；本轮将记录负方差对应坐标及最大绝对坐标。
