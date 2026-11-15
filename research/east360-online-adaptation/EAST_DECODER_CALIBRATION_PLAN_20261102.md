# EAST20：小容量直接诊断预测器的炮内规则校准

## Material Passport

ARS experiment-agent/run，TDD；执行前固定计划，开发诊断，不是独立确认。
上一goal turn为progress：两项源端试验及EAST20冻结诊断均真实完成且未过gate。

## 假设

源风险元教学不可靠地推广到EAST。保留JT forecast_only诊断头作物理测量任务先验，
仅用已经揭示的EAST冷启动20炮，训练如何用已成熟诊断修改内部表示。
不修改qualified c727基模、原forecast_head、risk_head、cold20、输入映射、阈值。
不读后340炮原始数据；没有J-TEXT在线分支。

先验是上一轮direct_decoder_source_v1/decoders.pt的forecast_only头，非失败risk-meta头。
固定取该575×1088权重前4个右奇异向量A；外部头输出prior(x)+(xAᵀ/2)Cᵀ。
C为575×4，2300可训练参数；先验与A冻结，C初始0。不是最终风险残差。
另一个4×64 W是每episode重置的首FFN临时参数，两者位置与作用不同。

## 匹配实验

20炮8D12S，复用原100个query与固定四折，每折15fit/5held。冷启动父模型早见过
全部20标签，因此这只是规则的按炮开发检查，不能称全系统独立留出验证。
拟合只用安全片段，以及破裂炮30<lead<=125ms的片段；保留125ms点标签。
较早D150/250ms片段不进入训练风险目标，但在held诊断中单列并评价总体D。

三臂forecast_only/mean_meta/worst_meta；同初始化、同折内样本序列，seed131。
每折128步，每步等炮抽1安全和1危险，再抽该炮合法query；Adam .001，clip1，仅末步。
共同目标.20绝对诊断SmoothL1+1e-4mean(C²)；meta另加两类风险变化的mean或max，/.1。
风险在固定报警门槛logit余量坐标计算，参考detach；max为两样本估计，非全局DRO。
内部W的更新完全复用direct_decoder.core.advance：仅25点已到达信号、中心化损失、
固定metric步、严格下降接纳、现有隐藏RMS上限。无当前炮结局或真实事件类别输入。

held共100query，三个拟合头+未校准先验，各real/reverse/hold，共1200配对比较。
按炮汇总D全40点、安全60点、D30–125ms24点；不作TPR/FPR或整炮首次报警声称。
预先候选worst_meta需三个组均改善，均衡结果胜forecast_only/mean_meta/prior及本头
reverse/hold，才有资格另做冻结全部20拟合与完整流。不能根据held换赢家或调预算。
若未过，保留失败并定位，不启动完整360来掩盖炮内失败。

## 验证

验证C=0预测精确等于先验，only C训练，先验/源/cold全不变；校准参数到更晚风险
的真实二阶路径通过有限差分/FFN插入检查；读文件限制20炮，HDF结局只校验已揭示
元数据。记录每折fit/held、实际训练样本、代码与先验哈希、参数和全部比较结果。
先验来自JT事件分层教学；A–F仍不是人工物理模态，也不声称学得新物理规律。
