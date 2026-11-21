# 协议文字更正（保留原始运行）

2026-10-05；依据独立时序审查16。原`protocol.json`、`summary.json`、候选模型及`code_snapshot.zip`均保留，不重写其哈希。

1. 原协议继承的`outer_positive: source D >30 to <=125ms auxiliary bag`是旧片段目标描述，不对应本轮whole训练目标。实际`partial_intra_20261005/source.whole_risk`调用`decoder_intra_20261005/whole_meta.select_peak`，对破裂炮选取所有严格lead>30ms有效端点，没有125ms上限。`objective`字段已正确写为whole-shot。源分类器原点标签未改。
2. 仅保留的later开发诊断沿用旧episode查询区间>30且<=125ms；新实现使用实际已发曲线，不以末态重算历史。
3. 新whole/later损失均经已发float32概率的有效float32端点裁剪，再取logit与固定阈值的相对softplus。旧later直接用raw logit，饱和时不可直接等同。本轮旧参照未触发该裁剪，且新P/D及signal/risk对照为同一口径；不变更已报告实验成绩或准入结论。

更正仅涉及描述和口径边界。后续`prepare.py`明确覆盖`outer_positive`并新增`later_positive`与`probability_loss`说明；不据此称重新完成一次独立训练。完整结果仍为所有20/40/60检查点未准入，未执行新的EAST回放。
