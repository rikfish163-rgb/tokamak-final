# 新副本的任务梯度相容性：离线CPU诊断

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent; results-analysis
- Origin Mode: run
- Origin Date: 2026-11-07
- Verification Status: PROTOCOL_FROZEN_BEFORE_DIAGNOSIS
- Version Label: cta_copy_gradient_v1

固定已完成的两个副本和原教师，不训练、不用GPU、不改阈值。仍用原源 VAL170 与已揭示 EAST20 的100窗。内层梯度只由原点过去窗口和已经到达的25点反馈计算；后续风险梯度只作离线审计，绝不传给实际更新器。

对共享编码器参数计算 g=grad L_signal 与 r=(1-2y)grad z_query。r 与二分类BCE梯度方向相同而去掉置信度倍率。若 g·r>0，小幅负梯度步一阶有利；若 g·r<0，一阶有害。报告夹角余弦、零梯度、按物理炮等权的有利比例及与已封存风险改动符号的关系，不将梯度夹角直接等同完整流收益。

此诊断用于区分‘1%步幅过大’与‘即便很小的反馈下降步也朝错误风险方向’。它只评价当前已训练副本，不能证明所有自监督任务不可用。整个数据流已有开发暴露，没有独立未来确认。

最多两CPU线程、300秒；只新增小型JSON，预算1MiB。为与用户本机Codex协调，GPU回溯诊断暂不启动。
