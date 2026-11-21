# 继续运行入口（2026-11-02）

## 当前状态

- HDF5结局权威口径已获用户再次确认；不按shot list覆盖，不按指标剔除冲突炮。
- J-TEXT合格基模、EAST cold20/input_map/阈值/炮序不变；在线适应仅EAST。
- 新训练完成：probe_outputs/innovation_metric_heads20_v1，128步、三辅助头、
  前20炮8D12S、后340炮读取0；263.4289秒，source_tensors_unchanged=true。
- heads SHA256：79e5cae2060683224419382bac86c5d078722032add49d879dd19859af2b3ba7。
- 完整360八臂实验运行中：probe_outputs/innovation_metric360_v2。
  本次最后读取progress为60/360、86.708秒；不是最终结果。
  主终端session_id=33269。先恢复此句柄，不要重复启动GPU实验。
- v1因快照包含仍在编辑的、未被推理导入的verify.py而被主进程停止，exit130；
  ABORTED.md及部分曲线保留。v2科学参数/权重/顺序完全一致，只修复快照范围。

## 完成后最小步骤

1. 读取实际进程退出码及completed.json，不能把progress或已有曲线当成完成。
2. 独立核读创新模块verify.py剩余最终合同部分和测试。Pauli子代理
   01a19c3f-7a5c-7681-a040-5b62d556f97d负责CPU verifier；无GPU权限。
3. 执行python -m experiments.risk_aligned_online.innovation_metric_20261102.verify。
4. GPU空闲后执行同模块.neural_replay，固定74087/73403/63656/62573，
   32整炮曲线及96真实层hook批次；不按表现挑样。
5. 全部核实后在MAIN的docs/research/source_event_rebuild_20261023写中文结果说明。
   预设FPR≤.20/TP不降/FP不增/J提高且超过四控制，未过不能升级。

本轮核心代码core/stream/prepare/run已冻结；不要在运行中修改它们或其导入依赖。
CPU核验脚本不在v2运行快照内，可以完成核验自身代码，但不得改实验合同。
已通过：104项主进程组合旧/新目标测试；新runner19项；CPU verifier开发测试23项
曾通过（最终文件仍需主进程复核）。无提交、无推送、无原始数据改动或删除。

## 已完成上轮结论

critical_shot_meta360_v1：cold125TP35FP，operational125TP34FP，reverse125TP33FP；
TPR均.694444，真实反馈未胜控制，科学验收失败。
MAIN报告：docs/research/source_event_rebuild_20261023/实际时钟炮级目标_完整360对照与损失含义审查_20261102.md。
本轮计划INNOVATION_METRIC_PLAN_20261102.md固定在新训练前；
新损失确实是实际成熟预测误差，不减含可训练预测的hold项。
表示度量步长η1、相对阻尼.01、绝对1e-8、单步隐藏RMS≤.05、总≤.25；
同钟159+128k，首影响160，W每炮清零。仅隔离炮内，尚不证明炮间持续学习。
