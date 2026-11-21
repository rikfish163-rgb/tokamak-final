# 当前继续入口：完整风险路径与匹配forecast控制

## 活动目标与权限

目标仍未达成：真实可归因的EAST炮内适应收益，随后炮内+炮间比较。
不以导数正确、训练完成或单炮边界翻转代替验收；现有数据全是开发数据。
HDF5 IsDisrupt/DownTime唯一结局依据，冲突不按成绩剔除。
J-TEXT已合格源固定c727，EAST-only；旧V3C/旧17+8不混表，不修改源权重。

## 当前真实进程（先恢复，不重开）

- 终端session_id=85431。
- PID197980，已用nvidia-smi确认真实rl_env CUDA进程。
- 命令：python -m experiments.risk_aligned_online.full_path_meta_20261102.run
  --output probe_outputs/full_path_forecast_controls360_v1。
- 最后观察29/360，22.317秒；不是最终结果。
- 本轮8臂中6臂重推后340炮，另2臂引用已核验cold和truncated_metric，前20成本保留。
- 逐位要求forecast_real复现innovation_metric360_v2/metric_forecast；每炮先封存再揭示。
- code snapshot只包含实际导入代码及计划，不含未导入的核验脚本。核心依赖不得运行中编辑。
- 所有子代理本轮已关闭；无需等待它们或假定它们仍在跑GPU。

## 已完成的主要实验证据

innovation_metric360_v2 exit0，1061.3989秒；CPU核验360炮2880曲线PASS，
固定四炮32曲线+96真实层hook PASS，source/head不变，代码快照当前漂移为空。
全360180D180S；前20为8D12S且7TP6FP，后340172D168S。

|方法|TP|FP|J|
|---|---:|---:|---:|
|cold|125|35|.500000|
|contrast_unit|125|34|.505556|
|innovation_unit|124|35|.494444|
|innovation_metric|126|36|.500000（数学相等）|
|metric_forecast|125|33|.511111|
|metric_zero|125|35|.500000|
|metric_reverse|126|35|.505556|
|metric_hold|126|35|.505556|

新metric救回74970但新增87689误报，净0；逆序/保持也救回该炮，且移除85599。
forecast移除73790/85599且无其他翻转，需补同forecast头的reverse/hold，不能借另一头的控制归因。
metric接受498/10910（4.56%）；forecast426，unit10854。
metric记录最大隐藏RMS.244137，contrast.046763；不能继续简单归咎于幅度过小。
旧严格门仍原样失败；用户允许FPR≤.20内FP小幅交换TP，需同时报告净收益。
metric126/36正好FPR=.20，但平衡流净正确数仍没有提高，不是实质退步或严格提升的浮点末位问题。

## 完整风险路径头已拟合，不是新性能结果

- full_path_preflight_v1/completed.json PASS，已结束69854的448点片段，数值/状态/收据严格相等。
- directional_derivative_v2.json：固定q416、3方向×2步长GPU有限差分PASS。
  更新接受[true,false,false]；full B梯度范数.456057，原TBPTT1图中无B路径；前向分数相同。
  只证明固定查询的风险分量导数；诊断入步状态及风险baseline有明确stop-gradient，不是整个显示标量的完全导数声明。
- full_path_heads20_v1完成128步，116.7237秒，20炮标签、0后续炮，source不变。
- 完成文件SHA=5b4baf0581fad9cbeeb3626dc161e63fd4bf5c71c6e6225c2031871539466542。
- **头文件**SHA=26370f1eb13a19fbd4f0200114fbca77b025cc2d8bdd1b7ae8f24d8e9ecbbf34。
  先前子代理拿头SHA去比完成文件才报不符；主进程已分别sha256sum确认，不是产物漂移。
- history SHA=22ef4b196d22d6120b9f89fe3b2c7224785adbb88e37a25528aadd50139e5d8f。
- training code117entries、当前drift为空。212支持抽样中199有过去接受，172显示前后风险损失相同；
  “损失相同”不单独证明零梯度或非零梯度；本实验恰要保留早期反馈的风险路径。

## 下一步实际工作

1. 恢复85431，读取实际退出码和完整completed.json。若仍活跃只观察，不重启。
2. 本次组合流的独立CPU重计核验器、固定四炮神经重放尚未实现/执行，不能套用旧8臂核验器。
   可重用旧innovation.verify的独立classify_curve/recount/paired_flips，但新reused规则必须显式处理。
   两个引用臂只有旧收据，不得把空新收据写成旧方法从未更新。
3. 新核验需核对8臂曲线、6新臂状态/真实梯度、两引用臂与父曲线逐位一致、forecast_real与父forecast逐位一致，
   先封存后结局、同掩码/时钟/重置、源及辅助头不变、训练/计划/代码hash绑定。
4. 固定74087/73403/63656/62573重放6新臂，另核对引用2臂及真实FFN hook；不可按改善挑样。
5. 按FULL_PATH_CONTROL_EVALUATION_20261102.md的整数TP-FP门，分别判断完整风险路径与forecast匹配控制；
   保留每炮翻转与>3000ms报警。不在不成功时调报警器。
6. 只有证明炮内收益后才组合炮后经验学习；本轮后340炮B冻结，不能声称已完成炮间持续学习。

另有可能需后续研究的实际限制：固定metric步大多被拒绝；full路径只补训练归因，不改变这条部署规则。
若仍不优于控制，应检验有界自适应步长/更新目标的风险辨识性，不盲目增大学习率。
operational外层在所有lead>30ms点取峰，并非125ms点标签，因此不能把其sigmoid当已校准125ms概率。
不因此改写当前已固定训练或已封存结果。

MAIN当前中文报告：docs/research/source_event_rebuild_20261023/成熟误差与表示步长_完整360对照及梯度路径诊断_20261102.md。
旧INNOVATION_METRIC_STATUS_20261102.md是早期60/360状态，已被本文件和完成证据取代。
没有删除数据、提交或推送Git。全部新输出在挂载盘，约9.1GB剩余。
