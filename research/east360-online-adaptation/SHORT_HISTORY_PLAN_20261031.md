# 短历史窗口监督：完整360单因素开发实验

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-10-31
- Verification Status: PLANNED_DEVELOPMENT_EXPERIMENT
- Version Label: short_history_context360_v1

## 假设与范围

现行EAST训练端点要求索引≥127，而部署从索引0预测。保持部署规则不变，给已结束炮增加短历史监督，检验这一覆盖缺口是否导致部分早期误报。不是宣称已经解决单位、标签或炮内创新问题。

参照运行：`probe_outputs/current_order_context360_v2`，completed SHA
`cb9cf19197ab25fb6e825687ec002bcad6135834dca1cb7dc3371234eb7883a9`。
同360身份、同仅电流顺序、同20炮冷启动、同density_gain映射程序、同J-TEXT固定基模与source-only context8先验、同阈值0.7258946895599365、严格lead>30ms。第一炮前不使用EAST标签；每炮全部预测封存后才揭示结局。前20炮计入完整流。

## 唯一算法改动

现有采样的全部≥127端点保留。额外添加索引0—126之间、valid为真且满足既有破裂炮lead>30ms要求的最多16个均匀端点，排序去重。不对早期端点按预测错误挑选。数据读出仍由过去窗口、左填充和可用性掩码构成。

继续使用`disruptive_positive_only`损失：破裂炮的早期负标签不会因此变成安全监督；仅其中实际正端点和安全炮端点参与loss。当前cold20短历史端点预计主要作用于两条安全炮，应保存端点清单以实际计数。未按未来结局补选安全炮，未移除标签争议。

新增窗口会改变每炮损失平均中的时间覆盖，并增加计算量；保持的是标签炮预算和优化步数，不宣称精确计算预算相同。它是覆盖策略干预，不是纯粹增加迭代次数，也不是只解冻更多参数。

冷启动仍20步、fast/off混合，Adam lr=.001、clip1、目标/源回放权重均不变。后续炮间每炮2步，同recent8和源事件回放、同guard。新增策略同时用于冷启动和炮间端点采样，其累积效应由完整流比较，而不能归给某一次更新。

## 实验臂与归因

沿用8臂：source、aligned、cold、intra、inter、both、both_shadow、intra_reverse。

- 新cold对旧cold：监督时间覆盖的结果，不是炮内增益。
- 新intra对新cold及新intra_reverse：炮内独立效果与实际反馈作用。
- 新both对新inter：联合系统效果；二者学习历史不同。
- 新both对新both_shadow：同一持久状态下的当前炮临时作用。
- source和aligned的全部360曲线应与参照逐点一致，证明输入、顺序和报警外部条件未漂移。

不搜索16个端点数量、学习率、阈值或预算。固定最终状态，完整跑一次。即使结果负面也保留。

## 检查与运行预算

本机rl_env CUDA，单GPU进程，外部timeout1800秒，TERM后10秒KILL。预计约12分钟；持续检查真实进程和每10炮进度。失败保留原因和产物，不覆盖目录；已获用户迭代授权的代码修复须明确披露并使用新目录。

运行命令：
`python -u -m experiments.risk_aligned_online.lifecycle_stream_20261025.run --output probe_outputs/short_history_context360_v1 --target-loss disruptive_positive_only --readout full --feedback progressive --context-memory experiments/risk_aligned_online/runs/context8_source_service_20261028/context8_epoch2.pt --order-policy current_only --endpoint-policy prefix16`

完成后独立核对封存曲线、源和原始文件SHA、全部指标、首报时间、实际端点、训练标签访问顺序。新旧比较使用同一完整360；不得只报修复炮。选定序号21、114、360重新推理aligned/cold/intra/reverse，要求与封存曲线逐点一致；无收益不追认成功。

## 接受边界

局部准备改动若改善J且FPR≤.20，仍只属于基础监督修正。炮内要求新intra相对新cold改善J、FPR≤.20，并检查逐炮收益及真实反馈对反序的优势；单一开发流优势不构成独立确认。总体科研目标不会因基础修复或炮间收益就被标记完成。

18个HDF5/shotlist争议继续披露，cold20中含74087、90874、83646三个冲突。阈值、集合、源标签与特征尺度不在本实验中调整。既有数据多次开发暴露，本计划不是独立预注册或前瞻验证。
