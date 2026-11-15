# 去除未认证Bt排序因素：完整360炮对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent / tdd-workflow
- Origin Mode: run
- Origin Date: 2026-10-31
- Verification Status: NOT_YET_RUN
- Version Label: current_order_context360_v1

## 问题与预先固定的决定

当前旧顺序以 `(初始Ip/100000)^1.2 × 初始|Bt|` 排序；基础审计发现Bt存储值分档，前113炮全部来自小数值组，cold20仅3个安全炮且19炮没有有效CIII。不能在未认证Bt单位时把该顺序称为真实低危害投运顺序。

本轮只新增一个顺序：依据每炮已保存信号前100ms的`median(abs(Ip))`，按 `(Ip/100000)^2`、炮号排序。不读取Bt，不设Bt次级排序，不使用结局、预测或后续全炮统计。平方是对正电流排序的等价表述，不是实测能量或破坏力。继续使用原固定360身份，不选择新集合，不剔除不利炮。

这是离线回放顺序干预。离线调度器使用所有候选炮前100ms构造顺序，但在线学习器不能访问未来炮；它不是原始历史时序，也不代表可以在真实运行前获得未知炮的前缀。结果是开发性方法诊断。

## 严格区分方法谱系

复用`lifecycle_context8360_service_20261028`这一谱系：

- 源模型SHA：c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02。
- 既有source-only context8先验：context8_source_service_20261028/context8_epoch2.pt，SHA 7cef5d4e9122b086886c3ee4b429b9cdf2663a921f4972b6687ceca5d74a81e5。
- 固定标签策略`disruptive_positive_only`、readout=`full`、feedback=`progressive`；不重新训练J-TEXT基模或该先验。
- 炮内为8个临时变量对已到达诊断变化的解析梯度更新，随后改变两个Transformer block之间的token；每炮重置。它不是最近14,316参数辅助目标＋Q/K矩阵更新的分支，不将两条路线合并归因。
- 炮间在当前炮预测封存、真实结局揭示后更新`slow.weight`和`memory.output.weight`；源权重与风险头冻结。

顺序改变后，新cold20的统计与监督准备必须仅由新前20炮形成，不能搬用旧cold模型。这种变化是顺序的实际后果，不是隐藏更改学习算法。训练步数、损失、回放长度、保护标准、学习率和源事件回放均不变。旧顺序结果保留为对照，不替换。

## 完整实验臂

沿用八曲线：source、aligned、cold、intra、inter、both、both_shadow、intra_reverse。source不应受顺序改变影响；aligned只包含20炮后密度映射；cold加入同预算监督准备；intra只做炮内状态更新；inter只做炮间持久更新；both开启两项；both_shadow在both持久状态下关闭当前炮的临时作用；intra_reverse保持幅度控制但改变反馈顺序。

`both`与`inter`长期状态由各自历史更新产生，因此二者是完整策略比较，不是固定参数下的单步因果效果；`both`对`both_shadow`用于观察同一持久状态下的当前炮临时作用。不能只挑其中有利差值报告。

主要结果：完整360炮TP/FP、TPR>30ms、FPR、J、提前量及>3000ms报警。保留前20试错成本，另报共同的新后340炮及各20炮块。固定原阈值0.7258946895599365；不做报警参数选择。检查intra相对cold、both相对inter及both_shadow是否有独立收益，不用排序收益代替炮内收益。

## 证据与执行

在读取新顺序的类别组成前先保存新order及SHA。完成后读取标签只作组成与评价分析。现行标签仍按HDF5权威；18炮shot list冲突不因此消失。源事件标签仍仅用于已有源事件覆盖的回放片段选择，不称为EAST机制真值。

新增排序函数先写测试、观察失败，再实现；验证只读ip前缀、无Bt/meta访问、时间边界、统一倍率的排序不变性和拒绝无效输入。旧order默认行为不变。

主进程统一调度本机rl_env/CUDA；预计参照旧完整运行约13分钟，外部硬超时1800秒，保留失败产物。命令使用外部`timeout --signal=TERM --kill-after=10s 1800s`；不改写旧结果。

不保证重新排序会改善成绩，不因不利结果再换一种顺序。当前全部数据都已参与开发，不能称为独立确认。

执行前披露：历史审计在排序规则已固定后、实际新order文件写出前，按已有电流统计计算了同一顺序，并报告新前20为18破裂/2安全。主进程不会因此更换规则或选择另一种有利顺序；新的排序实现仍只读取ip。由于组成已暴露，不称本轮为盲法或前瞻性预注册。
