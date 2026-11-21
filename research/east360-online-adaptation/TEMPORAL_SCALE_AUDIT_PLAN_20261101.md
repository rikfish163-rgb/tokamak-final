# 时间结构还是数值压缩：TRAIN1353与已结束EAST20炮审查

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent / results-analysis
- Origin Mode: run / validate
- Origin Date: 2026-11-01
- Verification Status: PLANNED_DESCRIPTIVE_AUDIT
- Version Label: temporal_scale_audit_v1

上一轮为progress：multi_token_context_v1完整360及复核完成，主候选127TP/34FP对cold125/35有小幅收益，但复制末片修正的对照相同；固定128点的片间差异能量中位数0.02097%。本轮不再次增大修正，而检查相同1kHz采样下源与目标的时间结构和变化对比度。标签窗口JT max(25ms,AdvancedTime)、EAST125ms不构成5倍物理时间伸缩的证据。

只读取JTEXT TRAIN1353和stratified_cold20_context360_v1已经预测封存并揭示结局的前20炮（8D12S）。不读VAL/TEST220信号或后340目标信号/结局，不训练、改阈值或改变原始时间轴。HDF5为结局/DownTime权威。源检查点与既有density映射仅提供原模型数值坐标；映射只作用目标asinh对比度，原始无量纲结构函数使用未映射值。

每炮在三类端点统计完整128点窗口：固定end127；整段preterminal有效记录内至多16个均匀端点；破裂炮在原正窗口内严格lead>10ms(JT)或>30ms(EAST)的至多8个端点。需要end>=127；不足记录明确缺席，短炮不补造全窗口。端点仅用训练／已结束炮标签作事后分层；不称在线未知阶段分类。三种视图重叠，不能把其窗口当作独立重复。

逐通道统计lag=1,2,4,8,16,32,64个采样间隔的变化：D(lag)=平均((x[t]−x[t−lag])²)/(2×该窗口可用值方差)。至少64个观测，某lag至少32对共同观测；常量无可辨时间尺度，D记缺失，不补0。不加具有单位的epsilon，不截断D；非平稳斜坡和振荡可使D>1或不单调，因此不把首个过0.5的lag称物理相关时间。不把模数等任意统计量的结果认证成物理过程。所有23列保留描述和缺失数，来源及语义不够时不推断。

另用既有源center/scale计算float64描述性asinh标准差，目标先按既有映射处理。它描述相同工程坐标下的变化幅度，不是逐位神经激活重放。对至少8个完整8点patch计算between/(between+within)的方差分解，衡量片均值保留了多少原观测变化。常量时分数缺失，缺测patch不作为真实观测。

输出每个窗口的数值矩阵、观测数与配对数、端点身份、输入哈希和源码快照。汇总先在每炮内平均窗口，再给各炮等权的中位数与10/90%分位；按HDF5安全／破裂分层，避免源1165安全/188破裂与目标12/8的类比例差异混成结论。source实际类别数以本次读取结果为准。

解释边界：D曲线差异支持统计时间结构错配假设，不认证具体物理时间单位；D相近而asinh标准差不同则支持对比度问题，而非自动选择一个时间倍率。任何后续时间伸缩或内层更新仍需完整流固定门槛对照，不能从本审查直接宣称炮内收益。CPU本机rl_env，timeout600秒，逐100炮记录进度，不自动修改正在运行的代码。
