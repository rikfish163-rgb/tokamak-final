# 已写入记忆的来源拆分：为什么炮间增加漏报？

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-11-05
- Verification Status: post-factorial diagnostic plan, before this intervention
- Version Label: layer0_memory_polarity_v1

局部四臂在开发流上减少误报，却丢失部分炮内及时预警。固定已完成局部回放的
全部记忆、原来的查询/接受决定、生效秩和键。并行重放只保留安全来源、只保留破裂
来源的128维向量，并从原始信号重放全部向量以核对。被移除来源的向量置零，
保留其键和核分母，避免把其余记忆同时放大。每种干预都分别检查关/开炮内，
不改变时钟、阈值、冻结源、原炮内动作或测量信号。没有新标签、优化或报警规则。

仅在对应父回放块结束后加载已接受的快照和已允许读取的来源标签，未查询
结局在重放期间不读。结束后按原 HDF5 独立配对计数；针对合用损失与获益个案，
比较两类来源各自对首报造成的变化。这里“安全/破裂来源”指**训练标签**，
不假设其对其它炮的实际 logit 位移必为负/正。父回放的写入决定来自完整策略，
因此即便某一拆分更好，也只是固定路径上的反事实机制诊断，不是新的在线学习结果。

执行 `run_layer0_memory_polarity_20261105.py --output <new-dir>`，CUDA 480 s
硬超时，每 20 炮记录进度。期待完整 360×8 曲线及独立原 HDF 重计。
