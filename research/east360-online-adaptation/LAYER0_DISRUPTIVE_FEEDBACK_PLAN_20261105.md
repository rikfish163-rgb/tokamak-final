# 仅破裂反馈写入：从头因果复跑

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-11-05
- Verification Status: pre-execution for the new full replay; after development diagnostic
- Version Label: layer0_disruptive_feedback_v1

锁定第一层炮内对照、局部核、原查询函数、同一16个额外标签/揭示秩、
候选20步与所有完整历史保护。从第1炮开始独立回放；查询仍看不到真实结局。
揭示后，破裂查询照旧按是否漏/迟报拟合候选；安全查询不写风险修正，
但仍正常进入最近历史的保护检查。这是方向干预后的开发候选，
不是在既有完整策略的记忆里事后删除向量：不同的历史可改变后续候选、
核分母和接受决定。共享炮间状态、四臂、源、炮内动作和报警器不变。

主要检验：相对锁定炮内，合用需同时保住或增加 30–3000 ms 内的及时预警、
在 FPR≤.20 下改善 J，并且同未查询集合结果不是靠使用过的炮驱动。
保留局部全反馈及同预算全局对照；该筛选曾用本开发流启发，不能作独立证据。
没有扫阈值或补标签。执行 `run_layer0_disruptive_feedback_20261105.py --output
<new-dir>`，同 CUDA 环境，硬超时480 s，逐块进度，独立 HDF5 重计。
