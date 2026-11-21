# 同预算全局对照修正：训练与推理均匀权重

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-11-05
- Verification Status: planned before corrected replay
- Version Label: layer0_uniform_matched_kernel_v1

已归档的均匀版本仅覆盖公开的推理核，候选目标仍调用私有的局部核。
保留其轨迹，作为“局部训练＋均匀部署”的混合干预，不再称作纯全局对照。
新版本保持同一源、炮内、炮序、查询、20步/次、保护和阈值；显式同时
替换两条调用路径，拟合时仍停止旧记忆的梯度。独立核实固定20＋16标签，
并核对实际目标函数核调用计数是否为每个提议22次（前、后评估各1次）。
均匀核只改变同层小参数的应用范围，不是重训源模型。
