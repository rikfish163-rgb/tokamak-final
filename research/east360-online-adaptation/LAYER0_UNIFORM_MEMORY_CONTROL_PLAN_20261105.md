# 同标签同释放时点：局部记忆和全局直接微调对照

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-11-05
- Verification Status: pre-execution plan
- Version Label: layer0_uniform_memory_control_v1

固定相同四臂、静态/炮内两条参考曲线、查询函数、16 个额外标签、
揭示/可用的块时钟、第一层参数位置、参数投影、训练损失、单次20步
及完整曲线保护。把每个历史修正对当前状态的相似度权重替换为常数1：
所有已采纳向量平均成同一个持续的128维参数，应用于每个因果端点。
没有选相似端点，也没有把任何未来标签移到开头。

这是**同预算的全局参数直接微调对照**，不是另训整个基座或对所有
可见标签集中重训的上界。相同标签与学习机会不等于实做的更新次数相同：
当前错误、保护决定、接受数可能因历史参数不同而不同，分别记录实际拟合数。
只有在两个回放独立复算并验证选中 ID 与标签释放秩完全相同后，才比较
局部/全局的 inter 与 both。在同一未查询子集与完整流同时看及时/误报
和保护副作用，不用开发数据选择新的阈值。

执行 `run_layer0_uniform_memory_20261105.py --output <new-dir>`，同 CUDA 环境、
480 s 硬超时、每块进度，使用同一独立 HDF5 评分器核验。
