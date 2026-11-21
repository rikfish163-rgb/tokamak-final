# 炮内临时状态按时间片读取：完整360炮配对试验

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent / tdd-workflow
- Origin Mode: run
- Origin Date: 2026-11-01
- Verification Status: PLANNED_DEVELOPMENT_EXPERIMENT
- Version Label: multi_token_context_v1

## 根据何在，改变什么

当前完整360诊断已完成并复核：cold125TP/35FP；保留当前行、反转较早历史121/35；把窗口每通道都保持为其最近观测144/52；保持原始均值88/25。后340的29个cold误报中，21个在索引>=128才首次报警。历史信息和最近状态的取舍会影响判断；这些人工窗口并非真实物理反事实，不能据其断言因果机制。

历史审查发现，多token神经适应及元训练自监督目标均已尝试。窄候选是在固定ContextMemory的同一c8状态上，利用各时间片自己的观测基B_k读取状态，再进入原第二Transformer块。未在有界检索中找到这一具体实现；不宣称首创或必然有效。VITA(arXiv2506.10085v5,§3)通过监督外层训练自监督更新，但其预测头共同训练；本项目固定既有风险头是额外限制。

## 固定合同

qualified JTEXT源SHA c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02。只在EAST进行在线适应，HDF5 IsDisrupt/DownTime为权威；JTEXT max(25ms,AdvancedTime)、EAST125ms不变。沿用stratified_cold20_context360_v1全部360身份与顺序及density映射、cold slow矩阵、风险头、门槛0.7258946895599365。前20为8D12S，原source-only预测7TP6FP计入所有总成绩。第20炮结束后读取标签准备，候选从第21炮起生效。这个排序是离线初始电流代理模拟，不是自然历史炮序或认证的低危害顺序。及时要求strict lead>30ms，>3000ms另报；所有数据已参与开发。

固定现有源训练的ContextMemory key/kernel/rate及c8写入规则，临时c每炮从0开始，按已到达全局8点patch更新；每个端点读取context[floor(end/8)-1]，负索引或空序列为0。未到达信号和当前炮结局不进入写入，炮间没有后340监督更新。

## 作用位置及支持规则

每个端点的原始128点因果窗口按模型实际右对齐方式分为16段，各8点；不能将全局patch索引直接当作滑动窗口token索引。对每段重算source-asinh均值和每通道8点完整观测mask，通过已有key/kernel得到B_k(92×8)。没有任何完整通道的段不施加修正，包括可能仅部分观测而仍是valid的token；这项支持规则先于训练固定。

每片修正为O(B_k c)，O无偏置64×92，CLS不被直接修改。原cold slow仍只作用最后一数据token；原第二块及risk head照常汇总。c=0时所有新修正为0，必须精确复现cold。虽然涉及16片，固定基与O后自由度仍至多8；这不是自由16×64 oracle。更大的整体修正范数不自动等于更多信息。

只拟合O的两个独立副本：aligned_last和per_token，均从同一cold O出发。每副本固定256个同样安全／破裂配对，种子2131，Adam lr0.001，clip1，1e-4×sum((O−O_cold)^2)正则，最后一步导出。每炮训练候选沿用安全全部有效点、破裂30<lead<=125ms的正标签点；安全峰值用softplus(maxz−b)，破裂峰值softplus(b−maxz)，b是固定报警logit边界。每步两炮各0.5。全候选扫描和选中点反传使用相同32窗口形状及梯度路径，峰值重算必须精确一致。模型其余张量不变。不按后340挑学习率、轮次、路线或阈值。

## 完整流实验臂

1. cold：delta0，必须匹配父cold曲线。
2. legacy：原cold O和原全局patch last读取，必须匹配父intra曲线。
3. per_token_real：per_token拟合O，各片B_k读取真实累计c，主候选。
4. same_O_aligned_last：同主候选O和token相位，只写末片，隔离作用范围。
5. same_O_tiled_norm：同主候选O，将末片修正复制到valid数据token，匹配主候选每例整体Frobenius范数；末片方向为0时保留0并记不匹配。
6. per_token_reverse：同主候选O，原已到达horizon反序反馈及原norm匹配。
7. per_token_local：同主候选O，每次写入从0开始并匹配真实c范数，检验累计经验。
8. per_token_constant：同主候选O，用准备20炮的等炮平均c方向，逐端点匹配真实c范数。其幅值仍来自真实反馈，只检验方向信息，不能称完全无反馈。
9. last_trained_last：aligned_last拟合O，末片读取，匹配训练预算。
10. last_trained_all：aligned_last拟合O，各片读取，与主候选构成权重×作用范围对照。

正序和破坏反馈不参与不同标签访问或炮间训练；首20所有曲线仍source-only。记录全360/前20/后340 TP、FP、J、首次报警、>3000ms及全部救回/改坏炮，实际状态与作用范数、零方向、支持数。

## 判定及执行

主候选必须FPR<=0.20、TP不少且FP不多于同父cold，至少一项严格改善，才满足本轮操作目标。作用位置归因还需J严格高于same_O_aligned_last和same_O_tiled_norm；当前反馈归因需胜过reverse和constant；若控制相同则归因不成立。local检验累计经验，不将相等强称记忆收益。此开发门槛不是总体显著性或独立未来验证。

先通过零状态恒等、非零O梯度、8个相位、空/短/缺失、完整记录与截断前缀一致测试。先做前20准备的2步CUDA预检，检查数值、旧曲线和代码接口，不看后340候选表现；不得根据预检成绩换参数。完整运行本机rl_env CUDA、外部timeout3600秒，30秒左右读进度/进程；源码运行中不修改，崩溃保留。内层为解析最小二乘梯度更新c，外层监督反传O；不将其说成原Transformer权重重新训练或已学到新物理。

当前GPU无训练任务；上一轮为progress，已完成4臂完整360历史诊断和独立重放。原代码派发句柄后续not_found且无代码文件，主进程接手，不冒称子代理实现完成。

实现前的精确前缀测试发现旧vectorized c8 trace在完整数组与截短数组之间有浮点舍入差异，尽管固定长度下扰动未来后缀不改变结果。新候选将独立key/basis行用固定32行padding分块计算，padding行不进入递推；保留原c8更新代数和参数，精确通过前缀测试。legacy臂仍执行旧trace以复现存档曲线。该数值执行差异明确记录，不说成已发现未来标签泄漏，也不用于解释大范围误报。空trace只允许尚无全局patch读出的端点；索引15对应已有首patch的零状态，不冒充空trace。
