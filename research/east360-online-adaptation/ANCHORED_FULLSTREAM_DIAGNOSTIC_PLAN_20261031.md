# 均值锚定与更新时点：完整360炮补充诊断

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent / results-analysis / tdd-workflow
- Origin Mode: run
- Origin Date: 2026-10-31
- Verification Status: UNVERIFIED（运行前登记）
- Version Label: anchored_full360_diagnostic_v1

## 为什么另做完整流

锚定目标与成熟窗口两项在cold20条件留炮筛查均失败，记录不变，候选未晋升。原计划“失败不跑360”在此作**公开的补充分析变更**，不是改写原门槛：逐窗BCE、真实反馈必须在每类都胜过扰乱反馈，并不等同于整炮首次报警收益；尤其125ms外的破裂炮窗口标0，却仍可能及时报警。完整流结果才能直接检验当前用户目标。已有全部数据参与开发，此项也仅开发诊断，不是确认性检验。

本项对两种规则均全量运行，不按结果选择时点或半径，不增加分数头/报警保护器。固定qualified J-TEXT、cold20输入映射/slow、原360炮序、阈值0.7258946895599365（以封存parent数值为准）。原17D/3S冷20成本复用封存预测；后340每炮临时状态归零，已到达25点反馈经真实表示损失梯度更新第一层Q/K，仅影响后续时点。没有cold20以后的持久更新。

## 拟合与各臂

用原conditional_weights的2707机会缓存及全部20个已结束炮，分别拟合origin/mature两种AnchoredTarget；每种256步、Adam .001、seed131、batch24（12safe/12danger，类内先均匀抽炮再抽窗口）。模型14,316参数，初始输出零；eta=.1、Q/K联合半径.1，末步保存，不选epoch。两种抽样一致。仅辅助目标网络拟合；基模、riskhead、slow不变。context缩放和反馈均值由全部cold20按炮等权拟合，来源与公式复用既有实现。

每规则四臂real/reverse/mean_feedback/donor，另有cold和identical，合计10条曲线。donor为cold20中原点128维缩放表示MSE最近的既往窗口，同分取最早缓存索引；后340不属于cold20。只替换目标网络反馈；mature求导输入仍真实，非完整移除反馈干预。reverse/donor非零动作匹配同规则real范数；mean恒等零只作结构检查。current outcome不可用于检索、目标或动作。两个规则均保留，无胜者挑选。

原点127，反馈maturity152，此后每25点更新；预测包括maturity的分数先保存，新状态从maturity+1生效。每次从零求新状态再替换，不累计。没有有效反馈则零替换。origin用原窗口求梯度，mature用截至maturity的128点窗口；目标context一致。改变窗口也改变分块/位置编码，不只新鲜度。

## 评价与判定

严格沿用EAST lead>30ms、第一次越阈值报警；全部破裂提前>3000ms另报，不撤回早先误报。报告full360、cold20、suffix340和预先固定last90，但不以子段代替全流结论。主表TP、FN/TARDY、FP、TN、TPR、FPR、J与median lead。额外≤.10/.15/.20为固定阈值结果是否满足上限，不事后扫阈值。

单个规则只有full360 J严格大于cold、TPR不低于cold且FPR≤.20才记为本次描述性运营改善；必须同时列全变化炮，不能将点损失下降称为救回。real与reverse/donor的J比较是机制对照，只有real均更高才支持本次干预下的额外反馈收益；均值本就等于cold。即使满足，也不撤销原组件筛查失败，更不声称物理发现/独立验证或目标完成。两规则同样判定，报告所有失败。

## 运行与核验

rl_env CUDA；主进程独占GPU，30分钟硬超时。全部360预测封存后才打开评价标签（cold20历史标签只在拟合阶段使用）；输入mapping/hash、原parent、source固定。保存每炮10曲线、更新时钟、状态范数、配置与模型；与旧cold逐点误差≤2e-6且过阈决定一致。定向单元测试时序、归零、无结局访问、求梯度窗口、对照与评价；完成后独立CPU复算全360，并从保存模型重放全部变化炮及预先固定70323。禁止删除失败产物或覆盖旧结果。

## 文献边界

Sagawa等ICLR2020 §2–3说明最差组拟合不等于最差组泛化，增加类别权重也不等于增加独立正常工况；本轮不照搬其DRO。CROIS（Nguyen等TMLR2023 §1–3）强调独立数据划分与较低容量分类层，但本项目不改变风险头，不是其复现。两篇不保证当前固定模型有效。主要验收来自用户的整炮预警目标，而不是从文献杜撰统一筛查门槛。

https://arxiv.org/pdf/1911.08731
https://arxiv.org/pdf/2204.09583
