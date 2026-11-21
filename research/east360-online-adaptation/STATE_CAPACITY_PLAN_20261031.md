# 固定错误端点的表示能力诊断（使用真值，不是在线实验）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent / results-analysis
- Origin Mode: run / validate
- Origin Date: 2026-10-31
- Verification Status: PLANNED_POSTHOC_DIAGNOSTIC
- Version Label: state_capacity_v1

## 问题和不可越过的解释边界

前轮新规则实际更新7001参数并降低训练BCE，但完整流134TP/43FP，劣于固定cold的134/42。现在检查其内部修正通路是否能表达所需变化。本诊断使用已经知道的结局选择错误端点和优化临时向量；这是非因果oracle探查，不是可部署策略，不能生成新在线TPR/FPR，也不能当作学会了从信号得到该向量。

局部优化找到越阈解，证明该端点存在一个测试过的可行修正；没有找到，不能证明全局不可能。尤其安全炮只优化最高风险端点，不能保证其他端点不报警，不能记为救回整炮。扩大内部可行空间也不证明新增自由度可被无标签数据可靠学习。

## 固定证据和病例

父：current_order_context360_v2，completed SHA cb9cf19197ab25fb6e825687ec002bcad6135834dca1cb7dc3371234eb7883a9。
新规则：east_operator360_v1，completed SHA 3edfa9b2a3025f327533ab252b61cde95d550f219447215eccce90a8cf147c33。
固定源、cold slow/output、映射及阈值0.7258946895599365；不改任何模型参数。

按父cold的完整360判定，记录全部46FN+42FP。前20有10FN，保持启动成本，只登记不使用后来的cold状态回算。后340的36FN和42FP共78个全部纳入：

- FN：选择lead严格>30ms且索引≥16中的最高cold风险端点；并列取最早。它不是所有可能有效端点的穷尽搜索。
- FP：选择整条安全记录的最高cold风险端点；并列取最早。若该端点<16，所有路径禁用修正；另记录任何首报<16的情况，后来端点改变不能撤销既有报警。
- 有争议标签不删除，只给出标记。不存在合法端点时列为无法探查，不另换炮。

## 约束与三个嵌套搜索

分别用原source operator与已训练EAST operator形成基B（92×8）。原插入末端token的改动为W_output B c，c是8维临时状态，其L2范数≤1。该约束来自现有工程实现，不是物理安全界限。

每种算子／端点计算r=最大奇异值(W_output B)。比较：

1. context8：优化c，范数≤1，仍经原B与W插入最后token。
2. free_last：自由64维末端token修正，范数≤r。
3. free_sequence：自由修正已观察的16个非CLS token，总Frobenius范数≤r；填充token及CLS不可改。

全部在原slow作用之后、第二Transformer block之前插入；不修改risk head、报警器或最终分数。自由控制的r是该context球最大可能改动尺度，不是本炮实际更新幅度，更不是物理能量。source与learned的r可以不同，跨两种算子不宣称严格同半径。

每条路径两个固定起点、各100步Adam lr=.1，每步投影回对应球。context起点为0与实际context；free_last起点为0与最佳context的实际shift；free_sequence起点为0与最佳free_last的shift。因此自由路径接受下级已找到解的热启动，这不是完全独立、相同初始化的优化器比较。保留每例best-so-far，使用已知标签BCE；不扫描次数、半径或学习率。

## 核验

从原32窗口推理批次获取第一层token，记录零修正概率与封存值的数值差；向量化诊断可能出现批大小导致的浮点差异，不能拿它作为改善。最终将每个最佳shift放回该炮原来的32窗口批次重算，只有这一重放跨越原阈值才记为“该选定端点找到可行翻转”；边界极近需另标，不评价新整炮FPR。

核对原始／封存SHA、全部78病例清单及10个启动错误、修正范数、有效token掩码、CLS不改、固定模型所有参数／缓冲区不变。保留每例起点、最佳迭代、best向量及重放结果。记录实际状态范数、最佳方向和实际方向夹角，区分改动量不足与方向不合适的可能性，但这些仍是事后描述。

## 执行

本机rl_env CUDA，单GPU进程，输出probe_outputs/state_capacity_v1；外部timeout1200秒，TERM后10秒KILL。没有原始数据修改、模型持久更新、阈值调整或结果替换。数据已参与开发，本次也不是独立验证。总体目标仍是可因果部署的EAST炮内增益，本诊断不替代该目标。

## v2数值修正（v1独立检查失败后）

v1完成了诊断计算，但独立CPU float64复算发现source规则61629的谱半径差约9.89e−6，略超过原定容差；保留v1及audit_failure.json，不放宽审计容差。v2仅将W_output×B及SVD改用float64，最后半径转回模型dtype。相同78病例、两规则、三个空间、两起点、100步和lr=.1全部不变；输出新目录probe_outputs/state_capacity_v2。此修正是诊断约束的计算精度，不是当前在线输入单位或原模型FPR的修复。
