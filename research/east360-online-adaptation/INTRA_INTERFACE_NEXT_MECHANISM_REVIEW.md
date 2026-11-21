# 炮内任务接口与跨通道假设：独立只读审查

2026-10-29。结论：**接口失配有源码和实验依据，但“分类器依赖跨通道关系、当前更新只拟合边缘幅值，因此失败”尚未被证明。** 暂不建议训练新的跨通道适应模块；先做下述一个只读反事实诊断。所有结果均为开发数据上的证据，不保证收益。

## 1. 固定源模型与实际计算图

本次直接读取了 `gradient_aligned512_20260929/source.pt` 的 CPU 元数据，SHA256 为 `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`；配置是 `source` 输入、W128/P8、d64、4头、2层、H25、`fast_site=tokens`，且 `forecast_trained=True`。哈希与本轮协议中的 `source_sha256` 一致；当前 `simple23.py`、局部读出、Q/K 与 factorial 更新源码的哈希也逐一匹配本轮协议。源模型加载和锁定见 [prepare_prior.py:22](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/redesign_20261004/prepare_prior.py:22) 和 [protocol.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/local_factorial_east360_v1/protocol.json)。

实际图是：23列源坐标 `asinh((x-center)/scale)` 与23列有效性标记，按8点拼为16个**时间 patch**；每个 patch 的368维经同一个线性 embedding 混合到64维，附加位置编码和CLS，经两层注意力/FFN；最终规范化后，风险头只读取CLS的64维。原H25预测头则读取17个 token 展平后的1088维、输出25×23。见 [simple23.py:55](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple23.py:55)、[simple23.py:97](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple23.py:97)、[simple23.py:118](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple23.py:118) 和 [simple23.py:127](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple23.py:127)。注意力的token轴是时间 patch，不是23个独立通道token；图中没有显式规定某对信号的比值或物理关系。

因此，该分类器**有能力**利用同一patch内的跨通道组合及跨patch演化，但架构本身不能证明它在EAST有效依赖这些关系，也不能证明这种依赖有正确的物理含义。输入中仍可有单通道水平、缺测模式和时间位置捷径。

## 2. 当前接口具体错在哪里，哪些还只是猜测

1. **读出与任务**：11臂不是直接用源H25头的输出做局部损失。程序挂钩源预测头的输入，从最终17个token中取**最后一个信号token**，另以冷20先验及本炮较老数据逐通道拟合 `64+1 → 25` 的线性读出。历史最多16行、含岭先验；新两段目标独立于拟合目标。目标是 `asinh(x_future)-asinh(x_origin)`，不是 `asinh(x_future-x_origin)`。每个有效通道要求该段25点及起点可用；MNM与mode_number_n不进入此辅助损失，但后者仍可进入风险输入。见 [local_readout/learning.py:22](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_readout_20261029/learning.py:22)、[local_readout/learning.py:79](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_readout_20261029/learning.py:79)、[local_readout/learning.py:108](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_readout_20261029/learning.py:108)、[factorial/learning.py:33](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_factorial_20261029/learning.py:33) 和 [11臂报告:21](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/炮内拟合与反馈分离_11组完整流及步长检验_20261029.md:21)。

2. **目标不是关系约束，但读出也不是单通道隔离网络**：当前目标是有效通道、时距上逐元素SmoothL1的平均，没有显式对通道对的协同变化、相对时延或联合预测分布施加约束。可是每个通道的线性读出都接收同一个已混合多通道的64维表示，不同通道梯度也汇入共享网络。故“目标按单元素求和”成立；“更新只能利用边缘幅值、完全不能编码关系”不成立。见 [factorial/learning.py:112](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_factorial_20261029/learning.py:112)。

3. **作用位置和方向**：11臂每次从零构造第一层Q/K各64×64、共8192维的临时增量，经注意力、冷20的固定 `slow` 末token算子、第二层后，同时影响最后信号token和CLS；它并非只更新辅助头，也没有修改长期源参数。当前损失不包含当前炮的风险标签或未来风险，Armijo只检验已到达信号损失；下降不意味着风险梯度方向正确。源检查点所记录的元训练快状态位于embedding后的token gain/bias，而不是当前Q/K矩阵，且原H25头已训练；位置差异不能单独解释失败。[MatrixAttention](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_basis_20261029/learning.py:12)、[实际Q/K前向](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_lowrank_20261029/model.py:31)、[更新与时钟](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_factorial_20261029/learning.py:130)、[接受规则](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_factorial_20261029/acceptance.py:92)。

形式上，对反馈窗口和稍后风险窗口，分别有 `g_signal = J_last,s^T · dL_signal/dh_last`、`g_risk = J_CLS,s^T · dL_risk/dh_CLS`。两者都经过相同临时Q/K，却通过不同token、读出、输入和时间；没有恒等式保证内积为正，也没有证据表明内积必为零。归一化、有限步、后来输入和首次报警进一步使一阶方向不等于炮级收益。

## 3. 已发生的事实与查重

本轮完整360中，cold为125TP/32FP；事前候选真实历史/真实反馈为125/33，0救回、新增64562误报；同一历史加反序反馈也为125/33。10个适应臂各10774次均接受非零更新、均降低当次已观察损失。后340炮的1352个回顾性抽样窗口中，候选真实下一窗口误差从零Q/K的0.034635到0.034345，直接保持最后观测为0.025006；反序反馈为0.034331。前者的“更新前”是零状态反事实，不是上一刻已部署状态，反馈和下一窗口的掩码也不同。[完整结果:55](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/炮内拟合与反馈分离_11组完整流及步长检验_20261029.md:55)、[辅助汇总](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/local_factorial_east360_v1/auxiliary_summary.json)。这否定本规则已取得可归因的炮内收益，不否定所有关系信息。

下面的诊断与旧方法不同，是因为它**不训练、不做在线更新、不对报警打分**；不是声称旧方法已穷尽整类思想：

| 已尝试 | 已有证据及与本诊断的区别 |
|---|---|
| featureaffine | 46维逐列输入gain/bias，由H25误差驱动；无回映在120炮为28/26，回映为28/18，后者纯intra FPR仍超限。这里不拟合坐标参数。[实现](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/feature_affine.py:18)、[结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/astra_design_20261005/47_预测坐标回映与炮内更新实测.md:47) |
| CORAL | 曾做输入U-CORAL重训及风险前64维Deep CORAL；后者静态133/41、intra137/47，FPR超限且J下降。这是跨域总体协方差/训练，不是逐窗口相对时序干预；旧实验也不等同完整S-CORAL复现。[结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261022/MNM恢复与表示对齐_执行记录_20261022.md:78) |
| eventteaching | 源端真实/打乱/无事件教学，EAST三者intra均129/35，扰乱时距同为129/35；不是证明事件信息无用。这里无事件头或标签训练。[结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/事件与参照梯度_五组完整360复核_20261028.md:18) |
| meta | 源端已做后续BCE/梯度对齐；Q/K匹配头的VAL危险阶段30/32炮变差，平均/最差分组与时钟四格均未同时改善安全和危险。这里不训练元梯度、头或新控制器。[结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/匹配更新位置与反馈时效_四格配对实验_20261029.md:46) |
| crosschannel | 已按通道整段遮挡、用其他通道重建，并更新embedding或末层FFN；240炮开发集的轮换/四视图共同反馈均增加误报且J低于静态。该240炮不是TEST360。这里不遮挡重建或更新embedding。[实现](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/cross_channel_prefix.py:100)、[结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/EAST单炮更新_四组反馈与步长约束对照_20261021.md:68) |
| QKlocalreadout | 就是本轮冷20先验＋本炮历史读出、真/反序/零目标和Armijo；不能再把重新加权、局部拟合或缩步称为未经测试的新接口。这里仅分析现有计算图，不改变其头、目标或接受规则。[实现](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_factorial_20261029/learning.py:88) |

已做的顺序判别将所有通道一起重排、训练判别器并更新token gain/bias，完整流125/32；它保持跨通道同刻组合，并未单独控制两个通道组之间的相对排列。[顺序实现](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/temporal_relation_20261029/model.py:29)、[结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/时间顺序辅助任务_完整360与更新方向失配_20261029.md:55)。

## 4. 唯一建议：两组相对时序的冻结图交互诊断（未执行）

这是**CPU只读诊断，不是新的适应方案**。检验的具体命题：在固定cold图中，某个跨组相对时序扰动是否产生不可由两组单独效应相加解释的CLS logit变化；同类扰动对现有局部H25反馈Q/K梯度及其投到风险方向的分量是否同样敏感。

- 样本：复用 [既有匹配方案:35](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/FACTORIAL_TASK_INTERFACE_REVIEW.md:35) 的事后41对、每炮一个反馈时钟 `c`，但不预称新诊断的有效配对仍有41对。该选点使用结局，仅可离线诊断。先复核映射、冷算子、源模型、前缀/封存哈希和未扰动cold输出；不能复现则停止。
- 固定两组为辅助先验活跃通道按原始列号奇偶分组；不含MNM和mode_number_n。只操纵在当前风险窗口和两个最近反馈输入窗口中全程有效的通道，其他通道和所有mask保持原样；两组各至少两个可操纵通道且各组实际发生非零改变，否则整对排除并报告。每个128点窗口切成16个8点patch，首末patch不动，中间14个patch分别固定循环移位 +2 或 −2 个patch。构造 `++、+−、−+、−−` 四个输入：第一符号对应奇数组，第二对应偶数组。对每个通道逐点值集合、8点patch集合、起点和mask做精确恒等检查；组划分不是物理家族。
- 在风险点 `c`，以固定冷20映射/slow和零Q/K算四个logit，记 `I_f=(f++ + f−− − f+− − f−+)/4`。如果该图对两组输入是可加的，这个交互严格为零；非零仅代表对**这一人造扰动**的计算交互，不自动是物理关系或预测有效性。
- 在同一个反馈时钟，从原样历史只读重建该次既有 `true_true` 局部头并在四格中冻结；对两个最近反馈输入窗口分别施加同样四格，已到达的真实目标、目标mask和原起点不动。分别计算零Q/K的 `g_ab=∂L_signal/∂s`，再算 `I_g=(g++ + g−− − g+− − g−+)/4`。以**未扰动**风险窗口在 `c` 的 `J=∂logit/∂s` 报告 `||I_g||`、`J·I_g`，以及真实输入下 `g_real` 和 `J·g_real`。这是局部梯度诊断，不执行Armijo或实际更新；必要时复现既有闭式读出，不拟合新的目标或训练任何模型。
- 逐炮报告四格及未扰动值、可用通道、排除数、实际扰动量和重复计算的数值误差；在至少20个完整匹配对上再按炮对汇总绝对交互量及配对重抽样区间，区间只用于探索。若不足20对或哈希/数值一致性未通过，判为证据不足，不调分组、时钟或移位量补样本。若 `I_f` 可辨且 `J·I_g` 相对 `J·g_real` 很小，可支持该扰动下的**接口敏感性差异**；若两者均可辨或均不可辨，则不能支持上述特定差异。这里没有预设“足够小”的效应量门槛，必须报告原始量，不能仅以显著性作决策。

风险：首末patch不变仍会制造拼接跳变，`+2/−2` 也改变各通道在具体位置上的形状；两组非线性交互可能来自LayerNorm、缺测或离分布伪影，并非真实跨诊断物理。奇偶划分只覆盖一种分区、不会覆盖组内关系；风险的非线性主效应也可能被交互抵消。该诊断与顺序判别、保持末值、频谱变换、元梯度训练都不同，但仅能检验固定操作，不能证明可部署更新或新增预警信息。

**缺口/停止决定：**目前缺少这个固定干预在真实源图/冷图上的可辨识性、同一时钟的反馈梯度是否携带风险可用的跨组差异，以及这种差异在未参与开发的新炮上是否稳定且能产生有益的有限步影响。即使诊断呈阳性，也不能据此直接设计或晋升一套关系损失；要先另立可因果执行的方案和对照，独立证明EAST intra收益，再评估intra+inter。不得给最终风险分数叠模块或通过报警调参补成绩。

## 5. 本次核验边界

本次只读了两处AGENTS、当前README、factorial审查及上述事实报告/源码；CPU读取源检查点元数据，核对源文件及5项当前源码SHA，检查 `complete.json`、`audit.json`、辅助汇总及其SHA。原有审计报告记录86项测试、11×360决定及107740条回执重计和74炮实际神经重放；**本次没有重跑这些测试、神经回放或新实验**。无GPU、无训练、无外传数据，也未修改除本报告外的文件。部分源训练脚本与检查点记录哈希不一致，本次不声称完整复现当年的训练，见 [来源限制](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/逐炮保护更新_完整流结果与纠错幅度诊断_20261029.md:109)。
