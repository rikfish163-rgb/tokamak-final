# 条件生成反馈：本地旧实验只读审查（2026-10-29）

## 结论与判定口径

在本次检索和追查范围内，**未找到已完成的“两个风险/阶段条件预测似然，在当前标签未知时对类别边缘化，再用已到达信号的损失更新内部表示”的同一实现及结果**。这是有范围限制的检索结论，不是整个仓库或所有历史代码不存在的证明。

最接近的是两个已有的阶段条件解释实验：一个用两套岭预测，另一个用两类历史路径能量；它们均停在反馈方向筛查，风险网络更新为零。它们的平方误差或能量**不是经校准的概率似然**；后者仅在每个类别内部对路径做 log-sum-exp，未对未知类别边缘化并反向更新。

旧实验**确实训练过依赖输入的反馈/梯度控制器**，不能概括为只试过一套全局投影或全局权重；但那些控制器的历史20炮并非当前工作树的冷20。下文将它们分别列明，不把旧结果移植到当前父模型。

## 最接近的六组旧方案

### 1. 两阶段条件直接预测：均值误差差，无神经更新

- **数据/训练**：EAST历史20炮（4破裂、16安全），四折15拟合/5留出；条件0包含安全炮和破裂前125 ms以外的窗口，条件1为破裂前125 ms内，不是整炮安全/破裂标签。每折冻结其已有父模型，用该折准备炮拟合两套岭回归，另有时间错置和无条件对照。[方法与边界](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261022/条件演化反馈_实测结果与泛化问题_20261023.md:21)
- **前向/损失**：冻结CLS表示分别输出未来25点×23通道变化，比较已到达5/10/25点的共同协方差马氏误差，取`E0-E1`；缺失通道用观测子矩阵。源码明确称其为未校准复合误差，非似然。[拟合与前向](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_forecast_20261023/run.py:45)；[误差](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_forecast_20261023/learning.py:68)
- **因果/范围**：预测发出后等观测到达才评分；没有把分数接到风险或通过它反向更新，神经更新0次。[实际执行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_forecast_20261023/run.py:196)
- **结果/相同性**：5点真实条件为破裂方向1/4、安全方向16/16；错置为1/4、14/16。这是逐炮中位反馈方向，**不是TPR/FPR**，也非类别边缘化更新。[结果JSON](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/conditional_forecast_cv20_20261023/summary.json:3)

### 2. 两阶段历史多路径能量：类内混合，无神经更新

- **数据/训练**：复用上项四折历史20炮与两种阶段标签；每个条件、每个历史炮按CLS或已观测信号上下文找最近片段。每折危险类仅3个独立破裂炮。[方法](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261022/历史路径保留与上下文_实测结果_20261023.md:11)
- **前向/损失**：每条历史路径提供直接未来25点变化；到达后逐路径算马氏误差，类内用`-log mean exp(-E_j)`聚合，再比较`E0-E1`。这是固定温度的未校准能量比较，**类内路径混合不等于未知类别的边缘化似然**。[实现](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/path_mixture_20261023/learning.py:50)；[两类差值](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/path_mixture_20261023/run.py:132)
- **因果/范围**：起点只用前缀，5/10/25点成熟后评分；风险模型与报警不变，神经更新0次，未进入TEST360。[协议源码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/path_mixture_20261023/run.py:96)
- **结果/相同性**：信号上下文、真实标签、多路径、5点为破裂1/4、安全13/16；错置为3/4、0/16。70868真实标签方向转正，但错置也为正；不是成功预警或两个类别边缘化更新。[结果JSON](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/path_mixture_cv20_20261023/summary.json:513)；[错置](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/path_mixture_cv20_20261023/summary.json:717)

### 3. 事件方向中的未来状态预测：有已到达反馈更新，但不是条件似然

- **数据/训练**：保留源分类器与预测头；源事件线性探针按J-TEXT TRAIN1353中有标注的事件端点拟合六类，形成5维事件对比方向。事件教师来自离线整段记录，来源与历史TEST220有重叠，不能当独立物理真值。[探针协议](/media/hetaisheng/044A81D94A81C83E/Fuison/outputs/event_prefix_probe_20261021/protocol.json:2)
- **前向/损失**：源头直接预测H25原信号；预测窗口和真实到达窗口分别送入固定教师，在5/10/15/20/25点的5维事件坐标上计算SmoothL1。不是每个风险/阶段各输出一个预测分布，也没有对当前事件类别做后验或边缘化。[源码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_dynamics.py:82)
- **因果/范围**：完整25点到达后才形成目标，在线只改原末层FFN的临时状态，预测信号不直接输入风险；每炮丢弃，无炮后监督更新。旧协议前30炮用于密度映射，不是现冷20。[运行协议](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_dynamics_experiment.py:71)
- **结果/相同性**：旧EAST240（101破裂/139安全）固定阈值下，事件坐标54TP/28FP，静态61/31，原信号损失57/29，随机方向62/33；不同臂不是等计算量。不是当前360或条件似然结果。[事件](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/event_dynamics240_20261021/event/comparison.json:2)；[静态](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/event_dynamics240_20261021/static/comparison.json:2)；[原信号](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/event_dynamics240_20261021/control/comparison.json:2)；[随机](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/event_dynamics240_20261021/random/comparison.json:2)

### 4. 内部表示补全：当前冷20父模型上的无标签表示更新

- **数据/训练**：两个补全器仅用J-TEXT TRAIN1353训练两轮，普通组为表示MSE，任务组另加保留冻结风险判断的KL；VAL160只作补全诊断。[训练说明](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/内部表示补全_炮内预警配对实验_20261028.md:33)
- **前向/损失**：互补遮挡已观察128点窗口中的8点块，残差MLP补全17×64内部表示；在线仅对同窗补全MSE求导，不是未来信号预测，也不分两类或计算混合似然。[模型与损失](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/latent_completion_20261028/model.py:12)
- **因果/范围**：先计算并封存当前块，再用已完成窗口更新输入嵌入的64增益＋64偏置临时变量；每128点一次，每炮重置，后续点才受影响。[更新与时点](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/latent_completion_20261028/model.py:60)
- **结果/相同性**：当前冷20父模型的EAST完整360中，cold、普通补全、任务补全及任务反序均125TP/32FP；各补全臂11114次实际更新。风险会变化但没有炮级收益；不是拟议的条件似然更新。[结果JSON](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/latent_completion_east360_v1/complete.json:4)；[更新数](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/latent_completion_east360_v1/complete.json:225)

### 5. 输入条件化反馈控制器族：真实存在，但并非生成似然

- **学习标量损失**：旧历史20炮（4破裂/16安全）四折15/5，学习`Lφ(h,c)`；`h`为64维状态，`c`为已观察信号水平、变化、可用率69维。输出非负标量，梯度更新临时第一编码器；历史标签只用于更新后25点的外层BCE，留出端点独立单步，不是完整流。安全BCE 0.033040→0.035925，及时破裂BCE 0.069717→0.061637，安全门槛失败。[实际网络与因果步](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/learned_update_loss_20261023/learning.py:11)；[结果JSON](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/learned_update_loss20_screen_20261023/completed.json:5)
- **上下文权重网络**：另一旧协议20准备炮（4破裂/16安全）＋10校准炮，185→32→23网络用当前已到达残差、水平、变化、可用性及当前分数，输出逐通道权重；加权SmoothL1的梯度改58,240个临时第一编码器参数，首次256点后、每128点更新；历史/炮后标签通过更新后BCE训练规则。360炮在线规则133TP/36FP，均匀及保持值均133/36，平均上下文134/35；不证明实时上下文收益。[源码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/contextual_feedback_20261023/learning.py:22)；[数据和结果](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261022/条件反馈持续学习_360炮结果与数据利用边界_20261023.md:19)
- **有限步条件权重**：另有20历史炮训练、10开发炮评估；23维已到达残差逐通道改变47参数策略权重及力度，组合各通道梯度，临时第一编码器每次从父模型重置。条件权重、全局权重、等幅均匀和置乱均为1TP/0FP；不更新为1/1。10炮只是开发筛查，不是360结果。[策略源码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/finite_conditional_loss_20261025/learning.py:17)；[结果JSON](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/finite_conditional_loss_cal10_20261025/completed.json:36)
- 这三种分别学习损失或按输入变化的梯度组合，**不输出两类预测似然，也没有对未知类别做无标签边缘化**。历史训练使用标签，不应称全程无监督。

### 6. 旧版前20炮跨炮经验：输入条件化选择更新动作

- **数据/训练**：旧360炮序的前20炮为2破裂/18安全；先封存这20炮预测，再用其结局和反事实重放构造1995条局部训练记录。它们与当前17破裂/3安全冷20不是同一组。[旧协议](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/feedback_selector_test360_20261023/protocol.json:5)；[训练数据说明](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261022/20炮经验选择反馈_结果与文献边界_20261023.md:44)
- **前向/损失**：185维已到达上下文输入线性岭模型，预测普通反馈和扣旧偏差反馈相对于不更新的后续风险BCE变化，选择三者中预测效用最小者；不是单一固定投影，也不是生成模型。[策略](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/feedback_selector_20261023/policy.py:41)
- **因果/范围**：前20炮不被重写，第21炮起规则冻结；每128点且至少256点后，基于已到达信号选择是否更新临时第一编码器，炮内累积、下炮重置，当前结局不输入选择器。[执行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/feedback_selector_20261023/execution.py:109)
- **结果/相同性**：旧完整360：上下文选择118TP/27FP，静态118/27，平均上下文118/27，固定偏差反馈119/27；6732次实际更新但无独立净收益。没有类别似然或类别边缘化。[结果JSON](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/feedback_selector_test360_20261023/completed.json:52)；[执行次数](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261022/20炮经验选择反馈_结果与文献边界_20261023.md:92)

## 当前冷20与旧输入条件控制器的区别

当前冷20的ID来自[当前协议](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/attention_weights_cold20_v1/protocol.json:15)，旧选择器前20来自[旧协议](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/feedback_selector_test360_20261023/protocol.json:5)；只读排序求交结果为**0个共同ID**。两组的父模型、炮序、训练和评价也不同，不能互相套用成绩。

对当前17破裂/3安全冷20，本次核到的注意力试验学习的是**一套固定的23通道权重**或一套共享的Q/K基与步长；当前输入会改变被加权/投影的原始梯度，但代码没有把当前上下文送入另一个已训练网络以动态生成权重/基。权重拟合与该差别见[计划](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/ATTENTION_WEIGHT_PLAN.md:5)、[权重源码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_weights_20261029/learning.py:20)、[Q/K基源码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_basis_20261029/learning.py:29)。

当前全局权重的完整360中cold、等权、学习权重、置乱、倒序全是125TP/32FP；每个适应臂10774次非零更新。[结果JSON](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/attention_weights_east360_v1/complete.json:4) 固定基学习最终选零步长，见[报告](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/历史反馈权重与注意力更新方向_配对实验_20261029.md:60)。

还需区分**广义的输入/历史条件化梯度**：当前冷20信号已用于拟合H25辅助读出先验；`local_readout`又用本炮较早、最多16行已到达观测局部拟合读出，再用不重叠的较新目标计算Q/K梯度。它确实让本炮数据改变反馈损失与梯度，不能说当前冷20只试过完全静态反馈；但它拟合的是**信号预测读出**，不是通过历史风险效用训练的上下文梯度控制器。[实现](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_readout_20261029/learning.py:79)；[时序和数据](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/炮内读出校正_完整360与预警收益分离_20261029.md:15) 完整360中本炮读出及倒序均125TP/33FP，cold及固定读出均125/32。[结果JSON](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/local_readout_east360_v1/complete.json:43)

**因此：旧20经验已试过直接学习输入条件化的更新策略；对当前这批冷20及其父模型，本次未找到已完成的同类风险监督动态控制器训练和评估，但已找到输入及本炮历史条件化的预测读出及梯度更新。**这不是全仓不存在的断言。

## 检索范围、核验与未覆盖

- 先检索主目录根README、`experiments/`下README及实验工作树可见README：`conditional`、`class-conditioned`、`energy`、`generative`、`latent dynamics`、`likelihood`、`phase`、`event`、`HMM`和条件、能量、生成、潜在动力、似然、阶段、事件、隐马尔可夫等中文词；再从[实验总索引](/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/README.md:875)追到上列源码、报告和JSON，并在两棵`risk_aligned_online`源码树搜索似然、分布及`logsumexp`相关标识。
- 本次只读核对了条件岭/多路径、输入条件权重及选择器、有限步策略、可学习损失、内部补全等代表性源码的SHA256，与各自运行`protocol.json`登记值一致；读了对应结果JSON。事件状态试验读了当前源码、协议和结果，但本次**未独立核对该试验历史源码快照的逐文件哈希**。上述旧报告中的测试和复算是历史记录，非本次重跑。
- 未全面审计README未索引的其他项目、所有`runs/`压缩快照、Git历史、二进制检查点、原始HDF5、整仓报告或外部文献；符号名称未命中的替代实现仍可能存在。仅分析已落盘开发结果，不作独立泛化或新效果承诺。
- 本次无GPU、训练、数据复制、旧代码修改、提交或新实验；唯一写入为本报告。
