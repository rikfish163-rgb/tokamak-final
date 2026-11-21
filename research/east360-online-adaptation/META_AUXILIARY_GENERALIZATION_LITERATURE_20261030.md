# 历史监督下学习无标签测试时梯度：面向固定 JTEXT 分类器的文献核查

核查日期：2026-10-30。独立文献侧任务；不阻塞主进程的 real-counterfactual 实测。

## 结论与适用范围

最值得迁移的是 **ARM-LL 的“用适应后监督风险学习无标签损失”与 ITTA 的“辅助梯度对齐、测试时只动新增内部参数”**。两者都真正执行测试时梯度，但原文均未证明：对一个已经达标、从离线阶段起就完全固定的 JTEXT 分类器，仅凭 EAST20 学到的辅助损失能够改善新炮预警。MT3 给出合成域上的明确双层训练协议；MedTS-TTT 给出时序分类中的单步快权重实例；Meta-TTT 提供合成域及留域选择参考，但存在需澄清的公式和训练范围问题。

用户报告的 EAST20 拟合有效、新炮 real-vs-mean feedback 优势反转，作为本任务的前提，而非本次独立复核的实验结果。它目前支持“反馈利用规则没有稳定泛化”这一判断；不能单凭该现象断定是优化器、损失容量或某一种域偏移造成。

以下五篇均查阅一手方法全文及对应训练/评估段落；不以摘要代替方法。2021–2023 的文章保留，是因为它们直接定义了本次要区分的机制。论文事实、可迁移判断和本项目提案分开书写。没有任何条目证明发现了物理规律，也不作新颖性声明。

| 方法 | 元训练究竟学习什么 | 部署时是否执行无标签梯度 | 固定已达标 JTEXT 的判断 |
|---|---|---|---|
| ARM-LL，NeurIPS 2021 | 模型初始化及损失网络 | 是，损失网络驱动模型梯度 | 必须改成仅对内部适应状态求梯度；原实验不是这一限制 |
| MT3，AISTATS 2022 | 经 BYOL 内步后易分类的初始化 | 是，单样本增强上的 BYOL 内步 | 固定源模型后已不是原文完整协议 |
| ITTA，CVPR 2023 | 辅助一致性度量；源网络也训练 | 是，仅更新新增适应块 | 测试阶段相容；全程冻结源模型的训练效果未验证 |
| Meta-TTT，arXiv 2024 v1 | mixed-BN 适应参数及初始化 | 是，伪标签和 minimax entropy | 原 BN 会改变；不能认定其离线协议全程冻结分类头 |
| MedTS-TTT，MICCAI 2026 | 经单步快权重更新后的慢参数 | 是，显式单步 SGD 公式 | 原架构端到端训练；改挂冻结风险头是新实验 |

## 1. ARM-LL：直接学习测试时无标签损失，但也学习基模

Marvin Zhang et al., **Adaptive Risk Minimization: Learning to Adapt to Domain Shift**, NeurIPS 2021。

官方：[论文页](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html)；[正文全文](https://proceedings.neurips.cc/paper_files/paper/2021/file/c705112d1ec18b97acac7e2d63973424-Paper.pdf)；[补充全文](https://proceedings.neurips.cc/paper_files/paper/2021/file/c705112d1ec18b97acac7e2d63973424-Supplemental.pdf)。定位：§3 的域生成假设；§4.1 Eq.(2)；§4.2 Algorithm 1 与 ARM-LL 段；§5.1；补充 Appendix B、C。

**原文协议。** 从一个训练域抽取一批样本，先隐藏标签适应，再用该批的历史标签优化适应后预测；同时训练模型参数和损失网络。不是必须另抽一个域作为每步 outer query。用统一记号重写：

\[
v_k=f_{\rm loss}(g_\theta(x_k);\phi),\qquad
\theta'=\theta-\alpha\nabla_\theta\|v\|_2,
\]
\[
\min_{\theta,\phi}\mathbb E_{d,B\sim d}
\frac1{|B|}\sum_{(x,y)\in B}\ell(g_{\theta'}(x),y).
\]

损失网络输入为预测模型的 logits，输出标量；部署固定损失网络，更新预测模型。FEMNIST 测试用户与训练用户分离；不同实验的“域”不全是未见过的域，例如 rotated MNIST 包含训练中稀少的旋转域。Appendix B 明确说 **ARM-LL 未做 streaming 实验**，流式梯度仅为可扩展设想。§3 的理论依赖输入边缘分布能够提供标签条件分布信息，以及域分布等假设，不能当作任意新域保证。[正文 §3–5](https://proceedings.neurips.cc/paper_files/paper/2021/file/c705112d1ec18b97acac7e2d63973424-Paper.pdf)、[Appendix B–C](https://proceedings.neurips.cc/paper_files/paper/2021/file/c705112d1ec18b97acac7e2d63973424-Supplemental.pdf)

**本项目判断与检验。** 可借鉴的是对“更新后风险”训练损失，而非把 forecast 拟合好就算完成。固定 JTEXT 时应把上式的可更新变量换成内部状态 a，θ 保持只读；φ 的监督从历史炮取得。这个限制改变了可达更新方向，原论文结果不随之自动成立。特别是只看 risk logit 的损失，可能学成类别先验纠偏，无法辨认 real 与 mean 信号的动态差异。应比较 logits-only 与内部时序残差条件化的同容量损失，并在未参与训练的整炮上测反馈替换效应。ARM 的批处理设定也不能用当前炮完整轨迹代替实时可见前缀。

## 2. MT3：训练“辅助内步之后”的分类能力，而不是联合损失而已

Alexander Bartler et al., **MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption**, AISTATS 2022。

官方：[PMLR 全文](https://proceedings.mlr.press/v151/bartler22a/bartler22a.pdf)。定位：§2.1 Eqs.(1)–(4)、Algorithms 1–2（PDF 第4–5页）；§2.2（第5–6页）；§3.1 的 joint-training 对照。

**原文协议。** 采用改造 BYOL：任务模型预测两个增强视图，目标投影来自元模型；内步不更新分类头。批级增强在一个任务内共享、任务间变化，以合成域差异。外层穿过内步更新包含特征提取器、分类及自监督分支的元参数：

\[
\varphi_i=\theta-\alpha\nabla_\varphi L_{\rm BYOL}^{i}(\varphi;\theta)
\big|_{\varphi=\theta},\qquad
\min_\theta\sum_i[L_{\rm CE}^{i}(\varphi_i)+\gamma L_{\rm BYOL}^{i}(\varphi_i;\theta)].
\]

实际训练和测试均用一个内步，α=0.1；训练每任务8图、每 meta-batch 4任务；测试使用单图的32组增强。预测后丢弃适应参数，返回元初始化；论文明确未研究持续 online 版本。监督来自 CIFAR-10，评估 CIFAR-10-C。部分训练增强与测试 corruption 类型重叠，作者亦提醒这些收益不能全归因于未见偏移的适应。[§2–3](https://proceedings.mlr.press/v151/bartler22a/bartler22a.pdf)

**本项目判断与检验。** 泛化动机是：训练时就让无标签更新服务于主任务；它没有证明每一次辅助下降都改善主任务。这里学的是易适应初始化及表征，不是任意固定网络外面的通用控制器。源模型和风险头全程冻结后，只能重新训练辅助网络或适应状态的规则，不能宣称复现 MT3。图像翻转、裁剪、BYOL 不变性不能直接移植到逼近破裂的时序；把不同风险阶段压成同一表示可能消除预警信息。可检验的是“固定增强、容量和步数，只改变外层是否穿过真实内步”，以及单次重置和炮内累计的区别；后者必须重新做因果轨迹评估。

## 3. ITTA：学习辅助梯度与监督梯度对齐，测试只更新新增内部块

Liang Chen et al., **Improved Test-Time Adaptation for Domain Generalization**, CVPR 2023, pp.24172–24182。

官方：[CVF 论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Improved_Test-Time_Adaptation_for_Domain_Generalization_CVPR_2023_paper.html)；[正文全文](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Improved_Test-Time_Adaptation_for_Domain_Generalization_CVPR_2023_paper.pdf)；[作者 arXiv 全文备份](https://arxiv.org/pdf/2304.04494v2)。定位：§3.1 Eqs.(1)–(5)、Algorithm 1；§3.2 Eqs.(6)–(8)、Algorithm 2；§4.1 的 training-domain validation。

**原文协议。** 源域标签训练分类；原始与增强表征 z,z′ 的一致性度量可学习。训练交替更新源网络/分类头，以及度量网络 w：

\[
L_w=\|f_w(z-z')\|_2,\quad
\min_{\theta,\psi}L_{\rm main}+\alpha L_w,\quad
\min_w\|\widehat{\nabla_\theta L_{\rm main}}-
\widehat{\nabla_\theta L_w}\|_2.
\]

原 Eq.(5) 的帽号是梯度坐标去均值后除标准差，并非简单单位 L2 归一化。测试时加入逐块内部适应参数 Θ，用 Lw 的梯度更新它们，冻结原特征网络、分类器及 w；采用 online setting。模型选择使用训练域内部8:2验证，目标域留出。它是**学习梯度匹配损失**，不应改述为 MAML 式适应后监督风险的同一套算法。正文用范数、伪代码用 MSE，复现应声明具体尺度约定。[§3–4.1](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Improved_Test-Time_Adaptation_for_Domain_Generalization_CVPR_2023_paper.pdf)

**本项目判断与检验。** 这是测试期冻结原模型最接近的例子，但离线阶段联合训练了 θ、ψ；“测试冻结”不等于“已达标模型从头至尾不动”。另一个移植风险是原对齐发生在 θ 空间，测试更新却在 Θ 空间。在本项目中，应直接在实际临时 Q/K 状态 a 的空间计算历史监督与辅助梯度，再评估有限步后的风险；在一个空间对齐不保证另一个空间获益。标准化后的匹配也不自动保证原始梯度内积为正。可做的关键对照是：同一辅助损失、同一 a，仅比较用源权重空间训练对齐与用实际 a 空间训练对齐。冻结源权重仍允许梯度经源网络传至 a；将整条特征路径 detach 则可能使更新消失。

## 4. Meta-TTT：合成域与元学习 BN，但须保留原文不确定性

Chen Tao, Li Shen, Soumik Mondal, **Meta-TTT: A Meta-learning Minimax Framework For Test-Time Training**。本次方法依据为 arXiv:2410.01709v1（2024-10-02），不把它当作已核验的后续出版定稿。

官方：[HTML 全文](https://arxiv.org/html/2410.01709v1)；[PDF 全文](https://arxiv.org/pdf/2410.01709v1)。定位：未编号的 “Our Method” → “Mixed-BN Adaptation”、 “Meta-Learning for Test-time Training”、 “Batch Domain Shift Synthesis”；Eqs.(1)–(7)、Algorithm 1；Appendix A / “Implementation Details”。

**原文协议。** 测试适应变量是 BN 的 γ、β 和源/当前批统计混合系数 α；其余参数测试时冻结。每个历史训练 batch 先进行无标签 minimax/伪标签内步，再用同批监督 CE 作外层；合成偏移是 ResNet stem 后的通道仿射扰动，调参使用训练域 leave-one-domain-out。不是学习一个任意损失网络。

\[
\mu=\alpha\mu_t+(1-\alpha)\mu_s,\quad
L_{em}=-\mathbb E H(p),\quad
\Theta_\beta\leftarrow\arg\min(L_{pseudo}-\lambda L_{em}),\quad
\Theta_\gamma\leftarrow\arg\min(L_{pseudo}+\lambda L_{em}).
\]

以上按原 Eq.(3)–(4) 记录：代入负熵后，β 的更新与正文“β 最大化熵”的说法相反。Algorithm 1 外层写更新 Θad，附录却设有分类器学习率；因此全程冻结头的训练范围不能确定。原实验一个元内步、batch64，不能视作单流单窗口已验证。[方法及 Appendix A](https://arxiv.org/html/2410.01709v1)

**本项目判断与检验。** 可借鉴的实质是用同域扰动构造训练任务，并用留域结果选规则；其 minimax 公式不宜未经澄清照搬。单炮的相关窗口不提供64个独立域样本，BN 统计也可能主要反映风险阶段变化。若已达标模型不允许改 BN，这条更新路径本身不相容。可以仅把“通道扰动训练＋留扰动家族验证”作为同一个内部状态方案的消融；需要先证明扰动不改变标签含义。它不能保证 real feedback 的方向比 mean 更好，更没有未来结局可用于部署时纠正错误伪标签。

## 5. MedTS-TTT：时序分类有真内层梯度，但原协议是完整片段分类

Mingzhi Chen, Yiyu Gui, Guibo Luo, **MedTS-TTT: Test-Time Training for Medical Time Series Classification**, MICCAI 2026。

官方：[MICCAI 论文页](https://papers.miccai.org/miccai-2026/0646-Paper5029.html)；[接收版全文](https://papers.miccai.org/miccai-2026/paper/5029_paper.pdf)。定位：§2.1–2.3，尤其 §2.2 Eqs.(2)–(5) 与 “Training and inference usage”（PDF 第4–5页）；§3.1–3.2 的 subject-independent split（第6页）。

**原文协议。** CLSA-TTT 在层内由当前片段产生 Q,K,V，并执行：

\[
Z=KW_{fast},\quad
L_{TTT}=\tfrac12\|LN(Z)-(V-K)\|_2^2,\quad
G=\partial L_{TTT}/\partial Z,
\]
\[
W_{fast}'=W_{fast}-\eta K^\top G,\qquad \Delta=QW_{fast}'.
\]

训练 forward 包含该内步，监督 CE 更新慢参数；测试保留相同无标签单步。它不是仅用前向网络直接生成矩阵，也不是预测未来信号的 ordinary forecast loss。原文为自建 tokenizer、gated convolution backbone 与分类器的端到端框架，四个 EEG/ECG 数据集按受试者划分 train/validation/test，以验证 macro-F1 选模型。§2.2 明确为 full-sequence alignment。[§2–3](https://papers.miccai.org/miccai-2026/paper/5029_paper.pdf)

**本项目判断与检验。** 可迁移假设是历史分类标签通过训练内步，使潜在自监督目标在新个体上有用；这仍依赖慢表征共同学习。原文没有验证任意已固定风险头下的插入式版本。片段 X 完整到达后分类可满足片段级因果性，但不能拿一个片段后部的 K,V 帮助预测其早期时刻；必须把输入截到决策截止点，并检查 tokenizer、卷积、归一化和池化都遵守这一边界。受试者留出不等于持续状态无漂移；未见原文给出本项目所需的逐时刻报警评估或跨片段累计验证。可检验对照为相同更新预算下的已到达前缀版、每窗重置版和炮内累计版，禁止用完整炮版本作为可部署结果。

## 算法边界：本任务需要的是哪一种学习

本节是对本项目计算图的分析，记冻结预测系统为 Fθ,ψ(x;a)，θ 为源网络、ψ 为风险头，a 为允许变化的内部状态。

| 路径 | 部署计算 | 历史标签用在何处 | 能否据此称为测试时梯度训练 |
|---|---|---|---|
| 既有 direct weight generator | a=Gφ(c≤t) | 离线训练 Gφ | 没有额外内步时不能 |
| 普通 forecast 辅助训练 | a⁺=a−η∇a Lforecast | 可只用于此前模型训练；辅助目标由信号定义 | 有梯度，但未必学过“此更新改善风险” |
| learned auxiliary loss | a⁺=a−η∇a Lφ(c≤t,a) | 训练 φ，使内步后的历史风险改善 | 是，前提是部署实际执行这一梯度 |
| learned fast-weight layer | W⁺=W−η∇W Lφ(X≤t,W) | 训练慢参数及内部目标 | 是；解析矩阵形式执行梯度也算，不要求调用某个 backward API |

TTT-MAE 候选未纳入五篇核心元训练方法：它的默认协议是 MAE 预训练后冻结 encoder 训练分类 probe，测试再更新 encoder/decoder 做重建；不是用适应后监督损失进行双层元训练。这个边界来自其 [§3、Eqs.(1)–(2) 全文](https://proceedings.neurips.cc/paper_files/paper/2022/file/bcdec1c2d60f94a93b6e36f937aa0530-Paper-Conference.pdf)，也说明“TTT”名称本身不足以判定训练协议。本次不重读 Khan，也不将其结果纳入新的有效性证据。

一次梯度步本身也是输入的确定性函数。若 a 固定初始化、只执行一步，它可能被展开成某种生成器；“代码里出现 gradient”并不证明表达力更强、利用了动态反馈或产生了新方法。必须核对部署图及干预：当前残差是否实际改变 ∇aL，累计状态是否影响下一步，更新后风险是否在新炮上获益。

## 对当前反转问题的可检验机制建议

建议只立一个候选：**在实际内部 Q/K 状态空间学习辅助度量，以历史炮的因果内步后风险作外层目标；源网络和风险头全程固定。** 这采用 ARM-LL 的目标思想，参考 ITTA 的梯度检查，但不是任何一篇的完整复现。以下是提案，本任务没有实施或验证。

选择既有内部适应位置和固定容量状态 a。设 r_t(a) 为主进程已经定义、在 t 时刻全部成熟的信号残差或有效双视图残差；φ 只参数化小型非负度量，例如：

\[
L_\phi(a;\mathcal F_t)=\tfrac12 r_t(a)^\top M_\phi(c_t)r_t(a),\quad
M_\phi(c_t)=R_\phi(c_t)^\top R_\phi(c_t)+\epsilon I,
\]
\[
a_{t+1}=\Pi_{\mathcal A}\big[a_t-\eta\nabla_aL_\phi(a_t;\mathcal F_t)\big].
\]

F、风险头、报警阈值和持续规则保持原样；φ 离线训练后也冻结，部署只更新 a。Π 使用主进程现有明确的状态约束，不借文献侧任务另增一套在线调参规则。固定 η、容量和更新时钟用于首轮比较。M 的非负性只能保证辅助二次型非负，不能保证风险改善；若 r 与 a 无导数，或两视图相同导致零残差，这个候选不会产生有效更新。低秩参数化也需避免两个乘子同时零初始化造成梯度全零。

对训练历史中的每炮 j，以真实反馈时钟回放前缀内步，再由同一历史炮的后续决策标签形成外层损失。可用以下形式表达相对冷模型的分组变化：

\[
\Delta_{j,k}(\phi)=L^{post}_{j,k}(\phi)-L^{cold}_{j,k},\qquad
\min_\phi\max_{k\in\{safe,danger\}}
\frac1{|\mathcal H_k|}\sum_{j\in\mathcal H_k}\Delta_{j,k}(\phi).
\]

这是避免只优化总体平均值的一个候选外层准则，不是文献给出的保证；阶段的权重、尺度和风险代理必须事前固定。不要只靠这个均值：留出集仍需逐炮列出收益与伤害、首次报警及有效提前量。不能用被评估的新炮结局挑选 φ、η、训练轮次或报警阈值。

ITTA 式梯度诊断应计算在**同一个 a_t**：历史监督参考梯度与 ∇aLφ 的未标准化内积、余弦和实际有限步风险变化。方向对齐是局部诊断；非线性、累计步数和风险阶段转移都可能使最终报警受损。它不应取代留炮的真实反馈反事实检查。

## 因果时序与泛化检验：供主进程独立采用

1. **单位为整炮。** 外层训练、模型选择和最终新炮评估按炮分离；同炮窗口、同炮增强不得跨分区。训练任务中可以用“前缀无标签内步＋该历史炮后续已知标签外层”；这只是离线模拟。验证炮标签只能打分/选配置，最终测试炮标签只作冻结方案的回顾评分。若 JTEXT 冷源或其他固定组件此前已看过这些 EAST 炮，应标为条件性验证，而非从未暴露的留出。
2. **截止点明确。** 建议采用 predict-then-update，并与主进程的反馈到达定义对齐：t 的输出先产生，t 新成熟反馈只作用于随后输出；已作出的历史预测不重算。r_t 只能由当时已到达的信号计算。若原预测 horizon 尚未成熟，该残差不能进入内步。模拟训练、real、mean、反事实及最终评估使用相同时钟。
3. **留扰动类型，不只留随机种子。** 在历史训练炮内部模拟有依据的测量扰动，例如已确认不改标签语义的增益偏差、偏置或短时缺测；先固定家族和范围，同一模拟域对适应前缀和外层预测输入保持一致。验证同时留出整炮及一种扰动家族；留强度只能说明强度外推。改变物理进程或报警时间定义的变换不得随意继承原标签。合成任务再多也不增加独立历史炮数量。
4. **主要负对照只替换反馈。** 保留当前实际预测输入、截止时刻、源权重和初始状态，把进入辅助更新的 feedback 换成预先定义、因果可获得的 mean 或匹配幅度的错配片段。mean 统计不得来自未来或测试炮标签；错配可取已到达过去片段，不能按当前真实 danger 标签挑选。单步干预从同一个 a_t 分叉；完整回放允许状态自然分化并另行报告。
5. **与当前结论直接对应。** 至少同时保留冷模型、无标签普通 forecast 内步、learned-loss real、learned-loss mean/错配，以及主进程已有的 direct-generator 参考。训练外层和更新预算保持可比。不要强迫反事实达到相同非零更新率；记录更新率、状态范数和风险 logit 改变量，区分方向、幅度及更新频率的作用。
6. **预先允许否定。** 若辅助损失下降而新炮报警受损，判候选失败；若 real 不优于 mean/错配，只能说明某种通用变换或正则化可能有用，不能声称有信号驱动适应收益。若 real 在某类炮改善、在 danger 反转，应完整报告，不能靠整体平均掩盖。整炮只有20时，验证方差与选择偏差会大；报告每炮差异比把重叠窗口当独立样本的显著性更有信息。

这里的泛化理由仅是：训练显式优化真实更新路径，历史任务覆盖可解释的偏移，测试规则在未见炮和未见扰动下保持有效。这是一条可证伪的经验路线，不是“元训练必能泛化”或“已学到物理规律”的论证。

## 核查与交付记录

本轮采用 deep-research 的一手全文核查方式，最终主报告选择五篇；另读 TTT-MAE 方法作为排除项。检索限于明确候选及其直接相关方法，没有泛检索100篇。来源为 PMLR、NeurIPS、CVF、作者 arXiv 和 MICCAI 官方文本；公开 PDF 通过网页解析或内存管道转成文本读取，没有保存论文副本。Meta-TTT 只据注明的 v1；未将检索到的后续出版元数据当成已读定稿。

本地仅只读工作树指令及历史审计中的相关段落；当前 EAST20/newshot 反转未读取原始数据复验。未运行 GPU、训练或项目代码；未上传任何私有文本、数据或模型。唯一写入文件为本报告。当前接口不提供模型/推理档位切换，不能确认 terra/max 或独立子代理审阅；来源核查由同一研究上下文完成。

给主进程的一项建议：优先把 **“实际内部状态空间的 learned auxiliary loss＋历史因果内步后监督训练”** 列为候选；决定其价值的主要负对照为 **同状态、同截止时刻，仅把 real feedback 换成因果 mean/错配**。只有新炮的风险与报警结果同时支持它，才升级为有效机制。
