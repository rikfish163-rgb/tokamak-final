# 补充证据登记：特权标签、在线学习与工况变化

## 2026-10-29续：炮内读出、有限步更新及任务归因

- 三名显式`gpt-6.1-terra/max`子代理完成代码审查、360炮画像和针对性原始文献核对；fast接口不可设置，没有声称开启。文献子任务的六篇阅读范围和局限在[备忘录](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/LOCAL_READOUT_NEXT_HYPOTHESES_TERRA.md)，不是六篇新增全文或百篇综述。
- 主进程复核[Zhu等情景适应作者稿](https://arxiv.org/pdf/2109.08956)第2节：其低／高工况依据全平台期的参数划分，本项目起始Ip/Bt代理不等同于原文工况或破坏力。
- 主进程复核[Keith等正式论文](https://link.springer.com/article/10.1007/s10894-024-00413-y)模型和比较方法节：逐炮首次报警与所需提前量不同于逐点分类误差；其同装置充分数据实验不是本项目跨装置性能保证。
- Murari论文主页成功读取，但后续特定段落定位受认证重定向影响；该条详细方法阅读为子代理记录，主进程不冒充又完成一遍全文核验。
- 新完整360：本炮读出125TP/33FP，cold125/32，倒序也125/33；候选不晋升。CPU前缀反例证明固定归一化步长可越过局部下降区间，但没有证明缩步会救回未来炮。
- [全部五组结果、公式、验证范围及下一步取舍](炮内读出校正_完整360与预警收益分离_20261029.md)。既有原基模、阈值、上游物理语义限制均保留。

## 2026-10-29续：固定表示的可读性与跨炮预测泛化

- [Zeng等AAAI2023正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/26317)：核读任务定义、直接多步线性方法、NLinear及顺序消融；未核验附录和作者代码，不新增全文阅读计数。只借鉴简单基线和相对最后观测的预测对照；其通用长时距预测不是EAST破裂任务。
- 本地固定神经表示，闭式岭回归训练误差很低，但EAST前20炮条件留炮仍不及保持末值；不是新在线成绩。固定映射和冷启动参数已使用20炮，不能称独立交叉验证。未读取TEST220、后340炮或任何新结局标签。
- [全部结果与边界](冻结表示读出与跨炮泛化_源域及冷启动诊断_20261029.md)。没有将简单线性探针包装成新物理机制；本轮科研目标未达成。

## 2026-10-29续：辅助头校准完整流与未来窗口基线

- [TTT++正式论文](https://proceedings.neurips.cc/paper/2021/file/b618c3210e934362ac261db280128c22-Paper.pdf)：重新核读§2.2–4.1、式1–6及三个假设，非新增全文计数。辅助任务与主任务关系、分布条件需要检验；不能把其理论直接作为单炮危险识别保证。本轮未核验作者代码。
- 普通预测头完整360为126TP/32FP，cold125/32，但倒序及保持末值同为126/32；前20炮校准头127/34，没有J净增益。
- 新的1352个下一窗口检查覆盖后340炮，更新后预测误差确有下降，但校准头仍为保持末值误差的9.95倍，逐炮0/340优于保持末值。属于本地抽样诊断，不是文献结论或物理机制证明。
- [全部对照、实现细节和验证记录](预测头校准_完整360与下一窗口基线_20261029.md)。原分类基模不变，科研目标未达成。

## 2026-10-29续：辅助头位置匹配与时效／分组目标

- [MT3官方原文](https://proceedings.mlr.press/v151/bartler22a/bartler22a.pdf)：重新阅读§2–2.1及相邻讨论，非新增全文计数。源训练优化适应后的主任务是设计启发，不保证本地冻结分类器的诊断反馈能改善EAST。
- [Sagawa等ICLR2020作者稿v2](https://arxiv.org/pdf/1911.08731)：阅读§1–2和式1–3；分组鲁棒训练与分组泛化需分开。本地max分组“风险变化+方向损失”只是受其启发的实验，不是原算法复现或理论保证。
- 两组源辅助头及后续四格真实训练均完成。位置匹配未解决风险普遍下降，最新反馈也未救回危险阶段；最差分组只减轻损害。没有新的EAST结果，完整负结果见[实验报告](匹配更新位置与反馈时效_四格配对实验_20261029.md)。
- 代码快照遗漏`__main__`入口已修复并完整同种子复跑；该复跑不是独立实验种子。没有“已学到物理规律”或“100篇全文新阅读”的声明。

## 2026-10-29续：保护方向完整流与幅度诊断

- 五臂完整360全部125TP/32FP，无炮内收益；训练内逐炮保护不代表未来不恶化。新增事后包络表明，这一轮观测到的概率改动幅度不足以纠正任何原错误炮。不是神经架构上限，也不是在线成绩。[结果与代码索引](逐炮保护更新_完整流结果与纠错幅度诊断_20261029.md)。
- Sun2020重新打开官方原文、核对相关定理检索结果；Fentazi2026作者稿补读§5.4–7。均为既有文献复核，不新增“全文已读”数量，未核验后者同行评审。
- 本地源检查点记录的token元适应与目前Q/K部署位置不同；待做匹配位置实验。文献不直接证明这是本项目全部失败原因。

## 2026-10-29续：梯度谱分解及类别损害抵消

- [SubGD，CoLLAs2022](https://proceedings.mlr.press/v199/gauch22a/gauch22a.pdf)：核读方法3.1–3.2及相邻推导；原方法使用未中心化梯度/更新自相关，非本地H/G交叉矩阵。未读全篇附录，未核验作者代码。
- [ATTITTUD，ICLR2021作者稿](https://arxiv.org/pdf/2108.11346)：核读第3–4节、式1–2、定理条件和SVD实现讨论；OpenReview直连遇到验证页，因此使用arXiv作者稿。当前炮缺少真实监督，不能照搬其当步主任务梯度或宣称理论保证。
- 同一cold20后的完整360谱方向候选125TP/33FP，cold125/32；方向倒序对照相同，无收益。历史平均损失下降却损害破裂类；随后逐炮约束诊断找到微小历史改善，尚未验证未来收益。详情：[本轮证据](炮内更新方向与类别抵消_完整流及约束诊断_20261029.md)。
- 已有gradient_transport和子空间方法保留为历史，不把谱投影改名成已证实的新物理机制，不新增100篇全文声明。

## 2026-10-29续：学习更新方向的文献复核与负结果

- [MT-net，ICML2018](https://proceedings.mlr.press/v80/lee18a/lee18a.pdf)：本轮核读§3.1–3.2、式3–13、算法1–2；借鉴学习适应子空间与共享变换，不声称逐炮Q/K低秩改写是原算法或全新概念。
- [MT3，AISTATS2022](https://proceedings.mlr.press/v151/bartler22a/bartler22a.pdf)：补读§2–2.1、式1–4及§3相关结果，非新全文阅读数量。文献讨论普通自监督测试时更新不一定改善分类；本地检验更新后的报警损失，不把图像结果作为EAST保证。
- 仓库已有源域FFN子空间和通道权重学习；本轮明确区别为已结束cold20指导当前第一层Q/K。两种方向拟合均将幅度置0，不伪称神经适应成功。
- [本轮三组完整360开发流与解释](历史反馈权重与注意力更新方向_配对实验_20261029.md)。未增加“100篇全文已读”声明，也未证明物理因果关系。

## 2026-10-29续：注意力更新位置与通道梯度方向

- [Hu等LoRA作者稿](https://arxiv.org/pdf/2106.09685)：本轮核读§4.1–4.2，前轮已读§7.1相关位置消融。借鉴冻结原权重、训练低秩增量；本地A行正交而非原文高斯，逐炮重置，不借用NLP结果证明EAST有效，不把LoRA称为本项目新方法。
- [Ren等ICML2018正式论文](https://proceedings.mlr.press/v80/ren18a/ren18a.pdf)：实际读§3.1–3.3、式1–12和算法1，未核验全部附录/作者代码。原文用小规模有标签验证集指导训练样本重加权；本地仅作通道梯度事后诊断，真实未来端点标签不能进入部署策略。
- 当前[完整360和通道方向结果](注意力炮内更新_完整360失败与通道反馈方向_20261029.md)为负结果。通道权重控制器在旧实验已有实现，不能换名宣称新创新；后续若复用，必须交代父模型、更新位置及反馈目标的差异。
- 本节不是100篇全文阅读证明。文献主张与本地实验发现分别列出，未证明新的物理规律。

## 2026-10-29续：反馈方向与当前更新位置的可达性

- [Sun等ICML2020作者论文](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf)：补读§4、Theorem1和相邻论述，非新增全文阅读。相关梯度的充分条件有光滑/凸及梯度有界等假设；本地仅据此设计方向诊断，不宣称Transformer保证。
- [Zhao等ICML2023论文](https://proceedings.mlr.press/v202/zhao23d/zhao23d.pdf)：本次读§4.2–4.3、算法1及相邻在线批依赖讨论，未把全文和附录计为已读。其oracle利用标签做模型选择；本地用标签直接选择梯度方向更宽松，不是同一算法或可比较成绩。
- [当前冷启动父模型的398端点、两位置配对结果](冷启动模型更新可达性_端点与位置配对诊断_20261029.md)：只检查访问过状态的功能作用。有限优化不证明全局上限；原峰值压低不等于整炮FP救回，不能改写成完整360在线增益。

## 2026-10-29续：时间关系辅助任务与更新后的下游目标

- Shuffle and Learn：[作者稿v2](https://arxiv.org/pdf/1603.08561)，本次补读§3.1–3.3、图2及§4.1开头，不重复计作新文献。原文将正向和完整反向三元组均作为有序正例；强调固定首末帧和运动采样，避免歧义。本地采用128点的首末片固定、内部片重排，所有通道联合移动；不是原三元组CNN的复现。已实现独立辅助头，训练不改源分类器；是否有EAST预警收益须看整流配对，不从辅助判别率推断。
- Ziakas与Russo：[VITA作者稿v2](https://arxiv.org/html/2506.10085v2)，2025-09-29；标题与v1的*Test-Time Adaptation for Generalizable Task Progress Estimation*不同。读§2–4.6及§6限制，未读附录实现或核验正式接收状态。冻结CLIP，学习适配器及K/V/Q投影和下游头，通过更新后的监督误差训练自监督任务；任务是机器人进度而非破裂。专家示范的t/T标签和VOC不能代替破裂标签、30ms预警或FPR。本项目只借鉴“辅助任务成功与更新后主任务成功分开检验”，不把已有元学习负结果换名成VITA成功，不引入最终风险修正模块。
- 本次检索另找到ST2ST和视频流TTT；未核读全文，本节不据其搜索摘要作实现或性能结论。OATTA为既有条目，不重复计数。

## 2026-10-29新增：补全目标的梯度路径

- [SimSiam作者稿](https://arxiv.org/pdf/2011.10566)：实际读算法1、§3、§4.1–4.2及相邻消融。固定其它条件比较stop-gradient的思路用于本地梯度路径消融；其ImageNet表示学习结果不能证明本地已经发生坍塌，也不保证EAST预警收益。CVF正文直连403，作者稿可读，不混淆读取渠道。
- [OATTA](https://arxiv.org/abs/2601.21012)：仅摘要；无梯度递归贝叶斯预测修正，不作为本项目神经炮内更新候选。
- [Shuffle and Learn](https://www.ri.cmu.edu/pub_files/2016/10/0161.pdf)：实际读摘要、引言及§3.1；顺序判别存在歧义。尚未实现，不宣称论文全文阅读或本项目实验收益。
- MAPU/E-MAPU：再次核对式14与算法1，原目标对完整/遮挡两路径均依赖目标编码器；本地另测停止一侧梯度，不声称这是论文原实现。
- 本轮诊断及独立360实验均保存在大盘worktree `Fuison-intra-trajectory-20261028`，事前约定为 `COMPLETION_GRADIENT_PLAN.md`、`GRADIENT_SPLIT_STREAM_PLAN.md`。文献条目不替代实测证据。

## 2026-10-28新增：内部时序补全与任务相关性

- MAPU/E-MAPU：arXiv:2406.02635v2，实际读§III-B2、§III-C1、式5/14、算法1及实验设置开头。用来构造固定源编码器的补全器教学和炮内一致性对照，不复制其完整E-MAPU、熵最小化或伪标签系统。
- CATI：doi:10.1016/j.ipm.2026.104658，仅出版社摘要及索引引言，全文直连403。不声称已读其完整梯度加权实现；本地概率蒸馏对照不是该论文复现。
- [方案、公式、阅读边界和实测记录](内部表示补全_炮内预警配对实验_20261028.md)。未将搜索到但未获得正文的其他条目计为已读论文。

## 2026-10-28续：成熟反馈与合成诊断

- Fentazi等arXiv:2610.12232v1：重新核读§3–4，成熟反馈时点和闭式RLS对照；不重复计新论文。不是本项目逐通道岭回归的复现或EAST收益保证。
- Liu等arXiv:2606.08462v1，*Experiment-free disruption prediction for new devices enabled by synthetic diagnostic data augmentation*：本轮读§1–3及结论，代码/正式出版状态未核验。使用NIMROD与诊断几何、高频数组，不能由23维派生量直接复现；EAST→J-TEXT方向及指标不能与本项目直接比较，目标实验信息参与CORAL的范围仍待核查。
- 沈呈硕IAEA event454/contributions43124前兆链条目仅检索摘要，正文403，不计全文。

链接、实际采用范围及试验结果见[本轮完整记录](炮内反馈失效_变化约束与突变放大实测_20261028.md)。未宣称100篇全文或这些文章证明了新物理规律。

## 2026-10-28再续：炮内背景适应与J-TEXT阶段监督

|标识|实际读取范围|设计影响与限制|
|---|---|---|
|YokoyamaJT60SA115440|[出版社](https://doi.org/10.1016/j.fusengdes.2025.115440)官方搜索索引§1–4、公式、表1和图示说明；直连403，未审原PDF图像|炮内背景学习与异常变化分开处理；专门磁模输入及统计检测不等于23维神经破裂适应。|
|Shen2026MILOnsets|[IAEA官方摘要](https://conferences.iaea.org/event/451/contributions/42899/)索引完整Description；直连403，未取得完整讲稿|与本地阶段标签相关，但原始onsets和最终合并区间不是同一个对象；不把J-TEXT阶段真值用于EAST在线输入。|
|BurliJET_readupdate|既有[作者v1](https://arxiv.org/html/2502.12182v1)，本次复核§2–6|完整诊断筛选、原始/调整后误报与报警持续条件分开；不是新文献计数。|
|Viewmakers_readupdate|既有[作者v1](https://arxiv.org/html/2410.11065v1)，本次复核§1–6、附录A/B和表4开头|零样本反例和非多种子限制保留，不将时序增强冒充炮内更新。|
|Koshizuka_readupdate|既有v1方法/评价及[当前v2](https://arxiv.org/html/2609.24443v2)引言；v2完整方法阅读沿用此前记录|不混用版本数字、40ms与本项目30ms计分，不新增计数。|

原始标签字段说明已完整核对，并对源TRAIN1353/VAL160清点；没有读EAST或TEST信号。
[本轮实测、信息边界与候选组件](炮内背景适应_跨团队文献与事件标签核查_20261028.md)。

## 2026-10-28续：状态条件、解释边界与低代价投运

|标识|本次实际读取范围|能借鉴什么；不能声称什么|
|---|---|---|
|FarreKaga2025v3|[作者稿2502.20294v3](https://arxiv.org/html/2502.20294v3)，§I–IV及附录A–C文字；核对arXiv版本页，相关正式DOI为10.1063/5.0311201，未将期刊版本与预印本逐字比对|使用剖面、控制及专门实验解释撕裂模风险；背景分布影响SHAP。6050炮、按炮80/10/10；不是少量EAST炮内更新。其100ms末次越阈口径不同于本项目首次报警30ms，不能混比成绩。|
|Perez2018FiLM|[AAAI官方PDF](https://cdn.aaai.org/ojs/11671/11671-13-15199-1-2-20201228.pdf)，§1、§2.1–2.2及相关工作开头；非全文复现|按条件逐特征调制是已有技术。本项目仅借鉴内部条件作用形式，不把FiLM、图像成绩或通道缩放本身当作炮内适应创新。|
|Yi2026DBC|[IAEA官方会议摘要](https://conferences.iaea.org/event/451/contributions/42900/)搜索索引完整Description及作者信息；直连403，未读挂载PPT|区分计划参数决定的potential DBC与诊断决定的actual DBC；低、中、高工况预算分开。这是会议方案，不是本项目经过验证的损伤公式。|
|Zheng2025FEC_DBC|[IAEA会议稿](https://conferences.iaea.org/event/392/papers/36206/files/13939-FEC2025_proceeding_WeiZehng2.pdf)搜索索引返回DBC定义、五类划分及图4讨论；直连403，非完整PDF阅读|原稿讨论Ip/Bt以及热能、CQ等量。索引公式存在分支文字重复，未据此复刻公式；不能把当前前100ms排序代理称为实测损伤。|
|ZhengShen2026DLD_readupdate|[出版社ae8d38](https://doi.org/10.1088/1741-4326/ae8d38)官方索引返回§1、3–7大段正文及表1–3、式2–3；此前已有同文献，属补读不新增计数|密度梯度、空间不对称和辐射共同参与模型归因；当前23列缺少对应空间诊断，不能由ne_nG臆造这些量，也不能把SHAP直接写成因果发现。|

本次取得的方法与本地实验对应见[研究记录](多团队文献_内部方向诊断与条件读出_20261028.md)。
未取得的Bardóczi2024正文和其他搜索候选不计为已读方法；KSTAR、MGTTA仅复核旧记录，不重复计数。

## 2026-10-28新增：轨迹预测、系统辨识与延迟反馈

本次增加两篇方法阅读及一项官方会议摘要；不把相关期刊书目算作已读正文。
论点、链接和实际读取范围集中见[本轮记录第2节](跨团队文献与通道反馈失衡_20261028.md)。

|标识|来源状态与范围|证据等级及限制|
|---|---|---|
|Luo2025Tearing|AAPPS-DPP2025官方单页全部文字；2026期刊ae9d94只取得书目信息|可支持直接多时距预测活动的设计动机；不作完整方法复现或本项目TPR依据|
|Laiz2025DCL|ICLR2025官方记录；作者v2的§1–5、§7、H及I部分|状态辨识存在假设；式6用下一时刻观测，不能无修改地用于当下因果路由|
|Fentazi2026Delay|arXiv2610.12232v1，§3–7及主要表格|未核实同行评审；支持检查反馈时点，不证明本项目更新有效|
|Wang2025TCV、Keith2024Risk、Murari2024Control|旧文献的方法/评价复核|不新增计数；不把物理状态、控制记录和不同计分口径直接移植到23维数据|

本轮检索再次找到Shen对齐、Xue物理先验及Zheng密度极限论文；已有登记不重复计数。
网页无法取得的期刊全文未被声称已读；没有声称穷尽所有论文或完成100篇全文审查。

核查日期：2026-10-26。这是阅读范围及拟用结论登记，不是全文阅读证明。
以下均为作者、出版社、官方会议或装置机构来源；搜索索引日期不作为发表日期。

| 标识 | 一手来源、版本与读取范围 | 拟用结论 | 禁止外推 |
|---|---|---|---|
| Vega2024LUPI | J. Vega等，*Real-time disruption prediction in multi-dimensional spaces leveraging diagnostic information not available at execution time*，Nuclear Fusion 64,046010，DOI10.1088/1741-4326/ad288a。[作者机构PDF](https://documenta.ciemat.es/server/api/core/bitstreams/c2824265-7263-41ae-b534-3b051a7a3319/content)。读摘要、§1–4、§5主要实验和表1–6；非逐字全文审阅 | 训练时可用执行时缺失的特权信息；时间点数不能当启动炮数；炮后再训练可改善后续预测 | JET同装置，不是J-TEXT→EAST；及时标准不是严格30ms；没有证明炮内自适应有效 |
| Chen2023ITTA | Liang Chen等，*Improved Test-Time Adaptation for Domain Generalization*，CVPR2023。[官方页面](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Improved_Test-Time_Adaptation_for_Domain_Generalization_CVPR_2023_paper.html)、[作者稿](https://arxiv.org/html/2304.04494v2)。读§3.1–3.2、算法1/2、§5.3–5.4；未完整审阅补充材料 | 辅助任务与可更新参数需共同设计；源训练中的梯度方向对齐是一种既有思路 | 图像域泛化，不保证时变单炮效果；本项目方向损失不是ITTA复现，也不能声称方向对齐首创 |
| Lee2025KSTAR | Jeongwon Lee等，*Machine learning based disruption prediction using long short-term memory in KSTAR*，Nuclear Fusion65,086017，DOI10.1088/1741-4326/adeb9b。[期刊入口](https://doi.org/10.1088/1741-4326/adeb9b)。已读取官方索引返回§1–5正文及表；再次直连被robots拒绝 | 利用不稳定起点构造训练区间，结构不复杂；标签时段及报警持续条件显著影响可比性 | 不是我们的跨装置30ms任务；使用的控制/平衡诊断不全在本项目23维中 |
| Barr2022Continual | Jackson Barr等，*Active and continual learning of fusion plasma turbulence surrogate models for digital twinning of a tokamak device*。[作者PDF](https://realworldml.github.io/files/cr/paper24.pdf)、[ReALML@ICML2022官方录用列表](https://realworldml.github.io/icml2022/accepted/)。读§1–4，未审附录 | 面对运行模式/壁材料变化，主动取样和回放有研究先例 | 是湍流代理回归与仿真标签预算，不是破裂预警，也不是主会论文 |
| Liu2026Synthetic | Zhiqiang Liu等，*Experiment-free disruption prediction for new devices enabled by synthetic diagnostic data augmentation*，[arXiv:2606.08462v1](https://arxiv.org/html/2606.08462v1)。2026-06-07预印本；读摘要、§2.1、§3主要实验及讨论，未完整读模拟细节 | 合成诊断及频域/协方差对齐用于跨装置探索；必须核对目标域数据访问 | 文中目标实验特征的访问预算与时点不充分清楚，不作为严格零目标样本证据；不能凭标题认定已正式发表 |
| Gupta2026Robustness | Neerav Gupta，*Benchmarking Sensor Robustness in Plasma Diagnostic Models: A Systematic Evaluation on TokaMark*，[arXiv:2607.11915v1](https://arxiv.org/html/2607.11915v1)。2026-07-05预印本；读摘要、§4.3、§6.7 | 传感器缺失/偏移值得单列压力测试；报警子实验缺少安全炮，不能算FPR | 不能把其高TPR当本项目应达到的对照性能；未独立验证其代码 |
| ITERCommissioning | [ITER官方Disruptions FAQ](https://www.iter.org/faqs?thematic=76)，读破裂后果及逐步投运部分；不是论文 | 低电流、低能量启动符合装置投运思路；损伤涉及热、电磁及逃逸电子负载 | 不能从这份机构说明推出本项目Ip/Bt乘积是经过验证的损伤量 |

## 拟写结论与证据属性

1. “标签在炮后可用”应与“炮内能拿到真实破裂标签”分开：工程合同；Vega2024LUPI提供相关范式，不是本实现证明。
2. J-TEXT事件标签作为训练教师的信息、EAST仅用前缀观测：本项目拟议设计，受Vega2024LUPI启发，尚未实现事件教师。
3. 更新信号损失下降不保证预警改善：本地诊断结果；Chen2023ITTA解释相关方法学问题，但不为本地具体计数背书。
4. 低参数不等于安全结局、低损伤：装置说明与本地冷启动构成共同约束，必须注明代理假设。
5. 文献成绩的样本单位、提前量、标签时点、是否含安全炮不一致：逐项核对，不作混合排行榜。

## 本轮新增与读取范围更新

| 标识 | 一手材料与实际读取范围 | 可用启发及限制 |
|---|---|---|
| Rajput2025Online | *Uncertainty Guided Online Ensemble for Non-stationary Data Streams in Fusion Science*，[作者稿v1](https://arxiv.org/html/2511.02092v1)，读§3.1–3.4、算法1及§4。作者主页列2026出版版为Machine Learning with Applications 24,100894；这里的方法细节仅据v1，不将其页首投稿期刊名称当出版信息 | DIII-D线圈变形回归，历史缓冲区取1/5/20/40/200炮；每次先预测，真值到达后更新。不是破裂分类，不能引用其80%误差下降作为本项目TPR增益，也不能用专家权重适应替代内部表示适应。 |
| Bonalumi2024XAI | L. Bonalumi等，*eXplainable artificial intelligence applied to algorithms for disruption prediction in tokamak devices*，[Frontiers in Physics 12,1359656](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1359656/full)，读§2、§3、§4及§5主要讨论 | 比较温度剖面相关的不同破裂路径，用遮挡和敏感度核查模型依据；也讨论热图偏置。其密度、温度、辐射空间剖面不同于本项目23维汇总特征。可借鉴“救回炮与新增误报都查依据”，不能照搬空间机制标签或把热图视为因果证明。 |
| FusionMAE2026ReadUpdate | [出版社全文](https://www.nature.com/articles/s42005-026-02626-3)本次可直接读取，已补读Methods的Dataset、Architecture、Training及下游任务部分；未完整阅读补充材料 | 88通道、10ms窗口、源训练统计归一化；重建损失区分真实缺失与人工遮挡。支持明确可观测性掩码，不证明单炮在线微调的预警收益。此前“直开失败”是旧读取记录，当前范围以本项为准。 |
| Coffee2025ALCC | 美国能源部[2025 ALCC项目说明](https://science.osti.gov/-/media/ascr/pdf/facilities/ALCC/ALCC_Factsheets_2025.pdf)，PDF第8页，*Real-Time Adaptive Disruption Forecasting at DIII-D*，Ryan Coffee/SLAC及David Rogers/ORNL；已读该页项目摘要 | 明确提出元学习快速优化、实时到达数据下的within-shot适配；这是项目计划，不是已完成实验或同行评审性能证明。对本项目的约束是不能把“首次炮内自适应”当作未经查证的创新声明。 |

本轮实现状态：`event_teaching_20261026`已加入源域六类事件辅助目标，原J-TEXT基模不变。
这只训练已知事件区间内的家族区分，没有正常类，不能称为完整事件起点检测器。
EAST不加载事件头，也不读取事件标签。是否有预警收益由三组完整360流检验，不能由上述文献代替。

## 2026-10-27：正常工况适应与有结构的动力学

| 标识 | 一手材料与实际读取范围 | 对本项目的具体约束 |
|---|---|---|
| Kim2026CANDI | H. G. Kim等，*CANDI: Curated Test-Time Adaptation for Multivariate Time-Series Anomaly Detection Under Distribution Shift*，[AAAI正式入口](https://ojs.aaai.org/index.php/AAAI/article/view/38524)，DOI10.1609/aaai.v40i17.38524；方法据[作者稿](https://arxiv.org/html/2604.01845v1)。读方法式1–18、主要消融及Limitations，未完整审阅所有补充实验 | 从正常验证数据建立参考，再筛选适应窗口；冻结检测器、更新输入输出侧模块。适用于重建型异常检测，不是破裂提前预警。正常训练集假设在本项目不能默认成立；不能照搬其距离门限或用点级FPR代替炮级FPR。 |
| Kim2024M2N2 | Dongmin Kim、Sunghyun Park、Jaegul Choo，*When Model Meets New Normals: Test-Time Adaptation for Unsupervised Time-Series Anomaly Detection*，[作者稿v2](https://arxiv.org/html/2312.11976v2)，[AAAI原文](https://ojs.aaai.org/index.php/AAAI/article/download/29210/30283)。读方法、消融和Discussion and Limitation | 趋势估计与模型更新分开消融，只对判为正常的样本适应。作者承认难区分正常偏移与异常。这里可借鉴更新样本选择，但EAST危险慢变不能被去趋势吸收，需独立验证；不复制其异常分数阈值。 |
| Wang2025TCV | Allen M. Wang等，*Learning plasma dynamics and robust rampdown trajectories with predict-first experiments at TCV*，Nature Communications，DOI10.1038/s41467-025-63917-x；[作者稿v2](https://arxiv.org/html/2502.12327v2)。读引言、实验流程、模型介绍、训练方法、数据分布及式1–4；未完整审阅全部补充材料。出版社直连重定向失败，方法据作者稿 | 先用311炮训练、131炮验证，再对44炮只微调约束时间相关部分；不是“只用5炮训练”。把物理状态与可调参数分开是重要启发，但其能量、加热功率、几何与控制量不全在我们的23维中。任务是降流轨迹设计，不是EAST零样本或炮内TPR证据。 |

研究推断而非文献结论：现有诊断重建目标应区分“可适应的装置/工况变化”与“必须保留的危险演化”。
完整能量守恒模型缺少必要观测时不能用代理量冒充；正常窗口的选择也不能偷用当前炮未来结局。
上述文章均不证明本项目已经实现这一分离，更不证明其物理可辨识性。

## 2026-10-27续：不同装置的事件演化与评价口径

| 标识 | 一手材料与实际读取范围 | 取舍与疑点 |
|---|---|---|
| Dong2025MARFE | Qianyun Dong等，*Physics-Informed Visual MARFE Prediction on the HL-3 Tokamak*，[作者稿v1](https://arxiv.org/html/2510.24347v1)。读§1–3、式1–11及表1–4；预印本，未核实后续期刊版 | 预测活动继续加剧，而非仅识别活动；701炮、按炮划分，依赖相机及温度等额外观测。实时刷新指推理，不是炮内参数学习。§3.2正文和表3的AUC不一致，故不引用其增益量作为设计依据；自动修订标签也不等同独立专家真值。本项目不能把A–F直接当MARFE、照抄密度温度门限，或把事件AUC当破裂TPR。 |
| Agarwal2025ADITYA | Jyoti Agarwal等，*Early Prediction of Current Quench Events in the ADITYA Tokamak using Transformer based Data Driven Models*，[作者稿v1](https://arxiv.org/html/2507.12797v1)。读§2–3及评价定义；未核实后续期刊版 | 提醒检查样本筛选和提前量：其高召回主要在8–10ms，不能移用于EAST>30ms。由1407炮筛选出415D+310S，后文又写721炮，计数差异未解释；CQ时刻聚类及过滤策略不等于本项目HDF5标签合同。保留为有局限的架构对照，不按其方式删难例以抬高成绩。 |
| Koshizuka2026ReadUpdate | [作者稿v2](https://arxiv.org/html/2609.24443v2)，本轮补读式1–8、§3.6、§4.2–4.5；旧记录已存在，不新增计数 | 明确TLS输出是平滑标签的条件期望，不是严格固定时距概率；40ms评价与按时间报警率也不同于本项目。相关启发是分清标签、风险表示和报警规则，不据此改动当前单因素实验的标签或阈值。 |

## 2026-10-27再续：活动是否恶化与破裂后果

| 文献 | 实际读取范围 | 对当前项目的约束 |
|---|---|---|
| Sweeney2017Locked | R. Sweeney等，*Statistical analysis of m/n=2/1 locked and quasi-stationary modes with rotating precursors at DIII-D*，Nuclear Fusion57,016019，DOI10.1088/0029-5515/57/1/016019；[作者稿v2](https://arxiv.org/html/1606.04183v2)。读§1、§2.1–2.2、§8.3–8.6及§9前半；未完整审阅附录 | 锁模出现、幅度大小与最终破裂不是同一件事。其判据依赖平衡重建，不能用qa_proxy替代q95，更不能凭n=1幅度还原磁岛宽度。§8.5的假警比例分母是全部已发报警，不是全部安全炮；不与本项目FPR直接比较。 |
| Hu2023MARFEmovement | Wenhui Hu等，*Prediction of multifaceted asymmetric radiation from the edge movement in density-limit disruptive plasmas on Experimental Advanced Superconducting Tokamak using random forest*，Chinese Physics B32,075211，DOI10.1088/1674-1056/acd2b0；[期刊全文](https://www.cpsjournals.cn/en/article/doi/10.1088/1674-1056/acd2b0)。读§2–3.5和结论 | 目标是MARFE移动，标签围绕移动时刻，而非所有破裂的DownTime；28个目标事件训练炮之外还有大量背景炮。其控制误差、约束与储能量不在本项目23维中。§3.2存在正负标签文字反写和负样本频率疑点；不照搬此处文字实现标签。 |
| Villone2026Loads | Fabio Villone等，*Reconstruction of electromagnetic loads during disruptions in ITER*，Nuclear Fusion66,016030，DOI10.1088/1741-4326/ae1b1a；2025年11月在线发表。[作者机构期刊PDF](https://www.iris.unina.it/retrieve/1f07de80-fd24-46e5-a129-685e2b252c6d/Villone_2026_Nucl._Fusion_66_016030.pdf)。读§1–3、§4主要讨论及§5；未审附录推导 | 低储能投运有依据，但真实电磁载荷需要结构电流、场和几何等信息。文中验证基于两类模拟VDE，方法用于炮间载荷重建，不是炮内破裂学习。本项目起始Ip/Bt排序只能叫运行参数代理，不是已测损伤。 |

## 2026-10-28 UTC：学习更新方向与时序形状对照

下列为本轮新增方法阅读；不是把检索条目数量算作全文阅读数。

| 文献 | 实际读取范围 | 对本项目的约束 |
|---|---|---|
| Deng2025MGTTA | Qi Deng等，*Learning to Generate Gradients for Test-Time Adaptation via Test-Time Training Layers*，AAAI39(15),16235–16243，DOI10.1609/aaai.v39i15.33783。[官方入口](https://ojs.aaai.org/index.php/AAAI/article/view/33783)、[作者稿v1](https://arxiv.org/html/2412.16901v1)。读§3、算法1、§4及消融；未核对作者代码 | 利用历史梯度学习更新方式，部署时主要更新归一化层；预训练使用128张无标签OOD验证图像，另用源图像统计，不能称零目标样本或直接换算成128炮。其目标仍含熵与统计对齐，不能自动解决破裂危险和正常变化混淆；图像成绩不是炮内预警证据。 |
| Artigues2023Shapelet | V. Artigues、P. C. de Vries、F. Jenko及JET Contributors，*A shapelet-based neural network for binary and multi-class disruption prediction for prevention at JET*，Physics of Plasmas30,082506，DOI10.1063/5.0151511。出版社直连失败；读取[公开原文转录](https://www.researchgate.net/publication/373378266_A_shapelet-based_neural_network_for_binary_and_multi-class_disruption_prediction_for_prevention_at_JET)中的输入处理、表III、§III.D、§V.C及结论。不是精读其博士论文 | JET局部归一化形状与窗口均值有明确消融：二分类和类型识别对绝对量的需求不同。可借鉴区分形状与水平，不能声称该文章已证明跨装置炮内适应，也不能把其7类破裂标签映射成本项目算法生成的A–F。 |
| Ferreira2022MHD | Diogo R. Ferreira、Tiago A. Martins、Paulo Rodrigues及JET Contributors，*Explainable deep learning for the analysis of MHD spectrograms in nuclear fusion*，Machine Learning: Science and Technology3,015015，DOI10.1088/2632-2153/ac44aa；2021-12-30在线发表。[作者稿PDF](https://web.ist.utl.pt/diogo.ferreira/papers/ferreira22explainable.pdf)因体积超限未完整打开；实际读取[出版社授权公开原文](https://www.researchgate.net/publication/357198091_Explainable_Deep_Learning_for_the_Analysis_of_MHD_Spectrograms_in_Nuclear_Fusion)§1–3、§4.2–4.3和结论。另读2021 IAEA会议幻灯片，不重复算另一项方法 | 预测锁模而非所有破裂，利用约1秒MHD频谱及CAM检查演化依据。主要曲线在49炮验证集上评估，作者明确没有充分测试集；锁模幅值阈值的高准确率不是CNN的严格提前预警成绩。本项目23维不能重建原始高频谱，不能直接照搬其模态判据；可检验的启发是保留活动的先后关系，而非只看幅度。 |

本轮也重新找到沈呈硕2026年数据质量与对齐文章，但其原文访问范围未扩大，不新增阅读或性能证据。
Artigues博士论文因文件过大及站点限流未获得完整正文；不将搜索片段记作完整阅读。
以上由AI辅助检索和整理；未代替作者判断或独立物理验证。

本节三项为新增方法阅读。既有条目再检索不重复计数；仅检索到、未读方法的论文不新增实现结论。

## 2026-10-28续：工况适配与等离子体状态表示

本节先登记证据，再形成设计判断。新增五篇，不重复计算此前的DisruptionBench、Viewmakers、
Keith生存分析或Murari持续学习。普通网页检索与官方/作者原文交叉检查；不是系统综述的穷尽检索，
也不是五篇逐字全文及代码复现。检索由AI辅助完成，未作人工专家确认。

| 标识 | 原始来源、版本与读取范围 | 允许使用的结论及边界 |
|---|---|---|
| Kirchmeyer2022CoDA | Matthieu Kirchmeyer等，*Generalizing to New Physical Systems via Context-Informed Dynamics Model*，ICML2022，PMLR162:11283–11301。[官方](https://proceedings.mlr.press/v162/kirchmeyer22a.html)、[作者v3](https://arxiv.org/html/2202.01889v3)。读§2–3.3、实验设置及§5主要结果；未完整审阅证明和补充材料 | 新环境冻结共享参数与线性映射，拟合少量工况变量；一条适配轨迹与32条不同测试轨迹分开。任务是动力学回归，不是破裂分类；一条完整轨迹不能换算成EAST当前炮已到达的短前缀，更不保证一炮适应成功。 |
| Nzoyem2025NCF | Roussel Desmond Nzoyem、David A. W. Barton、Tom Deakin，*Neural Context Flows for Meta-Learning of Dynamical Systems*，ICLR2025。[会议PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/f3398b76d17792893ce6d4f660546353-Paper-Conference.pdf)、[作者v6](https://arxiv.org/html/2405.02154v6)。读§3、算法2、§4.1–4.3、§5.1和附录A.2部分；v3后补读v6，不算两篇 | 训练时用不同工况之间的Taylor展开，适配时固定共享模型、拟合新工况变量。可辨识性结论有仿射物理参数、秩和训练误差等条件，不说明任意学习向量都是真实物理参数。不是EAST预警，也不是毫秒实时性证明。 |
| Yin2021LEADS | Yuan Yin等，*LEADS: Learning Dynamical Systems that Generalize Across Environments*，NeurIPS2021。[官方摘要](https://proceedings.neurips.cc/paper/2021/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html)、[全文](https://papers.nips.cc/paper/2021/file/3df1d4b96d8976ff5986393e8767f5b2-Paper.pdf)。读§1–2、§3.3、§4.1–4.3 | 将共享动力学与受正则限制的环境差异分开，新环境只学习后者。检验基于模拟动力学；这里借鉴内部动力学分解，不借此给最终风险分数再加修正头。 |
| Gozeten2025TTTTheory | Halil Alperen Gozeten等，*Test-Time Training Provably Improves Transformers as In-context Learners*，ICML2025，PMLR267:20266–20295。[官方](https://proceedings.mlr.press/v267/gozeten25a.html)、[作者v1](https://arxiv.org/html/2503.11842v1)。读摘要、§3及§4部分；没有完整核查证明 | 单层线性注意力等设定下研究测试时梯度更新，使用有标签上下文。只能说明预训练和适配问题要匹配；不能充当无标签炮内破裂适应的理论保证。 |
| Burli2025JET | A. Bürli、A. Pau、T. Koller、O. Sauter及JET Contributors，*Towards Transparent and Accurate Plasma State Monitoring at JET*，[作者v1](https://arxiv.org/html/2502.12182v1)，2025-02-14预印本；未核实正式出版版本。读§2–4、表1–5和§5前半 | 共享表示同时学习破裂分类、到边界时间、状态重建和轨迹平滑；初始520炮，完整诊断筛选后430炮。序列模型原始FA为23.3%，附近不稳定事件重新计数后FA′为8.18%；再加200ms持续报警条件才到6.13%。这些不是本项目严格炮级FPR或炮内学习收益。 |

计划采用的设计结论（研究推断，不是上述文章直接证明）：

1. 以少量内部工况参数作为可更新对象值得检验，但不预先把它们命名为密度、约束时间或真实破裂机制。
2. 信号预测辅助任务必须通过更新后的预警表现筛选；动力学误差下降本身不足以验收。
3. 活动识别、最终破裂预警、报警规则贡献分开报告，不能改写安全炮误报定义来制造改进。
4. 本项目继续使用完整在线流计分；不把文献离线适配轨迹的效果当成当前炮因果适配证据。

## 2026-10-28再续：状态表示与及时预警的区别

本次核读Poels等 *Plasma State Monitoring and Disruption Characterization using Multimodal VAEs*，
[作者稿v1](https://arxiv.org/html/2504.17710v1)，2025-04-24；未核实正式发表版本。
实际读§2–6（问题、数据、式5–36、实验和限制），未核读附录与实现。
TCV共1629炮，训练/验证/测试为1300/165/164；以重建、风险与混合先验组织低维状态。
其§6明确指出低维表示难以保留快速动态。该文支持分别检查工况表征与快速预警，
不证明EAST炮内更新有效；近破裂时刻的事后配对分析也不是30ms提前预警结果。
本文阅读不计为对作者代码的复现；当前八维向量与该文的二维表示、八个聚类不是同一结构。

再次检索到Viewmakers、Koshizuka有限时距预警、Shen2026数据对齐文章。
前两项已在旧表登记，不重复增加篇数；Shen2026仍未取得可完整核验的方法正文。
检索范围包含cross-tokamak alignment、plasma state representation及scenario-adaptive disruption prediction，
没有只检索Ai；搜索结果中的性能数字未直接用于本项目设计或结论。
