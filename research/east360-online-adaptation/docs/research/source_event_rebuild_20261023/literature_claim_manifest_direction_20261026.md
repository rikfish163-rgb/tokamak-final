# 文献核查增补：更新方向、事件弱监督与可解释性

核查日期：2026-10-26。下列“读取范围”是实际证据边界，不等于通读全文或复现。
公开检索未上传项目数据。未把搜索命中数记作已阅读论文数。

| 文献/材料 | 一手来源及实际读取范围 | 可以采用的证据 | 不可外推之处 |
|---|---|---|---|
| Bonalumi等，*eXplainable artificial intelligence applied to algorithms for disruption prediction in tokamak devices*，Frontiers in Physics 12 (2024)，1359656 | [期刊正文](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1359656/full)，§2–3及§4开头；非全文通读 | 结合遮挡与梯度敏感性，检查模型依赖的诊断信息；时间max-pooling对瞬变过敏，作者采用空间max/时间average的设计 | 依赖温度、密度和辐射空间剖面；本项目23维汇总信号不能复现其核心/边缘温度机制。敏感性不等于物理因果 |
| Liu等，*TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?*，NeurIPS2021 | [官方论文](https://papers.neurips.cc/paper_files/paper/2021/file/b618c3210e934362ac261db280128c22-Paper.pdf)，本次复核§2.2–4.1，未复核证明附录 | 自监督适应可能损害主任务；均值/协方差对齐及任务关联性是其设计重点 | 理论假定标签分布及相关条件分布对齐等；不能直接保证单炮时变分布、类别比例变化下的收益 |
| Montes等，*A semi-supervised machine learning detector for physics events in tokamak discharges*，DOI 10.1088/1741-4326/abcdb9 | [MIT作者机构页面](https://dspace.mit.edu/entities/publication/e800cf5e-03cf-4caf-a394-af232f5bce5b)，摘要完整；[作者稿PDF](https://library.psfc.mit.edu/catalog/reports/2020/20ja/20ja074/20ja074_full.pdf)获取超时，方法正文未核实 | 少量人工事件标记与label spreading可辅助构建事件数据库；论文也展示跨装置检测案例 | 事件检测不是破裂及时TPR；不能称本文已实现实时炮内学习。不能照搬未核实的图构建/阈值细节 |
| Shen等，*Autonomous Discovery of Disruption Precursor Event Chains in Tokamak Plasmas via Progressive Multiple-Instance Learning*，IAEA2026会议摘要 | [官方会议页](https://conferences.iaea.org/event/454/contributions/43124/)，直连403，已读取官方域名搜索索引中的完整摘要 | 炮级标签、多实例学习、逐步剥离强前兆及事件转移建模已被提出；与本项目想法高度相关，应纳入相关工作 | 截至核查日会议尚未召开（排期11月10日）；不是已读完整论文，无代码复现和本数据定量证据。双向上下文不能直接用作严格实时前缀输入 |
| Shen，*Predicting disruption in future tokamaks with fewer data by more physics-guided*，IAEA2023汇报摘要页 | [官方单页](https://conferences.iaea.org/event/335/contributions/28976/attachments/15708/26422/IAEA%20AI%20workshop%20-%20summary%20slide.pdf)，搜索索引全文 | 强调物理特征、目标少样本对齐以及数据质量/归一化；“10dis+100non”不是10炮总预算 | 幻灯片结果不是本项目同一测试合同；不把其FPR、TPR与当前30ms成绩直接排名 |
| *A Machine Learning Predictor for Vertical Displacement Events on EAST*，IEEE TPS53(1),136–145(2025)，DOI10.1109/TPS.2025.3528970 | [出版社摘要](https://ieeexplore.ieee.org/document/10852610/)，摘要完整，全文未读 | 对单炮前兆起点作区分，而不统一假定固定起点，具有相关性 | 特定VDE预测，不等于所有破裂类型；不能将其事件成功率照搬为跨装置总体TPR |

## 版本状态异常：单独记录，不与用户2026正式论文混淆

[arXiv:2404.08241](https://arxiv.org/abs/2404.08241)官方页面标为withdrawn，说明是提交时授权许可问题。
该记录不能当作未撤回的全文论文引用，也不能据此断言研究结论造假或用户提供的2026期刊论文被撤稿。
本次只核对版本状态，未拿其摘要成绩作为有效性能基准。
