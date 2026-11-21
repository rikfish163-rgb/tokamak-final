# 补充核查登记：正常工况覆盖、事件链与反馈延迟

日期：2026-10-26。不是系统综述，不声称通读100篇；既有文献复核不重复计为新增论文。

|条目|原始来源与实际读取范围|本项目可用的结论|必须保留的限制|
|---|---|---|---|
|Zhu、Rea等，Hybrid deep learning architecture for general disruption prediction across tokamaks|[作者预印本2007.01401](https://arxiv.org/pdf/2007.01401)，§2、3、4、4.1.1、5及附录采样说明；Table 3/4已读取|区分目标域少量破裂炮与大量非破裂炮；时序、多尺度和正常工况覆盖值得单独检查|20D+5995S不是总预算20炮；文中AUC与50 ms评价不能当成本项目TPR>30ms；全数据归一化不能直接照搬到在线协议|
|Sabbagh等，Physics of Plasmas30,032506(2023)，10.1063/5.0133825|[UKAEA公开期刊PDF](https://scientific-publications.ukaea.uk/wp-content/uploads/032506_1_ONLINE.PDF)，§I–II、III B、IV、V相关段；未声称逐式复现全部物理模型|活动存在与危险演化不同；事件起始和破裂末态不能混作同一状态；需全时段评估|包含转动剖面、平衡重建等当前23维没有的信息；其不同阶段、子集的预测率不能合并成我们的基准|
|Jia等，IAEA2024会议贡献113|[官方两页摘要](https://conferences.iaea.org/event/377/contributions/31666/contribution.pdf)，全文41行已读|专家事件标注可支撑事件识别与破裂预测联合建模|是会议摘要，不是已核查完整算法；其EAST专家事件与本项目J-TEXT自动A–F标签来源不同；跨装置应用仍列为后续工作|
|Kim等，TAFAS，AAAI2025|[作者稿2501.04970v1](https://arxiv.org/html/2501.04970v1)，Challenges/O2、PAAS、GCM与PA；沿用此前核查|部分真实未来信号到达即可构造监督，不必等待完整预测窗|诊断预测改善不保证风险改善；本轮不采用FFT或输入/输出校准，也不回写已发出的风险预测|
|Zhu等，Integrated deep learning framework…，NF63,046009(2023)，10.1088/1741-4326/acb803|[MIT机构摘要](https://library.psfc.mit.edu/catalog/reports/2020/23ja/23ja109/abstract.php)；本轮DOI全文访问仍不可用|可以列作事件识别与跨装置联合研究的相关工作|仍不能写成已读完正文、复现损失或其指标可直接比较；沿用先前读取限制|

所有“对本项目的建议”是工程推论，只有对应本地消融能证明是否有用。文献不是当前候选有效性的替代证据。
