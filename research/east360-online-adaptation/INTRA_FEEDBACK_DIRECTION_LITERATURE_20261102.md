# 炮内反馈方向：有边界的原始文献核查

2026-11-02。范围：四篇原始论文，不是系统综述；公开关键词检索，未向外部发送私有代码或数据。旧读状态只依据本次检索到的主目录 `docs/research` 与 WT Markdown 记录，不等于穷尽全项目。AI 辅助阅读，未作用户本人已读或独立盲审声明。

## 实验前提

已达标的 J-TEXT 模型冻结（100 TP/110 破裂、13 FP/110 安全），仅 EAST 炮内更新。按[现有回放记录](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/EAST20规则校准与完整360炮回放_20261102.md)，同 cold20 起点为 125 TP/35 FP；真实、反序、常值反馈的炮级结果相同。使用 128 点历史预测未来 25 点，待目标成熟后反馈；10910 次非零梯度尝试接纳 340 次。C2300 校准使用已结束 20 炮、128 步，mean/worst 元目标均未达标。另一轮 Armijo 全接纳为 122 TP/30 FP，不能推断多更新就有利；两轮不是同条件消融。

## 四篇核读

1. **Yunshu Du、Wojciech M. Czarnecki、Siddhant M. Jayakumar、Mehrdad Farajtabar、Razvan Pascanu、Balaji Lakshminarayanan，2018，*Adapting Auxiliary Losses Using Gradient Similarity*，arXiv 预印本；读的是 2020 年 v2，未核实为 ICLR 正式会议论文。** [原文 §§2–3、式(3)、附录 A.1](https://arxiv.org/html/1812.02224v2)。**旧读**：WT 的 `LOCAL_READOUT_NEXT_HYPOTHESES_TERRA.md` 等已核对其主结论。

   原式同时使用主任务梯度 `g_R` 和辅助梯度 `g_A`：`g_R + g_A max(0, cos(g_R,g_A))`；其小步论证不等于纯辅助更新必然改善主任务。原文称相似度是“a local quantity”，并指出收敛速度可能变差。**可否证什么**：对已结束炮，在同一参数状态检查真实实际步 `ΔW` 的 `∇R·ΔW` 及有限步风险损失；正值仅预示一阶不利，有限步风险确实上升才能否证该步改善风险损失。不能由局部符号保证或否定整炮首次报警收益；本炮结局梯度不能用于在线门控。

2. **Hao Zhao、Yuejiang Liu、Alexandre Alahi、Tao Lin，2023，*On Pitfalls of Test-Time Adaptation*，ICML / PMLR 202。** [原文 §§4.1–4.2、图3、§6 表4](https://proceedings.mlr.press/v202/zhao23d/zhao23d.pdf)。**旧摘要、新核正文**：原百篇清单仅标摘要级。

   图像实验中，在线适应受“over-adaptation on previous test batches”影响；图3同设定下 3 步准确率 70.9%，5 步降为 68.4%。表4 的时序相关 CIFAR10-C 中，不适应错误率 44.3%，SHOT-online 为 51.2±2.0%。这否定“无标签多步适应必然更好”，也说明历史状态和选择协议重要；它既不测 EAST，也不证明本次反馈错向。实际候选的诊断损失下降、元训练目标或接纳率均不能代替预先固定的逐炮报警评价。

3. **Tom Sherborne、Naomi Saphra、Pradeep Dasigi、Hao Peng，2024，*TRAM: Bridging Trust Regions and Sharpness Aware Minimization*，ICLR。** [正式论文 §§2–3、式(1)、(3)–(5)、(7)](https://proceedings.iclr.cc/paper_files/paper/2024/file/2c15b0221da28bc6f4373a7e78b896dd-Paper-Conference.pdf)。**新增**：上述本地文档范围内未发现该标题或作者记录。

   式(1)包含参数 L2 与参数邻域；式(3)–(4)用前后预测分布的 KL 约束函数变化；式(7)用信赖域量缩放 SAM 的对抗扰动。原文比较的是“previous step’s distribution”，不是证明任意隐藏层 RMS 等于输出 KL。它提供比泛称 natural gradient 更直接的空间区分，**不能证明本实验两项约束冲突**，也不证明迁移到聚变会有效。

4. **Wei Zheng、Fengming Xue、Zhongyong Chen 等，2023，*Disruption prediction for future tokamaks using parameter-based transfer learning*，Communications Physics 6, 181。** [出版方 Methods / Labeling](https://www.nature.com/articles/s42005-023-01296-9)。**旧读**：此前已有同节标注边界；本次重新核对其反例，不计为新增论文。

   出版方检索索引中的正文称，用前兆相关时间标注的 J-TEXT 模型“more false alarms occurred”，不如其固定窗标注的逐炮表现；它可反驳“更贴近前兆的标签必然带来更好报警”。直接打开出版页遇重定向，**表6具体数值和完整实验细节本次未核验**。论文不是本项目 A–F 事件类监督，也不是无标签炮内更新，不能据此推断事件标签普遍无效或创造新物理规律。

## 对当前实现的解释边界

`L_inner = L_centered-SmoothL1 + 0.01 mean(W²)`，`W∈R^(4×64)`；后项梯度为 `0.02W/256`。给定当前窗口 `φ`，`G=φᵀφ/(n_tokens·64)+ridge·I`，所以 `tr(ΔWᵀGΔW)=||φΔW||²_F/(n_tokens·64)+ridge||ΔW||²_F`。这是当前窗口隐藏变化与步长岭项的度量，不是最终风险输出的 KL；隐藏步 RMS 0.05、原点累计上限 0.25 也不是主任务改善保证。

因此，原始 `g_A` 与 `g_R` 的欧氏余弦不一定代表经过度量、裁剪及参数正则后的**实际步**方向。若事后实际步在相关窗口使风险损失上升，可否证该局部更新有利；若方向看似有利但有限步或首次报警不改善，也不能把错向当作唯一原因。真实/反序/常值同结果及 Armijo 反例均不能单独证明根因。A–F 为无序活动，既有 masked 拟合和乱标对照已失败；不建议把新增事件头改名重试。

核查计数：4 篇保留，其中 3 篇打开并读取正文，1 篇仅读取出版方索引正文；对“余弦保证报警收益”“多更新必然有益”“两种正则已证冲突”“前兆标注必然更好”四项扩张性断言不予支持。Du 的梯度加权不是学出的炮内更新器；本次未核验可部署的 learned-gradient 方案，C2300 的失败也不能否定该类方法全部可能性。未运行实验或验证新方案；已曝光的 360 炮及 cold20 内部分折不是独立未来验证。
