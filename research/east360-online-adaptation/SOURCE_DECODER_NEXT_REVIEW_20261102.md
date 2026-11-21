# 直接源端 diagnostic decoder：历史核查与最小三臂控制

## Material Passport（范围与结论）

2026-11-02，只读代码、计划和已保存完成材料；未运行训练、测试、GPU、原始 HDF 或新输出 audit。仅新建本文，不改旧报告及 context_pair 等冻结代码。沿用当前 terra/max，不调整 service。本文是历史/设计意见，不认证主进程随后实验的实现或成绩。

**在本次核查到的已完成材料中，未找到同时满足以下条件的直接 decoder 实验：c727 source-only、zero-slow、首层 FFN rank4 W、当前 source_event 固定归档 episode、训练原形状 forecast_head、无 risk-logit×B 桥。** 这是有边界的检索结论，不是对所有历史文件的“从未发生”证明。直接 decoder risk-meta 思路本身已经做过，尤其已有最后一层 FFN 的近邻先例，不能包装成全新原理。

## 历史差异核对

| 历史线 | 实际可训练对象／内层位置 | 与拟议实验的关键差异 |
|---|---|---|
| 20261004 forecast-prior、finite-prior、risk-prior | 原 forecast_head；token-fast 128 维，归一化 .05 步 | c727、无 B 桥；不是首层 rank4，也不是当前固定 source_event episode。三种目标分别是梯度对齐、真实有限步后 BCE、风险/预测梯度尺度平衡。 |
| 20261005 matched decoder | 原 forecast_head；最后一层 FFN 的 linear1/linear2 全权重与 bias 临时增量 | **已有 source-only、无桥的 forecast-only/risk-aligned 直接先例**，但位置、维度、episode 与更新预算不同。保存协议为 60 步，两臂 source_gate_passed=false。 |
| 20261029 matched Q/K head | 原 forecast_head；内层零起点 Q/K 8192 维，.05 步 | 已有 forecast/matched 成对训练，512 步，necessary_source_screen=false；不是 rank4 W。协议的 source_fast_site=tokens 描述原源模型，不表示其内层教学仍在 token 位点。 |
| 20261030 ContextTarget/source | 358→32→64 的目标向量网络；Q/K representation step | 与本轮 source_event 使用同一归档来源，但 decoder 不训练，source_screen_passed=false；不是物理 H25 decoder。 |
| 本轮 source_event_masked_tasks_v1 | 首层 FFN rank4 W；持久训练 shared B 或六张条件 B 表 | zero-slow、当前固定 episode 均匹配，但原 decoder 冻结，预测仍是 F(W)+a(W)B_eff；passed=false，不是直接 decoder 实验。 |

核查依据：

- token prior 的[匹配坐标与 head-only 训练](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/redesign_20261004/prepare_prior.py:107)，以及[finite-prior 的真实训练记录](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/runs/redesign_finite_prior128_20261004/training.json)（协议与 history 均为 128 步）。
- 最后 FFN 的[临时参数定义](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/ffn_intra_20261005/model.py:14)、[直接 decoder 目标](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/decoder_intra_20261005/matched_meta.py:35)、[保存协议](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/runs/ffn_matched_head_meta_20261005/protocol.json)、[结果](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/runs/ffn_matched_head_meta_20261005/summary.json)。
- Q/K 的[内层状态与 head-only 目标](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/qk_matched_head_20261029/learning.py:53)、[完成材料](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/qk_matched_source_v1/complete.json)；ContextTarget 的[网络定义](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/learning.py:51)、[controller-only 训练](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/source_run.py:52)。
- 当前[归档绑定](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_event_tasks_20261102/data.py:18)、[zero-slow 加载](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_event_tasks_20261102/trial.py:112)、[只训练 B 表](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_event_tasks_20261102/trial.py:269)、[已完成失败 gate](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_event_masked_tasks_v1/completed.json)。

## 建议的新最小控制：只改变 decoder 教学目标

原头为 17×64=1088 输入、25×23=575 输出的线性层，即每臂 626175 个参数；不是原 B 的 575 参数。使用原 head 权重的独立副本 Θ，不回写 c727 checkpoint；encoder、risk_head、输入尺度、阈值与 slow 全冻结。直接预测为 `Yhat=Theta H(W)+bias`，不出现 `a(W)B`、事件概率或最终分数叠加。[原 head 结构](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple23.py:66)、[当前 FFN 后 decoder 路径](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/dual_history_ttt_20261101/model.py:143)

三臂从完全相同 Θ0 出发，使用同一固定 archive/episode ID、相同 safe/danger 与事件分层 schedule、三种视图顺序、每步样本、优化器预算和末步选择。可直接沿用本轮 512 步、每步 safe/danger 各一条，而不是重新挑峰。三臂应相同地保留 W 零起点、成熟25点目标、隐藏目标屏蔽、metric 步长、RMS 上限与严格接纳规则。A–F 只沿用既有任务分层，不需要新 q 或新模块。

令 D 为预先固定、三臂共同的 origin 零状态 observed diagnostic 维护项；W⁺(Θ) 为由该 decoder 驱动、固定内层损失 L_in 下的同一更新步。D 与 L_in 是否同为 centered loss 必须预先说明。`r_g=BCE_g(after)-stopgrad(BCE_g(zero))`，组为 safe/danger，训练用同批次组估计：

| 臂 | 固定外层目标（沿用 .20/.1 尺度） | 唯一要检验的差异 |
|---|---|---|
| forecast-only | `.20 D` | 只拟合诊断是否已经足够；并非完全无标签，因为抽样仍用离线分层。 |
| average-risk-meta | `.20 D + (r_safe+r_danger)/(2*.1)` | 同一物理 decoder 是否能经实际 W 更新学到平均风险收益。 |
| worst-risk-meta | `.20 D + max(r_safe,r_danger)/.1` | 是否能减少平均目标以 safe 收益交换 danger 伤害的问题。 |

这里建议 max **风险增量**，不是 max 原始 BCE；两者选中的组可能不同，必须在运行前明确，不能事后互换。任何已有正则项应三臂一致。equal budget 指相同步数/episode 暴露，不声称 forecast-only 与二阶 meta 的 FLOPs 或耗时相同。新三臂之间可归因于教学目标；与旧 B 表直接比成绩则同时改变了容量和参数化。

## 三条具体风险与可证伪标准

1. **历史重复／归因混淆。** “训练原 decoder”以及“经 FFN 更新学风险”已有先例；本轮可新增的证据只能是当前 c727、首层 rank4、同归档条件下的匹配复核。若在新三臂中 meta 不胜 forecast-only，就不能把替换 B 后的整体变化归因于风险元教学；更不能把旧 last-FFN/QK 失败宣称为本轮已失败。比较必须保持原始 episode 和输入 contract，不重新按新 decoder 选 query。

2. **有 meta 名称，不一定有 meta 路径；minmax 也不保证每组改善。** 必须保留 `Theta → dL_in/dW → W⁺ → 后续风险` 的二阶路径，不能缓存成与 W 无关的最终 H、detach W⁺，或仅统计总 head 梯度。应分别记录风险项 head 梯度与接纳分支：拒绝时风险路径可能确实缺失，D 仍能更新 head。每组单条样本的 max 是“minibatch worst-risk 控制”，不是精确总体 GroupDRO；其噪声与组难度可能主导选择，训练 max 小也不保证 VAL/native 安全。按炮报告 safe/danger 两组与 native 保护，仍须真实反馈胜过匹配 reverse/hold。[Sagawa 等作者原文摘要](https://arxiv.org/abs/1911.08731)指出 worst-group 训练拟合不等于 worst-group 泛化；这里只借鉴分组目标，不声称复现其算法或取得其收敛保证。OpenReview 页面本次受浏览器验证限制，未声称读过全文。

3. **物理 decoder 不等于物理规律或跨时限迁移。** 输出对应23诊断、源尺度固定，是比显式风险桥更清楚的测量任务，但本质仍是源归一化/asinh 坐标中的 H25 预测。若 D 仍仅按通道去时间均值，任意随 H 变化、在25点恒定的通道项都被消去，decoder 的绝对水平不由该损失识别；不能据其误差称“完整物理预测已学会”。若采用绝对 forecast maintenance，必须三臂同用并与内层 centered loss 明确区分，不能暗中多给某臂监督。JT 的 max(25ms,AdvancedTime)、源 query lead>10ms 不等于 EAST125ms 与严格 >30ms；本轮源端局部改善即使成立，也只获得后续检验资格，不是 EAST 告警成功。

## 交接边界

当前旧成品仍为 safe_improved=true、danger_improved=false、native_protected=false、passed=false；不因新设计而改写历史。新控制尚无本文认证的结果，不等待主进程实验、不追加模块或调参。合格源不可改、不得最终风险叠加、先证明可归因炮内收益再谈运行操作点；既有 HDF 标签、125ms 与 >30ms 合同均保留。本文交主进程后冻结，不自行追踪或修改。
