# Origin vs maturity 第一层 CLS Jacobian：有界历史去重

2026-10-30。仅代码、报告及已保存 JSON 审查；未训练、未运行模型/GPU、未读原始 HDF5 或缓存信号张量、未上传。唯一写入本文件。按 results-analysis 区分已完成结果、代码先例和本轮计划；不是全仓库/全球新颖性证明。

## 结论：未检出精确重复，旧 fresh-clock 不等价

在下列直接调用链和保存结果内，未发现已经完成的 **同一 AnchoredTarget/context358、同一第一层 Q/K 临时参数、同一 CLS 目标向量 v，仅将求导输入从原点 o 的128点窗口改成成熟点 m=o+25 的128点窗口，并在 q=o+50 评分** 的配对实验。历史去重没有构成阻止本次组件诊断的依据；这不是效用或安全性背书。

工作树中本轮新出现的 [maturity_jacobian runner](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/maturity_jacobian_20261030/run.py:28) 及 [本轮计划](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/MATURITY_JACOBIAN_PLAN_20261030.md:10) 不计入历史重复，也不计为已运行结果。主进程表示尚未运行 GPU；本审查不替代新 runner 的实现验收。

## 最接近先例及区别

1. **已完成的 anchored 基线：明确仍是 J_origin。** [state_for 第58–68行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/anchored_feedback_20261030/run.py:58) 从零创建8192维临时 Q/K，在 `data['x']`/`mask` 上调用 `net.first`，对 `h[:,-1]` 求梯度；`later_x` 仅在后续风险评分使用。实际目标为 [learning.py 第24–36行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/anchored_feedback_20261030/learning.py:24) 的 `0.5*(network(context)-network(reference))`，不是直接生成 Q/K 参数。保存的 [complete.json 第173行起](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/anchored_feedback_cold20_v1/complete.json:173) 中，safe 平均 ΔBCE：real −0.0857068112、reverse −0.0907897205、donor −0.0954604219；[第328行 gate](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/anchored_feedback_cold20_v1/complete.json:328) 因未胜过两个安全控制而失败。它没有检验 J_maturity。

2. **已完成的 fresh-clock：刷新预测任务的起点和真值，不是固定 v 的 Jacobian 换窗。** [qk_clock_objective/learning.py 第13–31行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/qk_clock_objective_20261029/learning.py:13) 将可见信号截到 query−1，再重建四段已成熟 H25 任务；[forecast_examples 第92–106行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/deepcoral_prefix_20261022/learning.py:92) 的四个输入终点是 `query−1−25*(j+1)`。梯度通过这些**预测起点窗口**的 forecast SmoothL1 产生，[adapt 第53–74行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/qk_matched_head_20261029/learning.py:53) 并非成熟窗口 CLS 的 anchored representation loss。保存的同一 shot1051592/query285：[blocked receipt 第92230行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/qk_clock_objective_v2/validation.json:92230) 是起点230/205/180/155、最后真值255；[fresh receipt 第99526行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/qk_clock_objective_v2/validation.json:99526) 是259/234/209/184、最后真值284。**不存在输入终点284的同目标 CLS 求导对照。** [complete.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/qk_clock_objective_v2/complete.json:1) 保存四个 source screen 全失败；[audit.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/qk_clock_objective_v2/audit.json:1) 确认521配对端点、精确验证重放、无 EAST 结果。不能把它写成已经否证本次变量。

3. **更早的 maturity Jacobian 代码先例确实存在，不能声称“从未在成熟窗口求过 Jacobian”。** [event_gradient_projection.py 第60–67行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_gradient_projection.py:60) 已将原128点与真实已到达25点拼成成熟窗口；[第103–116行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/event_gradient_projection.py:103) 求五个事件状态坐标对临时参数的 Jacobian，将已有辅助梯度投到其零空间。其参数位置是 [persistent_ffn.py 第17行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/persistent_ffn.py:17) 的**最后一层 FFN**；状态是预测头前 CLS/均值/末 token 的192维组合映射到五方向，而不是第一层64维 CLS。它既不是 `J_mᵀv`，也不是 origin/maturity 同目标对照。本审查仅据实现认定近例，不把未在本次定位到的该分支保存结果当作已完成/成功证据。

4. **maturity 表示已有保存探针结果，但不是 maturity 更新梯度。** [source_diagnose.py 第107–119行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/source_diagnose.py:107) 先由 origin observation 算 action，再在 `no_grad` 内提取 maturity CLS+末 token；[probes.json 第183行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/context_feedback_diagnosis_v1/probes.json:183) 保存 maturity probe weighted AUC 0.921128、balanced BCE 0.772748，文件标注 `not_online_policy=true`。其更强可读信息不证明 J_mᵀv 更新有效。旧 label-aware target oracle 也仍在 [diagnose.py 第134–146行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/diagnose.py:134) 对 origin 窗口求导，再用 query 标签挑目标，不能充当本次无标签内步对照。

## 时间拼接核验：成立，但必须同时拼 mask

[缓存采集第86–112行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_weights_20261030/run.py:86) 明确 `m=o+25,q=o+50`，x 来自 origin、later_x 来自 query；[batch 第10–12行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/delayed_recurrent_memory_20261025/learning.py:10) 调用 [window 第41–48行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple_data.py:41)，终点为包含式。

| 张量/切片 | 实际时间索引（两端包含） | 点数 |
|---|---|---:|
| x | o−127 … o | 128 |
| later_x | o−77 … o+50 | 128 |
| x[:,25:] | o−102 … o | 103 |
| later_x[:,-50:-25]，即[:,78:103] | o+1 … o+25 | 25 |
| 两片按 dim=1 拼接 | o−102 … o+25 | 128 |

例如 o=127 时得到25…152，后续 query 为177；不是用完整 query 窗口求导。只读 [records.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/anchored_feedback_cold20_v1/records.json:1) 核验2707条、20炮，`o≥127`、`(o−127)%25=0`、两个偏移均成立，违反数0。未打开 observations.pt，故不声称已经验证实际缓存重叠值；主进程需保证 `x[:,50:] == later_x[:,:78]` 及 mask 同行一致。本轮窗口实现已有这些检查，但本侧车未执行其测试。

## 因果与控制限制（不扩展模型方案）

- **必须使用同一成熟表示自己的 detach reference。** [representation_step 第98–110行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/learning.py:98) 在 state=0、reference=h(0).detach() 时给出 `clip_radius(eta*Jᵀv)`。成熟 h 搭配 origin reference 会额外引入 `h_m−h_o`，不再只换 Jacobian。context、fit 均值、support 和实际目标的0.5系数仍须保持；这仍是实际内部梯度，不是直接前向生成参数。
- **reverse/donor 只干预目标支路，不再是“更新完全没有真实到达信息”的对照。** 若它们都共享真实 maturity 输入，则仍通过 J_m 看见真实到达信号；donor v 与真实 J_m 还存在配对不一致。可检验的是给定真实成熟状态下目标反馈的增量效用，不能据此宣称全部到达信息的因果必要性。mean v=0导致零更新是结构恒等式，不能作为效用证据。
- **相同 eta/radius 不等于相同实际更新幅度或方向空间扩大。** 两个 Jacobian 均为64×8192，秩上界均64；不存在已证的包含关系。旧 [evaluate 第109–124行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/anchored_feedback_20261030/run.py:109) 还把非零控制归一到本格 real 范数，故控制使用了 real 的幅度信息。保留该规则才是同协议比较；raw norm、最终范数及裁剪差异不能被隐藏。
- **窗口改变不只增加新25点，也删除旧25点并改变 patch 位置。** [first 第31–53行](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_lowrank_20261029/model.py:31) 为8点 patch、固定位置编码；25不是8的倍数。其 h 是第一层 attention+FFN 后 CLS。结果只能先归因于反传窗口改变，不能单独归因于“时间更新鲜”或“更大可达空间”。
- **同目标架构不等于同一个数值 v。** 重新训练成熟分支会改变目标网络权重。只有固定同一已保存网络的 origin/maturity 交叉执行才是数值同 v 比较；各自训练后的主格是两套训练/执行规则比较。本轮计划已列2×2交叉格，可按预定用途解释，不能事后换主格。
- **不能从缓存后续窗口引入未来。** 动作最多访问至 m；q 的最后25点及未来 mask 不得影响构造、检查或动作。mask 应按原输入拼接，不可误用辅助反馈中屏蔽第10列的 target mask。更新发生在 m 观测后，不能回写 m 或更早预测；q=m+25 仅作后续评分。每机会从零、既定折/预算/2707点不变。现有父模型已接触全部 cold20、只有3炮 safe，因此通过此组件诊断也不等于独立冷启动泛化或炮级安全达标。

## 本次证据一致性

已对照封存 [protocol.json 的 code_sha256](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/anchored_feedback_cold20_v1/protocol.json:90)：当前 anchored run/learning、context_vector learning、attention_lowrank model、conditional_weights run 五文件哈希全部一致；protocol、records、code_snapshot.zip 三个 artifact 哈希也全部匹配 complete.json。关键摘要：anchored run `e2c0e8cbd8ed096610c1dd2e39b41ee8d27ab50aa9dfc6fe2a383c953acfe618`，records `9dcab9569db22ae6c9d5c8a5b620f1e2b8b7c310517f78942fb377955d52c27a`。另检查 representation_step 的直接调用点及上述 fresh-clock/成熟状态近例；没有以目录名缺失断言未做过。不再扩展其他历史，不建议重扫步长，也不以辅助损失下降替代后续风险/安全控制证据。
