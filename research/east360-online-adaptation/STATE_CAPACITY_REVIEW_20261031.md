# 炮内状态表示容量：事后诊断设计审查

## Material Passport

- 日期：2026-10-31；工作树：`/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028`。
- 方法：`academic-research-suite / experiment-agent` 设计审查及 `results-analysis` 的样本、比较和证据边界核对。状态：**ANALYZED / PROPOSED**；不是新诊断的实验结果，也不是一次模型重放。
- 只读地查看工作树指令、代码、封存预测、结果和历史报告；仅写本报告。使用 CPU 对所选封存 NPZ、HDF5 结局和哈希做计数核对，**没有启动 GPU、训练、优化、模型推理或修改源码**。未重新执行既有验证或单元测试。
- 本审查由主控委派的gpt-6.1-terra/max子代理完成，审查者未再启动子代理；接口没有可设置的fast/priority参数。本路由说明由主控依据实际派发记录校正，不能将单个审查者包装成独立双重审查。

## 判定

**该设计可以回答一个有用但狭窄的问题：在固定 cold 模型、固定选中端点、给定输出扰动半径和事后已知正确方向时，有限搜索实际找到哪些跨阈值的端点见证。** 它不能直接回答能否在线救回炮、FP 是否消失、模型整体容量上限、物理机制，或可从到达信号学到这种干预。

设计应保留全部 **36 个后缀漏／迟报 + 42 个后缀误报**，但不可把 78 都当成可干预样本。按整炮最高峰选点，5 个 FP 的峰在 index `<16`，所有干预强制为零；另有 2 个 FP 虽然后续峰在 index `>=16`，其首次报警仍已在 16 之前发生。即使后者的峰被压低，也不可撤销先前告警。应对这 7 个炮单列，不从 78 中删除。

相同端点、相同基底规则之内，以 `sigma_max(W_output B)` 限定自由路径可保证**输出空间的球包含 context8 可能产生的单端点 shift**，这是一个有用的松弛比较；但它不是同一组可行动作、同等函数影响或同等可学习性的公平证明。两种规则各自的半径可能不同，不能将二者的翻转差异直接归因于基底方向。

## 样本与封存证据

固定 [父 completed](probe_outputs/current_order_context360_v2/completed.json)、[逐炮结果](probe_outputs/current_order_context360_v2/per_shot.json)、[原顺序](probe_outputs/current_order_context360_v2/order.json)、[map](probe_outputs/current_order_context360_v2/input_map.json) 和 [cold 状态](probe_outputs/current_order_context360_v2/models/cold.pt)，阈值 `0.7258946895599365`。破裂及时性采用首次有效越阈后 `round(lead_ms, 9) > 30`；安全炮任一点 `p >= threshold` 是 FP。这里的 FN 指未及时报警，既含从未报警，也含迟报，并非单纯无报警。

| 区间 | 组成 | cold 原始结果 | 本诊断处理 |
|---|---|---|---|
| 前 20 | 18D / 2S | 8 及时 TP、10 未及时、0 FP | 仅登记原 source 封存的启动成本；不用之后拟合的 cold 或新 operator 回算。 |
| 后 340 | 162D / 178S | 126 及时 TP、36 未及时、42 FP | 按固定规则从 78 炮各选一个端点；不使用已预测正确的 262 炮。 |

本次 CPU 只读逐个校验上述 78 炮父 NPZ 与各自 seal 的 SHA-256，并比较 [EAST operator 运行](probe_outputs/east_operator360_v1/completed.json)中相同炮的 cold 数组：**78/78 哈希匹配，78/78 cold 曲线逐点相同**。按 HDF5 `IsDisrupt/DownTime` 和原 scorer 口径复核这 78 个标签与报警，36 个 FN 均有 `index>=16`、严格 `lead>30 ms` 的候选端点；本次所选 FN/FP 峰均未出现并列。其余 282 炮及全套输入哈希没有在本次重新扫描；原有[独立验证产物](probe_outputs/east_operator360_v1/verification.json)记载对 2160 条六臂曲线重计、720 条父 cold/source 曲线匹配、全部臂前 16 索引未变及 18 条 CUDA 重放一致。这是**已有产物的记载**，不是本次 GPU 复现。

作为设计动机而非本诊断成绩，既有 operator 完整流记录 cold 为 `134 TP / 42 FP`、source_rule 为 `133 / 42`、learned_rule 为 `134 / 43`，其 `raw_intra_gate=false` 且 `independent_confirmation=false`。这说明已尝试的新规则没有通过原门槛，不足以判断是基底容量、实际状态更新方向、正常样本覆盖或其他因素导致。

7 个早期 FP 的 `(炮号, 首次越阈索引, 全炮峰索引)` 为：

| 首次报警 index <16 | 全炮峰也 <16，所选端点必须关闭干预 | 全炮峰 >=16，仍不能撤销先前报警 |
|---|---|---|
| 7 炮 | `85320 (15,15)`、`60239 (15,15)`、`85316 (15,15)`、`87363 (7,7)`、`85422 (5,7)` | `93983 (5,29)`、`93306 (15,4998)` |

索引是该炮保存的零基采样位置，不是经过认证的物理放电起点。`>=16` 是可能起效的最早位置，不保证已获得可用反馈、发生非零 write 或最后数据 token 有效。原实现的 8 点 patch、端点映射和 valid 掩码见 [endpoint_corrections / risk](experiments/risk_aligned_online/delayed_recurrent_memory_20261025/model.py:13)；规则实现见 [ContextMemory](experiments/risk_aligned_online/context_memory_20261028/model.py:52)。

## 精确定义建议

1. **选点和方向。** 后 340 中，仅对 cold 原始错误逐炮选点。D/FN：在封存 cold 曲线上，以 `index>=16` 且 `round((DownTime-time)*1000,9)>30` 为候选，取 cold 最高风险。S/FP：取整炮 cold 最高风险，即使它在 `<16`；不得改为选可干预的后期峰。并列时预先固定最早索引；缺失、NaN、无候选、损坏或哈希不符均显式登记并停止对应样本，不能静默删除或换点。D 向提高 logit，S 向降低 logit；按原判定 D 达到 `p>=threshold`、S 达到 `p<threshold` 才算该**端点**跨阈，单列数值边界。
2. **固定模型与两套 basis。** 两条分支共同使用源 base、原 cold 的 `slow` 和 `memory.output`、原 density map、同一阈值和输入。source basis 来自原 source context8 的 key/kernel；EAST basis 来自 [operator.pt](probe_outputs/east_operator360_v1/models/operator.pt) 的 key/kernel，不能用其替换 cold 的 slow/output。该文件另含 rate，但直接优化自由 `c` 时 rate 不参与基底生成，也不沿真实递推更新；所以两 basis 的 oracle 差异不是对两条完整真实在线规则的因果比较。basis 应由同一已到达 patch、同一标准化及有效性规则计算，并核对源／新实际 trace 的当前状态。
3. **作用位置和约束。** 每个启用端点、每个规则，令 `B` 为其 `92×8` basis，`W_output` 为原 cold 的 `64×92` 输出矩阵，`A=W_output B`。原 `context8` 取 `||c||₂<=1`，添加的末数据 token shift 为 `A c`，实际应用仍受 valid 掩码；它是该基底的松弛状态集合，不表示真实递推可到达每个 `c`。对照 `free_last64` 允许末数据 token 的任意 64 维向量，`free_all16tokens` 允许 16 个数据 token 的任意 `16×64` shift，两者在各自完整 shift 上分别施加**同一个总 L2/Frobenius 上界** `R=σ_max(A)`。对无效 token 置零；`index<16` 三条路径一律零干预，即便公式给出非零 `R`。若最后数据 token 无效但其他 token 有效，context8/free-last 的有效 shift 为零，free-all 仍可有 shift；这是更宽松的控制，应标记而不是称作同样的有效预算。`R=0` 须只允许零状态并报告。
4. **注入位置不可偏移。** 三条路径均插在第一、第二 Transformer block 之间、原 slow 对末数据 token 作用之后、第二 block 之前，其他冻结。直接写入 CLS 禁止；但第二 block 的注意力随后改变 CLS 是正常的下游效果，不应误判为直接修改 CLS。16 tokens 是**一个窗口里的数据 token**，不是对 16 个后续预测时刻作连续干预；逐端点独立优化更不构成整炮的可执行状态轨迹。[原 slow 路径](experiments/risk_aligned_online/lifecycle_stream_20261025/model.py:30) 与 [原风险读出](experiments/risk_aligned_online/delayed_recurrent_memory_20261025/model.py:122)给出了具体位置。
5. **有限搜索。** 每个启用端点、每个 basis、每条路径各 2 个初始化、各 100 步 Adam、学习率 0.1，每步投影到对应闭球；记录随机种子、dtype/backend、损失、梯度范数、实际状态/shift 范数、初末及最佳步。用二分类 BCE 的逐点目标等价于分别推动 D logit 上升、S logit 下降；独立逐样本保留已访问的最佳可行状态，包括零步和第 100 步。建议 context8 使用零和该规则的真实可行 context；free-last 用零和最佳 context8 shift；free-all 用零和最佳 free-last shift。这样较宽路径继承已找到的见证，避免有限优化把理论包含关系误报成反向结果；两个起点若实际相同要明示，不能冒称两次独立随机重启。继承见证也使搜索信息量不完全对等，须如实披露。

### 关于范数公平性

对任意 `||c||₂<=1`，`||Ac||₂<=σ_max(A)=R`，故同一 basis、同一掩码和注入点下，`context8` 有效 shift 可嵌入 `free_last64`，后者可嵌入 `free_all16tokens`。这是集合包含，不是集合相等：context8 是至多 8 维椭球像，两个自由球还含大量不可由它产生的方向。等上界也不保证实际使用范数相等、各方向增益相等、风险 logit 变化相等，或 Adam 在 8、64、1024 维上找到同样好的解。

`R` 是输入 `c` 单位球经输出矩阵映射的**最大可能**末 token 欧氏范数，不是每个方向的典型/实际范数、实际递推状态幅度、物理安全预算或跨时刻预算。应逐端点、逐规则记录 `R`、主要奇异值/有效秩、实际 context/shift 范数和最优见证的范数及 logit 变化。source 与 EAST 的 `B` 和 `R` 都可能不同；规则间的端点差同时混有半径、方向/秩、可用性和局部搜索差异。若将来需要隔离纯方向，可另立预先定义的共同半径或投影比较，不能事后把本设计自动解释成该比较。

## 最小验证与报告合同（未来执行，本次均未执行）

1. **溯源/选点。** 固定源、source memory、原 cold、EAST operator、order、map、各炮输入和封存曲线哈希；校验训练／预测模式、参数和缓冲区不变。重建 78 个炮的身份、first alarm、lead、峰索引、打分方向和基线概率；核对 36/42、36 个合格 FN、7 个 early FP、5 个不可干预峰、重复炮和无候选；保留前 20 原始预测。分别标注标签争议与基线在阈值附近的数值容差。
2. **零状态/真实状态。** 各规则同一端点，零状态原模型概率重现对应封存 cold；真实 `c` 的 `B c` 经固定 output/slow 路径重现各自封存 source_rule 或 learned_rule 点值。逐位一致优先，否则预先限定概率绝对误差 `<=2e-6` 且阈值侧不变；靠近阈值而不足以稳定判断的情况单列为数值不确定，不能硬判翻转。校验 `j=floor(index/8)-1` 及无可用写入的情况。不能仅比较最终 360 汇总，任何模型/代码版本不一致先停下定位。
3. **包含关系与冻结。** 合成和真实输入分别测试：CLS 在注入点逐位不变，padding 不可改；`index<16` 对三个路径均零作用；`||c||<=1`、`||shift||<=R`、有限性、零半径、近零/退化 `A`、误差容差；将 context 解嵌到 free-last、再嵌到 free-all，复算概率相同。包含继承见证后，逐点最佳方向损失应满足 `free_all <= free_last <= context8 <= zero`；FP32 下可先以逐点 BCE 绝对容差 `1e-6` 核对，仍须结合概率复放和阈值侧检查。若不满足先检查重放、掩码、投影和存储，不解释为结构容量倒置。
4. **优化和见证。** 每个样本独立运行两起点及约定 100 步，检查 Adam 状态不跨样本／分支泄漏，保留所有起点和迭代的最佳可行见证，回放最佳 shift 得到相同 score/范数/阈值判定，记录失败、非有限梯度和重复起点；不得根据后缀标签再选步长、半径、种子或胜出臂。记录实际两 basis 的 `R` 与范数，逐炮显示端点原值、最优值、margin、启用状态与见证索引。5 个前 16 峰明确为结构性关闭，不统计成优化失败。
5. **输出口径。** 分规则、分 D/S 列 78 个已选端点的状态和实际找到的端点翻转及分母，区分可干预/关闭、early FP 和标签争议；给出有效搜索失败及原因。不要报告 TPR/FPR 救回、全炮 FP 消除或整体容量上限，也不要把其他端点或 262 个原正确炮假定不变。若观察到某点在单独 oracle 优化下越阈，可称该点有一个所检约束下的**可复现见证**，不能称该炮被在线救回。

这些操作需要真实数据与模型的受控前向／梯度；本次审查并未执行。尤其优化后一个 FP 峰变低，不保证其他峰也低于阈值；为每个点分别挑选未来标签指导的状态，也不保证存在一条统一、可因果生成的整炮轨迹。

## 历史查重与当前在途文件

在本工作树及主树 `experiments/risk_aligned_online` 的 Python/Markdown、主树研究文档、相关工作树报告及 `probe_outputs` 命名范围内检索了 state/representation capacity、context oracle、自由 token、reachability 等关键词；这是**有界检索**，不是对全部压缩包、其他工作树、远程或删除历史的穷尽证明。没有发现完全相同的**已完成**诊断结果。

最接近的已完成先例并不相同：

- [2026-10-29 cold capacity 方案](COLD_CAPACITY_PLAN.md)及[结果](probe_outputs/cold_capacity_v1/complete.json)：旧顺序与旧 cold，覆盖后 340 炮、398 个选定峰；仅 `index>=128`，对 embedding gain/bias 做 64 步标签 oracle，另测半径 1 和 4，另有第一 block 后位置探针。其报告中半径 1 对 44 个漏／迟报找到 4 炮的端点越阈、对 31 个 FP 原峰压低 4 个；不是现在的 36/42、两 basis 与同一输出 shift 预算，亦不是在线救回。
- [低秩 Q/K 容量先例的汇总](UPDATE_GEOMETRY_GAP_REVIEW_20261030.md:28)：旧父模型、相同旧 398 个选点，目标是真标签指导的第一层 Q/K 或 V 参数因子而非现有 context8 输出椭球；其后真实反馈的完整流不曾自动继承 oracle 端点能力。历史显示“有见证”与“能学到真实更新”可以显著分离，但旧数字不能移植到当前 78 炮。
- [176 查询的 context-vector 对全 Q/K 诊断](probe_outputs/context_vector_cold20_v1/query_capacity_diagnostic.json)同样使用事后标签，但样本是 cold20 查询而非当前后缀错误炮，干预和优化预算也不同；不能替代当前比较。

本次查重还发现工作树中**未跟踪、当前出现的** [state_capacity 实现](experiments/risk_aligned_online/state_capacity_20261031/optimize.py)和[测试文件](tests/test_state_capacity_20261031.py)。读取时实现 SHA-256 为 `57130de065780527da16812f59da315e4c3e848afa1363480fab8f80a03be845`；该文件实现了三种路径、零/真实及嵌套见证起点和输出谱半径，但它本身不是已完成的 78 炮运行或验证结果。它及测试文件可能被其他工作过程继续修改；本次没有编辑或运行它们，也没有验证其端到端选点、basis 装配、快照与结果存储。执行前仍需按上面的合同核对，不以文件存在替代审计。

## 可得／不可得的结论

**可得：** 对这批开发暴露的固定错误样本，在指定点、指定已知标签方向、指定模型／basis、同一 `R` 和有限优化预算下，哪些见证被实际找到；若源/新两 basis 的结果不同，可描述为该具体两套事后 oracle 的观察差异。成功见证可以证明在其数值容差与受测约束内存在一个点状态；失败只能说这两次有限搜索未找到。

**不可得：** 局部 Adam 失败不证明不存在可行状态、不证明任何全局最优或不可救回；自由 token 集合更大或出现更多见证，不证明当前真实递推可达到、可从无标签到达信号辨识或可学习。用全炮峰、终局类别和 DownTime 选点／定方向是明确的未来信息泄露，只能称 label-informed post-hoc oracle，不能宣传为在线 TPR/FPR、实时延迟、部署收益或发现物理规律。只检查原本错误的炮，也不能知道相同策略是否损害原本正确的炮。

该平衡 360 炮和原方法已反复用于开发，且 HDF5 标签与 TEST 列表存在 18 炮争议，cold20 仅 2 个安全炮；见 [标签审计](TIME_LABEL_AUDIT_20261031.md)。这些限制不会因数学见证消失。若以后要检验可学习性，需另行授权并预先定义只用当下已到达信号、无需当前炮未来标签的规则，对完整封存流与正确炮共同评估，并在独立数据上确认；这不属于本次只读审查。

## 本次文件指纹

- 父 `completed.json`：`cb9cf19197ab25fb6e825687ec002bcad6135834dca1cb7dc3371234eb7883a9`；`per_shot.json`：`42be409a13d8c804b061dd047861516c39eee44ae19a3ae519d650df4e6ba66b`。
- 父 `order.json`：`ad44224b7f760eed43305420200c790ee893dc5511dd2f0c0bd280d1a8d5e5c9`；`input_map.json`：`7056d394be54463f6891c166e47de3918fb0d4c38a0585aad1341bb274725ff2`；`models/cold.pt`：`31ccf7a12567d3af4d4df93d41894339a81077e89d6d47aebcc24306673173a7`。
- source context8 memory：`7cef5d4e9122b086886c3ee4b429b9cdf2663a921f4972b6687ceca5d74a81e5`；EAST `models/operator.pt`：`1f27b0c3d74739a560f0e79bcd54c0f23a36c77d63f0262b31237dd9cf635824`。
- EAST `completed.json`：`3edfa9b2a3025f327533ab252b61cde95d550f219447215eccce90a8cf147c33`；`verification.json`：`20a5cda99d762973ad21c77332402807463314b3b5452ee2551447a86e14d54c`。
