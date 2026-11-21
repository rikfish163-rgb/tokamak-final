# Joint LoRA 设计与集成独立审查

审查时间：2026-11-04。范围为已冻结的计划、模型、训练、时间线、runner 和对应测试；本审查只写本文件。没有运行 GPU、读取新原始 HDF，或改动 producer。主进程报告的 67 项 CPU 测试、22 炮流程和 CUDA 四模式 preflight 不等同于本审查亲自复跑的结果；截至审查时，本人未检查完整流结果。HDF v4 标签未由本审查重新读取或验证。

## 结论与优先项

未发现已经证实会破坏每炮预测—反馈时序、或使 inter/both 各自维护不同 slow 轨迹的 critical 缺陷。以下是需要明确区分的具体限制与条件性故障：

1. **条件性全流中止，已用 synthetic 复现。** 对任一第 21 炮起的结束炮，如果没有满足窗口和标签条件的 `current` 记录，`eos_schedule` 明确抛错并中止，包括该炮信号很短、或破裂发生在首个训练起点之前的情况。参见 `training.py:14–47,130–143` 和 `run.py:335–339`。例如 200 点、1 kHz、破裂时间早于第 135 点的 synthetic 炮可产生空记录。真实 360 炮是否存在该条件，本审查未知；这不是静默错用未来标签，也不是建议在结果出现后擅自跳过。另一种未预声明的类别组合亦按计划明确报错（`training.py:152–163`）。应把异常原样视为运行失败，不把部分流当作完成。
2. **揭示前封存证明不完整。** `run.py:243–260` 在揭示前写 `_updates.json`，但 per-shot seal 只收录 curve 和 forecast 哈希，没有收录该 update-history 的哈希；`completed.json` 在结束时才汇总所有文件哈希（`run.py:421–434`）。因此可由日志重算、检查事件，但完成时哈希不能独立证明 update-history 在结局揭示前未被更改；这不是发生更改或标签泄漏的证据。不要把这一点说成所有逐炮产物均有揭示前哈希。
3. **“进程中绝无未来炮派生信息”并非字面成立。** `load_inputs()` 在流开始时加载锁定的 `order` 和 `input_map.json`（`mask_repair_20261104/east_evaluation.py:28–59`，`run.py:113–114`）。已存的 order 字段包括未来炮的 `initial_ip_A`、`current_energy_proxy` 等；原排序器明确由每炮约 100 ms 的 Ip 前缀生成这些量，并按 proxy 排序（`current_order_20261031/order.py:1–5,56–105`，`stratified.py:1–5,16–45`）。map 文件也含前 20 炮信号的汇总统计。前 20 炮 `prepare_actual` 不应用该 map，第 21 炮起才应用（`east_evaluation.py:62–81`）；逐炮信号读取受限于当前 sid，未见未来 order 字段作为模型输入或 HDF 结局替代。故这是**预先确定、利用全 roster 前缀构造顺序的离线回放**，不能写成严格按真实时间在线到达或事先完全不知道未来信号统计；现有代码证据也不支持将其直接认定为未来结局泄漏。
4. **W0 时序应精确表述。** `run.py:124–131` 开始前对旧 W0 文件做 SHA256，因而会流式读取其文件字节；但 `torch.load` 和得到 W0 张量只在第 20 炮封存及揭示后调用（`run.py:41–47,267–268,294–307`），并核对前 20 ID 和 source hash。W0 作为 buffer 被克隆，训练只允许 LoRA 和残差头，之后受 frozen digest 检查（`model.py:91–101`，`training.py:90–96`，`run.py:50–57,380–382,393–400`）。不能笼统声称 W0 文件字节此前完全未被读取。

## 512 参数的诊断梯度与风险路径

`model.py:91–108,117–147` 复制并隔离 source，只在末层 Q/V 的 packed 投影加入 rank-2 LoRA：Q/V 各 `A: 2×64`、`B: 64×2`，合计 512 个标量；K、原始 packed 权重、风险头和其他 source 权重被冻结。诊断为

`y_hat = W0 k + D h + b`，其中 `h = [CLS, last, last − previous]`，维度 192，诊断头 `D` 为 168×192（加 bias 共 32,424 参数）。风险从同一 encoder 计算，但走冻结的风险头，并不直接读取诊断输出。`simple23.py:124–140` 表明两者共享 norm 前的 encoder 表示；诊断 hook 捕获 norm 输出，风险还经过冻结的逐特征仿射与 padding mask，因此不应说二者的最终特征完全相同。

一个确定的初始化性质：`D=0` 时，`∂L_diag/∂θ_LoRA = (∂h/∂θ_LoRA)^T D^T (∂L_diag/∂y_hat) = 0`。`W0 k` 不依赖 LoRA，故第一步诊断损失只能先训练残差头，不能单靠此损失立刻推动 LoRA；风险损失可同时推动 LoRA。`tests/test_joint_lora_model_20261104.py:197–204` 覆盖该事实；`:301–315` 的诊断梯度正例先人为赋予 decoder 非零值，不能单独证明实际 warm 后必有有效梯度。又因初始 `B=0`（`cold20_compare_20261104/adapters.py:17–36`），初始 A 因子甚至对风险的梯度也可为零；“开放 512 参数”不等于每个标量每一步均有非零梯度。

本人用合格冻结 source checkpoint、合成窗口与真实 `direct_target` 公式做了一次 CPU 内存实验，W0 取合成零矩阵：

- 初始诊断损失对四个 LoRA 因子的梯度范数均为 0，而 decoder weight 梯度范数约 0.0657；同一 synthetic risk 损失对初始 Bq/Bv 的梯度范数约 0.00333/0.00847。
- 一次 synthetic 联合 warm 步后，decoder weight 范数约 0.1788；只用诊断损失时，Aq/Bq/Av/Bv 梯度范数约为 `4.25e-6 / 2.39e-5 / 2.21e-6 / 7.86e-5`。
- 将这一 synthetic 诊断梯度应用到独立 fast 克隆的一次 Adam 步后，同一目标上的诊断 MSE 从约 0.008523225 降到 0.008523139，风险 logit 最大变化约 `1.05e-5`，LoRA 因子位移约 0.00213。source、W0 和父模型未因此改变。

这只证明该路径**可以**在合成实例上从诊断损失影响风险 logit，不证明真实 warm 必然产生足够梯度，也不证明风险改善或泛化。慢训练 history 的 `gradient_norm` 是 decoder 与 LoRA 上的**联合**范数（`training.py:90–120`），非诊断到 LoRA 的单独范数。完整流后应在已保存 `warm.pt` 上检查 decoder 是否非零，并在有支持的 real 到达事件中分别统计 `gradient_norm`、`updated` 和 `parameter_displacement`；不要仅凭联合范数或 MSE 下降宣称 512 参数确实收到有效诊断梯度。`updated` 表示参数实际变化；Adam 的历史动量也可能造成变化，不能把它等同于当步梯度非零。

## 发布、成熟和窗口边界

时间均为零基样本索引。`timeline.py:143–158,216–231` 定义 `q=127+32k`；先把诊断预测 detach 后记录，到 `q+8<n` 时才从 `q+1:q+9` 建实际目标及逐通道 mask。`direct_fast_memory_20261104/memory.py:88–107` 要求 origin 末点与对应 future 点均可用，目标为 source asinh 坐标的未来值减末值，排除 RAW23 的 2、10 两列；不可用目标为零并被 mask。不会从预测值伪造真实标签，也不应将其说成物理单位预测。

| 事件 | 允许发生的事 |
| --- | --- |
| `q=127` | 用截至 127 的窗口发出并保存预测；此时不能用 128–135 训练。 |
| `t=135` | 128–135 均已到达后，以已发预测计算实际 pre-update MSE；如支持非零，则可重算可微预测、用实际/hold/reverse 指定目标更新本炮 LoRA，再预测时刻 135 的风险。 |
| `q=159 → t=167` | 同样先发第二个预测，后在 167 成熟；此前 135 的更新可以影响 q=159。 |
| 炮末尚未成熟 | `released_at=-1`，target 为 NaN、mask 全 false，不更新、不发明损失。 |

`timeline.py:163–201` 的 `loss_before_actual` 由**保存的已发值**而非重算值与真实 target 计算；hold/reverse 的反事实只改变训练目标，评估仍用同一真实 target 和 mask。无支持时跳过 Adam，history 的损失为 `None`。`timeline.py:85–98` 对风险固定 batch 32 且尾部 padding；`:131–142` 每炮 deepcopy、重建 Adam，fast 不写回父模型。`tests/test_joint_lora_timeline_20261104.py:126–193,243–313` 对前缀、135 边界、pending、实际目标评价及无支持的 Adam 状态给出 synthetic 检查。

这是代码层面的离散时间因果模拟，不是外部逐时刻发布证明：整个当前炮信号由离线 reader 放入 RAM，q 时预测先写本地 records，完整 forecast 文件直到炮末才写出并封存（`run.py:189–195,238–260`）。逐 q 的外部不可篡改时间戳及真实推理延迟均未在此证明。`issued_vs_recomputed_max_abs` 是值得保留的数值诊断：比较支持事件上已发预测与成熟重算的最大差；pre-loss 必须仍以已发预测为准。一次 CPU synthetic 检查中，改变 PyTorch MHA fastpath 会造成约 `1e-6` 级风险数值差，不能把 CPU 现象外推为 CUDA 误差界；runner 的 CUDA `configure` 禁用了 MHA fastpath 和 TF32（`east_context_operator_20261031/run.py:38–45`），真实 CUDA 仍应读实际日志。

## slow 轨迹与 EOS 日程

前 20 炮全部 arm 是 native 拷贝；第 20 炮封存和揭示后，才用这些结束炮和锁定 W0 准备 `fixed`，随后复制出 `slow`（`run.py:200–206,267–268,294–320`）。从第 21 炮起应理解为**两个 slow 制度**，而不是要求所有后续炮各臂有相同权重：

| 评估臂 | 炮初父模型 | 本炮 fast 目标 |
| --- | --- | --- |
| static / intra / hold / reverse | 同一个冻结的 `fixed=S20` | none / real / hold / reverse |
| inter / both | 同一个逐炮更新的 `slow=S_t` | none / real |

见 `run.py:208–228`。每炮各臂运行后校验父模型 digest 未变；预测和 forecast 封存、HDF 结局揭示后才对唯一 `slow` 执行 EOS，当前炮记录在训练后才进入历史 replay（`:243–268,335–379`）。代码 patch 在预测期间阻止 `read_outcome`，在 warm/EOS 阶段阻止新 `read_signals`/`read_outcome`（`:189–195,282–293`）。窗口本身只取截至 query 的 128 点（`simple_data.py:41–48`）；当前炮真实目标只能在样本成熟后参与 fast 更新。

EOS 在计划 `JOINT_LORA_PLAN_20261104.md:21–29` 明列：每结束炮 8 步、batch 64、各 32 个当前及历史记录、Adam `1e-4` 并重置、`risk + 0.1*diagnostic`、clip 1、seed `131 + stream_index`。标签与未来 8 点窗由结束炮生成（`training.py:14–76`），不是用本炮 fast 参数晋升训练。每通道是否可用仍由实际 mask 决定，要求 8 个时间点存在不等于 8×21 全部观测。`run.py:350–375` 保存被实际抽到的去重 records 并重映射 schedule；本人 synthetic 检查了 8×64 次抽样经映射后逐项相同、含放回重复，且每批当前/历史各 32。`replay_buffer_indices` 指向当次 `current + previous` 合并表，而非单独的 previous 表；重构训练时应按此解释。

## 完整流后的最小审计断言与科学解释

以下是检查建议，不是已取得的真实结果，也不要求变更冻结中的 producer：

1. 校验完成清单、全部 curve/forecast 哈希、源与计划代码哈希；首 20 炮七臂 curve 与 native 逐点相等。每炮验证 `origin=127+32k`、成熟恰为 `origin+8<n`、pending 的 `-1/NaN/false`，并从已发 prediction、真实 target、mask 独立重算 pre-loss；各反馈臂应使用相同真实 target/mask。对 static/intra/hold/reverse 和 inter/both，分别核对 `[:135]` 同父前缀；135 本身允许更新后不同。
2. 对第 21 炮 `seal.shot_start_parents.static`、`.slow` 均与 `warm.json.fitted_state_sha256` 对照；其后每炮 static hash 恒为 warm hash，且该炮 `inter.parent_before` 等于同炮 seal 的 slow hash，下一炮 seal 的 slow hash 等于上一炮 `inter.parent_after`。逐项复核 saved records/schedule 对应已结束炮、当前/历史各半；检查 fast 没有进入下一炮 parent。
3. 统计有支持事件数、各臂实际更新数、诊断 LoRA 梯度范数、因子位移、issued/recomputed 最大差，以及相同 mask 下已发预测相对 hold 的误差。`timeline.py:187–193` 返回的 gradient norm 是 clip **之前**的范数；`postloss_actual` 是刚用来训练的同一目标上的事后损失，不是下一次预测的泛化证据。半径 0.5 约束四个 LoRA 因子拼接向量的炮初位移（`timeline.py:31–57`），不是有效矩阵 `B_q A_q` / `B_v A_v` 的 Frobenius 半径，也不是物理安全界。
4. 最终按原固定阈值、HDF 结局、统一时钟和严格 `lead > 30 ms` 比较 intra−static、both−inter、both−intra，以及 real 对 hold/reverse；同时报告 TP、FPR、J、30–3000 ms、>3000 ms、逐炮救回/损失和新增/移除安全炮误报（计划 `:31–35`，评分实现 `hdf5_score_recheck_20261103.py:19–33`）。辅助 MSE 较优、同目标 post-loss 下降或真实梯度非零都不能代替风险对照；回放顺序和开发方案已接触既有结果，不能称独立未来验证。

另外，末两块 token 在被 mask 前就由 hook 捕获（`model.py:128–145`，`simple23.py:126–132`），即使末 16 点完全不可用，latent 仍可非零；对应测试明确允许这一行为（`tests/test_joint_lora_model_20261104.py:145–161`）。本人 synthetic 检查将不可用点的原始值替换为极大值，模型输出不变，但末 patch latent 非零。这是由位置/上下文形成的表示，不能当作“最近两块均有真实测量”或单凭非零值判为泄漏；解释结果时应结合可用性。

## 独立核验与冻结快照

本人在 CPU 且禁用 pytest cache、bytecode 写入和插件自动加载的条件下，运行 `tests/test_joint_lora_model_20261104.py`：**40 passed**；另对冻结版本 `tests/test_joint_lora_training_20261104.py` 和 `tests/test_joint_lora_timeline_20261104.py` 的因果、训练、EOS、pending 子集运行：**13 passed, 18 deselected**。最初误用无 PyTorch 的系统默认 Python 导致 collection error，改用现有 `rl_env` 后成功；该错误不计作测试通过。本审查未运行会创建 synthetic HDF 的 runner 测试。以上 synthetic/代码证据不是完整 360 炮或 CUDA 实测证据。

审查时的 SHA256（完整值）：

| 文件 | SHA256 |
| --- | --- |
| `JOINT_LORA_PLAN_20261104.md` | `d41903f0e57b8e4b2dc86200e136077c113d896d6898829faac3ee1d5b1e7d56` |
| `joint_lora_20261104/model.py` | `eaaa052179b2876de25d569f8077cda23bcb88c27f6bc168307a32ee6e288161` |
| `joint_lora_20261104/training.py` | `d67d1afd891f2c66e996fdbb140db208167d0dbdbb876f5fab1136f61ff0c79f` |
| `joint_lora_20261104/timeline.py` | `f4c23996d3b35a808598dabe18236c36098519a5a92dc828cf81a75dd8f5abdc` |
| `joint_lora_20261104/run.py` | `9be5ea33d83589dbc8686b2a2170a350ff031a1d1c4c192d81d8999efe67a506` |
| `tests/test_joint_lora_model_20261104.py` | `146f7de65b0057836c70555597cad34a034655af3528a157a2a897f334a8623e` |
| `tests/test_joint_lora_training_20261104.py` | `51c79d6e75505814cf8f25027e5640246d222c199e6dd3a30232b35cebaad8c3` |
| `tests/test_joint_lora_timeline_20261104.py` | `f7d4be3abaa3f7ae7a023573b783c7e27e7450f7eaffde4d3a62dda022db9211` |
| `tests/test_joint_lora_runner_20261104.py` | `51674fa7ae8eb516b34a96c2a198e0c51d7fb221f4df0e35369413ee7b7492b8` |
