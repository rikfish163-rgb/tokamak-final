# EAST 实际缺失 mask 的静态修复评价

状态：评价器和 CPU 合成测试；未执行真实 360 炮，不声称收益、独立验证或炮内更新。该开发序列、炮序和映射此前均已用于开发。

## 固定合同

- 使用 `qualified_source_static360_20261104_v1` 的固定 source parent、同一 360 炮及顺序、同一 map 和 256 窗口 batch。前 20 炮不映射，第 21 炮起使用已存固定 map；不重拟合 map，不读取 EAST110 或旧 cold 模型。
- 输入仅一套实际 EAST 可用性：沿用 `simple_data.read_shot` 的 MNM 不可用合同和 `apply_input_contract` 的逐点 CIII sentinel；数值映射不得改变 mask。不引入 `no_MNM_CIII` 反事实视图。修复只改变内部表示，不把估计当作原始物理信号。
- 对已读的整炮信号，在 input contract 之后、数值 map 之前，从原 mask 分别记录 CIII/MNM 的可用点数和比例；分母是该炮总采样点数。每炮 seal 和汇总 `coverage.json` 保存相同结果，输出哈希覆盖后者；不为统计重读原始数据或读取标签。整炮统计仅供事后诊断，不能作为炮内某一时刻可用信息：未来若评估依赖 CIII 的更新，必须使用该时刻已到达窗口中的真实可用性；没有 CIII 观测就不能编造 CIII 更新。
- 四臂为 native（不修复）、TRAIN constant、conditional、wrong_context，复用冻结的 `evaluate_batch`、`load_fit` 和 core 的资格/RMS 限制。native 常量选择沿用 source 评价器：若该 token 的 CIII 不完整，使用双缺失 TRAIN 均值，否则使用 MNM 缺失 TRAIN 均值。
- wrong_context 在同一当前窗口内逆序全部 token 上下文，含 padding；仅为非部署负对照，padding 是潜在混杂。base、repair、threshold 均固定，无神经更新、标签拟合或阈值搜索。
- 对每炮先完整预测四条曲线，将 native 及时间逐值比较基准已封存的 `map_after20`，并验证其 completed/逐炮 seal/曲线哈希。任何不相等均明确失败，不用容差替代；CPU 与基准 GPU 或不同运行环境发生数值差异时也必须失败，不据此推定模型收益。
- 一炮的四条曲线及 SHA seal 落盘后，才读取该炮 HDF5 `meta/IsDisrupt` 和 `meta/DownTime`；使用固定 parent 阈值及严格 `lead_ms > 30`。沿用已有 scorer 的 9 位小数毫秒 lead 舍入。保存逐炮、full360/startup20/suffix340 汇总和配对计数，及原始文件、输入、代码、基准和输出哈希、代码 ZIP。

## CPU 验证和运行交接

先写测试时新模块尚不存在，实际 RED 为 `1 error in 1.40s`。合成 HDF CPU 测试覆盖四臂、mask 和第 20/21 炮映射边界、未来修改不改变先前曲线、严格 >30 ms、native 与原静态预测器逐值一致、与 seal 绑定的参考曲线不一致时拒绝，以及封存后才揭示 HDF 结果。合成的 22 炮配对数据额外检验前 20 与后缀计数；CPU 测试不证明真实 360 炮或 GPU 逐值一致。

只读检查已验证既有基准 completed、输入、360 个 seal 元数据及前 20 map ID 的绑定；没有读取真实原始信号或运行 GPU。CUDA CLI 使用既有 `configure()`；CPU CLI 使用等价 seed、线程和 fastpath/TF32 设置。真实 GPU 运行由主进程决定：

主进程提供的独立 CIII 原始通道核查结果（本任务未重读或复核）：360 炮中 291 炮有可用点、69 炮无可用点；1,509,354 个采样点中 1,231,722 点可用；前 20 炮中 15 炮有可用点，共 58,790 可用点。真实运行后应使用 `coverage.json` 复核这些计数，而不是先将它们硬编码为假定的结果。用户另报告新 16-state 候选在 source VAL 的两个主要组 gate 失败；本 EAST 静态评价仍只作诊断。

```bash
python -m experiments.risk_aligned_online.mask_repair_20261104.east_evaluation --compute-device cuda --output probe_outputs/east_mask_repair_static360_20261104_v1
```

要求新输出目录。若环境数值或其他因素导致 native 精确对照失败，程序在相应炮的 HDF 真值揭示前停止，需审查原因；不能忽略该失败或以旧曲线代替计算结果。完整的 CPU 测试输出以交接回复为准。
