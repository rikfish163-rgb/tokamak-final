# 输入因子运行器代码审查

日期：2026-10-31。只读审查，不修改实验代码，不运行GPU。本报告适用于 `core.py` SHA256 `b03e6696d46918704cf54412c1d7c0820e0e3057c8a3bdde6f34dcc4c3877ccb`、`stream.py` `7cd3060bd84f9f68c82ee5cc3640494308bc967e0463b9b408ad8d8bb828b7cd`、`run.py` `9cec97b5cb354dd2051c95f88a2f5065af720516d5a879e2aaaa826a760251dc`。若主进程继续改动，须按新版本复核。

## 应在GPU完整流前修复：全局封存完整性缺口

**[P1] 全局屏障写入前未重新核验全部预测文件，结局访问后也没有最终整批复核。** [run.py:119](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:119) 在每炮写入时计算摘要，但 [run.py:133](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:133) 直接用内存中的 seal 列表创建屏障；`GlobalOutcomeGate.release` 只核对列表和数量，不重新读取360个预测文件。[run.py:137](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:137) 是**逐炮**重新算 SHA 后立即读该炮结局：若第360炮的文件在其写入后、屏障前损坏，程序会先读取1–359炮结局，再发现第360炮不符并失败。另外，已检查并评分的早期文件若在之后改变，写 `complete.json` 前没有全量复查，可能留下与摘要不符的交付物。当前每炮 seal JSON 也没有被屏障锁定或重新核验。

这不是已经发生文件损坏或已经泄露标签的证据；在文件始终不变的正常路径上，逻辑仍先封存360炮再读取本轮结局。但它不能保证计划要求的**严格全局已封存且仍完整**条件，因此建议作为运行前阻塞项。已有实现 [context_vector seal_predictions:222](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/run_full.py:222) 示范在屏障前重核全部 NPZ 和 seal；新运行器也应先记录各 seal 文件摘要、逐一核验后才 release，并在最终完成标记前复核封存物。补一个CPU模拟测试：篡改末炮后调用屏障，断言零次 outcome access；另测试评分后篡改早炮时不能标记 complete。

核验方式：除源码检查外，用合成 `GlobalOutcomeGate`、360个虚构摘要和模拟 barrier read，在不存在任何预测文件的情况下观察到 `released: True`、合成 reader 可调用。这只说明 gate 本身不验证文件，结合上述 runner 路径才构成缺口；没有读取任何真实结局。

## 其他具体问题

**[P2] 已锁定的历史完成记录未用于验证其 protocol/barrier。** [run.py:55](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:55) 只核对历史 `complete.json` 自身的固定 SHA，随后直接读取 [protocol.json:57](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:57) 和 [barrier:92](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:92)，并未把二者摘要与 complete 中 `artifact_sha256` 比对，也没有在运行末尾复核它们。每炮历史 NPZ 与 barrier 的自洽检查不等同于 barrier 已接入固定完成记录。**当前实际文件已手工比对，两项摘要确实匹配**，因此未发现现有归档发生漂移；问题是代码无法保证以后仍如此。建议读取前和完成前核验这两项及其固定 order/reference，并纳入针对性测试。

**[P2] 所谓1800秒硬时限不是硬中断。** [run.py:94](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:94) 在准备结束后开始计时，[run.py:103](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:103) 只在每炮开始前检查。单炮前向可以越过期限继续运行，末炮或之后评价也不会再触发该检查。若1800秒是GPU调度硬约束，需要外部可终止的超时机制或在批次及评价阶段实施有效限时；失败目录保留，不标记完成。

## 本次确认的合同与边界

- [core.py:4](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/core.py:4) 六曲线中 `identical` 和 `cold` 是同一恒等变换，另有四个单列因子；不应计作六个独立条件。
- `fit_maps` 检查360个唯一身份、冷启动映射的两个前20身份列表和源/目标 device；其统计来自哈希锁定的既有 density map。前20直接复用归档曲线，第21炮起在 density map 之后执行单列变换。该链是对已结束cold20统计的复用，不从当前炮拟合统计。
- `apply_map` 复制值与 availability，仅在目标列有效位置写乘法或加法，未改变其余列及掩码；`apply_input_contract` 先屏蔽 CIII 哨兵。`stream.py` 为所有条件沿用同一个掩码和全零Q/K状态。
- `chunks`、窗口、补齐、9×32送入模型及逐行 sigmoid 与归档批次几何一致；`identical` 与 `cold` 实时逐点相等。调用的 [check_cold_archive](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/operational_transport_20261030/run.py:115) 确实使用 `np.array_equal`，不是容差比较。**真实GPU实炮是否逐点相等尚未在本审查运行**；不匹配应直接停机，不能放宽容差。
- 预测阶段禁止 `read_outcome`、`labels_at` 和固定测试合同读取；gate 要求360个按序唯一 seal 才允许本轮结局访问。评价复用 `score_fixed_test`，要求完整锁定身份及180破裂/180安全，并按首次报警及严格 `lead > 30 ms` 计数。以上逻辑成立的条件是先修复上面的封存文件完整性缺口；它不否认 cold20 历史监督已经用于既有 cold 参数。
- 纯CPU、禁用pytest缓存的现有测试输出：`15 passed in 3.20s`；该测试文件不覆盖真实GPU精确重放、输入运行器全局屏障或归档哈希链。审查期间没有启动GPU、训练或改动共享代码。

## 后续接入复核与裁决（2026-10-31）

以上原始发现针对旧 `run.py`；本节适用于后续 `run.py` SHA256 `8813ea8aadfbc6d100f257e49a8d132fa8cfadc585ab1bd5b50958da691af3d3`，不改写旧审查历史或 v1 产物。

- **原 P1 的运行时复核缺口：未来运行已补强。** [run.py:129](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:129) 在每炮写入 seal 后将其 `seal_sha256` 放入 receipt；[run.py:137](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:137) 在写屏障及放行结局前检查全部已收集 NPZ 和 seal，[run.py:165](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:165) 在写完成记录前再次检查。`fit_maps` 和 gate 分别检查360个唯一身份和按序封存；helper 本身不单独证明列表是完整360炮。
- **原 P2 归档哈希链：未来运行已补强。** [run.py:56](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:56) 在读取归档 protocol/barrier 前核对锁定 complete 文件及它所声明的两项摘要；[run.py:164](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:164) 完成前再查一次。helper 在摘要缺失或不符时失败，不重建缺失摘要。
- **剩余可审计性边界，非本次 v1 数值失败：** future receipt 的 `seal_sha256` 在内存中记录，而 [run.py:139](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/input_factors_20261031/run.py:139) 写出的 barrier 只包含 `gate.seals` 的炮号和 NPZ SHA；`receipts.json` 到结局评价之后才写入。因此正常完成路径中有两次内存凭证复核，但若在放行后、保存 receipts 前中断，仅凭持久化 barrier 不能恢复事前 seal JSON 摘要，也不能仅从事后文件独立证明 seal 摘要何时记录。如需中断后也具备该证据，可在未来把预先计算的 seal 摘要一并持久化到 barrier 并测试；本次没有修改运行器。
- **v1 保留有效观测及其证据边界。** 已完成运行的 `complete.json` SHA 为 `a46dabcd514fec32d5b7f318605a2b9b590b596d298d24edccfd5353503e58e6`，原 code snapshot SHA 为 `b0a95e4107217dbdfd56a8112ec6f2572ec5ccb317a7554e74c1f6c9844382fa`，与完成记录一致；快照保留旧 runner 和执行前计划，不包含后加 helper。独立核验记录 [INPUT_FACTORS_VERIFY_20261031_v2.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/INPUT_FACTORS_VERIFY_20261031_v2.json) 报告2160条曲线、24个指标分段及此前3炮18条曲线GPU重放通过；本次复核未启动GPU。v1 原 receipt 无事前 `seal_sha256`，不能补造或把后续 helper 的保证追溯赋予它；没有发现实际损坏，不需因此重跑或否定已观察的结果。
- **1800秒：本次未触发，不是实际故障。** v1 完成记录为252.14秒。Python检查仍是逐炮而非硬中断；[计划的执行后补记](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/INPUT_FACTORS_PLAN_20261031.md) 明确未来才使用外部 `timeout` 并选新目录，未声称v1使用了该后加措施。
- **本次验证范围：**重新运行纯CPU的两个定向测试文件，`35 passed in 2.34s`；对四个模块及两个测试文件执行 Ruff，输出 `All checks passed!`。测试覆盖 helper 的缺失/篡改拒绝和 core/stream 合同；运行器实际调用次序本次通过源码核对，并未在这些测试中模拟整条 runner 的篡改或再次执行完整流。本次只追加本报告，未改 runner、测试或启动GPU。

主控后续整合补记：在上述复核之后，已将内存中预先记录的`seal_sha256`列表同时写入未来运行的全局barrier，处理了所述中断持久化边界。未修改v1归档；这属于后续运行器元数据增强，不是对v1追溯添加事前凭证。原审查中的行号和SHA仅对应各节注明的版本。
