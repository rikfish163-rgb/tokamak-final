# Source mask-repair 静态评价：实现与 CPU 审查

状态：source `val` / `test` 评价入口已实现；没有运行真实 source 评价，没有读取真实原始信号，没有运行 GPU，也没有证明模型收益。这里的合成测试及真实 fit 的只读权重检查不是独立验证。EAST 不在本次实现范围。

## 合同

- 固定已完成的 TRAIN fit 和合格 source 基模；不微调、不选择训练步或阈值。`completed.json` SHA256 固定为 `1a7304a619414508bfecb84dc710dbd5b6a15f46265de66f829914efe4a0bc85`；在 `weights_only=True` 加载修复权重前验证 completed、其产物、fit 输入和代码哈希。对照 fit 的 source 状态摘要与基模阈值。
- 原始 source 三个视图：`native`、只移除 MNM、移除 MNM 和 CIII；四臂分别为 `native`（无修复）、TRAIN 均值 `constant`、`conditional` 和 `wrong_context`。真实 native 中也可能有缺失；只有完整观测 token 及全 padding token 必然不修复。source 只按原有有限值 mask，不应用 EAST 的 CIII sentinel。
- 四臂都走未修改的 `core.forward`；三种修复由其统一实施资格筛选和每 token RMS 上限 0.25。native 视图的常量策略：CIII 有缺失时使用双通道缺失均值，否则使用 MNM 缺失均值；其余两个视图使用各自对应的 TRAIN 均值。
- `wrong_context` 把同一个当前因果窗口的 16 个 token 上下文（embedding 和可用比例）沿时间反转，然后按原位置 eligibility 修复；不跨 batch、炮或未来窗口，不使用标签。这是非部署负对照。**反转包含 padding，padding 与时间错配混杂，不能把该对照单独解释为纯粹的上下文依赖证据**；常量补偿仍是重要强对照。
- 每炮先把 12 条曲线与时间写入并封存，再从封存文件读取评分数据，随后才通过 `io23.read_outcome` 读取该炮 HDF5 `meta/IsDisrupt` / `meta/DownTime`。固定 parent 阈值、J-TEXT 及时报警严格 `lead_ms > 10`；沿用 `score_curve` 的 9 位小数毫秒 lead 舍入。安全炮的 DownTime 不作为破裂时间。
- 保存 `protocol.json`、`code.zip`、逐炮曲线及封存、`per_shot.json`、按 HDF 结局分组的 `summary.json`、逐炮和汇总的 `paired.json`、`raw_hashes.json`、含产物 SHA256 的 `completed.json`。完成前复核基模、修复参数、输入和代码哈希。

CLI 的 CUDA 分支调用原静态探针使用的 `east_context_operator_20261031.run.configure()`；该函数要求 CUDA。为保持 CPU 独立可用，CPU CLI 单独设置相同的 2 线程、seed 131、关闭 MHA fastpath 和 TF32。直接调用 `run_source` 的程序负责自己的数值配置。

## CPU 测试及只读检查

初次先写测试，评价模块尚不存在时 pytest 收集阶段得到预期 RED：`1 error in 1.48s`。后续增加逐炮 DownTime 检查时先得到 `2 failed, 7 passed in 3.49s`，实现后转绿。CLI 配置测试也先因 `configure` 尚未接入而失败，再实现配置。

最终命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -m pytest -q -p no:cacheprovider tests/test_mask_repair_evaluation_20261104.py tests/test_source_mask_repair_20261104.py tests/test_mask_repair_training_20261104.py
```

实际输出：`51 passed in 4.23s`。测试覆盖 CPU 合成 HDF 端到端、两个 source 角色、错误角色与错误 fit 哈希拒绝、先封存后揭示、预测失败不揭示、source CIII sentinel 不适用、padding/完整观测 no-op、修复 RMS、同窗口错配及 batch 独立、未来信号修改不影响已到达前缀、严格 10 ms、CLI 配置、相关 core 和 TRAIN 合同。测试产生的 HDF 和输出均为临时合成 fixture。

另外只读加载真实 fit 和固定 source 权重，在 CPU 合成输入上验证：fit 通过哈希和合同检查，修复器参数数目 2292，基模状态摘要和固定阈值与 fit 协议一致；四臂最大修正 RMS 分别为 `0`、`0.0748802198765841`、`0.08975622143535447`、`0.08975622143535447`，padding token 修正均为零。这不是原始信号评价。

## 运行交接

主进程可分别对锁定 `val` 和 `test` 使用独立的新输出目录：

```bash
python -m experiments.risk_aligned_online.mask_repair_20261104.evaluation --role val --compute-device cuda --output probe_outputs/source_mask_repair_eval_val_20261104_v1
python -m experiments.risk_aligned_online.mask_repair_20261104.evaluation --role test --compute-device cuda --output probe_outputs/source_mask_repair_eval_test_20261104_v1
```

CLI 默认 `--role val`、`--compute-device cpu`、`--batch-size 256`；现有输出目录不会覆盖。上面 GPU 命令只是交接示例，本任务未执行。此数据此前已参与开发，评价不是独立未来验证；静态离线结果也不是 J-TEXT 在线或 EAST 收益。

交接冻结时 SHA256：

- `evaluation.py`: `e412b8e9b357ec50d6e59e042591b8d40554e6605387413f30abfbcef6100d0f`
- `test_mask_repair_evaluation_20261104.py`: `621c671ebdb4f48d5798f5671569e9ea02da4cd1ff71c9dacdb1ea2def9de18d`
