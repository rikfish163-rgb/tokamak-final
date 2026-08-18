# Reproducibility protocol

## Locked pipeline

每个 seed 按以下顺序运行，任何阶段失败都停止，不使用后续阶段补救：

```text
1. A0/A1 predictor + chronological query audit
2. EAST-private dynamic hazard adapter
3. unified static360 → reset → online-active360
```

源域结果必须单独先跑，不能用迁移后的 EAST 指标代替：

```bash
for seed in 131 133 137; do
  conda run -n rl_env python \
    experiments/transfer_methods_2026/jtext_source_baseline.py \
    --data-root "$FUSION_DATA_ROOT" --seed "$seed" \
    --output "/path/on/ext4/jtext-source/seed${seed}"
done
```

该入口训练尚未接触 EAST110 的 rebuilt17 父模型，阈值只由 VAL160 确定，随后在
balanced TEST220 上按严格 `lead > 10 ms` 评估。参考摘要位于
`results/reference/jtext_source_multiseed_summary.json`。

入口是 `reproducibility/reproduce.py`，不是历史实验目录中的 shell 命令。编排器为每个 seed 写出三条实际 Python 命令、环境变量、数据审计和阶段日志。

## Fixed seeds and parameters

默认确认 seed 是 `131/133/137`，查询预算固定为 72 炮。锁定的在线参数来自 `LOCKED_BEST_V8`：

```text
private_support_epochs = 8
adaptation_events       = 64
intra_steps             = 1
inter_steps             = 2
intra_lr                = 1e-4
private_intra_lr        = 1e-4
intra_physics_weight    = 0.10
inter_lr                = 5e-4
private_inter_lr        = 2e-4
alarm_lr                = 0.02
```

`run_unified_online_active.py --locked-best-v8` 会在读数据或生成输出前校验这些参数、seed 和 360 炮上限；不匹配即失败关闭。

## Reference result

`results/reference/multiseed_summary.json` 是小型摘要，不是训练 checkpoint。它记录每个 seed 的 static/online confusion counts、FPR、总体 TPR、`TPR@30ms`、median lead、查询数和阶段隔离证明。复现器会把新摘要与它逐字段比较；若通过 `--seeds` 指定子集，只比较参考中的对应 seed 行，默认则比较 `131/133/137` 全部三行。

当前摘要的科学状态是 `ONLINE_DIAGNOSTIC`：360 炮曾参与方法开发，不能被描述为完全未见测试流。要生成论文最终结果，必须锁定代码和参数后另建一次性确认流，并保留独立数据/炮表哈希。

J-TEXT rebuilt17 源模型当前同样只是 `SOURCE_DIAGNOSTIC`：三 seed 均值为
`TPR@lead>10ms=0.8606、FPR=0.2061`，没有达到三个 seed 均需
`TPR≥0.90、FPR≤0.15` 的 S0 门。已提交的 EAST 在线摘要产生于本次源域切分与
采样修复之前，在重新跑完整链路前不得与新源域行拼接成同一次端到端结果。

数据可以位于 NTFS 或网络挂载；训练输出和 seal ledger 应放在支持稳定 fsync 的本地 ext4 目录。这样不会把外置盘的 `ntfs3` 写入阻塞误诊断为算法或 CUDA 失败。

## Determinism and hardware

参考运行使用 `rl_env`、PyTorch `2.11.0+cu128`、CUDA `12.8` 和单张 RTX 5070。GPU kernel、驱动和 BLAS 差异可能造成浮点末位变化；评估比较默认容忍度为 `1e-8`，而不是宣称跨硬件 bitwise 相同。每次运行应保存：

- Git commit、Python 和依赖版本；
- 数据审计 JSON 和炮表哈希；
- 每阶段命令及日志；
- 预测前模型哈希、冻结分支哈希和在线更新回执。

## What is intentionally absent

仓库不提交 HDF5、NPZ、PT、归档压缩包和历史运行目录。它们体积大、包含数据或只是过程产物；缺少合法 `data/` 时审计会失败，不能用空文件或已有摘要冒充复现。
