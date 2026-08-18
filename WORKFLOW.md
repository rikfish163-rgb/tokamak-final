# J-TEXT → EAST reproduction workflow

这份文件是仓库发布后的唯一操作入口。研究阶段的旧计划和历史失败方法不属于当前复现主线。

## Mainline

```text
data audit → A0/A1 chronological query audit
           → EAST-private dynamic hazard adapter
           → static 360 seal
           → reset to migrated parent
           → chronological online active 360
           → aggregate and verify
```

这里的 360 炮是当前固定的方法开发流。它用于复现当前最佳方法和在线更新收益，不是因“复现”而自动变成未见最终测试流。

## Immutable rules

- HDF5 `meta/IsDisrupt`、`meta/DownTime` 是结局标签权威。
- 训练/适配 support 炮按固定清单读取；EAST 360 炮只按 `config/splits/east_hdf5_authoritative_balanced360_v1.json` 读取。
- 报警按每炮最高风险事件评估；总体 TPR 纳入提前量大于 30 ms 的报警，并单列 `TPR@30ms`。
- 静态 360 必须先全部预测并封存；在线流从同一迁移父模型重置。
- 当前炮结局只在该炮 seal 和查询决定后揭示；揭示后的信息只更新后续炮。
- 运行输出只能写入 `results/generated/` 或用户指定的空目录。
- 缺少数据、CUDA、固定炮表或参考合同时非零退出，不生成假结果。

## Commands

```bash
uv sync --frozen --extra dev
uv run fusion-reproduce audit --data-root /path/to/data --full
uv run fusion-reproduce smoke --data-root /path/to/data
uv run fusion-reproduce reproduce --data-root /path/to/data \
  --output results/generated/locked-best-v8 --seeds 131 133 137
uv run fusion-reproduce verify \
  --actual results/generated/locked-best-v8/multiseed_summary.json
```

若要扩展方法，必须新建实验分支和新结果状态，不覆盖 `results/reference/`，也不能把在 360 上调出来的设置直接命名为最终部署结果。
