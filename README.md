# J-TEXT → EAST transfer and online adaptation

这是论文配套的可复现实验仓库。仓库只提交源码、固定配置、标签合同和小型参考结果；原始 HDF5 数据不提交到 GitHub。准备好同样的 `data/` 后，可以从零运行当前锁定的 J-TEXT backbone → EAST 迁移 → EAST 360 炮静态评估 → 按炮号顺序在线主动学习链路。

## 当前锁定结果

参考结果来自 `balanced360` 方法开发流，状态是 `ONLINE_DIAGNOSTIC`，不是未见未来流的部署确认。报警口径为每炮最大风险报警，`>30 ms` 的报警计入总体 TPR；同时保存 `TPR@30ms` 供论文报告区分口径。

| seed | static TPR | static FPR | online TPR | online FPR | online TPR@30ms |
|---:|---:|---:|---:|---:|---:|
| 131 | 0.9333 | 0.1444 | 0.9389 | 0.1444 | 0.8222 |
| 133 | 0.9111 | 0.1278 | 0.9111 | 0.1278 | 0.7722 |
| 137 | 0.9333 | 0.1389 | 0.9333 | 0.1389 | 0.8222 |

这三行是已提交的参考摘要，不代表把 360 炮当成独立最终测试集。`360` 既参与过方法确定，也按“先完成静态封存、再从同一迁移父模型重置并做时间顺序在线流”的合同运行。

## 1. 准备环境

推荐 Linux、Python 3.11、CUDA 12.8、RTX 5070，并使用项目约定的 `rl_env`。新环境可尝试：

```bash
uv sync --frozen --extra dev
```

锁定文件使用 PyTorch CUDA 12.8 wheel。若所在网络无法访问 PyTorch wheel 源，先在已有的 `rl_env` 中验证：

```bash
conda run -n rl_env python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

## 2. 放置数据

默认数据根目录是仓库下的 `data/`。也可以把现有数据目录通过参数或环境变量传入：

```bash
export FUSION_DATA_ROOT=/path/to/data
```

最小目录合同如下；详细 HDF5 字段和炮表见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。

```text
data/
├── J-TEXT/unified_hdf5/<shot//100*100>/<shot>.hdf5
└── EAST/unified_hdf5/<shot//100*100>/<shot>.hdf5
```

标签唯一以 HDF5 的 `meta/IsDisrupt` 和 `meta/DownTime` 为准；CSV/JSON 只用于固定炮表、AdvancedTime 研究记录或审计，不覆盖 HDF5 结局标签。

## 3. 复现命令

先执行不产生训练输出的数据审计：

```bash
uv run fusion-reproduce audit --data-root "$FUSION_DATA_ROOT" --full \
  --output results/generated/data_audit.json
uv run fusion-reproduce smoke --data-root "$FUSION_DATA_ROOT"
```

运行锁定的三个阶段。输出目录必须为空；程序会保存命令、环境变量、数据审计和每一阶段的日志：

```bash
uv run fusion-reproduce reproduce \
  --data-root "$FUSION_DATA_ROOT" \
  --output results/generated/locked-best-v8 \
  --seeds 131 133 137
```

只检查命令连接和炮表时可用 dry-run：

```bash
uv run fusion-reproduce reproduce \
  --data-root "$FUSION_DATA_ROOT" \
  --output results/generated/dry-run \
  --seeds 131 --dry-run
```

复现完成后，程序自动对比提交的参考摘要；也可以单独核验：

```bash
uv run fusion-reproduce verify \
  --actual results/generated/locked-best-v8/multiseed_summary.json
```

`--seeds` 可以指定确认子集（例如先用 `--seeds 131` 做 screen）；编排器只与
提交参考中的对应 seed 行比较。默认不指定时运行并核验 `131/133/137` 全部三条。

完整运行需要 GPU 和较大的临时磁盘空间，时间取决于 HDF5 读取速度。CI 只运行无数据、CPU 可完成的合同测试，不伪造 GPU 结果。

如果仓库位于 NTFS 外置盘，建议只把数据作为只读输入，使用 ext4（例如本机磁盘或 `/dev/shm`）作为 `--output` 目录。在线 seal ledger 为了断点安全会逐条 flush/fsync；部分 Linux `ntfs3` 挂载在这种写入模式下会长时间阻塞，这属于文件系统限制，不应被误判为模型训练失败。

## 4. 仓库结构

```text
reproducibility/                    # 数据审计、阶段编排、参考结果核验
experiments/transfer_methods_2026/  # 当前最佳 22 个运行模块
config/splits/                      # EAST/J-TEXT 固定炮表与切分合同
splits/                             # J-TEXT 炮号切分
shot_list/                          # 可读的炮号与标签辅助清单
results/reference/                  # 小型、可审计的参考摘要
docs/                               # 方法、标签和论文口径说明
tests/reproducibility/              # 仓库复现合同测试
```

旧的过程实验、失败方法、缓存模型、归档压缩包和数据特征缓存不属于发布仓库。它们不会被当前仓库的复现入口读取，也不应通过 Git LFS 或其他方式隐式加入结果。

## 5. 科学边界

本仓库复现的是当前最佳方法开发链路和其锁定参数。在线阶段遵循 `predict → seal → reveal → update → next shot`；当前炮的结局不能进入该炮预测。由于 360 炮参与了方法开发，它只能作为可复现的诊断/方法流结果。任何独立未来流、正式部署或论文最终泛化结论，都必须在参数、阈值和代码锁定后另行封存确认。

## Citation

仓库的引用信息见 [CITATION.cff](CITATION.cff)。发布前请补充作者、机构、版本 DOI 和许可证；当前副本不替用户选择法律许可证。
