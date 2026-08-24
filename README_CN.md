# J-TEXT → EAST：托卡马克破裂预警的因果迁移与时序在线主动适配

**默认语言：简体中文** · [English](README_EN.md) · [详细中文说明](README_CN.md)

[paper-v3c 主线技术报告](experiments/transfer_methods_2026/paper_grade_package/TECHNICAL_REPORT_zh.md) · [AdvancedTime 动态标签重跑附录](experiments/transfer_methods_2026/paper_grade_package/DYNAMIC_MAINLINE_TECHNICAL_REPORT_zh.md) · [实现细节](IMPLEMENTATION_DETAILS_CN.md) · [数据合同](DATA_CONTRACT.md)

![Python](https://img.shields.io/badge/Python-3.11-3776AB) ![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C) ![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900) ![HDF5](https://img.shields.io/badge/HDF5-authoritative%20labels-00599C)
![model](https://img.shields.io/badge/model-causal%20Transformer-0072B2) ![online](https://img.shields.io/badge/online-chronological%20active%20learning-009E73) ![reproducibility](https://img.shields.io/badge/reproducibility-hash--bound-2F80ED)

![J-TEXT 源模型、EAST 小样本迁移与在线主动架构](docs/figures/jtext_east_mainline_architecture_v2.png)

![evidence](https://img.shields.io/badge/evidence-TEST__DRIVEN__EXPLORATION-E69F00?style=flat-square)
![confirmation](https://img.shields.io/badge/independent%20confirmation-pending-D55E00?style=flat-square)
![labels](https://img.shields.io/badge/labels-HDF5%20authoritative-0072B2?style=flat-square)
![runtime](https://img.shields.io/badge/runtime-rl__env%20%2B%20CUDA-009E73?style=flat-square)

本仓库实现一条可审计的托卡马克破裂预警研究链：

```text
J-TEXT TRAIN1353 源模型
→ EAST support110 小样本迁移
→ EAST balanced360 静态封存
→ 重置到同一迁移后父模型
→ balanced360 按炮号顺序在线主动适配
```

J-TEXT TEST220 还运行一条独立在线诊断支路；该支路的在线参数**禁止**进入 EAST 迁移。所有整炮结局与破裂时刻只由 HDF5 `/meta/IsDisrupt` 和 `/meta/DownTime` 定义。

> 证据边界：当前状态为 `TEST_DRIVEN_EXPLORATION` / `DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION`。TEST220 与 balanced360 均参与方法开发，不是独立未来确认集，也不是装置部署成绩。

## 论文主结果：原 paper-v3c 谱系

以下结果来自封存的 `paper_grade_package/evidence/three_stage_results.json`。三个阶段使用原 paper-v3c checkpoint/标签/报警谱系；TPR 是满足提前量规则的 operational TPR，J-TEXT 要求 lead $>10$ ms，EAST 要求 lead $\geq30$ ms。

| 阶段 | TPR | FPR | AUC | $J=\mathrm{TPR}-\mathrm{FPR}$ |
|---|---:|---:|---:|---:|
| J-TEXT 原源模型 TEST220 | **0.851515** | **0.212121** | **0.921928** | **0.639394** |
| EAST 静态迁移 balanced360 | **0.840741** | **0.166667** | — | **0.674074** |
| EAST 原 v3c 在线 balanced360 | **0.848148** | **0.151852** | — | **0.696296** |

原 v3c 在线相对静态迁移：TPR@30 增加 `0.007407`，FPR 减少 `0.014815`，因此 $\Delta J=+0.022222$。三个 seed 合计多 4 个及时检出并少 8 个误报；查询数为 **102/90/99**。这是 balanced360 上的方法开发证据，不是独立未来确认。

| seed | 静态 TPR@30/FPR | 在线 TPR@30/FPR | 查询 |
|---:|---:|---:|---:|
| 131 | 0.822222 / 0.155556 | 0.844444 / 0.133333 | 102 |
| 133 | 0.838889 / 0.144444 | 0.838889 / 0.133333 | 90 |
| 137 | 0.861111 / 0.200000 | 0.861111 / 0.188889 | 99 |

## 查询预算曲线（seed 131）

预算曲线来自完整 balanced360 因果回放；不同预算版本只用于开发性比较。

| 查询数 | 查询率 | TPR@30 | FPR | $J$ | 相对上一预算 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.8222 | 0.1556 | 0.6667 | — |
| 16 | 4.44% | 0.8278 | 0.1556 | 0.6722 | +0.0056 |
| 35 | 9.72% | 0.8278 | 0.1556 | 0.6722 | 0 |
| 66 | 18.33% | 0.8444 | 0.1500 | 0.6944 | +0.0222 |
| 97 | 26.94% | 0.8389 | 0.1556 | 0.6833 | −0.0111 |

结果不是预算越大越好：本条流上 66 炮优于 97 炮。

## 30 炮查询预算与直接迁移对照

Q30 是按时间顺序、只用当时可见特征因果选出的 30 炮；选择完成后冻结炮号。为了避免直接迁移模型在自己训练过的 30 炮上评估，四个方法只比较同一剩余 330 炮：

| 方法 | TPR@30 | FPR | $J$ |
|---|---:|---:|---:|
| 原始静态模型 | **0.830189** | **0.140351** | **0.689838** |
| 在线炮内+炮间 | **0.836478** | **0.134503** | **0.701975** |
| 30炮直接迁移，原阈值 | 0.937107 | 0.274854 | 0.662253 |
| 30炮直接迁移，support110重校准 | 0.930818 | 0.263158 | 0.667660 |

在线方案相对最佳直接迁移的 $J$ 高 `0.034315`。该对照匹配了标签访问量，但 optimizer steps、replay 与 FLOPs 尚未完全匹配，因此只能支持“相同30个标签直接前置训练并不自动更优”，不能声称严格等算力胜出。

## 新增 J-TEXT 在线支路与动态标签附录

J-TEXT 在线主动是后来新增的独立诊断支路；其任何在线状态都不得进入 EAST。AdvancedTime 动态标签源在重新注册报警状态机初值后得到 `0.820370/0.157407 → 0.824074/0.146296`，属于另一 checkpoint/标签谱系，只保留在附录中，**不得覆盖原 paper-v3c 主结果**。

## AdvancedTime 附录标签合同（非 paper-v3c）

J-TEXT 破裂炮正样本上界为

$$
H_s^J=\max(25\ \mathrm{ms},\mathrm{AdvancedTime}_s),
$$

AdvancedTime 缺失时取 25 ms。EAST support110 仍使用固定 125 ms 上界。AdvancedTime 只改变窗口上界，不覆盖 HDF5 结局或 DownTime。当前臂名称为 `source_dynamic__east_fixed`。

在保持源 checkpoint、EAST support110 parent、balanced360 炮序、v3c 查询器和模型更新范围不变时，仅在预注册投影范围内筛选报警状态机初值。选定值为 `high=-0.1, physics=6.0, trend=13.5, early=7.5`。完整真实在线回放结果为：

| seed | 同参数静态 TPR@30/FPR | 在线 TPR@30/FPR | $\Delta J$ | 查询 |
|---:|---:|---:|---:|---:|
| 131 | 0.861111 / 0.155556 | 0.861111 / 0.133333 | +0.022222 | 101 |
| 133 | 0.761111 / 0.127778 | 0.766667 / 0.116667 | +0.016667 | 100 |
| 137 | 0.838889 / 0.188889 | 0.844444 / 0.188889 | +0.005556 | 95 |
| 等 seed 均值 | **0.820370 / 0.157407** | **0.824074 / 0.146296** | **+0.014815** | 98.67 |

三 seed 合计查询 296 次，逐炮转移为 9 次挽救、1 次伤害，净挽救 8 炮。机器证据见 [`dynamic_source_online_gain/report.json`](experiments/transfer_methods_2026/paper_grade_package/evidence_dynamic/dynamic_source_online_gain/report.json)。balanced360 同时用于参数筛选和回放，故这只支持“该开发流上存在在线增益”，不构成独立未来确认。

## 方法概览

- **源模型**：$128\times17$ 因果窗；patch 8/stride 8；两层 4 头 Transformer，$d_{\text{model}}=64$。
- **跨装置输入**：shared17 使用 J-TEXT TRAIN1353 的 median/IQR scaler；EAST 不重拟合共享 scaler。
- **EAST 私有输入**：8 个不适合强制同义对齐的通道形成 48 维因果摘要，进入有界目标残差支路。
- **小样本迁移**：冻结低层，只适配最后一层 Transformer、LayerNorm、独立 EAST head、共享动态与私有残差。
- **炮内更新**：每炮创建 fast 副本，只使用当前及过去端点做无标签适配；炮后丢弃。
- **炮间更新**：整炮预测和查询先封存；只有被选择炮才揭示标签并更新 persistent，失败时事务回滚。
- **报警**：任一因果端点满足冻结状态机即可，不要求连续 10 点；过早报警另行审计。

原 v3c 的损失、查询价值、报警状态机和证据边界见[主线技术报告](experiments/transfer_methods_2026/paper_grade_package/TECHNICAL_REPORT_zh.md)；动态标签重跑的公式—代码映射见[附录报告](experiments/transfer_methods_2026/paper_grade_package/DYNAMIC_MAINLINE_TECHNICAL_REPORT_zh.md)。

## 快速复现

### 1. 环境和数据

统一使用带 CUDA 的 `rl_env`：

```bash
export FUSION_DATA_ROOT=/absolute/path/to/data
PY=${CONDA_PREFIX}/bin/python

$PY -c 'import torch,h5py; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

公共 CPU CI 只安装合同审计依赖，不替代 GPU 训练或在线回放：

```bash
uv sync --frozen --only-group ci --no-install-project
```

GPU 开发环境保持在当前 Conda 环境中：

```bash
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --frozen --extra dev
export FUSION_PROJECT_ROOT="$PWD"
```

`fusion-reproduce` 是完整 checkout 的 editable CLI；本项目并不发布一个可脱离仓库独立运行的 wheel。它仍需要 `FUSION_PROJECT_ROOT` 指向的仓库配置、炮表和脚本。

数据目录：

```text
data/
├── J-TEXT/unified_hdf5/<shot//100*100>/<shot>.hdf5
└── EAST/unified_hdf5/<shot//100*100>/<shot>.hdf5
```

原始 HDF5 不提交到 Git。高频运行输出建议写入 ext4；不要把大量逐端点账本写到存在稳定性问题的 NTFS 路径。

### 2. AdvancedTime 附录三 seed 重跑

```bash
for seed in 131 133 137; do
  $PY experiments/transfer_methods_2026/run_selected_full_mainline.py \
    --seed "$seed" \
    --stage all \
    --data-root "$FUSION_DATA_ROOT" \
    --output "/ext4/runs/selected_full_mainline/seed${seed}"
done
```

固定阶段顺序为 `source-train → jtext-static → jtext-online → east-parent → east-static → east-online`。`jtext-online` 是独立诊断分支；`east-parent` 仍只读取同 seed 的**原始静态 J-TEXT checkpoint**，禁止读取 J-TEXT 在线状态。

动态标签源的报警初值筛选与三 seed 真在线回放复用上述封存 source/parent；每个 seed 使用独立 CUDA 进程：

```bash
$PY experiments/transfer_methods_2026/run_dynamic_source_online_alarm_search.py \
  --seeds 131 --exec-single \
  --parent-root /ext4/runs/corrected_mainline_screen \
  --output-root /ext4/runs/dynamic_source_alarm_v1

$PY experiments/transfer_methods_2026/paper_grade_package/scripts/validate_dynamic_source_online_gain.py \
  --report /ext4/runs/dynamic_source_alarm_v1/report.json \
  --run-root /ext4/runs/dynamic_source_alarm_v1 \
  --data-root "$FUSION_DATA_ROOT"
```

对 133/137 重复第一条命令，再用 `--reuse-complete` 聚合三 seed。不要在同一 Python 父进程中连续托管多个 CUDA replay。

### 3. J-TEXT 在线诊断

```bash
for seed in 131 133 137; do
  $PY experiments/transfer_methods_2026/run_jtext_online_active.py \
    --seed "$seed" \
    --arm full_active \
    --budget 20 \
    --data-root "$FUSION_DATA_ROOT" \
    --source-checkpoint "/ext4/runs/selected_full_mainline/seed${seed}/source-train/source_checkpoint.pt" \
    --output "/ext4/runs/jtext_online_v2/seed${seed}/full_active_b20"
done
```

### 4. 报告、测试与最终证据核验

```bash
$PY experiments/transfer_methods_2026/paper_grade_package/scripts/build_dynamic_report.py --check-only
$PY experiments/transfer_methods_2026/paper_grade_package/scripts/build_dynamic_report.py
$PY -m pytest -q tests/experiments/test_dynamic_mainline_report.py
```

附录报告构建器会拒绝把 paper-v3c 数字误写成动态标签同谱系结果；根 README 则由回归测试强制保留 paper-v3c 主结果、预算曲线和 Q30 对照。文档数值不得代替机器证据。

## 数据访问顺序

在线每炮必须遵守：

```text
predict
→ endpoint/shot seal
→ query decision seal
→ reveal selected outcome only
→ update or rollback
→ next shot
```

当前炮的 `IsDisrupt`、`DownTime`、AdvancedTime 和未来端点在当前炮预测时不可达。未查询炮的结局不能进入参数更新或表型记忆。

## 仓库主线文件

```text
experiments/transfer_methods_2026/
├── train_jtext_source_transformer.py
├── run_selected_full_mainline.py
├── run_unified_online_active.py
├── jtext_online_adapter.py
├── run_jtext_online_active.py
└── paper_grade_package/
    ├── DYNAMIC_MAINLINE_TECHNICAL_REPORT_zh.md
    ├── technical_report_dynamic/main.tex
    ├── configs/primary_chain_dynamic_source_v1.json
    ├── configs/dynamic_source_online_alarm_search_v1.json
    ├── configs/jtext_online_active_v1.json
    ├── evidence_dynamic/dynamic_source_online_gain/report.json
    ├── scripts/validate_dynamic_source_online_gain.py
    └── scripts/build_dynamic_report.py
```

## 历史结果隔离

paper-v3c 是本仓库论文主结果；AdvancedTime 动态标签与 J-TEXT 在线是独立附录。任何表格都必须显式标注 checkpoint、标签、报警口径和数据角色，禁止跨谱系拼接均值。

## 许可证与引用

许可证见 [LICENSE](LICENSE)，引用元数据见 [CITATION.cff](CITATION.cff)。提交论文或向技术部门交付前，仍须补充独立未来流确认；paper-v3c 与动态标签附录的机器证据和哈希分别固化，互不覆盖。
