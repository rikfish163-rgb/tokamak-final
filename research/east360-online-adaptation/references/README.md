# Fuison: 托卡马克破裂预测

本仓库面向实时托卡马克等离子体破裂预测，核心设计对齐相关前沿研究（paper_131）：
- 物理引导特征管线 (PGFE)
- 灰区感知标注 (FLS 兼容)
- 校准后的破裂概率输出
- 包含传统机器学习基线(XGBoost)和深度时序模型(Transformer, Mamba, GRU)
- 面向不同装置（EAST, J-TEXT）的迁移学习与适配。

> **当前唯一保留的研究主线（2026-08-31）**：J-TEXT Adv25 静态源模型
> → J-TEXT TEST220 独立在线主动诊断；同时使用另一组预先登记的 EAST
> `east_success` 静态源 checkpoint → EAST support110 迁移 → balanced360 静态
> 封存 → 从同一迁移父模型重置后的 EAST V3C 炮内/炮间在线主动诊断。
> J-TEXT 在线状态不进入 EAST。当前 TEST220/balanced360 都参与过方法开发，
> 因而所有在线数字均标记为 `DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION`，
> 不是独立未来流确认。

主线细节、谱系和可核验结果见
[`experiments/transfer_methods_2026/MAINLINE_STATUS_20260831_CN.md`](../EXTERNAL_ASSETS.md)、
[`experiments/transfer_methods_2026/MAINLINE_RECEIPT_20260831_CN.md`](../EXTERNAL_ASSETS.md)。
面向汇报和代码审查的中文技术报告见
[`代码核对版 PDF`](../EXTERNAL_ASSETS.md)、
[`LaTeX 源稿`](../EXTERNAL_ASSETS.md) 与
[`逐项代码证据索引`](../EXTERNAL_ASSETS.md)。报告专门解释
`17+8` 特征拆分、`delta-s17`、support 损失、EAST 125 ms 时间集合、物理代理、
查询/梯度保护和报警状态机；未完成的消融会明确标为待验证，而不会写成既成结论。
机器审计入口为
[`experiments/transfer_methods_2026/audit_current_mainline.py`](../EXTERNAL_ASSETS.md)，
审计产物固定写入挂载盘
`/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/`。

## 当前主线的已封存诊断结果

J-TEXT 使用 `lead > 10 ms`；EAST 使用 `lead ≥ 30 ms`。数值来自同一目录中的
静态/在线配对，不跨谱系拼接：

| 分支 | 静态 | 在线主动 | 查询/说明 |
|---|---:|---:|---|
| J-TEXT TEST220 | TPR 0.836364 / FPR 0.109091 | TPR **0.854545** / FPR 0.109091 | 44/220；延迟反馈专家权重更新 |
| EAST seed131 | TPR 0.794444 / FPR 0.100000 | TPR **0.800000** / FPR 0.094444 | 94/360；V3C 系统诊断 |
| EAST seed133 | TPR 0.761111 / FPR 0.116667 | TPR **0.761111** / FPR 0.111111 | 99/360；TPR 不下降 |
| EAST seed137 | TPR 0.816667 / FPR 0.144444 | TPR **0.816667** / FPR 0.138889 | 98/360；TPR 不下降 |

J-TEXT 当前在线增益来自查询后的专家权重更新，`representation_updated=false`；
EAST 的 `ΔJ=+0.007407` 是同时包含炮内/炮间更新、endpoint memory、episode gate
和告警保护的完整 V3C 开发诊断，尚不是干净的单模块神经增益。最新的
seed131 同 parent 归因筛查进一步显示：禁用炮间神经更新后，决策/记忆侧仍能
减少一个 FP，而固定计算的纯神经炮间更新没有改变炮级边界；详见
`experiments/transfer_methods_2026/MAINLINE_STATUS_20260831_CN.md` 和
`fuison_ext4_runs/east_v3c_timescale_ablation_20260831/seed131/aggregate.json`。
EAST support
历史 JSON 角色表虽声明 `10D+100N`，但 HDF5 `meta/IsDisrupt` 权威标签实际为
`15D+95N`；入口和审计器会显式记录这一差异。

详细的两层因果协议是：

```text
炮内：当前因果信号前缀 → 端点预测/封存 → 无标签 fast-state 更新 → 下一端点
炮间：整炮预测/封存 → 不读结局的查询 → 仅被选炮 HDF5 揭示 → 下一炮前持久更新
```

正式论文或部署结论前，必须补齐固定 parent、阈值、报警器和 query manifest 下的
正交消融，并在未参与方法开发的 EAST future stream 上一次性确认。

当前主线的统一结果回执由
`experiments/transfer_methods_2026/build_mainline_receipt.py` 生成：

```bash
/home/hetaisheng/miniconda3/envs/rl_env/bin/python \
  experiments/transfer_methods_2026/build_mainline_receipt.py
```

它只读取已封存的 JSON 结果，不重新计算或重新选择指标；生成的机器可读总表和
中文结果表位于挂载盘 `fuison_ext4_runs/mainline_receipt_20260831/`。

J-TEXT 的“真正神经在线更新”另有独立诊断报告
[`JTEXT_EPISODIC_NEURAL_DIAGNOSTIC_20260831_CN.md`](../EXTERNAL_ASSETS.md)：
原型局部适配器在严格因果条件下被验证为安全但无稳定增益；放宽到单 episode
路由会产生负迁移。因此当前可复现的 J-TEXT 正增益仍只属于决策层专家权重更新，
EAST V3C 的系统级增益也继续与该诊断分开归因。

## 目录结构

```text
/
├── src/
│   ├── data/
│   │   ├── build_dataset.py       # 数据集构建脚本
│   │   └── read_hdf5_structure.py # HDF5数据读取工具
│   ├── models/
│   │   ├── baseline/
│   │   │   └── train_xgb.py       # XGBoost 等传统简单模型训练脚本
│   │   └── advanced/
│   │       ├── train_sequence.py  # 包含 Transformer, MambaLite, GRU 等前沿时序模型训练脚本
│   │       └── sequence_arch.py   # 深度学习网络架构定义
│   └── evaluation/
│       ├── eval.py                # 评估指标脚本
│       ├── calibrate.py           # 概率校准脚本
│       └── generate_reason_report.py # 生成破裂原因报告
├── docs/
│   ├── architecture.md            # 系统架构文档
│   ├── literature/                # 参考文献与前沿调研
│   │   └── paper_131.txt
│   ├── notes/                     # 项目会议、个人随记、图片整理
│   └── research/                  # 方法调研与框架分析
├── data/                          # 数据集存放处（软链接或本地挂载）
├── shot_list/                     # 各装置的炮号列表 (EAST, J-TEXT 等)
├── splits/                        # 数据切分配置文件
├── artifacts/                     # 训练产生的模型文件与数据集特征字典
└── reports/                       # 生成的指标报告与图表
```

## 数据路径配置

本仓库推荐的数据根目录：

```powershell
$env:FUISON_DATA_ROOT = "G:\我的云端硬盘\Fuison\data"
if (-not (Test-Path $env:FUISON_DATA_ROOT)) { throw "FUISON_DATA_ROOT not found: $env:FUISON_DATA_ROOT" }
```

如果 HDF5 仅存放在云端根目录，请创建一次链接：

```powershell
if (-not (Test-Path "data/EAST/unified_hdf5")) {
  cmd /c mklink /J data\EAST\unified_hdf5 "$env:FUISON_DATA_ROOT\EAST\unified_hdf5"
}
```

## 环境搭建

推荐使用 Python 3.10+ 创建虚拟环境，并安装相关依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas h5py scikit-learn torch matplotlib joblib xgboost shap
```

## 运行说明

### 1) 构建数据集
```powershell
python src/data/build_dataset.py --target-dir artifacts/datasets/jtext_v1
```

### 2) 训练 Baseline (XGBoost)
```powershell
python src/models/baseline/train_xgb.py --dataset-dir artifacts/datasets/jtext_v1 --output-dir artifacts/models/best
```

### 3) 训练 Advanced Models (Mamba / Transformer / GRU)
```powershell
python src/models/advanced/train_sequence.py --dataset-dir artifacts/datasets/jtext_v1 --output-dir artifacts/models/iters/adv_mamba_lite --models mamba_lite
```

### 4) 生成评估和归因报告
```powershell
python src/evaluation/generate_reason_report.py --run-dir artifacts/models/iters/adv_mamba_lite --dataset-dir artifacts/datasets/jtext_v1
```
