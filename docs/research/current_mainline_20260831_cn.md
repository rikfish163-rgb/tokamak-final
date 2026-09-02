# 当前工作树主线增量与结果索引

发布日期：2026-08-31（发布整理：2026-09-03）  
仓库状态：`TEST_DRIVEN_EXPLORATION / DIAGNOSTIC_ONLY`

本文是本仓库已有 paper-v3c archived-parent 实验包之外的当前工作树增量索引。
它收录 2026-08-31 版本的 Adv25 标签合同、J-TEXT 独立在线分支、EAST
`shot-first → endpoint` V3C 分支、代码审计和封存结果。两套谱系在结果目录和
checkpoint profile 上明确分开，不能把表中的数字拼接成一个未经运行的端到端结果。

## 1. 当前主线

```text
J-TEXT TRAIN1353（Adv25：H=max(25 ms, AdvancedTime)）
        ├── jtext_exact 静态源 checkpoint
        │       ├── J-TEXT TEST220 静态评估
        │       └── J-TEXT TEST220 炮序在线主动（独立分支）
        │
        └── east_success 静态源 checkpoint
                └── EAST support110 适配
                        ├── balanced360 静态封存
                        └── 从同一迁移父模型重置后的 V3C 在线主动
```

J-TEXT 在线状态、查询记录和专家权重不进入 EAST 迁移。J-TEXT 与 EAST
下游使用各自由验证集登记的 profile；它们遵循同一 Adv25 标签和模型合同，
但不是同一个 checkpoint 文件。

J-TEXT 破裂炮的弱 lead-time 窗口为

\[
H_i=\max(25\ \mathrm{ms},\;\mathrm{AdvancedTime}_i),
\]

缺少 `AdvancedTime` 时取 25 ms。炮级破裂结局和 `DownTime` 以 HDF5 的
`meta/IsDisrupt` 和 `meta/DownTime` 为准；JSON/文本 shot list 仅提供炮号、
顺序和差异审计信息。

## 2. 封存结果

J-TEXT 以 `lead > 10 ms` 计 timely TPR，EAST 以 `lead ≥ 30 ms` 计
operational TPR。以下结果来自同一目录中的静态/在线配对：

| 分支 | 静态 TPR / FPR | 在线主动 TPR / FPR | 查询数 | 解释边界 |
|---|---:|---:|---:|---|
| J-TEXT TEST220 | 0.836364 / 0.109091 | **0.854545 / 0.109091** | 44/220 | 延迟反馈专家权重更新；`representation_updated=false` |
| EAST seed 131 | 0.794444 / 0.100000 | **0.800000 / 0.094444** | 94/360 | 完整 V3C 系统诊断 |
| EAST seed 133 | 0.761111 / 0.116667 | **0.761111 / 0.111111** | 99/360 | TPR 不下降 |
| EAST seed 137 | 0.816667 / 0.144444 | **0.816667 / 0.138889** | 98/360 | TPR 不下降 |

EAST 三 seed 算术平均由 static `TPR@30=0.790741, FPR=0.120370`
变为 online `TPR@30=0.792593, FPR=0.114815`，对应
`ΔJ=+0.007407`，其中 `J=TPR@30−FPR`。J-TEXT 20% 预算的
`ΔJ=+0.018182`。

这些数字不能写成独立未来流确认：`TEST220` 和 `balanced360` 都参与过
方法开发或参数选择。EAST 的正变化是同时包含炮内/炮间更新、端点记忆、
episode gate 和告警保护的系统级开发诊断，不能单独归因于 Transformer
底层表示学习。J-TEXT 当前正增益明确属于查询后的专家权重更新，不是神经
表示已经被在线重学的证据。

## 3. 代码与审计入口

当前增量代码位于
`experiments/transfer_methods_2026/`，核心入口为：

- `run_jtext_adv25_active_mainline.py`：J-TEXT TEST220 炮序在线主动分支；
- `run_east_v3c_success_mainline.py`：EAST `shot-first → endpoint` V3C 诊断回放；
- `audit_current_mainline.py`：只读主线和封存回执审计；
- `build_mainline_receipt.py`：从既有 JSON 回执生成机器可读索引，不重新选阈值；
- `jtext_label_contract.py`、`jtext_query_policy.py`：标签与查询策略合同；
- `mainline_paths.py`：拒绝把最终产物写入 `/home`、`/dev` 或本机根文件系统。

同目录中的 `v3c_*_compat/`、`adv25_attribution_audit_*` 和根因审计文件是
可追溯诊断材料；它们不覆盖 `shot-first` 主线，也不应被当作新的独立确认结果。

## 4. 结果文件布局

当前工作树的紧凑证据收录在：

```text
results/current_mainline_20260831/
├── receipt/                         # 总表、审计、编译后的中文报告
├── sources/                         # jtext_exact 与 east_success 小型源 checkpoint
├── jtext_active/                    # canonical replay 与 10/20/30% 预算结果
├── east_v3c/                        # 三 seed 静态/在线/query/reveal/update 账本
└── diagnostics/                     # 明确标注的诊断 JSON
```

其中每个 EAST seed 保存 `mainline_manifest.json`、support 适配回执、静态预测、
在线逐炮预测、查询/揭示/更新账本和最终在线状态；另有 seed131 同 parent 的
炮内/炮间归因摘要。J-TEXT 保存 canonical replay、预算曲线和 EAST profile
一致性诊断。编译好的中文报告为
`receipt/jtext_east_academic_technical_report_cn_20260831.pdf` 和
`receipt/jtext_east_readable_technical_report_cn_20260831.pdf`，可编辑源文件在
`docs/research/`。

## 5. 可复现边界

公开仓库不包含原始 HDF5、NPZ/NPY 特征数组、风险轨迹缓存和历史大规模运行目录。
这些文件可能包含敏感或不可移植的数据，也会绕过项目的数据合同。复现时设置：

```bash
export FUSION_DATA_ROOT=/absolute/path/to/data
conda run -n rl_env python experiments/transfer_methods_2026/audit_current_mainline.py \
  --output results/generated/mainline_audit.json
```

GPU 运行统一使用 `rl_env` 和 CUDA PyTorch；最终运行账本应落在挂载盘，并在
manifest 中记录 Python、PyTorch、GPU、源 checkpoint SHA、炮表 SHA 和代码 SHA。
在线协议始终是：

```text
当前因果信号 → 预测并封存 → 不读结局地决定查询
→ 仅揭示被选炮标签 → 下一炮前更新
```

任何未查询炮的标签、未来窗口或当前炮结局进入查询、预测或更新前，都属于合同
违规，审计应失败关闭。

