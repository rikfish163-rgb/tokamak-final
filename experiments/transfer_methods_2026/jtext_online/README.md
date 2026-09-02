# J-TEXT Adv25 在线主动分支

这是当前仓库中与 EAST V3C 归档主线严格隔离的 J-TEXT 扩展。它使用

\[
H_i = \max(25\ \mathrm{ms},\; \mathrm{AdvancedTime}_i)
\]

作为 J-TEXT 破裂炮的正样本时间窗；如果炮表没有该炮的 AdvancedTime，则退回
25 ms。破裂结局和 DownTime 始终以 HDF5 `meta/IsDisrupt`、`meta/DownTime` 为准。
JSON 只提供正样本窗口边界，不能覆盖 HDF5 结局。

## 与 EAST 主线的关系

| 分支 | 源标签 | 在线变化 | 是否可作为 EAST 父模型 |
|---|---|---|---|
| `paper-v3c-archived-parent-replay` | 固定发布合同 | EAST 双时间尺度 V3C | 是，按归档合同 |
| `jtext-adv25-online` | `max(25 ms, AdvancedTime)` | 冻结 J-TEXT 专家之间的延迟反馈权重 | 否 |

J-TEXT 在线状态不会写入 EAST 迁移。EAST 迁移若使用 Adv25 源模型，只能读取
`paper_grade_package/evidence/jtext_adv25_source_lineage/` 中的静态 checkpoint。
这条 J-TEXT 在线结果是 `TEST_DRIVEN_EXPLORATION`，因为 TEST220 已参与过方法开发，
不能表述成独立未来确认。

## 运行

原始 HDF5 不随仓库发布。使用项目 CUDA 环境 `rl_env`，并把数据根目录放在仓库
外部：

```bash
conda activate rl_env
python -m experiments.transfer_methods_2026.jtext_online.runner \
  --data-root /absolute/path/to/data \
  --output /absolute/path/to/empty/jtext-adv25-budget20 \
  --budget 20 --device cuda
```

预算按完整的 20 炮块执行：10%、20%、30% 分别是每块 2、4、6 炮，TEST220
总查询数分别为 22、44、66。当前炮块先完成全部信号预测并写入 SHA-256 seal，
随后只对选中的炮揭示 HDF5 结局，再更新下一炮块使用的专家权重。

输出包括：

- `summary.json`：静态/在线指标、预算、查询数和块级权重轨迹；
- `predictions.json`：每炮静态与在线预测、首次报警和提前量；
- `query_ledger.jsonl`：查询价值、选择原因和预测 seal；
- `mainline_manifest.json`：标签、炮表、checkpoint、阈值和运行环境哈希。

`--query-policy random` 是同预算因果随机对照；它使用相同块大小、相同输出格式
和相同标签揭示边界。默认 `active` 只是已登记的开发策略，不代表它在未见未来流
上已经证明优于随机。

## 代码边界

`io.py` 中 `load_signal_only` 只打开 `data/*` 和时间属性，`OrderedOutcomeOracle`
把 HDF5 `meta/*` 读取放在 `finish_prediction_block` 和显式 query 之后。每个端点
传给 Transformer 的窗口只包含该端点之前的 128 个采样点；MHD 通道的可用性也用
前缀累计，而不是用整炮非零检查。

`controller.py` 的三个专家均为冻结 Transformer，在线只更新三元凸组合的
log-weight：

\[
\log w_{k}^{(b+1)} = \log w_k^{(b)} - \eta\,\ell_k^{(b)},
\qquad
\pi_k = \frac{\exp(\log w_k)}{\sum_j\exp(\log w_j)} .
\]

因此这条分支可以证明“查询后发生了因果决策适应”，但不能声称在线过程改变了
Transformer 底层表示或重新学习了新的物理规律。需要神经参数更新的实验应使用
仓库中单独的 EAST V3C/双时间尺度入口，并在消融中与本分支区分。
