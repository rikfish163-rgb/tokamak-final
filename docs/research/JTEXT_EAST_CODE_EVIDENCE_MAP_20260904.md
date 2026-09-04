# J-TEXT → EAST 技术报告代码证据索引（2026-09-04）

本索引用于约束 `jtext_east_academic_technical_report_cn_20260831.tex` 中的实现性陈述。它区分：

- **实现事实**：可直接从当前主线代码或测试读取；
- **历史实验事实**：来自封存回执，协议变化后不能外推；
- **设计动机**：有物理或文献依据，但不等价于本仓消融证明；
- **待验证主张**：当前不能写成已证实结论。

## 1. raw23、rebuilt17、raw8 与“25 维”

| 结论 | 代码/结果证据 | 边界 |
|---|---|---|
| 原始输入为固定 raw23 | `experiments/transfer_methods_2026/rebuilt_harness.py:35-39` | 23 个 HDF5 聚合列，不是 25 个传感器。 |
| rebuilt17 的名称和顺序 | `rebuilt_harness.py:41-45` | 15 个原始语义经重构 + `Bt_over_Ip`、`Ip_diff` 两个新增派生量。 |
| 单位、比值、差分、掩码 | `rebuilt_harness.py:66-100` | J-TEXT 电流恰好一次 kA→A；除零和非有限值失败关闭。 |
| 在线可用性只看过去 | `rebuilt_harness.py:103-115` 及后续函数体 | 防止未来非零 MHD 样本反向改变早期可用性。 |
| raw8 私有列 | `experiments/transfer_methods_2026/v3c_success_compat/run_unified_online_active.py` 中 raw8 loader/sidecar；`FEATURE_REBUILD_PLAN.md:11-38` | raw8 是目标私有使用，不是物理无效。 |
| 特征组历史 AUC | `REBUILT_RESULTS.md:12-34` | 支持 rebuilt17 作为当前工程入口；不证明两个派生量逐一不可删，也不证明 J-TEXT 正迁移。 |

必须保留的论文边界：`17 + 8 = 25` 是当前双分支的逻辑槽数；它不是 Zheng et al. 论文 47 路诊断/FFE 输入的逐列复刻，也不是完整 PGFE25 的等价重建。当前没有 `no_Bt_over_Ip`、`no_Ip_diff` 两项单因素消融。

## 2. support110 静态迁移谱系

| 阶段 | 实际更新对象 | 证据 |
|---|---|---|
| 载入 J-TEXT Adv25 静态 checkpoint | J-TEXT encoder/head | `v3c_success_compat/grid_search.py:350-371` |
| adapt-high | 解冻最后一个 Transformer encoder layer、最终 norm；新建 EAST head | `grid_search.py:395-413` |
| 在线起点封存 | 载入 support parent 后冻结 encoder/head | `v3c_success_compat/run_unified_online_active.py:2620-2703` |
| shared17 动态残差 | 目标 projector、next head、residual head | `v3c_success_compat/dynamic_hazard_adapter.py:72-134` |
| raw8 私有残差 | EAST-private dynamics residual | `v3c_success_compat/east_private23_sidecar.py:19-90`; `east_private_continual.py` |

因此报告中的 `s_base` 是 **support110 适配后的 EAST 基模型 logit**，不是未经适配的 J-TEXT 源风险分数。J-TEXT 在线分支从不进入 EAST parent。

## 3. delta-s17 与 raw8 标量维度

- `v3c_success_compat/run_dynamic_hazard_adapter.py:31-33`：共享冻结特征维度为 `64*3=192`，物理家族数为 4。
- `run_dynamic_hazard_adapter.py:59-84`：`CLS(64) + last(64) + current_last-previous_last(64)`。
- `dynamic_hazard_adapter.py:90-131`：输入 `192 + physics4 + availability4 = 200`，投影到 16；与一个 base logit 共同产生一个标量 residual；`tanh` 上界为 ±1；至少两类物理家族可用才打开门。
- `v3c_success_compat/east_private23_sidecar.py:19-90`：raw8 每通道 6 个因果统计，共 48 维；与一个 shared/base logit 组成 49 维输入；产生一个标量 residual。
- `v3c_success_compat/run_unified_online_active.py:881-918`：先得到 `shared_logit = base + delta17`，再由私有分支得到 `combined = shared_logit + delta8`。
- 测试：`tests/experiments/test_dynamic_hazard_adapter.py`、`test_east_private23_sidecar.py`、`test_east_private_continual.py` 核对形状、零初始化与上界。

实现只保证所有项为同一 logit 维度。是否校准为破裂风险必须由 HDF5 标签、阈值、逐炮指标和分支消融证明。

## 4. 三套不能混写的训练损失

1. **adapt-high 基模型**：support110 上 BCE，见 `grid_search.py:395-413`。
2. **shared17 support adapter**：`BCE + 0.20 next + 1.00 anchor + 0.05 delta²`，见 `run_dynamic_hazard_adapter.py:148-200`。
3. **raw8 private support adapter**：`sealed_temporal_residual + 0.20 reconstruction + 0.02 residual²`，见 `run_unified_online_active.py:801-856`。

`sealed_temporal_residual_loss` 的精确定义在 `v3c_success_compat/east_private23_sidecar.py:93-120`：以封存 logit 为基准，目标移动最多 ±0.5 logit，再加 0.25 倍 BCE。

## 5. lambda-out 与时间集合

- `v3c_success_compat/two_timescale_online_adapter.py:148-188`：`shot_level_mil_hazard_loss` 使用 smooth top-k，默认 `top_k=4`、`temperature=0.25`、`hard_negative_weight=0.25`。
- `v3c_success_compat/dynamic_hazard_adapter.py:57-69`：EAST 窗口标签为 `0 <= (DownTime - t)*1000 <= horizon_ms`。
- 选定 V3C runner 在 `run_unified_online_active.py:2672-2676` 明确准备 `fixed125` EAST 标签。
- `two_timescale_online_adapter.py:229-242`：batch 若包含逐点 `alarm_boundary`，使用 operational alarm-boundary loss；否则才回退到上述 MIL 损失。

`lambda_out` 是硬负窗口权重，不是“外部域”权重，也不属于 J-TEXT Adv25 标签。其必要性仍缺同协议权重消融。

## 6. Huber/SmoothL1 与物理代理

- `v3c_success_compat/run_dynamic_hazard_adapter.py:141-145`、`east_private23_sidecar.py:117-120`、`two_timescale_online_adapter.py:817-825` 等处实际调用 `torch.nn.functional.smooth_l1_loss`。
- 默认 `beta=1`；代码没有 MSE/L1 对照，因此只能主张稳健动机，不能主张实验已证明必要。
- `v3c_success_compat/online_active_audit.py:39-113`：四族因果创新量的逐项定义。
- `online_active_audit.py:116-183`：support 第 90 百分位尺度、缺失感知家族平均、energy×concurrence、至少两族可用。
- `v3c_success_compat/two_timescale_online_adapter.py:335-369`：无标签伪监督只处理 `q<=0.10` 的 FP-like 和 `q>=0.65` 的 FN-like 强冲突。

当前代理不含密度、精确 q95，也不是 MHD 方程。`intra-only` 的炮级零变化意味着这些项的独立贡献尚未被证明。

## 7. 查询价值与 gradient protection

- `v3c_success_compat/unified_online_value.py:1334-1425`：当前 operational 查询模型只用不确定性与物理冲突，权重投影到 `[0.55, 0.75]`。
- `v3c_success_compat/run_unified_online_active.py:1190-1218`：开发特征中的 `expected_gradient = 4*peak*(1-peak)`，只是边界不确定性代理，没有 `autograd.grad`。
- `unified_online_value.py:854-938`：六信号 `OnlineQueryValueModel` 是另一开发类，不应冒充当前 V3C 操作查询公式。
- 真实更新保护：参数白名单、价值缩放、梯度范数 0.5、残差限幅、replay、guard anchor、更新前模型与优化器快照、失败回滚。共享回滚见 `two_timescale_online_adapter.py:1071-1172`；私有回滚见 `east_private_continual.py:473-537`。
- `two_timescale_online_adapter.py:654-676`：训练使用可微 raw increment；部署仍使用硬物理门，避免零初始化处梯度被硬门杀死。

仓库没有单独名为 `GradientProtect` 的类，也没有 operational `P_protect` 标量。

## 8. 报警状态机参数与预算谱系

- `v3c_success_compat/unified_online_value.py:483-550`：逐端点双路径状态机；快/慢窗为 5/100 点。
- `unified_online_value.py:561-817`：连续参数边界、predict/seal、揭示后更新方向、guard 审核。
- `DUAL_THRESHOLD_ALARM_RESULT.md`：开发点的静态风险轨迹结果。
- `DUAL_THRESHOLD_ROBUSTNESS_RESULT.md:5-39`：37 个去重配置，31 个通过；注册点 `physics=8, trend=12, early=7.9` 位于稳定区。
- `fuison_ext4_runs/adv25_v3c_success_shotfirst_replay_20260831/seed131/mainline_manifest.json`：当前 V3C 命令为 block20/quota6/history160/quantile0.70/uncertainty-weight0.55。
- `fuison_ext4_runs/mainline_receipt_20260831/mainline_results.json`：三 seed 实际查询 94/99/98；4/20=72 属于旧 locked-v8，不能混用。

状态机初值、上下界和查询配置均由 balanced360 开发得到，必须标记 `DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION`。

## 9. 当前可写与不可写的结论

可以写：

- 代码遵守 predict → seal → query → reveal → update；
- 存在真实的炮内/炮间目标侧反向传播；
- 三 seed balanced360 开发回放中，系统平均 TPR@30 从 0.790741 到 0.792593，FPR 从 0.120370 到 0.114815；
- 开发消融表明当前可见收益是神经、状态机、记忆与保护共同构成的系统级结果。

不能写：

- rebuilt17 的每一维都已证明必要；
- 当前 25 个逻辑槽等于公开论文的完整 PGFE25；
- `4p(1-p)` 是真实 expected gradient；
- 每个 loss 项都已证明不可冗余；
- 当前增益证明 Transformer 学到了新的 EAST 底层物理规律；
- balanced360 是完全未见的最终测试集。

## 10. 本次核对与文档构建回执

- 代码回归测试（`rl_env`）：

  ```bash
  PYTHONPATH=experiments/transfer_methods_2026/v3c_success_compat:$PWD \
    /home/hetaisheng/miniconda3/envs/rl_env/bin/python -m pytest -q \
    tests/experiments/test_dynamic_hazard_adapter.py \
    tests/experiments/test_east_private23_sidecar.py \
    tests/experiments/test_east_private_continual.py \
    tests/experiments/test_two_timescale_online_adapter.py \
    tests/experiments/test_unified_online_value.py
  ```

  上述五个核心模块结果为 `76 passed, 1 warning`；再加入标签合同、主线审计、
  路径约束、查询策略和归因审计测试后，完整相关集为
  `133 passed, 1 warning`。该 warning 来自 PyTorch Transformer 的
  nested-tensor 优化条件，不是测试失败。

- 报告使用 XeLaTeX 连续编译三遍；最终 PDF 为 29 页，`pdfinfo`、Ghostscript
  空设备解析和 MuPDF 信息解析均通过，最终日志无未定义引用、无 LaTeX error、
  无 overfull box。
- 编译产物：
  `docs/research/jtext_east_academic_technical_report_cn_code_verified_20260904.pdf`。
- PDF SHA-256：
  `36049122c12c53e288c1c3686f16ac0f264d1e0b1cefa8660e2702992df2e984`。
