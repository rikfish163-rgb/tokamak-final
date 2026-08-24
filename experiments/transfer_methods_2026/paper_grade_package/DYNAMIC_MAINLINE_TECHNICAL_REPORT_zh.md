# J-TEXT→EAST 动态主线技术报告

## 文档状态

- 主线：J-TEXT 动态正样本边界源模型 → EAST support110 小样本迁移 → EAST balanced360 静态评估 → 同父模型顺序在线主动适配。
- 证据状态：`TEST_DRIVEN_EXPLORATION`；`DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION`。
- 结论边界：J-TEXT `TEST220` 与 EAST `balanced360` 均参与方法开发，不能表述为独立未来确认、装置部署成绩或因果机理证明。
- 数值来源：证据装配脚本已从封存预测与 HDF5 真值逐炮重算，并核对 80 个输入产物的 SHA-256。紧凑报告 SHA-256 为 `35915b782b6c247ff768e643d02d370f4776586fab10ce54f9f8c1305a9cdbff`；若文档与机器证据不一致，应失败关闭并修正文档，不能修改评价口径迁就正文。

## 1. 研究问题与严格主线

目标是在低 EAST 标签需求下保留 J-TEXT 源表示，并允许目标装置进行因果在线适配。执行顺序固定为：

```text
J-TEXT TRAIN1353 源域训练
→ VAL160 选择 epoch 与阈值
→ TEST220 静态评估
→ 只把原始静态 J-TEXT checkpoint 送入 EAST
→ EAST support110 拟合目标高层、独立风险头与目标残差
→ balanced360 静态封存
→ 重置到完全相同的 EAST 父状态
→ balanced360 按炮号顺序在线执行
```

J-TEXT 在线主动实验是独立诊断分支。它从各随机种子的静态源 checkpoint 出发，在 TEST220 上执行 `predict → seal → query → reveal → update`；其在线状态不得进入 EAST 迁移。

## 2. 数据与标签合同

| 装置 | 数据角色 | 炮数 | 构成 | 用途 |
|---|---|---:|---:|---|
| J-TEXT | TRAIN1353 | 1353 | 188 破裂 | 源模型参数与 scaler |
| J-TEXT | VAL160 | 160 | 80/80 | epoch 与阈值选择 |
| J-TEXT | TEST220 | 220 | 110/110 | 开发期静态比较和在线诊断 |
| EAST | support110 | 110 | 15 破裂/95 安全（HDF5） | 小样本迁移与目标阈值 |
| EAST | balanced360 | 360 | 180/180 | 静态和顺序在线开发流 |

`support110` 的 15/95 来自实际 HDF5，不采用旧 JSON 标题的 10/100。support110 与 balanced360 炮号互斥，但 balanced360 参与过方法设计，不能重新命名为独立测试集。

对任一炮 $s$，整炮结局 $d_s\in\{0,1\}$ 只由 HDF5 `/meta/IsDisrupt` 给出；破裂时刻 $t_s^{down}$ 只由 `/meta/DownTime` 给出。JSON AdvancedTime 仅控制破裂炮正样本窗上界，不决定结局，也不覆盖 DownTime。

J-TEXT 当前选定上界为

$$
H_s^J=\max(25\ \mathrm{ms},A_s),
$$

其中 $A_s$ 为该炮 AdvancedTime，缺失时 $H_s^J=25\ \mathrm{ms}$。定义

$$
\mathrm{TTD}_{s,t}=1000(t_s^{down}-t)\ \mathrm{ms},
\qquad
y_{s,t}^{J}=\mathbf 1\{d_s=1,\ 0\leq\mathrm{TTD}_{s,t}\leq H_s^J\}.
$$

EAST support110 保持固定 125 ms：

$$
y_{s,t}^{E}=\mathbf 1\{d_s=1,\ 0\leq\mathrm{TTD}_{s,t}\leq125\ \mathrm{ms}\}.
$$

因此当前臂准确名称为 `source_dynamic__east_fixed`，不是“两端都使用 AdvancedTime”。该臂按开发期 $J=\mathrm{TPR}-\mathrm{FPR}$ 选择，仍属于测试驱动探索。

## 3. 报警与指标

令 $a_s$ 表示是否报警，$t_s^{alarm}$ 表示首次合格报警时刻，提前量

$$
\ell_s=1000(t_s^{down}-t_s^{alarm})\ \mathrm{ms}.
$$

J-TEXT operational TP 要求 $\ell_s>10\ \mathrm{ms}$；EAST 要求 $\ell_s\geq30\ \mathrm{ms}$。当前主线不要求连续 10 点；任一因果端点满足冻结状态机即可。提前超过 3000 ms 的报警仍计入主口径，但必须单列审计炮长、campaign 和 padding 捷径。

$$
\mathrm{TPR}=\frac{TP}{TP+FN},\qquad
\mathrm{FPR}=\frac{FP}{FP+TN},\qquad
J=\mathrm{TPR}-\mathrm{FPR}.
$$

raw TPR 只表示发生过报警；提前不足的破裂炮在 operational 口径仍是 FN。本文主表只报告 operational TPR。

## 4. 输入表示与 J-TEXT 源模型

共享主干使用 rebuilt17：`Bt, ip_A, Bt_over_Ip, Ip_diff, v_loop, Z_proxy, sxr_kurt, sxr_skew, sxr_var, xuv_kurt, xuv_skew, xuv_var, Mir_avg_fre, n=1 amplitude, mode_number_n, dx_a, dy_a`。J-TEXT 电流只执行一次 $I_p^A=1000I_p^{kA}$，并重构

$$
B_t/I_p=B_t/I_p,\quad
\Delta I_p(t)=I_p(t)-I_p(t-1),\quad
Z_{proxy}=V_{loop}/I_p.
$$

TRAIN1353 拟合每通道中位数 $m_c$ 与四分位距 $q_c$：

$$
\tilde x_{t,c}=\operatorname{clip}
\left(\frac{x_{t,c}-m_c}{\max(q_c,10^{-6})},-10,10\right).
$$

EAST shared17 不重新拟合该 scaler。不可用值由 availability mask 管理；短前缀左侧补零并通过 `valid_lengths` 生成 patch mask，不读取端点之后数据。

输入是 $128\times17$ 因果窗。宽 8、步长 8 的卷积 patch embedding 产生 16 个时间 token；加入 CLS token 与正弦位置编码。主干是两层、4 头、$d_{model}=64$、FFN=256 的 Transformer；风险头为 `64→64→GELU→Dropout(0.1)→1`。

源损失为

$$
\mathcal L_{src}=-\frac1B\sum_{i=1}^{B}
[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))].
$$

$B$ 是 batch 大小，$y_i$ 是窗口标签，$z_i$ 是风险 logit，$\sigma(z)=1/(1+e^{-z})$。训练使用 AdamW：学习率 $3\times10^{-4}$、weight decay $10^{-4}$、batch 128、梯度范数上限 1。最多 30 epoch、每 epoch 128 step、至少 12 epoch、patience 7。每个 epoch 只在 VAL160 中寻找 timely TPR≥0.90 的阈值，再按最低 FPR、最高 TPR、最高 AUC 排序；锁定前 `test_shards_read=0`。

## 5. J-TEXT→EAST 小样本迁移

载入同 seed 的静态 J-TEXT checkpoint 和源 scaler；冻结 patch embedding、CLS 与第一层 Transformer，只解冻最后一层和最终 LayerNorm。丢弃 J-TEXT 风险头，初始化独立 EAST 风险头。support110 训练 12 epoch、每 epoch 32 batch、batch 128、AdamW 学习率 $10^{-4}$：

$$
\mathcal L_{adapt}=\operatorname{BCEWithLogits}(z^E_{s,t},y^E_{s,t}).
$$

support110 同时选择阈值：先满足 timely TPR≥0.80，再最小化 FPR；因为没有独立目标验证队列，这只是开发期校准。

共享动态 adapter 的有界修正为

$$
\delta_t^{shared}=g_t\tanh f_{shared}(h_t,q_t,a_t,z_t^0),
\qquad|\delta_t^{shared}|\leq1,
$$

其中 $h_t$ 是隐藏状态，$q_t$ 是共享物理向量，$a_t$ 是可用掩码，$z_t^0$ 是基础 logit，$g_t$ 是物理可用门。支持集损失为

$$
\mathcal L_{DHA}=\mathcal L_{BCE}+0.20\mathcal L_{next}
+1.00\mathcal L_{anchor}+0.05\,\mathbb E[(\delta_t^{shared})^2].
$$

$\mathcal L_{next}$ 是下一因果状态 Smooth-L1；$\mathcal L_{anchor}$ 保护已判对支持窗口。

不适合强制跨装置对齐的 EAST raw8（`CIII, MNM, Mir_VV, Mir_avg_amp, ne_nG, qa_proxy, sxr_mean, xuv_ratio`）进入私有支路。每通道提取 6 个因果摘要，共 48 维。私有修正为

$$
\delta_t^{private}=0.5\tanh f_{private}(r_t,z_t^{shared}),
\qquad|\delta_t^{private}|\leq0.5,
$$

最终 $z_t^E=z_t^0+\delta_t^{shared}+\delta_t^{private}$。这不是丢弃 23 维，而是让 shared17 承载跨装置可比较信息、raw8 通过小容量目标支路利用 EAST 表象。

## 6. EAST 在线主动适配

每炮严格执行：`predict → endpoint seal → 可选炮内 fast 更新 → shot seal → query → 仅选中炮 reveal → persistent update 或 rollback`。当前炮结局、DownTime、AdvancedTime 与未来端点在预测时不可达。

每炮开始复制 `fast=deepcopy(persistent)`。炮内无标签梯度只改变 fast，炮后丢弃，persistent 哈希必须不变。更新使用已经到达的新端点做下一状态、teacher 一致性和物理支持约束；不是把当前预测直接当真值，也不读取未来端点。

EAST 查询值组合报警边界不确定性与模型—物理矛盾，并由滚动 token 预算消费。当前配置：每 20 炮名义配额 4、历史 80、分位数 0.8、不确定性权重 0.65、初始 cutoff 0.35、token capacity 12。它不是看完 360 炮后选全局 top-k；实际查询数为 63/63/62。

只有选中炮封存后才读取标签。监督目标面向部署报警边界，并加入 replay、物理、下一状态与 guard。更新前保存模型和优化器；非有限损失/梯度/参数/优化器或 guard 失败均触发事务回滚。更新后的 persistent 只影响后续炮。

## 7. J-TEXT TEST220 在线主动诊断

J-TEXT 在线只训练 2241 个参数的小残差，源 encoder/head 冻结：

$$
z_t=z_t^0+\Delta z_t,\qquad
\Delta z_t=0.5\tanh f_{res}(x_t,a_t,\sigma(z_t^0)).
$$

零初始化保证初始输出等于静态父模型，并限制最大 logit 位移为 0.5。

已到达相邻端点 $t<t'$ 的炮内损失为

$$
\mathcal L_{intra}=1.0\mathcal L_{next}
+0.5\mathcal L_{teacher}+0.25\mathcal L_{anchor}
+0.0\mathcal L_{physics}.
$$

physics 字段虽有代码与回执，但本次权重为 0，不能写成已生效贡献。

对风险概率 $p_1,\ldots,p_T$，定义

$$
U=T^{-1}\sum_t(1-|2p_t-1|),
\quad
\Delta_x=\frac{\operatorname{mean}|x_T-x_1|}
{1+\operatorname{mean}|x_T-x_1|},
\quad
C=\operatorname{clip}(|\,|p_T-p_1|-\Delta_x|,0,1).
$$

炮表型为 $\phi=[\operatorname{mean}(p),\max(p),p_T-p_1,\Delta_x]$。只在历史已查询并揭示的表型库 $\mathcal M$ 中定义新颖度

$$
N=\operatorname{clip}
\left(\min_{\psi\in\mathcal M}\sqrt{\operatorname{mean}(\phi-\psi)^2},0,1\right),
$$

空库时 $N=0$。查询价值

$$
V=wU+(1-w)C+\eta N,\qquad w=0.65,\quad\eta=0.10.
$$

20% 预算在每个连续 20 炮块恰好查询 4 炮，共 44 炮；先按 $V\geq0.5$ 主动选择，块末以 causal forced-fill 补足。10%/30% 对应 22/66 炮。

查询后取当前 8 窗、replay 8 窗、guard 8 窗，执行 2 个 optimizer steps：

$$
\mathcal L_{inter}=1.0\mathcal L_{sup}
+0.5\mathcal L_{replay}+0.25\mathcal L_{anchor}
+0.0\mathcal L_{physics}.
$$

三个 seeds 均查询 44 炮、尝试 88 步、无回滚；active 与 random 的窗口数和报告 FLOPs 匹配。

## 8. 当前结果

| seed | J-TEXT TPR@10 | J-TEXT FPR | J-TEXT AUC | EAST static TPR@30 | EAST static FPR | EAST online TPR@30 | EAST online FPR | EAST queries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 131 | 0.900000 | 0.154545 | 0.978926 | 0.866667 | 0.138889 | 0.861111 | 0.138889 | 103 |
| 133 | 0.836364 | 0.100000 | 0.961736 | 0.766667 | 0.122222 | 0.766667 | 0.116667 | 100 |
| 137 | 0.863636 | 0.136364 | 0.963058 | 0.838889 | 0.166667 | 0.844444 | 0.172222 | 97 |
| 等 seed 均值 | **0.866667** | **0.130303** | **0.967906** | **0.824074** | **0.142593** | **0.824074** | **0.142593** | 100.00 |

EAST online 相对 static 的 $\Delta J$ 按 seed 为 $-0.005556、+0.005556、0$。seed131 少 1 个及时真阳性；seed133 少 1 个假阳性；seed137 同时增加 1 个及时真阳性和 1 个假阳性。三 seed 变化相互抵消，不能表述为稳定在线增益。该结果来自恢复后的历史炮优先 support110 采样器和未经删改的原 v3c 在线配置。

$J$ 的均值为：J-TEXT static **0.736364**，EAST static **0.681481**，EAST online **0.681481**；EAST 在线相对静态 $\Delta J=0$。seed133 TPR 未达到 0.80，因此 `T0_ADMITTED=false`。

J-TEXT 20% full-active 相对 static 的 $\Delta J$ 为 0、−0.009091、0，故 `ONLINE_SUPPORTED=false`。seed131 的 10%/20%/30% 均未改变炮级 operational 指标。active 与 equal-compute random 均查询 44 炮，集合只重合 4 炮，却都只覆盖 6 个静态错误型炮，显示当前价值函数对可纠正错误富集不足。44 个前缀标签直接训练的同后缀对照从 $J=0.763025$ 降至 0.751261；它排除“同样标签直接加到迁移必然更好”的简单解释，但不证明在线策略优越。

## 9. 消融裁决与局限

1. J-TEXT `intra_only` 的炮级指标不变，虽有最大概率位移约 0.0239。
2. `inter_active` 在 seed131 丢失一个 operational TP，炮间监督并非单调有益。
3. `full_active` 与 `full_random` 在 seed131 均不改善，主动查询未稳定优于等计算随机。
4. EAST 在线三 seed 平均 $\Delta J=0$，不能写成“在线机制已稳定提升”或“已解决误报”。
5. 贡献是可审计的特征拆分、因果封存、目标私有支路与两时间尺度实现；不是部署通过，也不是所有破裂因果机理证明。

## 10. 符号与缩写

| 符号 | 定义 |
|---|---|
| $s,t$ | 炮编号与炮内因果端点 |
| $d_s,t_s^{down}$ | HDF5 结局与破裂时刻 |
| $A_s,H_s^J$ | AdvancedTime 与 J-TEXT 正样本上界 |
| TTD | time to disruption，距离破裂时间 |
| $x_t,a_t$ | rebuilt17 与可用掩码 |
| $h_t,q_t,r_t$ | 隐状态、共享物理向量、raw8 摘要 |
| $z,p$ | 风险 logit 与 sigmoid 概率 |
| $\delta^{shared},\delta^{private}$ | 共享与 EAST 私有 logit 残差 |
| $U,C,N,V$ | 不确定性、矛盾度、新颖度、查询价值 |
| $w,\eta$ | 查询混合权重与新颖度系数 |
| fast/persistent | 炮内临时副本/跨炮慢状态 |
| TP/FP/TN/FN | 真阳性/假阳性/真阴性/操作假阴性 |
| TPR/FPR/AUC/J | 真阳性率/假阳性率/ROC AUC/TPR−FPR |
| SHA-256 | 绑定代码、数据、模型与证据字节的哈希 |

## 11. 公式与代码映射

| 内容 | 代码 |
|---|---|
| HDF5 与 AdvancedTime 标签 | `advanced_time_labels.py`, `train_jtext_source_transformer.py` |
| rebuilt17 | `rebuilt_harness.py` |
| Transformer/目标高层适配 | `grid_search.py::Enc/Head/train_adapt` |
| 动态主线 | `run_selected_full_mainline.py` |
| shared hazard | `dynamic_hazard_adapter.py`, `run_dynamic_hazard_adapter.py` |
| EAST raw8 | `east_private23_sidecar.py`, `east_private_continual.py` |
| EAST 在线 | `run_unified_online_active.py`, `unified_online_value.py` |
| J-TEXT 在线残差/损失 | `jtext_online_adapter.py` |
| J-TEXT 查询与顺序流 | `run_jtext_online_active.py` |
| 最终证据重算 | `paper_grade_package/scripts/build_dynamic_mainline_evidence.py` 与最终装配脚本 |
| 报告合同/编译 | `paper_grade_package/scripts/build_dynamic_report.py` |

## 12. 复现命令

```bash
cd /path/to/tokamak-final
export FUSION_DATA_ROOT=/path/to/data
PY=${CONDA_PREFIX}/bin/python

$PY -c 'import torch,h5py; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'

for seed in 131 133 137; do
  $PY experiments/transfer_methods_2026/run_selected_full_mainline.py --seed \"$seed\" --stage all --data-root \"$FUSION_DATA_ROOT\" --output \"/ext4/runs/selected_full_mainline/seed${seed}\"
done

$PY experiments/transfer_methods_2026/paper_grade_package/scripts/build_dynamic_report.py --check-only
$PY experiments/transfer_methods_2026/paper_grade_package/scripts/build_dynamic_report.py
$PY -m pytest -q tests/experiments/test_dynamic_mainline_report.py
```

跨机复现必须同时归档代码 SHA、配置 SHA、HDF5 内容清单、炮表 SHA、checkpoint SHA、阈值回执、逐炮预测和查询/更新回执。单独复现汇总表不等于复现完整主线。

## 13. 当前裁决

工程链路已经打通并可审计，逐炮证据重算与哈希核验也已完成，但 EAST admission 和 J-TEXT `ONLINE_SUPPORTED` 均未通过。下一步应进行独立未来流预注册确认；不能继续在 balanced360 或 TEST220 上追逐参数并把结果升级为确认性结论。
