# J-TEXT 炮间局部神经适配诊断报告

**日期：** 2026-08-31  
**状态：** `DIAGNOSTIC_ONLY / NEGATIVE_TRANSFER`  
**适用对象：** J-TEXT Adv25 静态源模型的独立 TEST220 在线分支  
**运行环境：** `rl_env`，PyTorch 2.11.0+cu128，NVIDIA GeForce RTX 5070 Laptop GPU

## 1. 结论摘要

本轮验证了一个严格按炮序运行的局部神经适配器：查询炮的 HDF5 结局只有在
`predict → seal → query → reveal` 完成后才可读；适配器只从该查询炮的误差窗口
拟合，并从下一炮开始通过过去端点路由。J-TEXT 源编码器和源风险头始终冻结，
当前炮不会使用刚刚揭示的 episode。

结果没有支持“已揭示错误能够稳定泛化到后续炮”的假设。在同一静态父模型、同一
阈值和同一 20% 查询清单下：

* 普通端点条件局部适配器和隐藏原型适配器均没有改变炮级告警；
* 将路由器的端点集合相似度与适配器内部局部性门对齐后，适配器开始产生可见作用，
  但把安全炮错误地变成了报警，`FP: 17 → 19`，`J: 0.745455 → 0.727273`；
* 要求至少两个严格过去、同类错误 episode 才允许路由后，负迁移被抑制，
  但也没有产生任何净救回。

因此，本轮不把局部神经适配器加入 J-TEXT 的已封存主动主线，更不把它导出为
EAST 迁移父模型。旧的 J-TEXT 主动结果仍单独保留为决策层专家权重更新结果；
它不能被重新表述为 Transformer 表示学习。

## 2. 固定谱系与数据合同

本诊断从下列静态 Adv25 source checkpoint 开始：

```text
fuison_ext4_runs/v3c_adv25_exact_20260830/seed131/source/source_checkpoint.pt
```

固定项如下：

| 项目 | 固定内容 |
|---|---|
| 数据流 | J-TEXT TEST220，按炮号升序 |
| 源标签 | 破裂炮 (H_i=\max(25\,\mathrm{ms},\mathrm{AdvancedTime}_i))，缺失时取 25 ms |
| 阈值 | source checkpoint 在 VAL160 固定的 0.684680700302124 |
| 查询 | 已登记的 canonical budget20 manifest，44/220 炮 |
| 结局权威 | 逐炮 HDF5 `meta/IsDisrupt` 与 `meta/DownTime` |
| 在线更新 | 仅影响后续炮；J-TEXT 在线状态不进入 EAST |
| 报警器 | 单次越阈的源分数，J-TEXT operational lead (>10\,\mathrm{ms}) |
| 适配参数 | 每个 episode 约 789 个可训练参数，远低于 5000 |

TEST220 已参与方法开发，因此这里的所有数值都只是诊断结果，不是独立未来确认。

## 3. 局部适配器的定义

### 3.1 端点表示与路由

对第 (t) 个因果端点，路由描述符只使用当前端点的隐藏表示、风险 logit
增量和已观测时间：

\[
d_t=\operatorname{norm}\left[
\operatorname{norm}(h_t),
\tanh(z_t/4),
\tanh((z_t-z_{t-1})/2),
\tanh(t/(2f_s))
\right].
\]

其中 (h_t,z_t) 来自被冻结的源模型，(f_s) 是当前炮采样率。路由器只遍历
满足 (i_{\mathrm{source}}<i_{\mathrm{current}}) 的历史 episode，并先检查
风险区间、端点相似度和正确查询炮 blocker。

### 3.2 原型条件适配器

每个被揭示的 FP、FN 或 TARDY 炮建立一个局部适配器。适配器包含一个 rank-(r)
低秩隐藏变换：

\[
\tilde h=h+\alpha W_{\uparrow}
\tanh(W_{\downarrow}h),
\qquad r=4.
\]

风险残差由隐藏原型位移、端点上下文和局部相似度决定：

\[
z'=z+\delta(h,c),
\qquad
\delta=\ell(d)\,\Delta_{\max}
\tanh\big(g([\tilde h-p,c,s_h,s_d])\big),
\]

其中 (p) 是揭示错误轨迹的隐藏原型，(s_h) 和 (s_d) 分别是隐藏空间和
端点描述符空间的相似度，

\[
\ell(d)=\sigma\{12(s_d-0.8)\}.
\]

FP 使用负方向，FN/TARDY 使用正方向；方向只在结局揭示后生成。所有权重更新
均执行有限性检查和相对参数半径投影。更新前后源编码器/风险头哈希必须一致。

### 3.3 训练目标与事务边界

查询结果揭示后，从该炮的因果端点选取最多 16 个监督点，优化加权二元交叉熵：

\[
\mathcal L=
\frac{\sum_t w_t\,\operatorname{BCEWithLogits}(z'_t,y_t)}{\sum_t w_t}
+0.01\|\delta_t\|_2^2+0.001\|\tilde h_t-h_t\|_2^2,
\]

\[
w_t=1+4\left[y_t(1-\sigma(z_t))+(1-y_t)\sigma(z_t)\right].
\]

拟合完成后，候选适配器仍先处于事务状态；只有通过 guard 才能写入有序 episode
bank。任何非有限值、超出半径、重复提交或非法炮序都会拒绝提交。

## 4. 实验矩阵与结果

所有实验均使用同一 source checkpoint、TEST220 roster、44 炮 query manifest、
VAL160 阈值和报警器。强诊断设置为学习率 0.05、32 个拟合步、最大相对参数变化
1.0、最大 logit 变化 4.0；这只是测功能上限，不是正式部署设置。

| 实验臂 | 局部结构 | 路由门 | 静态 TP/FP/FN/TN | 在线 TP/FP/FN/TN | ΔJ | 结论 |
|---|---|---|---|---|---:|---|
| generic strong | endpoint-conditioned residual | 单 episode | 108/17/2/93 | 108/17/2/93 | 0 | 无炮级作用 |
| prototype strong | 隐藏均值原型 | 单 episode | 108/17/2/93 | 108/17/2/93 | 0 | 路由与作用坐标不一致时欠驱动 |
| prototype set-kernel | 隐藏原型 + 端点集合 max-kernel | 单 episode | 108/17/2/93 | 108/19/2/91 | -0.018182 | 负迁移，停止 |
| prototype probation | 同上 | 至少 2 个严格过去同类 episode | 108/17/2/93 | 108/17/2/93 | 0 | 安全但无增益 |

对应 operational 指标为：

| 实验臂 | operational TPR（lead (>10\) ms） | FPR | AUC | 中位提前量 |
|---|---:|---:|---:|---:|
| 静态基线 | 0.900000 | 0.154545 | 0.979091 | 68.875 ms |
| prototype set-kernel | 0.900000 | 0.172727 | 0.978430 | 64.057 ms |
| prototype probation | 0.900000 | 0.154545 | 0.979091 | 64.366 ms |

set-kernel 臂提交 15 个 episode、应用 818 个端点路由；probation 臂提交 14 个
episode，但只有 45 个端点满足重复历史支持门。两者都没有使用未查询炮结局。

## 5. 失败定位

本轮结果把问题分成两个层次：

1. **作用函数不足。** 单一隐藏均值原型不能复现路由器使用的“历史端点集合最大
   相似度”，所以即使路由成立，未来端点在适配器内部仍可能被判为局部外样本，
   造成近似零作用。
2. **可泛化错误证据不足。** 修复为端点集合 max-kernel 后，适配器确实改变了
   风险，但主要把安全炮推过阈值；说明相似轨迹中同时存在正确预测与错误预测，
   单个错误炮的监督方向不能直接推广到同类表观轨迹。

“至少两个过去错误 episode”可以抑制这个负迁移，却同时把可用路由压缩到 45 个
端点，未发现任何救回炮。因此当前数据和查询预算下，不能把局部相似性当作共享
破裂机制的充分证据。

## 6. 与已封存主线的关系

### 6.1 J-TEXT 决策层主动分支

canonical J-TEXT active-Hedge 分支仍为：静态 operational
`TPR=0.836364, FPR=0.109091, J=0.727273`；在线为
`TPR=0.854545, FPR=0.109091, J=0.745455`，查询 44/220。
其回执明确记录 `representation_updated=false`，因此增益归因于延迟反馈的
专家权重更新，而不是本报告的局部神经适配器。

### 6.2 EAST V3C 分支

EAST V3C 使用独立登记的 EAST source profile 和迁移父模型；J-TEXT 本诊断的
episode、查询记录和参数绝不进入 EAST。已封存 EAST V3C 结果仍按其原始谱系
单独报告。本报告不能用 J-TEXT 的失败改写 EAST V3C，也不能借用 EAST V3C 的
系统级开发增益证明 J-TEXT 神经在线学习成功。

## 7. 结论与允许的下一步

本轮正式裁决为：

* **禁止**把 prototype set-kernel 适配器加入当前主线；它产生了可复现的负迁移。
* **禁止**用 TEST220 继续搜索相似度、半径、学习率或报警阈值来补救。
* **保留** probation 版本作为安全上限诊断：它证明了重复支持门可以抑制伤害，
  但没有证明正向泛化。
* **保留** canonical J-TEXT decision-level active-Hedge 和 EAST V3C 结果，
  但维持各自的谱系、父模型、指标口径和 claim boundary。

若继续推进“真正神经在线学习”，下一次只允许在新的、未参与结构选择的按炮序流上，
预注册 delayed promotion：先建立 probation episode，再用后续已揭示炮验证跨炮
收益，确认不伤害 TP/TN 后才晋升共享 LoRA。该验证前不得把任何 TEST220 结果写成
机制发现或独立未来确认。

## 8. 机器证据与复现命令

代码：

```text
experiments/transfer_methods_2026/jtext_episodic_adapter.py
experiments/transfer_methods_2026/run_jtext_episodic_online.py
tests/experiments/test_jtext_episodic_adapter.py
```

主要结果目录：

```text
fuison_ext4_runs/jtext_episodic_online_20260831/
  seed131_b20_prototype_setkernel_diagnostic/
  seed131_b20_prototype_probation_diagnostic/
  seed131_b20_prototype_locality_guarded/
```

复现其中一个诊断臂（输出目录必须为空且位于挂载盘）：

```bash
PYTHONPATH=experiments/transfer_methods_2026 \
/home/hetaisheng/miniconda3/envs/rl_env/bin/python \
experiments/transfer_methods_2026/run_jtext_episodic_online.py \
  --seed 131 \
  --source-checkpoint fuison_ext4_runs/v3c_adv25_exact_20260830/seed131/source/source_checkpoint.pt \
  --data-root data \
  --output fuison_ext4_runs/jtext_episodic_online_20260831/replay_output \
  --device cuda \
  --no-use-guard \
  --adapter-kind prototype \
  --minimum-episode-support 2 \
  --learning-rate 0.05 --fit-steps 32 \
  --max-relative-change 1.0 --max-logit-delta 4.0
```

回归证据：

```text
293 passed, 6 warnings
audit_current_mainline.py: status=DIAGNOSTIC_ONLY
```

上述测试和审计只证明代码合同与已封存谱系一致，不改变 TEST220 已参与方法开发
这一事实。
