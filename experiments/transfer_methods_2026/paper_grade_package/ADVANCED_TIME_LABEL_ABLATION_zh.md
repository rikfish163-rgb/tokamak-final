# AdvancedTime 正样本上界对照实验（修正版，seeds 131/133/137）

## 结论先行

先前 v1 报告中的 EAST `TPR@lead≥30 ms=0.50–0.53` 无效，不能继续引用。v1 实际评估的是裸 `encoder/head` 加单阈值，没有执行当前静态迁移模型中的 shared hazard adapter、EAST-private raw8 residual 和双阈值物理/趋势状态机，因此它与主线 `0.8x` 指标不是同一个模型。修正版用完整主线重新执行后，固定标签基线精确恢复为三 seed 平均 `TPR@30=0.8407、FPR=0.1667`。

按用户指定的逐炮标签，仅改变正样本时间上界后，`max(25 ms, AdvancedTime)` 在 J-TEXT TEST220 上改善了分类与 FPR，但使中位提前量平均缩短约 49.3 ms；迁移到 EAST 后，seed 131 改善而 seed 133 明显下降。三个 seed 中最差 `TPR@30` 为 0.7667，未满足每 seed `TPR≥0.80` 的项目门槛。因此 AdvancedTime 硬替换标签目前仍是 `DIAGNOSTIC_ONLY`，不能替换固定 125 ms 主标签。

## 1. 唯一变量与标签合同

炮级结局只读取 HDF5 `meta/IsDisrupt`，破裂时刻只读取 HDF5 `meta/DownTime`。AdvancedTime JSON 不能把安全炮改成破裂炮，也不能改变 DownTime；它只为已经被 HDF5 判定为破裂的炮定义正样本时间上界：

\[
L_i^{\mathrm{J}}=\max(25\ \mathrm{ms},A_i),\qquad
L_i^{\mathrm{E}}=\max(125\ \mathrm{ms},A_i).
\]

其中，\(A_i\) 是炮 \(i\) 在原工程 JSON 中记录的 AdvancedTime，数值单位直接按毫秒解释。若 JSON 中缺少炮号，J-TEXT 回退到 25 ms，EAST 回退到 125 ms。固定对照在两个装置上均使用 125 ms。

其余因素保持不变：rebuilt17 输入、128 点因果窗口、patch 8、`d_model=64`、2 层 4 头、J-TEXT TRAIN1353/VAL160/TEST220、EAST HDF5-authoritative support110、balanced360 顺序、随机种子、optimizer、训练步数、support110 阈值选择器和完整双阈值报警状态机均相同。动态标签被一致传入四个实际受监督位置：

1. J-TEXT `BalancedSourceSampler`；
2. EAST 历史 shot-first 迁移采样器；
3. shared hazard adapter 的窗口标签；
4. EAST-private raw8 residual 的 binary/soft temporal target。

阈值仍只由 support110 的 HDF5 结局选择，目标为炮级 raw TPR 0.80；balanced360 不参与训练、权重、epoch 或阈值选择。每个实验臂先完整封存 360 炮预测，再读取 HDF5 结局计算指标。大于 3000 ms 的早报警仍按项目合同计入及时报警，同时单列计数。

## 2. 修复后的完整迁移链

修正版实际评估的模型为：

```text
J-TEXT encoder/head
→ EAST adapt_high 顶层适配
→ shared hazard adapter
→ EAST-private raw8 residual
→ support110 冻结阈值
→ 双阈值 physics/trend 状态机
→ balanced360 预测封存
```

seed 131 的固定标签源模型用当前代码重新训练后，encoder 与 head 均和归档模型逐张量位级相等，模型哈希同为 `214eb476fb26fe1efc606d0b0ce70bee794ca6558a0a410a6ed6602da23a832c`。三个 seed 的 fixed/fixed 目标父模型哈希、阈值和 360 炮逐炮报警字段均与现有主线归档完全一致。逐炮核验字段包括 `shot`、`stream_index`、`alarmed`、`alarm_index`、`route`、`early_score` 和 `alarm_time_s`。

## 3. 标签覆盖与窗口变化

J-TEXT TRAIN1353 含 188 个 HDF5 破裂炮，其中 187 炮在 JSON 中有 AdvancedTime，只有 37 炮的值大于 25 ms。按当前生产采样器的实际边界计算，固定标签有 21,912 个正端点，动态标签有 5,556 个，减少 74.64%。

EAST support110 按 HDF5 实际为 15 破裂/95 安全，而不是旧 JSON 标题所写的 10/100。15 个破裂炮中 10 炮能匹配 AdvancedTime，只有 2 炮大于 125 ms。固定标签有 1,865 个正端点，动态标签有 1,899 个，仅增加 34 个。因此 EAST 侧标签改动先天很弱。

## 4. J-TEXT TEST220 结果

正式源域 TPR 要求提前量至少 10 ms，阈值只由 VAL160 选择。

| seed | 标签 | TPR@10 ms | FPR | AUC | 中位提前量 |
|---:|---|---:|---:|---:|---:|
| 131 | 固定 125 ms | 0.8818 | 0.2273 | 0.9307 | 107.2 ms |
| 131 | AdvancedTime | 0.9000 | 0.1545 | 0.9791 | 68.9 ms |
| 133 | 固定 125 ms | 0.8182 | 0.1909 | 0.9253 | 120.7 ms |
| 133 | AdvancedTime | 0.8364 | 0.1000 | 0.9617 | 63.4 ms |
| 137 | 固定 125 ms | 0.8545 | 0.2182 | 0.9098 | 122.1 ms |
| 137 | AdvancedTime | 0.8636 | 0.1364 | 0.9629 | 69.8 ms |
| **均值** | **固定 125 ms** | **0.8515** | **0.2121** | **0.9219** | **116.6 ms** |
| **均值** | **AdvancedTime** | **0.8667** | **0.1303** | **0.9679** | **67.4 ms** |

动态标签在源域提高了平均 TPR 0.0152、降低了 FPR 0.0818、提高了 AUC 0.0460，但平均中位提前量缩短 49.3 ms。它更集中于临近破裂的终端状态，这能提高可分性，却减少了 25–125 ms 区间内的早期前兆监督。

## 5. EAST balanced360 完整静态结果

下表 TPR 均为 `lead≥30 ms` 的 operational TPR，\(J=\mathrm{TPR}_{30}-\mathrm{FPR}\)。表中的 TP/FN 若在机器结果出现，默认是“是否发生过报警”的 raw 计数；及时混淆计数另以 `timely_TP_at_30ms/timely_FN_at_30ms` 表示，二者不得混用。

| J-TEXT 标签 | EAST 标签 | seed | TPR@30 ms | FPR | J | AUC | 中位提前量 |
|---|---|---:|---:|---:|---:|---:|---:|
| 固定 | 固定 | 131 | 0.8222 | 0.1556 | 0.6667 | 0.9322 | 208.0 ms |
| 固定 | 固定 | 133 | 0.8389 | 0.1444 | 0.6944 | 0.9362 | 175.0 ms |
| 固定 | 固定 | 137 | 0.8611 | 0.2000 | 0.6611 | 0.9315 | 224.0 ms |
| **固定** | **固定** | **均值** | **0.8407** | **0.1667** | **0.6741** | **0.9333** | **202.3 ms** |
| AdvancedTime | 固定 | 131 | 0.8667 | 0.1389 | 0.7278 | 0.9539 | 166.0 ms |
| AdvancedTime | 固定 | 133 | 0.7667 | 0.1167 | 0.6500 | 0.9376 | 136.0 ms |
| AdvancedTime | 固定 | 137 | 0.8333 | 0.1556 | 0.6778 | 0.9490 | 209.0 ms |
| **AdvancedTime** | **固定** | **均值** | **0.8222** | **0.1370** | **0.6852** | **0.9468** | **170.3 ms** |
| 固定 | AdvancedTime | 131 | 0.8611 | 0.1722 | 0.6889 | 0.9341 | 210.0 ms |
| 固定 | AdvancedTime | 133 | 0.8389 | 0.1500 | 0.6889 | 0.9363 | 177.5 ms |
| 固定 | AdvancedTime | 137 | 0.8389 | 0.1778 | 0.6611 | 0.9298 | 208.0 ms |
| **固定** | **AdvancedTime** | **均值** | **0.8463** | **0.1667** | **0.6796** | **0.9334** | **198.5 ms** |
| AdvancedTime | AdvancedTime | 131 | 0.8556 | 0.1444 | 0.7111 | 0.9536 | 167.0 ms |
| AdvancedTime | AdvancedTime | 133 | 0.7667 | 0.1222 | 0.6444 | 0.9360 | 138.0 ms |
| AdvancedTime | AdvancedTime | 137 | 0.8333 | 0.1611 | 0.6722 | 0.9455 | 217.0 ms |
| **AdvancedTime** | **AdvancedTime** | **均值** | **0.8185** | **0.1426** | **0.6759** | **0.9450** | **174.0 ms** |

只改 J-TEXT 标签时，平均 FPR 下降 0.0296，AUC 上升 0.0135，平均 \(J\) 上升 0.0111；代价是平均 TPR@30 下降 0.0185、提前量缩短 32.0 ms，并且 seed 133 的 TPR@30 只有 0.7667。只改 EAST 标签的均值变化很小，且没有在三个 seed 上一致改善 \(J\)。两边同时修改也保留了 seed 133 的灵敏度下降。

## 6. 科研裁决

1. 当前迁移主线的真实固定标签基线确实是 `0.8x`，不是 v1 的 `0.5x`。v1 结果已被本报告和 v2 机器证据明确撤回。
2. J-TEXT AdvancedTime 硬标签在源域有明显判别收益，但更偏向终端状态；其跨装置迁移收益对随机种子不稳定，不能宣称已经学到更可迁移的共同破裂机制。
3. EAST AdvancedTime 在 support110 中只实际扩展两炮、34 个端点，统计支撑不足，当前没有稳定证据表明它优于固定 125 ms。
4. 三个 seed 的 fixed/fixed 均满足 TPR≥0.80、FPR≤0.20；含动态 J-TEXT 标签的两组均因 seed 133 TPR<0.80 失败。因此当前主线继续使用固定 125 ms。
5. balanced360 已参与方法开发，本实验仍属于 `TEST_DRIVEN_EXPLORATION`，不是独立未来流确认。均值变化不能升级为部署结论。

更合理的后续方向是把 AdvancedTime 用作 horizon/onset 辅助任务或连续阶段权重，而不是删除 25–125 ms 的早期正监督；若继续验证，必须预先固定融合权重并在独立未来流上一次性确认。

可复现入口为 `experiments/transfer_methods_2026/run_advanced_time_label_ablation.py`；固定合同见 `paper_grade_package/configs/advanced_time_label_ablation_v2.json`；三 seed 机器结果、逐炮预测封存和哈希见 `paper_grade_package/evidence/advanced_time_label_ablation_v2/`。
