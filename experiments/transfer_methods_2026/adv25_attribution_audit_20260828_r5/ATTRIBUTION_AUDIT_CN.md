# Adv25 在线主动学习物理归因与双时间尺度审查

## 结论

当前方法的因果访问顺序基本合理，但证据必须拆成三层：完整系统、神经更新、物理归因。
Adv25 seed131 的完整 EAST V3C 在 balanced360 开发流上降低了 FPR；然而固定查询的神经四臂和物理开关实验没有把这一收益归因到炮内神经更新或物理损失。
因此当前最严格的表述是：**受约束的在线决策系统有诊断增益；神经表示学习和物理规律学习尚未得到证明。**

| 审查命题 | 裁决 |
|---|---:|
| 完整 V3C 决策系统有开发流增益 | 通过 |
| 炮内神经增益 | 未通过 |
| 炮间神经增益 | 未通过 |
| 在线物理损失带来可测增益 | 未通过 |
| 查询器具有错误富集能力 | 通过 |
| 已证明学到物理规律 | 未通过 |
| 已完成独立未来流确认 | 未通过 |

单 seed 固定查询审计中，炮内是否观察到至少一个炮级 J 变化：未通过；炮间：通过。这只是诊断证据，不是跨 seed 支持。当前干净 2×2 只有 1 个 seed，因此正式跨 seed 裁决仍为未通过。

## 一、同父模型 2×2 炮内/炮间消融

令 M00、M10、M01、M11 分别为 static、intra-only、inter-only、intra+inter。
主效应为 ΔI=M10−M00、ΔR=M01−M00，交互项为 ΔIR=M11−M10−M01+M00。
所有四臂必须共享 parent、阈值、查询清单、报警器和炮序。

### J-TEXT TEST220（Adv25，J=TPR@10−FPR）

| 因子 | ΔJ | ΔTPR@10 | ΔFPR |
|---|---:|---:|---:|
| 炮内主效应 | 0.000000 | 0.000000 | 0.000000 |
| 炮间主效应 | 0.000000 | 0.000000 | 0.000000 |
| 联合总效应 | 0.000000 | 0.000000 | 0.000000 |
| 交互项 | 0.000000 | 0.000000 | 0.000000 |

J-TEXT 四臂的炮级指标相同；8295 次炮内尝试均被变化上限回退，炮内臂在功能上等同于静态臂。

### EAST balanced360 神经残差路径（J=TPR@30−FPR）

| 因子 | ΔJ | ΔTPR@30 | ΔFPR | ΔAUC |
|---|---:|---:|---:|---:|
| 炮内主效应 | 0.000000 | 0.000000 | 0.000000 | 0.000093 |
| 炮间主效应 | 0.005556 | 0.005556 | 0.000000 | 0.000123 |
| 联合总效应 | 0.005556 | 0.005556 | 0.000000 | 0.000247 |
| 交互项 | 0.000000 | 0.000000 | 0.000000 | 0.000031 |

炮间路径在单 seed replay 中只把一个破裂炮从 TARDY 推到 lead≥30 ms；原始 TP/FP/FN/TN 不变。该单炮变化在 physics-off 中同样出现，不能解释成物理损失收益；它也不足以构成跨 seed 神经增益。

## 二、完整 V3C 与神经路径不能混称

| 指标 | 静态 | 完整在线 | 差值 |
|---|---:|---:|---:|
| TPR@30 | 0.7667 | 0.7667 | 0.0000 |
| FPR | 0.2333 | 0.1944 | -0.0389 |
| J | 0.5333 | 0.5722 | 0.0389 |
| >3000 ms 过早报警数 | 12 | 13 | +1 |

完整 V3C 同时包含 fast/slow 神经更新、EAST-private raw8、端点记忆、四个报警参数和 delta60 延迟/否决头。
故 7 个 FP 的减少是系统级收益，不是 Transformer 或物理损失的独立收益。

### 与此前 frozen v2.1 主线的严格对照

此前 frozen v2.1 的三 seed 汇总必须单独保留：static TPR@30=0.84074、FPR=0.16667；online TPR@30=0.84259、FPR=0.16667，ΔJ=+0.00185。它与 Adv25 clean/V3C 使用的 parent、报警器和实验合同不同，不能作为本轮神经 2×2 的重复。
frozen 证据表是否成功绑定：通过；其 SHA256 和路径已经写入 JSON lineage_comparison。

## 三、物理负对照

physics-on 与 physics-off 使用同一查询清单：True，查询数 72。
联合臂 physics-on 相对 physics-off 的 ΔJ=0.000000、ΔAUC=0.000093。
当前结果未通过物理归因门。physics-off 只是第一层消融，不能单独证明物理信息的因果作用。下一轮必须在完全相同的优化预算下增加：物理时间反转、炮内时间打乱、物理族置换、raw8 置零和标签错位；真实物理臂需跨三个 seed 稳定优于最强负对照。

## 四、查询是否具有智能性

| 分支 | 查询数 | 查询错误率 | 全流错误率 | 富集倍数 |
|---|---:|---:|---:|---:|
| J-TEXT | 22 | 0.2273 | 0.1273 | 1.7857 |
| EAST 神经四臂 | 72 | 0.4306 | 0.3028 | 1.4220 |
| EAST V3C | 86 | 0.2442 | 0.2333 | 1.0465 |

EAST 神经路径有一定错误富集，但这只说明查询优于均匀抽样的方向，不等于查询具有未来纠错价值。
真正的主动性必须用同预算 random、uncertainty-only、physics-only 和预期未来收益选择器比较，并统计每次查询对后续相似炮的净收益。

## 五、正式门槛是否已经满足

当前协议状态：未通过。要求至少 3 个 seed 的 clean factorial、完整负对照套件和独立未来流；当前失败原因：clean_factorial_has_one_seed, balanced360_participated_in_method_development, physics_off_is_not_the_full_negative_control_suite, current_v3c_combines_neural_and_decision_layers。异常停止规则为 TPR<0.60 或 FPR>0.40。

只有在相同 parent、scaler、阈值、报警器、query manifest、炮序和更新预算下，才允许计算炮内/炮间主效应。

## 六、下一轮唯一合理的正式矩阵

1. 固定 Adv25 静态源 checkpoint、EAST110 parent、阈值、报警器、炮序和查询 manifest。
2. 运行 static、intra-only、inter-only、intra+inter，先证明神经路径本身有收益。
3. 在同一神经轨迹上正交比较 alarm-only、neural-only、neural+alarm，禁止把决策保护算作表示学习。
4. 对 neural-only 重复 real-physics、zero、time-reverse、within-shot-shuffle、family-permutation。
5. 对查询重复 active、equal-budget random、uncertainty-only、physics-only；每臂更新次数、窗口数和 FLOPs 相同。
6. seeds 131/133/137 仅用于开发流稳定性；配置冻结后必须在新 EAST chronological stream 一次性确认。

## 七、论文表述边界

可以写：在 balanced360 开发流上，受约束 V3C 系统降低了误报，并保持 lead-qualified TPR。
不能写：在线模型已经学习到新的破裂物理规律，或该结果已经得到独立未来流验证。

机器可读证据见同目录 `attribution_audit.json`。
