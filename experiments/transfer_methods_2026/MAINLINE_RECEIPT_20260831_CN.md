# J-TEXT → EAST 主线复现回执

**生成日期：2026-08-31**  
**状态：`DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION`**

这份回执只记录当前能够从代码和已封存产物中复现的主线。它不把历史
实验、严格冻结负对照和当前 V3C 诊断结果拼成一个新结果。

由封存 JSON 自动汇总的机器可读结果和中文短报告见：
[`mainline_results.json`](../fuison_ext4_runs/mainline_receipt_20260831/mainline_results.json)
和
[`MAINLINE_RESULTS_CN.md`](../fuison_ext4_runs/mainline_receipt_20260831/MAINLINE_RESULTS_CN.md)。

## 1. 当前唯一主线

```text
J-TEXT TRAIN1353（静态 Adv25；同一训练/标签/架构合同）
        ├── jtext_exact profile → J-TEXT TEST220 静态/独立在线主动
        │                         （不进入 EAST）
        │
        └── east_success profile → EAST support110 适配
                                    ├── EAST balanced360 静态封存
                                    └── 从同一迁移父模型重置
                                            └── EAST V3C 炮内/炮间在线主动
```

当前两条下游分支属于同一 Adv25 训练合同，但使用不同的、预先登记的
VAL-selected checkpoint profile；它们不是同一个文件 SHA。J-TEXT 在线状态仍
完全隔离，不得作为 EAST 的迁移输入。

J-TEXT 破裂炮的源域标签窗口为

\[
H_{\mathrm{JTEXT},i}=\max(25\ \mathrm{ms},\;\mathrm{AdvancedTime}_i),
\]

缺少 `AdvancedTime` 时取 25 ms。结局和 `DownTime` 以逐炮 HDF5 元数据为
准；JSON/文本炮表只用于炮号、顺序及差异审计。

EAST 迁移严格使用 **EAST 白名单登记的 J-TEXT 静态 checkpoint**，不读取
J-TEXT 在线分支的权重、查询记录或在线状态。若需要声称“单一 checkpoint
端到端主线”，还必须用同一 SHA 重新完成 J-TEXT 和 EAST 两条下游验证；当前
回执不作这一声称。

## 2. 存储与运行核验

最终产物根目录是挂载盘：

```text
/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/
```

本次核验中 `/home/hetaisheng/fuison_ext4_runs` 不存在。挂载核验为：

```text
/media/hetaisheng/044A81D94A81C83E  /dev/nvme0n1p3  ntfs3  rw
```

`/dev/shm` 只作为运行时临时空间；入口在发布前执行哈希和文件清单校验，
并拒绝把 `/home`、`/dev` 或本机根文件系统作为最终产物根目录。

统一运行时：

```text
/home/hetaisheng/miniconda3/envs/rl_env/bin/python
PyTorch 2.11.0+cu128
NVIDIA GeForce RTX 5070 Laptop GPU
```

## 3. 已封存结果

### 3.1 J-TEXT 独立在线主动分支

入口：[`run_jtext_adv25_active_mainline.py`](./run_jtext_adv25_active_mainline.py)

20% 查询预算、TEST220 炮序流的封存产物：

```text
/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/
jtext_adv25_active_canonical_replay_20260831/budget20/
```

| 视图 | 查询炮数 | TPR（lead > 10 ms） | FPR | J=TPR−FPR | AUC |
|---|---:|---:|---:|---:|---:|
| 静态 | 0 | 0.836364 | 0.109091 | 0.727273 | 0.974215 |
| 在线主动 | 44/220 | **0.854545** | **0.109091** | **0.745455** | 0.975620 |
| 在线−静态 | 44 | +0.018182 | 0 | +0.018182 | +0.001405 |

逐炮协议是：

```text
当前炮信号前缀 → 预测并封存 → 主动策略选查询炮 → 只揭示被选炮 HDF5 结局
→ 炮间更新 → 预测下一炮
```

这条当前 J-TEXT 实现的在线增益应准确归因于**查询后延迟反馈的专家权重
更新**；`representation_updated=false`，不能写成 Transformer 底层表示已经
被在线重新学习。它是 J-TEXT 独立附加分支，不是 EAST 的迁移父模型。
因此表中的 J-TEXT 静态行是“三颗静态 Adv25 source experts 的固定等权组合”，
不是单独 seed131 source checkpoint 的 TEST220 单模型记录；两者不能直接横向
比较。三颗单独 source checkpoint 的 TEST220 结果已经在机器回执的
`jtext_source_test220.per_seed` 中登记；canonical 在线表仍保持专家组合口径，
不把两个统计对象混为一谈。

### 3.2 EAST 静态迁移与 V3C 在线主动

入口：[`run_east_v3c_success_mainline.py`](./run_east_v3c_success_mainline.py)

该入口复现已经验证出在线净增益的 V3C 双时间尺度实现：EAST 白名单登记的
静态 Adv25 checkpoint 经 EAST support110 形成历史兼容的 `shot-first → endpoint` 迁移
父模型；balanced360 先完整静态预测并封存，然后重置到同一父模型执行按炮序
的 V3C 在线主动。

support 的组成需要单独说明：历史角色表声明为 `10D+100N`，但兼容实现对这
110 个炮号逐一读取 HDF5 `meta/IsDisrupt`，权威标签实际为 **15D+95N**。其中
`68928、71316、71393、92933、93888` 与 JSON 的安全角色声明不一致。当前回放
保留这一历史 sampler 以复现既有父模型；它不会把角色表中的文字标签当作权威，
也不会把这组结果写成干净的 10D+100N 小样本实验。对应的组成审计见机器回执
`mainline_audit_20260831_v2.json` 的 `east_support_label_audit` 字段。

在线协议包含两个不同层级的屏障：炮内端点先用当前因果前缀预测并封存，再执行
不读取结局的 fast-state 无标签更新；整炮结束后才封存炮级轨迹、计算不含结局的
查询价值，并只对选中的炮揭示 HDF5 结局，随后在下一炮到来前执行持久炮间更新。
未查询炮的结局不会参与查询、炮内更新或后续回放。

三 seed 的 balanced360 诊断结果：

| seed | 静态 TPR@30/FPR | 在线 TPR@30/FPR | ΔTPR | ΔFPR | 查询炮数 |
|---:|---:|---:|---:|---:|---:|
| 131 | 0.794444 / 0.100000 | 0.800000 / 0.094444 | +0.005556 | −0.005556 | 94 |
| 133 | 0.761111 / 0.116667 | 0.761111 / 0.111111 | 0.000000 | −0.005556 | 99 |
| 137 | 0.816667 / 0.144444 | 0.816667 / 0.138889 | 0.000000 | −0.005556 | 98 |
| **算术均值** | **0.790741 / 0.120370** | **0.792593 / 0.114815** | **+0.001852** | **−0.005556** | **97** |

三颗 seed 的新鲜复现目录为：

```text
/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/
adv25_v3c_success_shotfirst_replay_20260831/seed{131,133,137}/
```

seed131 静态结果为 `TP=170, FP=18, FN=10, TN=162`，在线结果为
`TP=170, FP=17, FN=10, TN=163`；seed133 在线相对静态减少 1 个 FP，
seed137 在线相对静态减少 1 个 FP。三颗 seed 的 TPR@30 均不下降、FPR 均下降。

三 seed 结果的正确解释是：这次已登记的开发复现中，完整 V3C 决策系统产生了
FPR 减少且 TPR@30 不下降；它仍不能被写成独立未来流确认，也不能据此声称
某一个神经模块学习到了破裂因果机理。该运行是 `run_mode=development`，带有
`--development-search`，并同时启用了炮内/炮间更新、endpoint memory、episode
gate 和告警状态机保护；查询数 94/99/98 也不是固定百分比预算。因此当前
`ΔJ=+0.007407` 是系统级开发诊断，不是 clean neural ablation 的结果。

同一 source/support parent 下的 seed131 配对归因回执已经完成，见
[`east_v3c_timescale_ablation_20260831/seed131/aggregate.json`](../fuison_ext4_runs/east_v3c_timescale_ablation_20260831/seed131/aggregate.json)。
它包含 `intra-only`、`inter-only`、`intra+inter`、固定计算量
`neural-only` 和禁用炮间神经更新的 `side-only`。该筛查显示：炮内反向传播
发生但没有改变炮级边界；纯神经固定计算对照也没有改变边界；保留查询后的
决策/记忆侧更新仍可减少一个 FP。因此完整 V3C 的可见增益不能单独归因于
Transformer 表示变化。多 seed 的正交归因复现仍属于后续工作，seed131 不冒充
多 seed 显著性证据。

J-TEXT 另有一个真正执行局部神经 episode 更新的独立诊断，但 guarded 版本
没有改变 TEST220 炮级指标；它不进入 canonical J-TEXT 在线结果，也不进入
EAST parent。该边界和数值见
[`JTEXT_EPISODIC_NEURAL_DIAGNOSTIC_20260831_CN.md`](./JTEXT_EPISODIC_NEURAL_DIAGNOSTIC_20260831_CN.md)。

### 3.3 Source-profile 一致性诊断

将 EAST 的 `east_success` profile 直接用于 J-TEXT TEST220 的对照目录为：

```text
/media/hetaisheng/044A81D94A81C83E/Fuison/fuison_ext4_runs/
jtext_adv25_active_east_success_replay_20260831/budget20/
```

该诊断的 static/online 均为 `TPR=0.800000、FPR=0.109091`，`ΔJ=0`。
它证明 source profile 变化会改变 J-TEXT 结果，因而本回执分别报告
`jtext_exact` 和 `east_success`，不把两者的数字拼成单一 checkpoint 结果。

## 4. 为什么之前会出现“在线没有增益”

`adv25_v21_shotfirst_replay_20260831` 是严格的 `shot-first + frozen v2.1`
负对照。它和上述 V3C 结果的 support parent 哈希不同，故静态父模型也不同。
它在 balanced360 上的静态与在线结果均为
`TPR@30=0.866667, FPR=0.138889`（`TP=175, FP=25, FN=5, TN=155`），说明
该冻结合同下没有发生可观测净变化；这不等于 V3C 兼容谱系不存在增益。

`class-first` 与 `shot-first` 不能互换；入口会检查 source checkpoint、迁移
父模型哈希、炮序和因果字段。README 中复现正向 V3C 诊断时必须显式使用
`--mode development`；默认 `locked` 是负对照。

## 5. 因果与论文边界

已通过机器回执核验的内容：

- J-TEXT 与 EAST 的源模型、在线状态和结果目录相互隔离；
- EAST 静态阶段先完成 360 炮预测并封存，在线阶段从相同迁移父模型重置；
- 查询选择不读取未查询炮结局；
- 被查询炮才在 `reveal` 阶段读取 HDF5 结局；
- 在线更新只影响后续炮；
- 最终结果和运行时环境落在挂载盘，非 `/home` 本地结果根目录。

仍然不能写入正式论文为独立确认的内容：

- balanced360/Test220 已参与方法或参数开发；
- 当前 J-TEXT 在线增益不是神经表示学习增益；
- 当前 EAST 三 seed 并非每颗都同时降低 FPR、提高 TPR；
- 当前 EAST seed131 的同 parent 归因仍是单 seed；不能扩写为神经模块的多 seed 显著增益；
- 尚未有完全未参与方法选择的 EAST future stream；
- 现有结果不能证明模型学习了全部破裂因果机理。

正式论文之前仍需锁定：源 checkpoint SHA、support parent SHA、阈值、报警
规则、查询策略和版本化代码，然后在未参与开发的 EAST 顺序流上一次性确认。
