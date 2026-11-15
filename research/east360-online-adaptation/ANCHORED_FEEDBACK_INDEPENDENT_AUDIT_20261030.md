# Anchored feedback 独立 CPU 审查 — 2026-10-30

## 结论

**数值复算及隔离/哈希核验通过；ordinary、anchored 均未通过本轮候选 gate。** 未发现封存汇总、供体选择或监督抽样的实际实现错误。anchored 留炮危险组相对 reverse/donor 的优势在三个折均保留，但安全组总体输给这两个对照；相对 ordinary，三个安全炮全部变差，危险炮仅 6/17 更好。不能据此接受新方法或进入完整 360 炮验证。

仅 CPU、保存数组与代码审查；未导入实验模块或调用原 summary/gate/donor 函数，未执行模型前向、反向、拟合、GPU、原始 HDF5 或模型重放。主进程负责 neural replay。沿用当前角色，未切换模型、未启用 fast。按 results-analysis 流程区分直接复算、代码解释与能力边界；本轮只新建本报告及下面的只读复算脚本。

## 数据来源与可复算方法

[复算脚本](audit_anchored_feedback_20261030.py#L1) 默认 JSON 输出到 stdout，不写文件；加 `--rows` 输出全部 **1,296 条逐炮/组/臂记录**（fit 864，held 432），包括 before/after/delta BCE、窗口数、支持数和 logit 变化。运行：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -B /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/audit_anchored_feedback_20261030.py --rows
```

逐炮预测来源是 [封存目录](probe_outputs/anchored_feedback_cold20_v1) 内 `{ordinary|anchored}_fold{0|1|2}_{shot}.npz`。NPZ 第 j 行对应 [records.json](probe_outputs/anchored_feedback_cold20_v1/records.json#L1) 中按原顺序筛选该 shot 的第 j 行；[folds.json](probe_outputs/anchored_feedback_cold20_v1/folds.json#L1) 决定 fit/held 身份。不存在将同炮两次 fit 与一次 held 混为独立样本的池化。

独立 float64 公式：`BCE(z,y)=logaddexp(0,(1−2y)z)`；先在每炮、每组内平均 `BCE(arm)−BCE(cold)`，再对该组炮等权平均。safe/early 的 y=0，danger/late 的 y=1；Δ<−1e−6 为改善，Δ>+1e−6 为变差。每个已存机会均有支持，没有事后排除窗口。

实际覆盖：**120 NPZ / 2,280 数组 / 16,242 个 family×fold 预测行**，来自同一 2,707 个机会；43,312 个 fit 和 21,656 个 held 臂-窗口 BCE。复核 96 个 fit 组汇总、32 个 held 组汇总及全部 432 个封存 held 逐炮记录。最大绝对误差：fit 汇总 **6.66e−16**、held 汇总 **4.44e−16**、held 逐炮字段 **8.88e−16**；两个 gate 与封存结果完全一致。原始分组窗口数：safe 527、danger 63、early 2,095、late 22。共 **17D/3S**，不是新增独立数据。

## Held 全部对照结果

下表均为炮等权 ΔBCE；safe 3 炮，其余各 17 炮。负值表示相应点标签 BCE 下降。

| 规则 / 臂 | safe | danger | early | late |
|---|---:|---:|---:|---:|
| ordinary / real | -0.140617035 | -0.291599500 | 0.078818907 | -0.032591499 |
| ordinary / reverse | -0.140678021 | -0.272414383 | 0.078221774 | -0.046710608 |
| ordinary / mean_feedback | -0.142841989 | -0.313652212 | 0.078012689 | -0.055288079 |
| ordinary / donor | -0.138421961 | -0.288769502 | 0.068252457 | -0.046998630 |
| anchored / real | -0.085706811 | -0.255399906 | 0.028591985 | -0.167720296 |
| anchored / reverse | -0.090789720 | -0.193481194 | 0.030544314 | -0.180972542 |
| anchored / mean_feedback | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| anchored / donor | -0.095460422 | -0.093232950 | 0.024753466 | 0.048556951 |

两个 real 均改善 2/3 safe、14/17 danger。primary 变差炮如下（列出的均是相对 cold 的 ΔBCE，而非整炮报警变差）：

- ordinary：safe 70404 +0.081369366；danger 70327 +1.555240681、70861 +0.034590565、71370 +0.078091905。
- anchored：safe 70404 +0.083635306；danger 69454 +0.004062077、70327 +1.481838128、71370 +0.033432615。

anchored−ordinary 的 paired mean：safe **+0.054910224**（0 改善/3 变差），danger **+0.036199594**（6/11）；early −0.050226923（15/2），late −0.135128797（7/10）。**early BCE 上升不自动等于运营伤害，下降也不自动等于报警改善**：其 label-0 定义为 lead>125ms，而当前及时报警允许 lead>30ms、没有 125ms 上界，>3000ms 另报。本审查不计算整炮 TPR/FPR、救回或改坏。

## 全部门槛复核

比较项写 real−control；须 <−1e−6。mean-feedback 按计划不参加效用 gate。

| 判定项 | ordinary | anchored |
|---|---:|---:|
| safe 均值 <−1e−6 | −0.140617035，通过 | −0.085706811，通过 |
| safe 改善 ≥2/3 | 2/3，通过 | 2/3，通过 |
| danger 均值 <−1e−6 | −0.291599500，通过 | −0.255399906，通过 |
| danger 改善 ≥9/17 | 14/17，通过 | 14/17，通过 |
| safe real−reverse | 0.000060986，失败 | 0.005082909，失败 |
| safe real−donor | -0.002195074，通过 | 0.009753611，失败 |
| danger real−reverse | -0.019185117，通过 | -0.061918713，通过 |
| danger real−donor | -0.002829998，通过 | -0.162166956，通过 |
| real/reverse 非零方向余弦中位 <0.99 | 0.995219111，失败 | 0.965353549，通过 |
| 后续 logit 差 >1e−4 的比例 >50% | 2640/2707=97.524935%，通过 | 2697/2707=99.630587%，通过 |

故 ordinary 失败：safe 不胜 reverse、方向余弦不达标；anchored 失败：safe 不胜 reverse、safe 不胜 donor。[gate 实现](experiments/risk_aligned_online/anchored_feedback_20261030/run.py#L39)；[ordinary 封存 gate](probe_outputs/anchored_feedback_cold20_v1/complete.json#L162)；[anchored 封存 gate](probe_outputs/anchored_feedback_cold20_v1/complete.json#L328)。ordinary 从上一轮“五项失败”变成本轮“两项失败”是**对照/gate 定义改变**，不是同一模型表现改善；共同保存输出严格未变。

## Fit / held 对照：本轮训练端也已独立复算

| 规则 / 折 | fit safe | held safe | fit danger | held danger |
|---|---:|---:|---:|---:|
| ordinary / 0 | -0.256532 | 0.081369 | -0.823919 | -0.514265 |
| anchored / 0 | -0.245553 | 0.083635 | -0.813727 | -0.377871 |
| ordinary / 1 | -0.097024 | -0.322486 | -0.797496 | -0.301697 |
| anchored / 1 | -0.092834 | -0.208918 | -0.776421 | -0.336380 |
| ordinary / 2 | -0.169600 | -0.180734 | -0.578834 | -0.012285 |
| anchored / 2 | -0.165260 | -0.131838 | -0.571873 | -0.011258 |

六个 family×fold 中，所有 fit safe/danger 炮的 real BCE 都改善；anchored 在 fit 的 safe/danger 均胜 reverse 和 donor。其危险组 real−donor，fit 三折为 **−0.347936/−0.482460/−0.098073**，held 为 **−0.030416/−0.296218/−0.159406**；held real−reverse 亦三折为负（−0.005298/−0.069045/−0.121311）。本轮危险组相对这两个非平凡对照的优势未在留炮整体反转；这与上一轮 real−mean-feedback 的比较不是同一检验。

但 anchored 安全组 real−donor 在 fit 均为负，held 为 **−0.001326/+0.002723/+0.027864**；real−reverse held 为 −0.001151/+0.000171/+0.016229。因此失败具体落在安全组反馈优势的转移，而不是完全无法拟合。这个解释仅限共享父模型的组件比较，不能宣称独立泛化。全部 fit 对照数值和逐炮例外均可通过脚本输出；[fit_groups](probe_outputs/anchored_feedback_cold20_v1/fit_groups.json#L1) 已逐项验证，非只引用原汇总。

## Donor、抽样、零对照与哈希

- **Donor：8,121/8,121 个索引重算一致。** 每折 2,707 个，候选只能是该折 fit 炮且排除当前整炮；排同炮前的 fit 上下文总池分别 1,830/1,548/2,036 行。独立用标准化后的前 128 维（CLS64+末 token64）的 float64 均方距离穷举，不输入反馈230维、类别、lead、query 或结局。保存实例中并列最近距离 **0 个**；最小一二名距离差约 1.23e−7/9.82e−7/3.51e−6。代码按全局索引递增构造池并 argmin，符合并列时选最小原索引；并列分支本次数据未触发。[donor 实现](experiments/risk_aligned_online/anchored_feedback_20261030/learning.py#L39)
- **折/抽样：** fit/held 炮数 13/7、13/7、14/6，折内无交集，held 恰覆盖全部20炮各一次。records/folds 与上一轮逐项相同，全部固定时钟及唯一键通过。三个新训练记录共 **768 步、18,432 抽样位置**，与原 representation 逐项相等，也与独立 seed131“先均匀选炮、再选窗口”抽样完全一致；每批12 safe+12 danger，零 held、early、late 教学抽样。[训练实现](experiments/risk_aligned_online/anchored_feedback_20261030/run.py#L71)
- **Scale/mean：** 三个 anchored 模型的 scale、mean、spec 与原 ordinary 模型完全相等，六个模型均为14,316参数且有限。独立从保存的 fit-only 上下文复算所有尺度/标志，float32 结果精确相同；炮等权 mean 最大差1.52e−7。所有 fit 组可参与无监督尺度/均值计算，不等于 early/late 被用于监督抽样。[复用统计量](experiments/risk_aligned_online/anchored_feedback_20261030/run.py#L188)
- **全部控制一致性：** 16,242 行支持掩码与保存上下文一致；120 份 cold 数组与上一轮 cold 严格相同；ordinary held 的 **300 个共同数组**（每炮15个）与上一轮 representation 逐元素相同；ordinary fit 的三个共同臂、各组与上一轮 train_groups 最大差6.66e−16。reverse/donor 的所有非零方向按 real 范数匹配，最大范数差1.49e−8，半径上限0.1仅有 float32 舍入误差。[控制构造](experiments/risk_aligned_online/anchored_feedback_20261030/run.py#L28)、[范数匹配](experiments/risk_aligned_online/anchored_feedback_20261030/run.py#L103)
- **Anchored mean-feedback：8,121 行全部精确零。** 匹配前范数、匹配后范数、保存的对应余弦均为0，logit逐元素等于cold，所有 fit/held ΔBCE=0；未将零方向伪造为非零。ordinary 的 mean-feedback 不为零是预期设计。该恒等式不是反馈效用证据，gate 正确排除了它。
- **哈希全部一致：** 本轮 manifest **136/136** 文件；上一轮输入 **60/60**，与其 completion manifest 一致；当前代码/计划/测试 **73/73**；ZIP成员 **73/73**，成员集合亦完全相等；冻结父引用 **6/6**；source checkpoint SHA256 `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`。没有扫描或读取原始 HDF5。manifest 不自哈希 complete.json，也不包含之后新增的主进程 replay 产物。

## 代码解释与剩余边界

[共享网络相减](experiments/risk_aligned_online/anchored_feedback_20261030/learning.py#L21) 符合 `0.5*(f(origin,feedback)−f(origin,fit_mean_feedback))`；两次计算共享参数，输入 detach，支持掩码作用于结果。每机会从零矩阵状态起步，沿原点表示 Jacobian 更新第一层 Q/K（[state_for](experiments/risk_aligned_online/anchored_feedback_20261030/run.py#L58)）。对照只替换反馈，保留真实原点；donor 反馈来自异炮拟合历史，而不是部署政策或有物理可实现性保证的轨迹。

**相减只锚定参考反馈为零，不消除 origin×feedback 的非线性交互或真实原点 Jacobian。** 控制还继承 real 更新范数，故不能把它们描述为完全无 real 信息的独立策略。保存的 raw_norm 是控制范数匹配前、但半径投影后的范数。以上为解释约束，不是本次发现的实现 bug。

复算脚本通过；无数值审查 blocker。保存标量余弦只复核聚合，未重建神经方向；未运行含优化步骤的单测，未独立 neural replay 或确认运行中内存权重，source unchanged 在本审查中指磁盘哈希一致。父模型已见全部20炮（[封存范围声明](probe_outputs/anchored_feedback_cold20_v1/protocol.json#L76)）；这是 **17D/3S 的共享父模型开发材料，不是独立数据、不是完整360、不是TPR/FPR或FP救回结果，也不是方法获准**。

