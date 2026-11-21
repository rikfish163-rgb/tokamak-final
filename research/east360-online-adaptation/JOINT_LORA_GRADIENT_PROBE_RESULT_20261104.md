# JointLoRA 已结束 20 炮：诊断梯度作用性探针

## 结果与适用范围

本次固定 `warm.pt` 模型、只做梯度计算、不更新任何参数。按规则选出 **60 个窗口，恰好每炮 3 个**。在失稳正窗口，真实反馈与未来风险 BCE 梯度同向 **4/8**，hold 为 **8/8**、reverse 为 **6/8**；在失稳较早窗口分别为 **7/16、7/16、9/16**。这批训练内坐标没有显示真实反馈在这些分组中一致优于伪反馈。

这是相同 20 个已参与 warm 训练的炮上的小样本梯度诊断，有明显训练内乐观偏差；同炮窗口也不是独立炮。它不能估计新炮泛化、实时告警效果或实际 Adam 更新后的收益，不应据此修改正在运行的策略。

## 方法与因果范围

- 只从运行中的 `warm.json.label_shots` 读取 20 个已结束 EAST 炮的 HDF signal 和 `meta/IsDisrupt` / `meta/DownTime`。读取器白名单检查设备、ID 与顺序；不读取第 21 炮以后的 HDF、J-TEXT VAL/TEST 或事件 CSV。先按原契约处理实际可用 mask，再对这 20 炮使用旧的已拟合 density map，和 warm 训练输入相同。
- 每炮从 `q=135+32k`、`q+32` 存在且未越过失稳时刻的候选中，按索引均匀选至多 3 个，未按模型错误或梯度排序。标签是 **q+32** 的 125 ms 真实硬结局窗口；失稳正窗口的提前量先按毫秒小数点后 9 位规范化，严格要求 `>30 ms`。较早失稳窗口为负标签；安全炮提前量记 `null`。
- 对每例，固定 warm 模型参数，仅对 512 个 Q/V LoRA 坐标求导。`g_diag` 来自 q 的预测与真实 `q+1..q+8` 目标之间的 observed-mask MSE，目标由既有 `levels` / `direct_target` 构造；hold 使用零增量，reverse 使用真实目标的相反数，mask 完全相同。decoder、W0、原 source 与两个原生 head 均冻结。
- `g_risk` 是 q+32 风险 logit 的真实标签 BCE 梯度，使用与 `train_schedule` 相同的 `logit − τ`，其中固定阈值 `0.7258946895599365`，`τ=0.9738925713822922`。定义同向为 `<g_diag,g_risk> > 0`，此时假设原坐标中的微小普通梯度步 `−η g_diag`，风险 BCE 一阶变化为 `−η<g_diag,g_risk>`；结果文件保存单位步的 `−<g_diag,g_risk>`。

**这个一阶式针对普通梯度步，不等同正在运行的 Adam 更新。** Adam 的坐标预条件、历史动量、梯度裁剪、半径投影以及有限步非线性都可能改变实际变化；余弦和范数也依赖 LoRA 因子参数化。q+32 的真实标签仅用于已结束炮的离线探针，并不在 q 或 q+8 提供给在线策略。

## 按组、同一 mask 的结果

所有 60 例均有支持（每例 160 或 168 个目标坐标），所有真实和伪反馈的诊断梯度范数及风险梯度范数均非零；无支持、零诊断范数、零风险范数分别均为 **0**。下表分母只包括有支持且两个范数非零的可比较例；无零点积。

| HDF 分组（独立炮数） | 例数 | real 同向 | hold 同向 | reverse 同向 |
|---|---:|---:|---:|---:|
| safe（12 炮） | 36 | 24/36，66.7% | 28/36，77.8% | 22/36，61.1% |
| earlier，失稳前负窗口（8 炮） | 16 | 7/16，43.8% | 7/16，43.8% | 9/16，56.3% |
| positive，失稳前正窗口（8 炮） | 8 | 4/8，50.0% | 8/8，100% | 6/8，75.0% |

与 hold 配对看，safe 例中 real 独有同向 3 例、hold 独有同向 7 例；earlier 各 4 例；positive 分别为 0 例和 4 例。8 个 positive 例各来自不同失稳炮，q+32 实际提前量为 **37–62 ms**。`cases.json` 保留每例 shot、q、arrival、future risk point、真实提前量、标签、支持数、三个控制的 loss、范数、内积、余弦和单位步一阶量。

下表为**窗口等权的描述性均值**，不是炮级置信区间；风险梯度在三个控制间相同。内积均值受范数影响，与余弦均值或同向比例不是同一量。

| 组 | mean ‖g_risk‖ | mean ‖g_diag‖ real / hold / reverse | mean cosine real / hold / reverse | mean 内积 real / hold / reverse |
|---|---:|---|---|---|
| safe | 0.9336 | 0.02263 / 0.01435 / 0.03084 | 0.2536 / 0.3788 / 0.2480 | 0.003831 / 0.006101 / 0.008372 |
| earlier | 4.6838 | 0.03454 / 0.01826 / 0.04747 | −0.0238 / 0.0221 / 0.1087 | −0.032023 / −0.020016 / −0.008009 |
| positive | 2.2690 | 0.03750 / 0.03670 / 0.07516 | −0.0441 / 0.3789 / 0.3471 | 0.014994 / 0.015735 / 0.016477 |

例如 `71304` 的 positive 例在 q=3399、risk=3431、提前 62 ms：real 余弦 `−0.2883`，hold `+0.1916`，reverse `+0.5702`。这仅描述同一固定 warm 状态中的梯度方向，不表示该炮真实在线更新一定使风险变差或 hold 在未来炮一定更优。

## 校验与产物

- 先进行了缺失模块 RED；合成单测 **7 passed**，覆盖均匀取点、125 ms、严格排除 nominal 30 ms、三臂 mask、BCE 中心和内积符号及无支持/零范数分项。Ruff check 与 format check 通过。
- CPU 单线程、`CUDA_VISIBLE_DEVICES=''`；没有优化器或参数步。模型 state digest 在分析前后不变，`warm.pt` 加载后 digest `69a3b183a5ce708293cfc23ff4871db5e2e9c29b8c0fa7f174ccf1b193033582` 与 `warm.json` 收据一致；source state digest 同样不变。warm、map、source 文件哈希在运行前后复核一致。由于 GPU 全流仍在运行，warm 文件哈希及该收据不是对整个运行已完成的声明。
- 三控制共享模型和 mask；独立检查 MSE 的梯度线性恒等式，所有例中 `dot_reverse − (2·dot_hold − dot_real)` 绝对残差最大约 `1.34e-6`，量级与 float32 计算相容。这是控制实现一致性检查，不是物理规律证明。
- 明细：[cases.json](probe_outputs/joint_lora_gradient_probe_20261104_v1/cases.json)，SHA-256 `72f8d8c7ac970f53d4ee6239037970cc3abf1c0363915d2da3a7aac5e28dac27`；[summary.json](probe_outputs/joint_lora_gradient_probe_20261104_v1/summary.json)，SHA-256 `6dbb58bcc02b3ea18392c45b572ce3ebffabbe42d36ec45bc06ef659b41ce728`；`completed.json` SHA-256 `5d9ef8ff763a6f501ebe513806e2c37df402290c89164407484061b8cabc36b5`。
- 探针脚本 SHA-256 `14d450e8effebcee84c61f0b06fd8101bd09e9448dc9610e29621daede8dbd70`；单测 SHA-256 `c43a8c7e8ee91fbbf40ebccd3d1828b372138c4a574ba4c0f116b65dadd08c00`。
- 输入绑定：`warm.json` `2e908bc822d633c2ec009548317467e30f6a1cb03cf1e5b48873d812035d8e09`；`warm.pt` `d237001ff8febf82f17eb974850c48db115fd81f212d7695b78bc76b078407b5`；density map `43c37ad981261e059ac8bf8d11d89324bad375dd7405decc194c51e53614aff2`；qualified source checkpoint `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`。

未修改正在运行的 producer、模型权重或策略；没有上传数据，没有进行新 warm 训练或 J-TEXT 在线实验。
