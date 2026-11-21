# Maturity Jacobian 独立 CPU 审查 — 2026-10-30

## 独立结论与具体例外

**主格 MM 仍未通过 gate；所有保存统计、gate、哈希及本次时序检查均复核一致，未发现实际实现或数值差异。** MM 相对旧 anchored OO 的 safe/danger 炮均 ΔBCE 差为 **−0.004113882 / −0.010610780**；safe 3/3、danger 11/17 相对旧格更好，但仍不胜安全组 reverse/donor。

重点定位结果：固定 origin teacher 只换求导窗口（OM−OO），safe 仅改善 **−0.000868005**（3/3），danger **变差 +0.001314830**（7改善/10变差）。固定 mature teacher 换窗口（MM−MO）同样是 safe −0.000898239、danger +0.001323917（4改善/13变差）。**本结果不支持把主格的危险组改善归因于执行时改用更近的窗口。** 改 teacher 而保留 origin 求导（MO−OO）为 safe −0.003215643、danger −0.011934697；这是本次配对数值定位，不是机制因果证明。

MM 相对 cold 仍变差的 primary 炮：**safe 70404 +0.077966160；danger 70327 +1.497934562、71370 +0.030610636**。70327 相对旧格又差 +0.016096434；70861 虽仍优于 cold，却相对旧格差 +0.042742039。69454 从旧格 +0.004062077 变为 −0.068719336，解释了相对 cold 改善数从14/17到15/17，但不等于整炮“救回”。

范围：17D/3S 共享 cold 父模型开发诊断，父模型已见全部20炮；仅3安全炮，每折fit仅2安全炮。**不是独立数据验证、不是完整360、不是TPR/FPR或FP-rescue证据，不接受新方法。** 不提出新研究设计。

## 2×2 数值与全部 gate

缩写：OO=`origin_teacher_origin_step`（旧 anchored）；OM=`origin_teacher_mature_step`；MO=`mature_teacher_origin_step`；MM=`mature_teacher_mature_step`（唯一预定主格）。以下为 real 相对同一 cold 的炮等权 ΔBCE。

| 格 | safe Δ（改善/3） | danger Δ（改善/17） | early Δ | late Δ |
|---|---:|---:|---:|---:|
| OO | -0.085706811（2/3） | -0.255399906（14/17） | 0.028591985 | -0.167720296 |
| OM | -0.086574817（2/3） | -0.254085076（14/17） | 0.029206157 | -0.156337732 |
| MO | -0.088922455（2/3） | -0.267334603（15/17） | 0.035111850 | -0.151702033 |
| MM | -0.089820693（2/3） | -0.266010686（15/17） | 0.035854906 | -0.137935240 |

三新增格均满足 safe/danger 均值<−1e−6、至少2/3与9/17改善。下面比较项均为 real−control 或新格−OO，须<−1e−6；mean-feedback 恒等零，不参加效用 gate。

| 剩余判定项 | OM | MO | MM |
|---|---:|---:|---:|
| safe real−reverse | 0.004673063 失败 | 0.004651526 失败 | 0.004143478 失败 |
| safe real−donor | 0.009581949 失败 | 0.009593456 失败 | 0.009454987 失败 |
| danger real−reverse | -0.065494420 ✓ | -0.059102087 ✓ | -0.059828089 ✓ |
| danger real−donor | -0.162315669 ✓ | -0.143421951 ✓ | -0.143878044 ✓ |
| safe real−OO real | -0.000868005 ✓ | -0.003215643 ✓ | -0.004113882 ✓ |
| danger real−OO real | 0.001314830 失败 | -0.011934697 ✓ | -0.010610780 ✓ |
| real/reverse 非零方向余弦中位 <.99 | 0.965345621 ✓ | 0.968238652 ✓ | 0.968590319 ✓ |
| 后续 logit 差>1e−4 的比例>50% | 2699/2707 ✓ | 2700/2707 ✓ | 2699/2707 ✓ |

失败列表与封存逐项一致：**OM三项**（safe不胜reverse、不胜donor，danger不优于OO）；**MO、MM各两项**（safe不胜reverse、donor）。OO旧gate亦独立重算为同样两项安全失败；不把新增加的“胜OO”条件套回OO自比较。依据：[promotion](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/maturity_jacobian_20261030/run.py:37)、[封存OM](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/maturity_jacobian_cold20_v1/complete.json:164)、[封存MO](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/maturity_jacobian_cold20_v1/complete.json:332)、[封存MM](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/maturity_jacobian_cold20_v1/complete.json:499)。

early 使用 lead>125ms 的 label-0；其 BCE 上升不自动构成运营伤害，下降也不自动表示告警更好。当前及时报警为 lead>30ms、无125ms上界，>3000ms另报；本审查未计算整炮报警。

## 主格与旧格逐炮配对

列“差”为 MM−OO；负值更好。末列为固定旧teacher仅换求导窗的 OM−OO。全部来自 held 保存数组，不以不同窗口数池化。

| 组/炮 | OO ΔBCE | MM ΔBCE | MM−OO | OM−OO |
|---|---:|---:|---:|---:|
| safe/70392 | -0.208917640 | -0.210641856 | -0.001724216 | -0.000731404 |
| safe/70394 | -0.131838100 | -0.136786383 | -0.004948284 | -0.001180063 |
| safe/70404 | 0.083635306 | 0.077966160 | -0.005669146 | -0.000692549 |
| danger/69454 | 0.004062077 | -0.068719336 | -0.072781412 | -0.001413566 |
| danger/69458 | -0.008212568 | -0.007675764 | 0.000536804 | 0.000198741 |
| danger/69765 | -0.816334703 | -0.818929609 | -0.002594906 | 0.006465270 |
| danger/69768 | -0.463635980 | -0.459514788 | 0.004121192 | -0.000249885 |
| danger/69781 | -0.771117919 | -0.790663428 | -0.019545509 | -0.009691470 |
| danger/69791 | -0.177432507 | -0.185061069 | -0.007628562 | 0.001053625 |
| danger/69854 | -0.444221805 | -0.445521119 | -0.001299315 | -0.001445629 |
| danger/69855 | -0.433649297 | -0.435452946 | -0.001803648 | -0.001551252 |
| danger/69859 | -0.391778510 | -0.394942653 | -0.003164143 | 0.002652285 |
| danger/70327 | 1.481838128 | 1.497934562 | 0.016096434 | 0.000447753 |
| danger/70418 | -0.488418983 | -0.483547684 | 0.004871299 | 0.014284484 |
| danger/70861 | -0.111352781 | -0.068610742 | 0.042742039 | -0.000069158 |
| danger/71265 | -0.294524907 | -0.294463892 | 0.000061014 | 0.000110463 |
| danger/71304 | -0.142932645 | -0.170777185 | -0.027844539 | -0.002199710 |
| danger/71370 | 0.033432615 | 0.030610636 | -0.002821979 | 0.000593409 |
| danger/71415 | -0.529490508 | -0.541122231 | -0.011631723 | 0.007509547 |
| danger/74087 | -0.788028109 | -0.885724417 | -0.097696308 | 0.005657211 |

两个固定teacher的换窗均未带来危险组平均收益，且同半径不等于同动作幅度：OM−OO 的 real 范数差均值 +0.000048440、最大绝对差0.036799632；MM−MO 为 +0.000043908、最大0.035195127。未保存动作向量/目标向量，不能凭这些标量重建方向或声称独立验证了数值 v；同teacher、同context、同mean使 v 保持相同，是代码路径判断。

## Fit 复算与覆盖

每格每折 fit 的 safe/danger 全部炮均改善；这里每项格式是 safe / danger 炮均 ΔBCE。

| 折 | OO | OM | MO | MM |
|---|---:|---:|---:|---:|
| 0 | -0.245553 / -0.813727 | -0.245932 / -0.815648 | -0.243644 / -0.808341 | -0.244202 / -0.816207 |
| 1 | -0.092834 / -0.776421 | -0.093477 / -0.776961 | -0.092083 / -0.775863 | -0.092678 / -0.777169 |
| 2 | -0.165260 / -0.571873 | -0.165336 / -0.572675 | -0.164681 / -0.569979 | -0.164839 / -0.577230 |

因此不是“无法拟合”的简单失败；但 fit 效用不保证 held 安全组胜过对照。全部 fit/held、四臂、四组（含early/late）的逐炮与组汇总均由脚本重算，未只复算表中的primary字段。

实际执行 exit 0，**14,265 项检查，issues=[]**：

- 新增 **180 NPZ / 3,420 数组 / 24,363 cell×fold预测行**；旧OO **60 NPZ / 1,140数组 / 8,121行**。每格仍只有同一2,707机会，非独立新增样本。
- 新格 fit/held 逐炮组臂记录 **1,296 / 648**；旧格 **432 / 216**；合计 **2,592**。新格 fit/held 臂窗口BCE计算64,968/32,484；旧格21,656/10,828。
- 新格 **144 fit / 48 held 组汇总**和648条封存held逐炮记录全字段对照；旧格48/16组汇总及216条held逐炮记录复核。最大误差：新fit汇总6.66e−16、新held汇总8.88e−16、新held逐炮9.99e−16；gate误差0。未发现缺失、非有限数组或长度不符。
- 180份新增 cold 与同折同炮OO严格一致；全部支持掩码匹配原缓存。新增24,363行 mean-feedback范数/对应余弦精确零、logit精确等于cold，旧OO8,121行亦如此。非零控制范数匹配最大误差1.49e−8；所有保存范数满足0.1上限至float32舍入。

## 时钟、均值、抽样、哈希

**窗口/未来边界（非神经重放）：** 将缓存origin与query组成独立0..177时间轴，取25..152，对照仅单独加载的 `window.py` 纯拼接函数。2,707窗口的78点重叠全部精确一致：**4,856,358个值及同数mask**；成熟窗 **7,969,408个值及同数mask** 全部一致。绝对区间为[o−102,o+25]，未混入[o+26,o+50]。

对所有窗口分别把query最后25点换成NaN、+Inf、−Inf、−12345并翻转mask，共 **10,828个窗口扰动实例**，输出值和mask精确不变；丢弃的旧前25点另做2,707例同样通过。4个使用区非有限值用例、2个重叠值/mask错误用例均拒绝。检查对象仅是输入拼接，**不是模型输出未来不变性的GPU验证**。[window.py](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/maturity_jacobian_20261030/window.py:6)

代码 `mature_data` 只替换 x/mask，context/support与query缓存不变；`choice` 正确选取三格teacher和求导输入；动作沿选中窗口、query仍只用于后续评价/历史教学监督。[mature_data](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/maturity_jacobian_20261030/run.py:27)、[choice](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/maturity_jacobian_20261030/run.py:82)、[执行路径](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/maturity_jacobian_20261030/run.py:127)。使用query缓存的前缀来复原成熟窗不等于使用query未来后缀。

**不是 only freshness：** origin窗移25点会删除最旧25点、加入已到达25点；25不是8的倍数，因此重新分8点patch，并改变保留样本的token内位置和位置编码对应。[patch=8](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple23.py:19)、[reshape及位置编码](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple23.py:107)。两个Jacobian输出维数同为64，不由此次变化推导秩增大或可达空间包含；成熟格的reverse/donor仍保留真实成熟输入，只置换目标网络反馈，非消除全部成熟信息。

**隔离/配对：** 三折fit/held13/7、13/7、14/6，折内无交集，held恰覆盖20炮；records/folds与两级前序缓存相同，固定时钟/唯一记录通过。768个新训练步、18,432抽样位置同时匹配旧anchored及更早representation，并独立seed131重现；每批12S+12D，仅fit，零early/late/held教学样本。三个新模型mean/scale/spec与原模型严格相同，均14,316参数；从原fit上下文重新求炮等权mean，最大差1.52e−7。8,121个沿用donor索引的fit-only/异炮约束复核通过，原供体文件哈希不变；本轮不重复上一审查的全最近邻穷举。

**哈希：** 新manifest194/194、旧OO manifest136/136、原缓存输入60/60、冻结父引用6/6、当前代码/计划/测试75/75、旧调用链73/73、ZIP75/75（成员集合相同），全部匹配；旧complete.json也按新protocol所存hash复核。source checkpoint SHA256仍为 `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`。source unchanged 在本CPU审查中是**磁盘字节不变**，不是独立验证运行中内存权重。新complete不自哈希，后续replay文件不属于其原manifest。

## 复算入口与限制

[审查脚本](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/audit_maturity_jacobian_20261030.py:1) 只写stdout；复用本人的 [旧审查算术函数](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/audit_anchored_feedback_20261030.py:48)，未导入任何实验runner的summary/gate，不执行训练/神经模型前向或反向。唯一执行的实验函数是独立加载的CPU窗口拼接函数。没有GPU、原始HDF5、删除或修改封存产物。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -B /media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/audit_maturity_jacobian_20261030.py --rows
```

`--rows` 已验证输出2,592行，含cell/fold/split/shot/group/arm、before/after/delta BCE与窗口/支持数。新来源为 [新输出目录](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/maturity_jacobian_cold20_v1) 下 `{cell}_fold{fold}_{shot}.npz`，旧来源为 [旧输出目录](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/anchored_feedback_cold20_v1) 下 `anchored_fold{fold}_{shot}.npz`；NPZ第j行映射records中按原顺序筛选该shot的第j条，folds确定fit/held。公式为float64 `logaddexp(0,(1−2y)*z)`，先炮内组均值再组内炮等权，safe/early y=0、danger/late y=1，改善阈值−1e−6。

无CPU审查blocker或发现的实现bug。未重建保存标量余弦对应的参数方向，也未执行含runner的单测。主进程报告GPU重放3,420数组exact、无refit/HDF5、source unchanged；**这是主进程验证，不计入我的独立重放覆盖**。本报告不扩展实验范围，主候选依旧未过预定门槛。

