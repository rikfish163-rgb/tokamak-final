# 更新后风险目标：同起点对照结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent + results-analysis
- Origin Mode: run
- Origin Date: 2026-11-07
- Verification Status: ANALYZED
- Version Label: source_meta_objective_v1

## 结论

加入更新后风险损失后，没有获得独立炮内收益。新旧训练目标得到的两个副本，在完整
EAST 360炮上各自都是静态118个及时检出、30个误报；开启炮内更新后数字不变。
原达标模型为116／29，开启同一炮内更新也不变。不能把额外源训练带来的2个检出
归因于炮内适应，更不能把它作为物理规律学习的证据。

保留的旧首层适应参考仍为120及时／33误报，满足FPR≤.20且比静态多4个及时，
但J不变。本轮新副本没有超过这一参考；“本轮无新增收益”不等于否定已有的小幅运行收益。

## 为什么做这一组

原`gradient_aligned512_20260929`协议声明after-risk权重0.5，但512步日志全部符合
`before + 0.20 signal + 0.10 alignment`，最大标量残差4.17e−8。半before、半after
公式只1步在1e−7内吻合。当前代码已经包含after项；旧日志不足以证明旧反向图缺项，
所以本轮用新运行而非推测来检验这一解释。

本轮在`mode=aligned`内只改变after权重0／0.5。两组都从原合格权重副本起步，
固定512步、batch16、AdamW lr1e−5、seed131、signal系数0.20、alignment系数0.10。
原始J-TEXT TRAIN1353及`max(25 ms, AdvancedTime)`标签不变，不凭TEST挑轮次。
两组保存的同批次记录一致；原达标权重没有覆盖。

新增回归直接捕获反向传播标量与裁剪前梯度，验证两组梯度之差确为
`0.5 * grad(after - before)`，不是只改日志。包括原有测试在内，17项定向测试通过。

## 完整EAST结果

180破裂／180安全，包含原前20炮。两组共用原炮序、冷启动输入映射和阈值
0.7258946895599365；先封存全部预测，再用原HDF5结局评分。及时首报严格>30 ms。

| 父模型 | 预测方式 | 及时检出 | 误报 | TPR | FPR | J |
|---|---|---:|---:|---:|---:|---:|
| 原达标模型 | 静态 | 116 | 29 | 0.644444 | 0.161111 | 0.483333 |
| 原达标模型 | 炮内 | 116 | 29 | 0.644444 | 0.161111 | 0.483333 |
| before目标副本 | 静态 | 118 | 30 | 0.655556 | 0.166667 | 0.488889 |
| before目标副本 | 炮内 | 118 | 30 | 0.655556 | 0.166667 | 0.488889 |
| after目标副本 | 静态 | 118 | 30 | 0.655556 | 0.166667 | 0.488889 |
| after目标副本 | 炮内 | 118 | 30 | 0.655556 | 0.166667 | 0.488889 |

每个父模型另跑同幅错序反馈和仅分数偏移对照，均与自身静态有相同炮级结果。
原模型30–3000 ms检出100，新副本102；>3000 ms均16。所有臂满足0.20上限，
均不满足固定阈值下0.10／0.15上限，未通过阈值回扫改变这一结论。

## 源模型与实际更新

J-TEXT TEST220按严格>10 ms核算：原模型100/110及时、13/110误报；两副本在固定原阈值
均为99/110、13/110。若使用各自新VAL阈值，则两副本均为98/110及时，低于0.90要求。
这些副本没有自动替换原基模，也没有继承它的达标声明。

部署使用与源训练一致的tokens／normalized／直接H25接口。每个反馈从原临时状态重建，
H25真实信号全部到达后才更新，首次影响origin+26；不是第一FFN方向实验，也不累计炮间状态。
每个父模型均执行43372次反馈更新，后340炮全部发生概率变化，但没有有利或有害的炮级翻转。

| 父模型 | 每炮最大概率变化的中位数 | 全流最大变化 | 未及时检出的后续破裂炮 | 这些炮的门槛差中位数 |
|---|---:|---:|---:|---:|
| 原模型 | 0.000262 | 0.002684 | 63 | 0.399036 |
| before副本 | 0.000310 | 0.003714 | 61 | 0.418561 |
| after副本 | 0.000310 | 0.003706 | 61 | 0.418852 |

这里的门槛差是在严格>30 ms区间内，取静态最高风险点计算的差值，属于事后诊断，
不是在线查询依据或全模型可救上界。该结果支持“当前更新作用很小”，不证明所有更强
的内部适应都不可行。

## 因果访问与可复用产物

9个真实截断见证覆盖三个父模型，概率最大差3.28e−7；原静态360复现116／29。
GPU封存完成耗时136.64s。全部风险网络在炮内回放过程中保持不变，仅临时状态更新。
没有EAST标签拟合、炮间更新或报警参数调整。

- 源拟合：`probe_outputs/source_meta_objective_20261107_v1/{before,after}/`
- 完整封存、原HDF5评分与逐炮结果：`probe_outputs/source_meta_native360_20261107_v1/`
- 执行：`/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.joint_meta --checkpoint experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt --output probe_outputs/source_meta_objective_20261107_v1/before --mode aligned --risk-after-weight 0 --fast-rule normalized --steps 512 --batch-size 16 --lr 1e-5`；after臂将输出目录改为after，权重改为0.5，仍读取原checkpoint。
- GPU预测：`/home/hetaisheng/miniconda3/envs/rl_env/bin/python run_source_meta_stream_20261107.py predict probe_outputs/source_meta_native360_20261107_v1`
- CPU评分：`CUDA_VISIBLE_DEVICES='' /home/hetaisheng/miniconda3/envs/rl_env/bin/python run_source_meta_stream_20261107.py score probe_outputs/source_meta_native360_20261107_v1`

## 局限与下一项

本轮仅一个固定种子，且使用已参与开发的数据，不是独立未来确认。它排除“补上after项
就能解决现有炮内失败”的简单解释，不证明源训练日志完全无关。

已复核`sequence_fast.py`、`sequence_train.py`及`decoder_intra_20261005/whole_meta.py`：
连续多窗口、真实信号反馈、截断反向传播等并非尚未尝试的新构想，不直接重做。
下一项检验预测头读取位置：风险头读CLS64维，而原预测头读17×64维；以同参数量的
CLS／末端data-token预测头做结构对照，保持原风险模型不变。不再扫描本轮的after权重。

## 方法依据

[MT3（AISTATS 2022）](https://proceedings.mlr.press/v151/bartler22a/bartler22a.pdf)
区分了自监督内层适应和评价适应后分类效果的外层学习。本轮只是时序机制对照，
不是其BYOL图像算法复现；原文不构成EAST有效性或物理因果性的证据。
