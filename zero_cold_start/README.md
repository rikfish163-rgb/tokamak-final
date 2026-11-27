# 零冷启动跨装置破裂预警

本包从首炮开始运行冻结的 J-TEXT 源模型，以有限炮数校准装置尺度，通过每炮独立的风险参照和炮后倍率反馈适配 EAST，不要求先积累 20 炮。

## 运行

运行目录为 Git 仓库根目录，即本地的 `Fuison-final/code/`。需要 Python、NumPy、h5py、scikit-learn 和 PyTorch。当前实验使用 Python 3.12、PyTorch 2.7.1 和 RTX 3070，未修改仓库原有复现主线。

原始数据和 shotlist 放在 Git 仓库的上一级目录：

```text
Fuison-final/
    code/zero_cold_start/
    data/{EAST,J-TEXT}/unified_hdf5/<shot_group>/<shot>.hdf5
    shot_list/EAST/AdvancedTime_EAST.json
    shot_list/J-TEXT/AdvancedTime_J-TEXT.json
```

模型权重和 `artifacts/east_meta.json` 不上传 Git。运行评估前，将本地源权重放到 `artifacts/jtext_source.pt`。也可按锁定划分重训源模型：

```bash
python -m zero_cold_start.backbone --seed 133 --epochs 40 --patience 20
```

这条命令会覆盖本地 `artifacts/jtext_source.pt`。重训权重不保证与封存实验权重逐字节一致。现有封存权重 SHA256 为 `45b17a4d56b710c8ba349af53778c175a46112776ed1525dcd508953bb1d51e7`。

运行自检和评估：

```bash
python zero_cold_start/engine.py
python -m zero_cold_start.experiments one_shot
python -m zero_cold_start.experiments one_shot_extension
python -m zero_cold_start.experiments dev360
python -m zero_cold_start.experiments indep2593
```

评估命令会覆盖对应的本地结果 JSON。

## 状态边界

- `one_shot` 在归档 360 炮上运行。首炮沿用源模型原输入约定（含电流单位换算），计入评估；首炮预测完成后，用该炮开头 200 ms 内的有效样本（破裂前至少留出 250 ms，正常结束前留出 50 ms）的中位数与四分位距估计六个通道的映射，后续整流冻结。首炮缺测或尺度无法估计的通道保持单位映射。每炮前 100 ms 用于估计风险参照。开发对照比较分数倍率（当前分数达到本炮参照乘以 M）与原有的赔率倍率，并搜索初始倍率与持续确认时间，以 FPR 严格低于 20% 为约束选及时 TPR 最高项。
- 以下为先前封存的 `dev360` 和 `indep2593` 配置。炮内只使用已到信号，建立本炮对齐统计、初始风险参照和报警状态，不修改跨炮参数。
- 炮间在结局揭示后更新目标统计记忆和相对倍率。先诊断，再用同炮反事实回放决定是否提交倍率更新。
- 骨干权重冻结。最终阈值在初始参照形成后固定于本炮，并在不同炮之间随参照和反馈改变。最终版本没有持续移动的炮内风险基线。
- 破裂标签取自原 HDF5 `meta/IsDisrupt`。只有破裂炮的 `meta/DownTime` 用作破裂时刻。
- `AdvancedTime` 仅从 shotlist 读取，缺失值不补造。严格提前超过 30 ms 的及时率与逐炮 `AdvancedTime` 对比是两个指标。
- DBC 排序使用初始电流平方代理，不是经设备损伤系数标定的实际破裂预算消耗。`dbc_asc` 与 `ip_asc` 是同一排序。

### 新实验的两种倍率

`score_ratio` 从 100 ms 起使用 `M × max(本炮初始参照, 0.0001)` 作为报警线。初始参照由前 100 ms 分数的 logit 中位数反变换得到。分数范围为 [0, 1]，若该线大于 1，本炮余下时段无法越线。前 100 ms 沿用冻结源模型的阈值。M 只在结局揭示后按同炮回放证据更新。

`logit_margin` 对赔率施加倍率，并叠加初始波动裕量与上下限。两者使用相同的冻结风险头，开发结果均包含第一炮。持续确认时间以毫秒计。文献中的初始重建误差与本工程的分类分数有不同的值域，倍率需分别验证。

一炮冻结映射实验未通过扩展流验收。在开发 360 炮的 23 个工作点中，按 FPR < 20% 最大化及时 TPR，选出分数倍率初值 909、持续 10 ms。开发流及时 49/180、误报 31/180；同一配置从独立的首炮重新启动，在扩展 2,593 炮上及时 885/1,018、误报 1,487/1,575。其中 1,466 个误报炮在参照形成后的报警线高于 1，所以首报已发生在启动段。两次回放使用各自的第一炮估计装置尺度。详见[开发结果](results/dev360_one_shot.json)和[扩展结果](results/extension2593_one_shot.json)。

## 封存结果与限制

[开发集对照](results/dev360_locked_comparison.json)包含 360 炮开发流。[扩展集评估](results/final_holdout_indep2593.json)包含与开发集互斥的 2,593 炮历史 EAST 数据。模型仅在 J-TEXT 上训练，扩展集评估中仍允许预测后使用已结束炮的结局更新下一炮参数，因此这是序贯在线评估，不是全程无目标标签测试。

主反思方案按炮号排序，在扩展集取得及时 TPR 58.74%、FPR 17.52%、AUC 0.8501。电流平方代理升序取得及时 TPR 60.81%、FPR 15.75%、AUC 0.8684。排序实验是历史回放，不代表真实装置可以重新排列已经发生的放电。

这些结果不证明现场部署可靠性，也不证明反思在所有误报预算下优于盲更新。扩展集上盲更新的 FPR 更低，主反思方案的及时召回更高。开发集和扩展集分别只有 26 与 395 炮破裂炮有 `AdvancedTime`，不能把其子集结果推广到全部炮。
