# 零冷启动跨装置破裂预警

本包从首炮开始运行冻结的 J-TEXT 源模型，通过因果前缀统计对齐和炮后反馈适配 EAST，不要求先积累 20 炮。

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
python -m zero_cold_start.experiments dev360
python -m zero_cold_start.experiments indep2593
```

评估命令会覆盖对应的本地结果 JSON。

## 状态边界

- 炮内只使用已到信号，建立本炮对齐统计、初始风险参照和报警状态，不修改跨炮参数。
- 炮间在结局揭示后更新目标统计记忆和相对倍率。先诊断，再用同炮反事实回放决定是否提交倍率更新。
- 骨干权重冻结。最终阈值在初始参照形成后固定于本炮，并在不同炮之间随参照和反馈改变。最终版本没有持续移动的炮内风险基线。
- 破裂标签取自原 HDF5 `meta/IsDisrupt`。只有破裂炮的 `meta/DownTime` 用作破裂时刻。
- `AdvancedTime` 仅从 shotlist 读取，缺失值不补造。严格提前超过 30 ms 的及时率与逐炮 `AdvancedTime` 对比是两个指标。
- DBC 排序使用初始电流平方代理，不是经设备损伤系数标定的实际破裂预算消耗。`dbc_asc` 与 `ip_asc` 是同一排序。

## 封存结果与限制

[开发集对照](results/dev360_locked_comparison.json)包含 360 炮开发流。[扩展集评估](results/final_holdout_indep2593.json)包含与开发集互斥的 2,593 炮历史 EAST 数据。模型仅在 J-TEXT 上训练，扩展集评估中仍允许预测后使用已结束炮的结局更新下一炮参数，因此这是序贯在线评估，不是全程无目标标签测试。

主反思方案按炮号排序，在扩展集取得及时 TPR 58.74%、FPR 17.52%、AUC 0.8501。电流平方代理升序取得及时 TPR 60.81%、FPR 15.75%、AUC 0.8684。排序实验是历史回放，不代表真实装置可以重新排列已经发生的放电。

这些结果不证明现场部署可靠性，也不证明反思在所有误报预算下优于盲更新。扩展集上盲更新的 FPR 更低，主反思方案的及时召回更高。开发集和扩展集分别只有 26 与 395 炮破裂炮有 `AdvancedTime`，不能把其子集结果推广到全部炮。
