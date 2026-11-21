# 下载 EAST360 逐炮结果与历史试验记录

[GitHub Release：EAST360 研究快照](https://github.com/rikfish163-rgb/tokamak-final/releases/tag/east360-research-snapshot-20261117)

直接读仓库里的[研究说明](README.md)、[主结果图与绘图数据](outputs/east360_main_nature_20261117/README.md)或[逐炮图谱](outputs/east360_row5_70320/README.md)即可了解主结果。Release 附件保存不适合放入 Git 的封存预测、逐炮评分、对照与负结果的明细、曲线、代码快照及大文件。

## 按路径查找

Release 内的 `results-index.jsonl.gz` 按研究目录中的相对路径记录文件去向：`git` 在本分支，`release` 在记录所列的 `asset` 中，`local_*` 留在本地。可以只下载包含所需结果的分卷。比如查逐炮绘图 ZIP 或 2,593 炮扩展：

```bash
gzip -dc results-index.jsonl.gz | grep -E '逐炮绘图数据|commissioning_frozen_extension_east2593'
```

每个分卷都是单独可解包的 tar.zst，无需拼接。把命令里的文件名换成索引指向的附件名：

```bash
tar --zstd -xf east360-research-snapshot-20261117-results-01.tar.zst
```

解包后的根目录为 `east360-research-snapshot-20261117/`。`field-changes.jsonl.gz` 记录哪些公开副本省略了字段；如果原文件因序列化中断而不完整，公开副本只记录失败位置，不作为有效成绩。原始路径、字段含义和结局口径见各实验协议。

## 公开范围与重放条件

原始 HDF5、模型权重、训练输入、训练缓存和本机环境保留在本地。混合文件只发布可公开的结果部分；例如逐炮绘图 ZIP 可用于读取封存风险分数，但省略实测输入及部分拟合系数，不能直接重画依赖这些字段的内部波形图。可查看已导出的图谱，或用已提供 CSV 完整重画 [Nature 风格主图](outputs/east360_main_nature_20261117/README.md)。完整模型重放仍需[外部资产](EXTERNAL_ASSETS.md)。

该快照归档既有实验，不把 360 炮开发回放视为独立验证，也不改变 2,593 炮扩展在主阈值上误报超限的结论。
