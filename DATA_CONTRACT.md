# Data contract

这个仓库不包含原始数据。用户需要把数据目录放在仓库外，然后通过 `--data-root` 或 `FUSION_DATA_ROOT` 指向它。这样 GitHub clone、论文归档和本地大数据盘可以分离。

## Required layout

```text
<data-root>/
├── J-TEXT/unified_hdf5/<bucket>/<shot>.hdf5
└── EAST/unified_hdf5/<bucket>/<shot>.hdf5
```

其中 `bucket = (shot // 100) * 100`。例如炮 `12345` 位于 `unified_hdf5/12300/12345.hdf5`。

仓库根据以下固定清单检查所需炮号：

- J-TEXT：`splits/train.txt`、`splits/val.txt`、`splits/test.txt`，分别固定为
  `1353（188D/1165N）`、`160（80D/80N）`、`220（110D/110N）`，三者按炮号
  零交叉；
- EAST support：`config/splits/east_paper131_split_all_in.json` 的固定110炮；清单名义
  分组为10D/100N，但当前权威 HDF5 结局是15D/95N，训练与评估必须以后者为准；
- EAST 方法开发流：`config/splits/east_hdf5_authoritative_balanced360_v1.json` 的 test 炮。

`audit` 默认快速检查每个装置的首尾炮；`audit --full` 检查全部清单炮号，并打开每个 HDF5。缺炮、打不开或字段不完整时命令非零退出。

## HDF5 authority

每个文件至少需要：

```text
data/ip
meta/IsDisrupt
meta/DownTime
```

完整审计还要求当前 raw23 读取路径用到的 23 个通道：

```text
Bt, CIII, MNM, Mir_VV, Mir_avg_amp, Mir_avg_fre, Z_proxy,
dx_a, dy_a, ip, mode_number_n, n=1 amplitude, ne_nG, qa_proxy,
sxr_kurt, sxr_mean, sxr_skew, sxr_var, v_loop, xuv_kurt,
xuv_ratio, xuv_skew, xuv_var
```

`meta/IsDisrupt` 和 `meta/DownTime` 是结局标签的唯一权威来源。炮表 JSON、CSV、AdvancedTime 和 onset 文件不允许覆盖 HDF5 结局；它们只在相应研究代码显式请求时作为辅助信息使用。

J-TEXT 源模型阈值只由 VAL160 选择，TEST220 只评估一次。破裂炮只有第一次因果
越阈严格早于 `DownTime` 超过 10 ms 才计 TP；恰好 10 ms 不计。短炮只允许复制
已经观测到的最早历史值作左补齐，不能读取当前端点之后的数据。

## Leakage boundary

数据审计不把标签复制到输入特征。在线运行要求：

1. 当前炮先只读因果观测，完成预测并写入 seal；
2. seal 完成后才允许按查询预算揭示当前炮结局；
3. 只有揭示后的标签才能用于炮间更新；
4. 更新后的模型只服务后续炮。

静态 360 结果和在线 360 结果分两个阶段保存；在线阶段从相同的迁移父模型重置，避免把静态推理状态带入在线更新。

## Data audit output

`fusion-reproduce audit --full` 输出的数据审计 JSON 包含：数据根目录、预期炮数、打开的 HDF5 数、文件大小摘要哈希和标签权威声明。该快速哈希只用于发现明显的清单变化。

论文归档或跨机构复现应额外运行：

```bash
fusion-reproduce audit --full --content-hash --portable \
  --data-root "$FUSION_DATA_ROOT" \
  --output required_hdf5_content_manifest.json
```

该模式对每个必需 HDF5 的原始字节计算 SHA-256，并输出相对路径、文件大小、逐文件哈希和排序后的聚合哈希。`--portable` 把机器本地数据根替换成 `<external-data-root>`，使提交到仓库的 manifest 不暴露工作站路径。它可以验证受控访问数据是否与论文版本逐字节一致，但仍不替代数据 DOI、访问许可或原始数据本身。
