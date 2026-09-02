# Adv25 EAST-success 静态源模型 bundle

该目录只保存三颗用于当前 EAST `shot-first` V3C 诊断谱系的 J-TEXT 静态
源模型 checkpoint。它不是新的训练结果，也没有读取 TEST220 或 balanced360
来选择模型。

| seed | 来源 checkpoint | 内容 SHA-256 |
|---:|---|---|
| 131 | `v3c_adv25_epoch3_20260830/seed131/source/source_checkpoint.pt` | `1ab6b8e3b30992400ea6df8ffe235cb9d0023730f5a0c77d4de0f087ae85912d` |
| 133 | `v3c_adv25_epoch3_all_20260830/seed133/source/source_checkpoint.pt` | `c8c5945f58a938d9c7d8197650b01b9dacf2be7cf0ae317825f422d1283142da` |
| 137 | `v3c_adv25_epoch3_all_20260830/seed137/source/source_checkpoint.pt` | `5183dc09239409325dd4f10fde8a6af4b6da4e28accc9ff88b11ff07a4d826cf` |

三个 checkpoint 的训练标签合同都是
`H_JTEXT=max(25 ms, AdvancedTime)`，结局权威为 J-TEXT HDF5 的
`meta/IsDisrupt` 与 `meta/DownTime`。复制到本 bundle 后只做文件级 SHA 校验；
任何内容改变都必须新建 bundle 和新谱系，不能覆盖旧目录。
