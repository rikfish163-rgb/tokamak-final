# Locked split manifests

当前发布主线只使用两个 EAST manifest：

- `east_paper131_split_all_in.json`：固定 110 炮 EAST support；JSON 的历史名义分组是
  10D/100N，但当前 HDF5 `meta/IsDisrupt` 的有效计数是15D/95N。JSON只固定炮号，
  不覆盖 HDF5 标签；
- `east_hdf5_authoritative_balanced360_v1.json`：按炮号排序的 360 炮方法开发/在线流（180D/180N，标签最终以 HDF5 为准）。

运行时由 `experiments/transfer_methods_2026/common.py` 读取 support 和 360 roster；`reproducibility/data_contract.py` 只负责检查这些炮号在外置 HDF5 数据中存在且可读。

本目录不再保留旧的 F110/F220/旧 paper131 测试入口。需要改变炮表时，必须新建版本化 manifest 和新的参考结果，不覆盖这两个锁定文件。
