# Joint LoRA 恢复状态辅助模块简评

2026-11-04。模块和测试已冻结；没有读取真实 HDF、运行 GPU 或修改原 producer。此模块只恢复和验证状态，不代表完整流已经恢复或完成。

**接口现状须注意：** 根据随后被更改的前一条需求，当前 `state.py:88–98` 允许已揭示 prefix 末尾最多一炮尚无 EOS 日志；它不会训练该炮。用户最新要求 main 将 prefix 截到最后一个已完成 EOS，待处理炮由 runner 单独补一次：这一调用方式可用，但 helper 自身目前并不拒绝单个 pending。冻结后本审查没有再改代码；不要把 helper 误述为强制严格等长。

- `restore_warm` 接收已解析的 warm receipt，读取 `W0`、严格加载 state，并校验拟合状态 digest 与冻结 source 一致；main 仍需核对 warm 的前 20 炮身份和源文件可信性。
- `replay_updates` 按日志逐炮校验顺序、parent digest、当时已结束的真实记录、实际使用的记录集合及每步预算，再以原 `train_schedule` 和 `lr=1e-4` 重放；结果 digest 不同即报错，没有容差或回退。返回的 slow 是独立副本。日志和 receipt 的自洽性不等于外部不可篡改证明。
- 与原 GPU 运行逐字节一致需要 main 使用原设备与数值后端配置；本审查未进行 CUDA 验证。若最后一炮 pending，必须在消费下一炮前由 runner 显式且仅一次补 EOS。
- 测试先因缺少模块 RED；补充 pending 需求的测试也先 RED。最后一次 CPU synthetic 测试输出 **32 passed**；main 的集成重跑应以以下冻结哈希为准。本次测试不创建 HDF。

冻结 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `experiments/risk_aligned_online/joint_lora_resume_20261104/state.py` | `0d75ceebc8122f0f04d3b33e79fb8a31626912185b0d511742c9917857620a45` |
| `tests/test_joint_lora_resume_state_20261104.py` | `48957d1b9dce97da57251ac56fcf05bbd434f5999859358fcbd3725e512c37c1` |
