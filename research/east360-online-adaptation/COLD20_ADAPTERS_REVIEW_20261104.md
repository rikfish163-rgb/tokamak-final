# Cold20 受限适配器：CPU 实现与验证

## 范围

仅提供已合格、已冻结 `Simple23` 的独立克隆及其可训练参数列表；没有读取原始炮信号或标签，没有拟合、GPU 实验、报警变更、收益结论或新增最终风险修正。源权重由调用者加载和校验；本模块不重新读取检查点，不能自行证明传入对象的哈希。

实现：[adapters.py](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/cold20_compare_20261104/adapters.py)。测试：[test_cold20_adapters_20261104.py](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_cold20_adapters_20261104.py)。

## 接口与约束

调用 `make_adapter(base, mode)`，其中 `base` 须已由上游验证为合格模型、所有参数冻结、所有子模块处于 `eval`；不满足时抛错，不擅自改变源模型状态。返回独立 `Simple23` 深拷贝，源模型参数、缓冲区、flags 和已有梯度不被修改，且与克隆不共享参数/缓冲区存储。

| mode | 唯一可训练范围 | 标量数 |
|---|---|---:|
| `norm` | 最后 encoder 层 `norm1`、`norm2` 的 weight/bias | 256 |
| `qv_lora` | 最后 encoder 层 Q/V 各自 rank-2 的 A/B | 512 |

LoRA 通过 `torch.nn.utils.parametrize` 放在 packed `in_proj_weight` 上：`ΔQ=Bq@Aq`、`ΔK=0`、`ΔV=Bv@Av`，`alpha/r=1`。A 使用独立 CPU generator、seed `20261104`、高斯标准差 0.02 初始化，再转到原权重 device/dtype；B 严格为零。原 packed 权重、K、out projection/bias、风险头、预测头及其他参数均冻结。注册后再次 `eval()`，避免新参数化模块默认处于训练模式。

`named_trainables(model)` 返回当前 `requires_grad=True` 的 `(name, parameter)` 列表；`trainable_parameters(model)` 返回**标量总数**。主进程可从前者取参数交给自己的 optimizer；训练完成后对克隆调用 `model.requires_grad_(False).eval()`。克隆后的原源对象不参加该操作。适配器不选择训练目标、数据、阈值、步数或优化器。

LoRA 的 `state_dict` 使用 PyTorch 参数化名称，原始权重位于 `encoder.layers.1.self_attn.parametrizations.in_proj_weight.original`，四个增量位于对应 `.0.lora_*`。需要恢复时，应先用同一合格父模型创建同模式克隆，再严格加载对应适配器状态；不能把该状态当成未参数化的原始 source checkpoint。

## 实际验证

按 TDD 先写测试：模块不存在时收集失败；另先复现未冻结/非 eval 源未被拒绝，再增加边界检查。实现后，使用 `CUDA_VISIBLE_DEVICES=''`、禁用 pytest cache/插件自动加载及 Python bytecode，运行指定适配器测试和一项既有原生前向基准：

```text
43 passed in 3.27s
```

其中 42 项为新测试参数化用例，1 项为既有 source-native 零状态前向检查；不是全仓测试。测试使用经 `load_parent` 校验 SHA-256 的合格源检查点及合成输入，分别覆盖 CPU float32/float64：

- 固定 MHA fastpath=false、TF32=false 时，完整窗口和左 padding 的部分前缀上，两模式的 logit、forecast 与源模型 `torch.equal`；克隆全部子模块为 eval，参数与 buffer 无共享存储。
- 两模式分别只开放 256/512 个标量；反向时冻结参数无梯度，合成输入上的一次 SGD 步只改变允许的克隆参数，不改变源模型、buffer 或其已有梯度。此为单元测试内的合成一步，不是数据拟合。
- Q/V 的 B 首次反向均有非零梯度，A 此时梯度为零；B 的一步更新后 A 梯度非零，下一合成一步可以改变 A。构造非零 B 后，Q/V 权重严格等于原值加各自 `B@A`，K 保持不变。
- 合成一步后的两种 `state_dict` 均可严格加载到同模式新克隆，恢复后的风险与预测输出逐值相等；测试不写检查点。
- 无效 mode、错误对象、叠加参数化、未完全冻结或非 eval 源均拒绝；拒绝不改变源 flags/状态。适配器初始化不消耗调用者全局 RNG；以 meta 默认设备构造的 CPU 合成检查验证初始化显式使用源设备，而非错误依赖全局默认设备。

## 交接风险

模块不会替调用者修改全局数值后端设置。要复用本次严格零状态比较，主进程需在比较的整个进程内一致设置 MHA fastpath=false、CUDA matmul TF32=false、cuDNN TF32=false，并固定数据、dtype、设备和执行路径。本次只在 CPU 验证，不能据此声称 GPU 或其他后端的 bitwise 等价。所给适配器仅隔离参数范围；是否产生有效或安全的预警收益仍须主进程按既定盲态与对照协议实测。
