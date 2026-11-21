# JointLoRA 模型部件：CPU 交接

## 实现与接口

新增 `experiments/risk_aligned_online/joint_lora_20261104/model.py`，不生成 target、标签、loss、优化器或训练日程。调用方负责按既有 SHA 加载并冻结合格 J-TEXT source，提供已确定的 `initial_forecaster`：

```python
model = JointLoRA(frozen_source, initial_forecaster)
result = model(x, availability)
# result: logit [B], diagnostic [B,168], key [B,70], latent [B,192]
fast_scope = model.neural_parameters()       # 4 tensors, 512 scalars
decoder_scope = model.decoder_parameters()   # 2 tensors, 32,424 scalars
```

- 通过既有 `make_adapter(base, "qv_lora")` 深复制 source，仅打开最后 encoder 层 Q/V rank-2 LoRA。原 source、原 risk head 和原生 25 步 forecast head 的参数均不被改写；新诊断 decoder 不在 risk 路径上。LoRA 的初始 B 为零，故零初始化时风险与 source 精确一致；LoRA 更新后，由于共享 encoder 表征变化，风险和原生 forecast 的输出都可能变化。
- `W0` 是与调用方张量不共享存储、无梯度、注册在 state dict 中的 float32 buffer，shape `[168,70]`。新 `Linear(192,168,bias=True)` 的 weight/bias 均为零；诊断输出为既有 `dynamic_key(levels(clone,x,a),a) @ W0.T + residual_decoder(latent)`。168 是 8×21 直接诊断坐标，并未将 source 原有 horizon 从 25 改成 8。
- clone 每次 forward 仅调用一次。局部 hook 捕获 `clone.norm` 的 `[B,17,64]` 输出后立即移除，异常亦通过 `finally` 移除。Simple23 的 CLS 在序列末尾，latent 顺序严格为 `norm[:, -1]`、`norm[:, -2]`、`norm[:, -2]-norm[:, -3]`。返回的 latent 保留必要的 autograd 图；模块本身不缓存本次图。
- `model.train()` 允许参数求梯度，但始终将 source clone 保持在 eval；两个参数 scope 在调用方临时冻结后依然能返回对应参数，便于分开安排 LoRA-only 和联合训练。两个 scope 不包含 `W0` 或原 source 参数。

## 主进程集成边界

1. **零 decoder 的首步梯度**：初始化时，仅以 diagnostic loss 反向传播，decoder 权重可收到梯度，但其对 latent 的导数为零，Q/V LoRA 的 diagnostic 梯度也为零。decoder 权重非零后，测试确认 diagnostic 梯度可到达 Q/V 的 B。另一方面，risk loss 在零初始化时即可传入 Q/V 的 B；由于 B 初始为零，LoRA A 的首步梯度也为零。这是初始化的数学后果，不是已验证的学习收益；如何进行 warm/EOS 及炮内更新属于主进程。
2. **捕获点不是后处理 token**：`norm` hook 位于源模型后续的 fast-origin 仿射和无效 patch 清零之前；若末两个 patch 全缺失，其 hook 输出仍可能非零。测试保留了这种既有源模型行为，不能把 latent 非零解读为那些位置有真实观测。
3. 本实现严格调用现有 `levels`：availability 为真的非有限值会报错；原 `Simple23` forward 自身会将非有限值视为不可用。调用方应按已有数据契约正确标记缺失，不能把两者的容错语义混同。
4. `W0` 不被优化器发现、无存储别名，并且模型 forward/测试优化步不修改它；但 PyTorch 的公开 buffer 并非语言级只读，外部原地写入或显式加载不同 state dict 仍可更改它，主训练器需保护并核验固定值。
5. hook 是短生命周期，但同一 `model` 实例的并发或嵌套 forward 可能交叉触发彼此的 hook；当前接口应按实例串行调用。返回的图由调用方正常释放。构造零 decoder 时 `nn.Linear` 的常规初始化会先消耗全局 RNG 再清零，公平对照应在模型构造后明确设置训练采样随机种子。
6. 构造器检查 actual cfg 为 dim64、2 层、4 heads、W128/P8、source view、token fast site、normalized fast rule、原生 horizon25，并检查 embedding、头和注意力几何、source float32 scaler/参数、冻结/eval 及 W0 的 shape/dtype/有限性；**不能仅凭内存对象证明 checkpoint 哈希**，调用方仍须执行合格 checkpoint 校验。GPU、autocast 和真实训练流未在本任务验证。未修改旧 density map、共同代码或任何现有 producer。

## 验证

- TDD 先看到目标模块不存在的 RED，再实现；使用 `load_parent(DEFAULT_PARENT)` 的 SHA 锁定合格 source、合成输入和 CPU，无 raw 数据或 GPU。
- 本模块回归 **40 passed**：包含 full/partial-prefix、autograd enabled 和 disabled 的零 LoRA source 风险精确相等；原 forecast 相等；零残差诊断等于 `W0@key`；decoder 修改不改 risk/key；参数及 buffer 存储不共享；优化器一步只修改 LoRA 且原 source 数值、flags、已有 grad 不变；零/非零 decoder 梯度边界；hook 顺序、调用次数、异常清理及图释放；缺失末尾 patch；严格 state_dict 往返和无效输入拒绝。
- 与既有 cold20 adapter、direct-memory key/model/timeline 联跑 **140 passed in 2.99s**；Ruff check 和 format check 通过。测试中的单次合成优化器步只是隔离/梯度回归，不是数据拟合或风险实验。
- 模型 SHA-256：`eaaa052179b2876de25d569f8077cda23bcb88c27f6bc168307a32ee6e288161`；测试 SHA-256：`146f7de65b0057836c70555597cad34a034655af3528a157a2a897f334a8623e`。未写 `__init__.py`，主进程负责集成与 GPU 调度。

这是可训练的内部 Q/V 权重路径与独立诊断 decoder 的实现验证；尚无任何本部件带来 EAST 告警收益或学到物理规律的证据。
