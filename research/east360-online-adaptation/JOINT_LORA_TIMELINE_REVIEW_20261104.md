# JointLoRA 炮内 timeline：CPU 交接与冻结

## 接口与时序

`run_shot(model, shot, mode, *, device, lr=1e-4, radius=.5)` 返回 `(probabilities, history, forecasts)`。`shot` 使用既有 signal-only `values: float32[T,23]`、`available: bool[T,23]`；其他字段不会用于计算。仅支持 `none / real / hold / reverse`，不读取本炮结局、不做 EOS 或跨炮更新。

- 每次调用深复制输入 JointLoRA；在临时副本上只开放 512 个 Q/V LoRA 参数，冻结 decoder、源模型其余参数和 W0。每炮新建 Adam、从传入模型状态重置；不写回模型或 shot。
- `q=127+32k` 单独 forward，实际产生的 168 维预测立刻 `detach().clone()` 保存；到 `t=q+8` 后才使用真实到达的八点，通过既有 `levels` 和 `direct_target` 得到真目标及 mask。不能用到达时重算的预测替换已发布预测。
- 成熟时重算 q 的可微诊断，按有效坐标 MSE 更新：real 用真实目标；hold 用零；reverse 用真实目标的相反数。三臂均使用同一个真实可用 mask。`none` 仍发布预测并计算成熟统计，但不优化。
- 有支持的更新先做 LoRA 梯度全局裁剪 1，再做 Adam，最后把**所有 LoRA 参数拼接后的位移**投影到距本炮起点不超过 `radius`；投影触发时采用很小的内缩以抵消 float32 舍入。无支持不做 optimizer step，也不推进已存在的 Adam 动量。`updated` 指投影后参数确实变化，零梯度不伪报更新。
- 事件顺序为**到达→更新→计算该时刻风险**；因此 `t` 的风险可受 t 时刻成熟反馈影响，`t-1` 及以前不可回改。风险每次都用 32 行 batch，不足的部分重复最后一个实际窗口填充并丢弃填充输出；每个窗口仍只含其端点以前的数据。此实现是已载入信号上的离线因果回放，不等于真实采集延迟认证。

## 输出约定

`probabilities` 为 `np.float32[T]`。`history` 只包含已经成熟的事件：`arrival, origin, loss_before_actual, postloss_actual, hold_mse, gradient_norm, parameter_displacement, updated, supported`；`supported` 是有效目标坐标数，`gradient_norm` 是裁剪前总范数。实际损失始终使用已发布预测/真实目标或更新后预测/真实目标，即使更新目标为 hold/reverse。没有支持时损失和梯度范数为 `None`；none 不做梯度计算，其梯度范数也是 `None`。额外记录 `issued_vs_recomputed_max_abs` 供数值审计，不用重算值覆盖发布值。

`forecasts` 含 `origin`、`released_at`、`prediction`、`target`、`mask` 五个 NumPy 数组；这里 `released_at` 表示**目标成熟索引**，未成熟为 `-1`。pending 的 prediction 已真实保存，target 为 NaN、mask 全假；它既不进入 history，也不能当成已验证预测。成熟的 target/mask 在所有臂均是真目标/真实支持，未支持坐标的 target 值不能参与误差统计。

## 集成及注意点

- runtime 必须沿既有 runner 调用 `configure()`，关闭 MHA fastpath；CPU 测试也关闭它。issue 使用单窗口、eval 和 grad-enabled dispatch，保存时立即 detach；成熟时使用相同 dispatch 重算，以减小 issued/recomputed 后端差异。timeline 记录 `issued_vs_recomputed_max_abs`，但**不对偏差设自动拒绝阈值**，主进程须审计；本审查进程没有做 GPU 验证。
- risk 的 32 行固定填充独立于 q 预测的单行 forward。合成测试覆盖短炮/长炮和改变后续信号时，已发布风险前缀逐点相等；测试不代表其他硬件的性能或实时吞吐。
- W0 与 decoder 在当前炮均被冻结；若 decoder 权重为零，仅 diagnostic loss 的 LoRA 梯度为零，history 显示 `updated=False`。这一情况不是有效的神经更新。
- 与原 source 模型一样不在本模块处理 CIII sentinel、设备缺失或固定 density map；调用方须提供已经按旧契约准备好的实际 mask/映射。禁止根据 target 结局裁剪或改写已发布风险。
- `postloss_actual` 是更新后对**同一已成熟目标**的拟合内诊断，不是新一窗口的前瞻预测误差。参数投影约束 LoRA 参数位移，但不会投影或清除 Adam 动量；本模块也不独立验证传入模型的 checkpoint 哈希、后端配置或全流告警效果。

## 验证与冻结 SHA

先运行缺失 timeline 的 RED，随后 CPU 合成测试 **27 passed**。覆盖四臂 target/mask 一致、实际发布值而非回算值、首个成熟更新点、已发布前缀不变、pending 和无支持、zero-gradient 防伪更新、LoRA-only/decoder 与 risk head 冻结、全局梯度裁剪、投影半径、Adam 不支持时不前进、非零初始 LoRA 重置、原模型及输入不变。与冻结 model 和既有 cold20 adapter 联跑 **109 passed in 7.84s**；Ruff check/format check 均通过。合成测试中的临时 Adam 步是行为回归，不是数据拟合。

- 冻结 producer `timeline.py` SHA-256：`f4c23996d3b35a808598dabe18236c36098519a5a92dc828cf81a75dd8f5abdc`。
- 测试 `test_joint_lora_timeline_20261104.py` SHA-256：`f7d4be3abaa3f7ae7a023573b783c7e27e7450f7eaffde4d3a62dda022db9211`。
- 已有冻结 `model.py` SHA-256 不变：`eaaa052179b2876de25d569f8077cda23bcb88c27f6bc168307a32ee6e288161`。
- 模型测试 `test_joint_lora_model_20261104.py` SHA-256：`146f7de65b0057836c70555597cad34a034655af3528a157a2a897f334a8623e`。

主进程另报告独立完成 67 项 CPU 测试、22 炮端到端及 CUDA 四模式合成验证；这些是主进程报告，不计入上面由本审查进程亲自运行的 27/109 项，也不等同完整 GPU 流或真实 EAST 收益。本审查进程此前写入 timeline、其测试及本审查；收到冻结指令后只更新本审查，没有修改任何 producer 或测试文件。没有读 raw HDF、访问网络或使用 GPU。此处不声明真实 EAST 告警收益。
