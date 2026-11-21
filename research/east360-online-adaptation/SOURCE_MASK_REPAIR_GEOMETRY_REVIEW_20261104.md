# 梯度几何模块：CPU 合同与边界

状态：只实现和测试数学模块；未读取原始数据、未运行 GPU、未拟合预条件器、未评估 EAST 或声称收益。source 模型、修复器和所有既有 producer/fit 均未修改。用户提供的既有静态和炮内结果是提出本假设的背景，不是本模块产生的证据。

## 接口和因果责任

```python
from experiments.risk_aligned_online.mask_repair_20261104.geometry import (
    ContextGeometry, context_from_embedding,
)

context = context_from_embedding(embedding, actual_availability)  # B x 130
geometry = ContextGeometry().to(device=embedding.device, dtype=embedding.dtype)
new_state, matrix = geometry.propose(state, gradient, context)   # B x 16, B x 16 x 16
```

调用方必须保证 `embedding` 和 `actual_availability` 来自当前已到达窗口，梯度来自真实允许的重建任务；模块没有读取未来、标签或风险分数的接口，但无法证明调用方所传张量的来源。模块也不持有 source 模型或修复器，因此冻结这些模型、监督数据边界及何时 detach 必须由后续训练/在线代码明确负责；本模块不暗中 detach，允许训练信号经过 `propose` 反向传播。

每个有效帧定义为该帧 23 通道中至少一个真实可用，patch 由 8 帧组成且至少有一个有效帧时才参与 embedding 均值。对每个样本按时间顺序选最早与最晚各 4 个**有效** token，若少于 4 个则两端都使用现有有效 token；全 padding 的样本显式报错。最后两维是 CIII/MNM 实际可用帧数除以该样本有效帧数，padding 不进入分母。该表示不把估计值当作物理观测。

## 数值合同

- 上三角按 `torch.triu_indices(16,16)` 顺序有 136 项，经 `.125 * tanh` 形成对称 S；对角只计一次。模型是 LayerNorm130 → Linear130×32 → SiLU → Linear32×136，最后层权重与偏置均零初始化，共 8,940 参数。
- 计算 `torch.linalg.matrix_exp(S)`，再对其和转置取平均以消除浮点矩阵指数的非对称舍入；数学上不改变对称 S 的指数。每项绝对值不大于 0.125，因此理论上 S 的谱在 [-2, 2]，M 为 SPD 且谱在 [exp(-2), exp(2)]。浮点测试允许相应的舍入误差。
- `propose` 只用 `M @ g` 作为下降方向，没有独立状态偏置或风险头。方向及 L2 范数以 float64 计算；方向范数小于 `1e-8` 时按既有规则保持输入 state，否则步长为 0.025，逐样本投影至半径 0.25。投影时对上界留 4 个 dtype epsilon 的余量，并复核转换回原 dtype 后的范数：若仅因转换舍入而越界，再向内调整。输入 state 必须已在半径内，否则显式拒绝；这比旧代码允许的 `0.2500001` 检查容差严格。在旧实现因 float32 舍入产生超半径 state 的极端边界上，新结果有意略有不同，以满足硬半径合同；普通单位初始化及已有投影测试仍逐值匹配旧步骤。
- 支持 float32 和 float64；shape、设备、dtype、finite、空 batch、全 padding 及 state 半径均检查，不以其他 dtype 或设备暗中替代。无数据的 `meta` 张量明确拒绝。真实 GPU 尚未验证，后端对 `matrix_exp` 不支持时异常会直接暴露。
- 零末层确保初始 M 与单位矩阵逐值相同，136 个对称槽的末层参数均有非零梯度；但数学上此时 M 对输入 context 的导数必然为零。末层获得非零权重后，测试确认可向 context 回传有限非零梯度；不能把初始化瞬间的 context 导数误述为非零。

## TDD 与验证

先写测试，模块不存在时得到预期 RED：`1 error in 1.26s`。随后增加全 `meta` 设备边界测试，得到 `2 failed, 10 passed in 3.18s`，加入显式拒绝后转绿。另一个确定性的 float32 投影舍入边界测试先失败，随后补充转换后复核并转绿。测试覆盖旧规则的单位初始化逐值一致、小梯度与零梯度 no-op、matrix exponential 对照、float32/float64 SPD 和谱界、所有上三角槽梯度、预条件方向、可训练性、batch 独立、投影半径、输入不变及非法输入拒绝。实际最终测试输出以交接回复为准。
