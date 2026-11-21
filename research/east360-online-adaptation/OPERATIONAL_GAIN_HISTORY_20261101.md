# 操作性目标与正倍率输入：有界历史核对

2026-11-01，主控只读核对。此前派出的Terra/max子代理后续工具返回not_found且没有留下复核文件；下列内容由主控读取实际代码和结果整理，不冒称子代理已完成审查。没有重新运行历史模型。

## 已经做过的相近实验

1. `MAIN/experiments/risk_aligned_online/runs/shot_objective_alignment20_test360_20261023/completed.json`：外部20炮4D16S，训练encoder，固定风险头和阈值，源回放加目标mean/safe_peak/peak。完整360中原静态117TP/27FP；mean静态142/68，safe_peak静态129/53，peak静态111/46。其各自炮内结果为142/74、131/61、116/53。它证明整炮峰值目标已经被试过，不能说本轮首次提出；也不能说仅改峰值目标就已经改善。
2. `MAIN/experiments/risk_aligned_online/runs/supervised_density_gain20_test360_20261023/completed.json`：相同外部20炮背景下训练一个正密度倍率，风险网络不动。gain从约0.00036155变为0.19436；静态117/27变为141/88。随后forecast头单独准备，炮内140/90。它不是9参数结构倍率，也不是当前流前20准备；它已经否定“给密度倍率监督就会自然解决误报”的简单推断。
3. `WT/probe_outputs/operational_transport_east360_v1/complete.json`（注意名为complete而非completed）：前20为17D3S，学习把已到达预测梯度映射为任务方向，临时Q/K改动。完整360 cold125/32，operational125/32，point124/31；reverse125/32。该路线没有产生可靠炮内报警收益，不等于本轮原始输入正倍率拟合。

上述源父SHA均见其对应protocol，为c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02。第三项顺序含未经认证Bt的旧排序，不能同当前仅电流分层准备的数值直接相减作归因。

## 本轮差异与不能声称的内容

当前`operational_input_gain_v1`使用同一stratified前20（8D12S）及其成本计入的完整360，固定源网络，比较1个密度参数与9个正倍率参数、平均与峰值两种目标。没有源回放更新encoder，没有外部EAST准备炮，没有炮内更新。新的是这个受限参数空间和准备合同的配对实验，不是峰值损失本身新颖，更不是整套目标已经实现。

9个参数作用于10列；电流、磁场、qa_proxy、v_loop、Z_proxy、频率、模数、偏度、峰度、比值不拟合。SXR mean乘g/var乘g²只是共同响应增益的声明假设，不能认证原始仪器关系已重建。

主控对50个源TRAIN炮和已结束前20目标炮抽查Z_proxy与v_loop/Ip：相对误差中位数分别约0.14279与0.04582，并不严格相等；未据名称强行绑这两列。该小范围探查不证明所有文件的定义，更不把两者不相等说成原数据错误。

本轮历史检索是有界的，不保证其他目录、压缩快照或已移除产物中没有类似方法。未做新的文献首创性声明。
