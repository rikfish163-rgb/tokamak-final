# 固定源模型的较长历史：d=1与d=2完整360对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent / tdd-workflow
- Origin Mode: run
- Origin Date: 2026-11-01
- Verification Status: PLANNED_DEVELOPMENT_DIAGNOSTIC
- Version Label: causal_span_v1

前序TR1353+已结束EAST20的结构函数审查没有支持全通道统一变慢：软X射线在首128点中相对变化更快、源asinh幅度更小，Mir_VV/v_loop的短lag变化则更慢。当前不据标签25/125比例选择5倍时间伸缩，也不从不同通道的统计比值编造物理时间常数。历史有界检索未发现当前source/cold的输入跨度对照，原TCN dilation、更新时钟、变化损失不是同一操作。

本轮只问：相同冷启动模型是否能利用更长的已到达历史？固定d=1与d=2，不扫描或选择最优d。不是统一物理标定，也没有神经炮内参数更新。它决定后续是否值得设计可学习的时间跨度模块，不能替代核心炮内收益。

沿用同一stratified_cold20_context360_v1的360身份和顺序、HDF5结局/DownTime、source SHA c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02、原密度映射、cold状态、风险头和概率门槛0.7258946895599365。前20依旧source-only，7TP6FP全计入；后340使用同一cold。EAST训练标签125ms、及时strict lead>30ms不变，>3000ms另报。

对每个实际预测端点e：d1为原始[e−127,...,e]；d2仅在e>=254时为[e−254,e−252,...,e]，不足历史时两个方法都用原d1窗口，保留原预测而非跳过端点。所有23列同刻一起取样及其原掩码，最新行保持当前真实观测。没有插值、均值滤波或修改原文件采样率/时间戳；评价仍使用原实际端点时刻。此抽取同时改变跨度、样本间距及可能的混叠，结果不能单独认证物理时间错配。以前已发生的报警不可撤销。

CPU测试先验证d1逐值等于旧window、d2精确索引/掩码、短前缀和254边界、当前值与未来扰动不变。GPU运行必须精确复现全部360的cold曲线，并精确保留所有e<254的原分数；不能放宽误差容差替代。保存源码/哈希、完整曲线和全部逐炮结果。只运行一次d2，rl_env CUDA、timeout1200秒，逐20炮报告进度，原始代码运行中不编辑。

按完整360、前20、后340报告TP/FP、J、所有救回改坏和>3000ms；新增变化必须能按实际首报解释。无论结果好坏均不替换基线、不声称炮内神经更新成功。数据已反复用于开发，不是独立未来验证。
