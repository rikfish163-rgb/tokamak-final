# 首次匹配辅助任务拟合的固定预算

2026-11-04，在本次拟合开始前固定。使用`source_anticipation_design_20261104_v2`，V1因实际训练曝光的正例数不匹配而不得使用。

三个辅助头从同一合格源检查点的原生诊断头独立复制；所有源模型参数和原生头冻结。三个头共用V2日程的128步，每步同4个source TRAIN例子，每头512次例子曝光。`point_contract`、`anticipation125`、`time_shuffled125`只改变明确保存的外层标签。缺测视图按已保存日程循环。时间控制必须同时保持每炮语料正例数及实际训练曝光正例数。

用公开PyTorch Adam，lr=1e−3、无weight_decay，头梯度裁剪范数1。外层目标按SOURCE_ANTICIPATION_PLAN及meta.py的余弦方向＋0.20诊断拟合＋1e−4相对初始化保护。固定最终第128步，不按VAL或EAST挑epoch。临时参数仍是源原生128维token接口的0.05归一化步和半径1，原J-TEXT训练窗不改。

这一阶段只拟合并保存辅助头，状态必须标为`TRAINED_NOT_VALIDATED`。没有VAL实际更新验证、倒序／保持／时间标签控制和诊断精度检查，就不得接入EAST或声称收益。只读取日程需要的J-TEXT TRAIN原始信号；不读取TEST/EAST原始信号，不开展J-TEXT在线流。本阶段的时间和标签来自已完成的源域监督数据，不是在线未知标签。
