# 已结束EAST炮监督炮内更新规则：完整360对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-10-31
- Verification Status: PLANNED_DEVELOPMENT_EXPERIMENT
- Version Label: east_operator360_v1

## 问题与固定父模型

当前context8在炮内用诊断变化估计8个临时变量，但该估计规则的key/kernel/rate仍来自J-TEXT训练；EAST仅学slow和output，缓存delta还截断了对规则的梯度。拟检验：在已结束EAST炮上，让风险监督真正穿过这条更新递推，能否使后来炮的真实反馈有用。

父运行固定为`probe_outputs/current_order_context360_v2`，不是已否定的prefix16运行。completed SHA=`cb9cf19197ab25fb6e825687ec002bcad6135834dca1cb7dc3371234eb7883a9`。源模型SHA=`c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`；相同360身份、仅电流顺序、cold20、density_gain、阈值0.7258946895599365，及时lead严格>30ms。

首20炮由原源模型预测，全部成本计入整流。第20炮封存／揭示后，才激活已审计、只由相同cold20形成的父cold状态；重新按这20炮构建密度映射并与父映射精确比较。旧cold在上一轮已逐位重训复现，本实验复用该固定父状态，不提前回算首20炮。

## 唯一可训练范围

冻结base、slow、memory.output、risk head、source scaler及报警阈值。仅解冻memory.key.weight/bias、kernel、rate，共7001参数。第一Transformer block输出h可缓存；每步必须重算memory.trace，风险梯度穿过trace和固定的output／第二block／risk head，不得缓存或detach可学习delta。

这不是改最终风险分数，也不是在线修改Transformer主干权重；部署时8维临时状态按已到达诊断更新，间接改变内部token，每炮清零。训练规则持久保留，但后340炮不再作跨炮监督更新，本阶段先单独检验炮内作用。J-TEXT模型不重训，也不开展J-TEXT在线更新。

## 数据和优化预算

仅使用当前前20的已揭示结局（18D/2S）；原full_window端点策略≥127、破裂lead>30ms，继续disruptive_positive_only损失。83646安全标签有争议，不改写；2S不是足够正常工况覆盖，结论须保留局限。

固定2epoch。每epoch按炮序遍历18D，与2S交替配对，每步loss=0.5L_D+0.5L_S；共36次Adam step，lr=.001、clip1。每条D总用2次，每条S总用18次；这是类均衡加权，不是增加独立炮。目标损失直接用现有window_loss；不增加预测损失、teacher约束或新正则。预测误差只记录，不用于选模型。固定最终状态，不按后340指标选择epoch或超参。此规则准备比父模型多36次优化，同标签预算但非同计算预算。

## 六臂完整流

1. cold：固定父模型，关闭临时作用。
2. source_rule：相同父模型，原源域规则的真实炮内反馈；应逐点等于父intra。
3. learned_rule：目标训练后的规则，真实已到达反馈。
4. reverse_feedback：新规则，对已到达目标的horizon指派反序；沿用逐patch范数匹配。
5. negative_shift：将新规则真实delta取负，比较相反的内部作用方向；不是反向时间或可部署策略。
6. zero_feedback：新规则不写入状态，应与cold逐点相同。

所有当前炮结局在六臂预测封存之后才读；后340结局仅计分。风险变化必须由临时状态引起，不能改变off预测。不得撤销早期报警，已有7个cold FP在index<16，这些不能由该fast路径修复。

## 验证及接受标准

先做参数梯度和冻结范围检查，真实cold20各参数组有限且至少出现非零梯度；保存训练步、所用炮号、loss及梯度。验证全套非operator参数／缓冲区不变，cold20 off风险逐位不变；每炮完整cold/source_rule分别与父cold/intra逐点一致；zero_feedback与cold逐点一致。

主指标使用完整360，不剔除启动或争议炮；另报告early>3000ms、首次报警、块级与标签敏感性。learned_rule须相对cold增加J、FPR≤.20，并检查对原source_rule及反序／零反馈的优势。仅训练损失下降、预测重建下降或相对一个变差基线的改善都不够；一次开发流优势也不构成独立确认。

后验独立计分所有6×360条封存曲线，核对SHA与当前标签、固定序号21/114/360做6臂CUDA重放，检查未来扰动与每炮reset的单元测试。不声称发现新物理规律。本轮不加入炮间规则更新，只有炮内作用成立后才比较intra+inter。

## 执行与停止

本机rl_env CUDA，主控单GPU进程，输出`probe_outputs/east_operator360_v1`。外部timeout1800秒、TERM后10秒KILL；监控真实进程与训练／炮序进度。若纯计算超时则保留已完成阶段证据，不能报完整流成绩。出现错误保留原因，不覆盖既有输出。用户已授权自主迭代；任何修复均披露，不暗中改预算。

本数据集和父方法均参与过开发。本约定用于限制这一次干预，不是盲法预注册。基础单位和18个标签冲突尚未解决，本轮不能以训练成功宣称数据已经无问题。

审查补充（运行前）：即使不增加显式logit修正，中间状态也可能学成近似恒定的风险偏移。六臂只检验这次规则拟合及推理反馈干预，不能穷尽隐式校准解释，也不能证明风险标签特异性或新物理规律；必要时以后另立同预算训练对照，不根据本轮结果临时加臂或选最优配置。
