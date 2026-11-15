# 首越线前的最晚保护时刻：等一帧是否真能降低预测难度

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run, fixed before last-write audit
- Origin Date: 2026-11-06
- Verification Status: previous true rescue and resolution diagnostics known; no last-write counts
- Version Label: patch_utility_crossing_last_write_v1

只用八帧保护下已核验的真实可救迟报（包括不完整尾段）。
保持同一动作、阈值与有限作用窗口，不把它变成新的选择器。
在真实局部首越阈时刻 k 以前，逐点检查 d 从原 m+8 到 k−1：
此前的原生成保护仍有效、静态截至 d 未越阈、两种符号相对
静态的已到达改动都不超过原保护限幅。先不做任何拟合。

- 记录每个动作的全部合法决策点、最晚决策到真实首越的
  帧数；按原炮去重汇总。个别时刻允许不等于一直允许。
- 在每个合法决策点，以真未来静态加上最后已达的该动作
  改变量作保持对照，按原真实首报效用计分。静态真未来、
  可救案例身份和真实首报都只用作事后诊断。
- 比较真实首越帧距阈值与当时保护上界，区分“时钟允许等到
  前一帧”和“已观察改变量已经足够预测下一帧”。不把任何
  保护放松或临时写入计算成在线成绩，不读新 VAL、TEST、EAST。
