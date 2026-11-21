# 已到达信号预测与破裂风险：任务接口审查

2026-10-29。**只读审查与下一步提案；开始审查时本轮 11 臂正在运行，未读取或编造其结果。** 所有已使用数据均有开发暴露；本文不是独立验证，也不承诺收益。保持合格 J-TEXT 分类基模、风险头、冷启动映射和报警阈值不变；在线更新仅限 EAST。

## 已知、未知与不能重复的尝试

预测损失衡量的是已到达信号在某个坐标、掩码及平均权重下的拟合，不是破裂风险的判别损失。对临时状态 \(s\) 的小步近似是 \(\Delta L_{risk}\simeq-\eta\nabla_s L_{risk}(t_{later})^\top\nabla_s L_{signal}(t)\)：信号梯度下降，不要求它与**更晚时刻**的风险梯度同向。有限步、后续输入变化、首次报警及提前量又进一步打破这种近似；风险头本身没有因预测误差下降而自动学会利用残差。

- 已否定的具体主张：过去一组 Q/K 更新有 96.59% 降低当次信号损失，完整流却从 cold 的 125TP/32FP 变为 121TP/32FP；不能把当次下降当成可靠预警增益。局部读出由固定读出的下一窗口误差 0.087975 降到 0.034635，但普通保持末值为 0.025006，完整流为 125TP/33FP，倒序同为 125/33；其固定步 6325/10774 次下降，尚混有优化越界问题。[局部读出报告](</media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/炮内读出校正_完整360与预警收益分离_20261029.md>)、[Q/K报告](</media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/注意力炮内更新_完整360失败与通道反馈方向_20261029.md>)。
- 已有元对齐、Q/K 匹配头和分组/时钟四格：VAL 危险阶段 30/32 炮恶化，四格均未同时改善安全与危险；不应把“加后续 BCE、梯度余弦或更新位置匹配”当作未试新方案。[匹配头与四格报告](</media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/匹配更新位置与反馈时效_四格配对实验_20261029.md>)。顺序判别任务全流与 cold 同为 125/32；26 个及时危险窗口虽都改善辅助得分，只有 5 个稍后风险上升。[顺序任务报告](</media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/时间顺序辅助任务_完整360与更新方向失配_20261029.md>)。
- 已有“到达信息是否额外判别危险”的 5 点探针：4 个留出破裂炮全部 BCE 变差，不能再原样训练一个到达信息分类器。[记录](</media/hetaisheng/044A81D94A81C83E/Fuison/docs/research/source_event_rebuild_20261023/到达信息判别反馈_结果与跨炮泛化缺口_20261024.md:3>)。此前也已有带事后真实标签的方向及有限步诊断；标签 oracle 不能用作在线选择。
- 仍未知：本轮**实际** Armijo 接受率和风险结果；这套 H25 局部读出的真/反序损失差在接近决策的破裂与安全炮中是否不同；即便不同，它是否超出完整前缀包含的信息、能否通过 Q/K 作用于风险、能否跨炮泛化。上述失败仅约束各自方法及数据，不能推出所有预测目标或所有炮内学习均无效。

## 当前 11 臂若失败，能与不能说什么

[计划](</media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/LOCAL_READOUT_FACTORIAL_NEXT_PLAN.md:15>)与[实现](</media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_factorial_20261029/learning.py:33>)一致：历史 16 行以内、较老两段目标拟合读出；新两段 H25 目标产生一阶梯度；每次从零 Q/K 临时状态开始，最大范数 0.05，按到达目标的 Armijo 规则接受或精确归零。反序只是翻转已到达的每段 25 点。各臂可能接受不同步长、不同弃权，因此比较的是**完整规则**，不是纯粹等范数方向。

如果封存重计后 `true_true` 的整炮 J 未严格优于 `true_reverse`/`true_hold`，只能说在当前父模型、数据、时钟、损失、掩码、接受规则和固定报警合同下，未建立真实最近目标的额外炮级收益；不能断言逐点风险曲线相同、所有有效信息不存在、局部读出必然无用或任何在线策略不可能有效。若确有接受更新且这些接受步严格降低**已观察损失**，也只能说这些局部下降不足以建立报警收益；弃权不是下降，也不能证明剩下唯一瓶颈是任务信息或表示。分别报告每臂接受/弃权、步长、下一窗口信号误差、逐炮救回和损害。

一个实现细节：当前必要筛选代码检查的是 `candidate.tp >= cold.tp`，即**总及时 TP 数不减少**；它不等于逐炮零损失，可能救回和损害互相抵消。须另外报告逐炮损害，不能将二者混写。[筛选代码](</media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/local_factorial_20261029/audit.py:523>)。

## 候选建议（主进程尚未采纳）：封存后检验顺序拟合差

这是**只读二次诊断，不训练新模型、不新增在线更新、不筛学习率**。它与旧 5 点判别探针的问题相关，但不重复其训练：直接复用本轮所有实际反馈时钟的封存 `loss_before`，检验**此 H25 读出—目标接口**的一个固定标量；不重做顺序分类器、未来风险梯度或元训练。只有本轮完整封存且哈希及逐炮结果通过主进程既定核验后才可能做；是否实施仍由主进程决定，不自动展开后续诊断。

令 \(H\in\{true,reverse,hold\}\) 为历史拟合目标，\(k\in\{true,reverse,hold\}\) 为最近目标。使用本轮真实损失
\[
L_{H,k}(t)=\operatorname{mean}_{M_t}\rho\bigl(W_H\phi(h_{s=0})-T_k(y_t)\bigr),\qquad
q_H(t)=L_{H,reverse}(t)-L_{H,true}(t),
\]
其中 \(\rho\) 是现有 SmoothL1，\(y_t\) 是 asinh 坐标中已成熟的两个 25 点**增量**，\(M_t\) 是该反馈的完整通道掩码，\(W_H\) 是较老历史拟合出的读出，\(h_{s=0}\) 是零 Q/K 状态的 64 维末信号 token；\(\phi\) 表示各通道按先验均值、尺度标准化并拼接常数截距，\(W_H\phi\) 是各通道 25 个时距的预测。每个 \(q_H\) 的两项只换新目标真/反序，不换读出、输入或掩码；再用 `reverse`、`hold` 历史拟合的 \(q_H\) 作完整对照。另报 \(L_{H,hold}-L_{H,true}\)，但它是**目标设零的损失差**，不是“预测器是否胜过保持末值”的误差差。

参数位置必须说清：既有在线损失的梯度只作用于第一层 Q/K 的临时 \(2\times64\times64=8192\) 状态；各次反馈从零计算，风险头和源分类器冻结。[Q/K实现](</media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/attention_lowrank_20261029/model.py:29>)。这里所用 `loss_before` 是零状态反事实，**不是**上一时刻实际状态或已接受步后的风险效果；\(q_H\) 也不是新的在线损失、报警分数或物理机制标签。

### 固定取样、对照及停止条件

1. 只分析冷启动后的 340 炮。每个破裂炮，在其真实破裂时刻之前、下一预测点提前量满足 `np.round(lead_ms, 9) > 30` 且原始 `lead_ms <= 125` 的反馈时钟中，事后选**最后一个**；该时钟前 cold 不得已报警。无合格时钟或无有效目标则记录并排除。此选点使用结局，**仅为事后诊断，不能作为在线时钟**。
2. 对每个合格破裂炮按炮号升序，从尚未配对的安全炮全部反馈时钟中选 cold 尚未报警、同一个两段完整通道有效性模式、时钟相差不超过 128 个采样点、冷模型在前一采样点的概率差不超过 0.05 的候选；按概率差、采样点差、安全炮号、时钟依次升序取一个。每安全炮最多一次；不按 \(q_H\)、真实更新风险或结果挑配对。按真实时间戳算提前量，128 是采样点而非预设毫秒。
3. `loss_before` 与反馈时钟保存在每炮封存回执；须核对 6 项 `H_true/H_reverse` 的有效元素数、有限值与同一掩码。回执只记录总元素数，**不足以证明掩码完全相同**：若需要逐通道模式，由获准人员只读原始已到达前缀和固定先验重建；无法核对则报告不可评估，不默认为一致。不可把信号数组复制成新数据集。
4. 每炮只贡献一个配对，计算 \(D_H=\operatorname{mean}_{pairs}(q_H^D-q_H^S)\)。预先固定 2000 次、`numpy.default_rng(131)` 的配对重抽样百分位 95% 区间；成对重抽样不重新匹配。仅当不少于 20 对，且 \(D_{true}\) 的区间排除零，且 \(|D_{true}|-|D_{reverse}|\)、\(|D_{true}|-|D_{hold}|\) 的区间下限均大于零，才称这个**特定标量**的筛查有支持。报告所有配对、排除数量、三种 \(D_H\)、区间、`hold` 目标对照和每臂接受/弃权分布，不挑最有利通道或时钟。
5. 少于 20 对、掩码或封存核对失败为**证据不足**；达到样本下限但任一条件不满足即停止该标量路线，不根据这批结果换配对阈值或补扫步长。区间仅描述固定匹配样本的波动，不含匹配过程或反复开发选择的不确定性，不能称为确认性统计检验。

若筛查有支持，也只说明在这一回顾性匹配子集中，该已到达损失差与事后结局有条件关联；冷概率和采样时钟不等于完整前缀匹配，不能称为新增条件互信息、可部署危险分数、物理因果或预警收益。若无支持，只否定上述标量/匹配/样本的筛查，不能否定其他诊断量或表示。任何进一步方法都需另行设计、固定因果合同并在未用于开发的新炮上检验；不将该标量直接接入风险或加可靠性门控。

## 既定 20 对门槛的样本可行性核验

2026-10-29，按上述原条件，仅用旧版 local_readout_east360_v1 的 cold 曲线、时间戳、炮级结果及原始信号掩码计算，未读本轮新实验结果。后 340 炮中共有 163 个破裂、177 个安全；113 个破裂炮有落在严格 (30, 125] ms 的反馈时钟，要求此前 cold 未报警后剩 **45 个合格破裂炮**，其中 32 个在旧 cold 的整炮结果中为漏报或迟报，另 13 个在所选时钟或之后由 cold 及时报警，因此 45 不等于漏报／迟报数量。没有合格时钟因全无目标而被排除；166 个安全炮至少有一个合格候选时钟。

按原先的炮号升序贪心规则，保持两段完整通道模式完全一致、时钟差至多 128 采样点、概率差至多 0.05，得到 **41 对**；其中 29 对的破裂炮在旧 cold 中漏报或迟报。未配对破裂炮为 60999、65944、73403、73405。41 ≥ 20，故原样本数量门槛在这份旧 cold 数据上可行；20 不是经过功效计算的阈值，不能据此证明关联、预警收益或方案应当执行。当前本轮 cold 与旧版相同采用用户提供的前提，未读新结果自行核验；因未读取新损失，亦未确认本轮各臂损失有限。

30 ms 边界复核：首版直接比较未舍入浮点数，三个反馈点的原始提前量略大于 30 ms，但按固定评价口径四舍五入到小数点后 9 位均为 30 ms，应排除。71086 和 93218 此前未报警，原各占一个匹配对；69324 此前已报警，本来就不进入匹配。因此窗口覆盖由 116 降为 113、合格破裂炮由 47 降为 45、匹配对由 43 降为 41，其余匹配对不变。125 ms 上限仍按原始提前量判断，未放宽。复算命令已更正。

只读复核了后 340 份旧曲线封存哈希、340 份原始 HDF5 哈希、固定读出先验哈希，并逐炮将重建的两个窗口有效元素总数与旧回执记录对齐。重建严格沿用 EAST MNM 不可用、CIII −10±0.01 缺测、先验激活通道、每段起点和 25 个目标点均有效的合同；时间差来自原始时间戳，严格 30 ms 下界按上述口径判断。全部资料已有开发暴露；这是关联筛查的**样本可行性**，不是独立验证，也不表示主进程已接受候选或要求继续其他诊断。

下面是在本工作树根目录执行的原命令；只读数据并打印匹配对，不写代码文件或结果文件：

<details>
<summary>复算命令</summary>

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -B - <<'PY'
import hashlib, json
from pathlib import Path
import h5py
import numpy as np
from experiments.risk_aligned_online import io23

root = Path('probe_outputs/local_readout_east360_v1')
def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)
protocol = load(root / 'protocol.json')
order = protocol['order']
rows = load(root / 'per_shot.json')['cold']
assert len(order) == len(rows) == 360
assert [r['shot'] for r in rows] == order
prior = Path(protocol['readout']) / 'cold_readouts.npz'
assert sha(prior) == protocol['readout_sha256']
with np.load(prior, allow_pickle=False) as z:
    active = np.array([f'last_token_{c}_coef' in z.files for c in range(23)])
assert not active[2] and not active[10] and active.any()
ref = Path(protocol['reference'])
hashes_path = ref / 'input_hashes.json'
assert sha(hashes_path) == protocol['reference_hashes']['input_hashes.json']
hashes = load(hashes_path)
threshold = float(protocol['threshold'])
positives, negatives = [], []
counts = dict(disruptive=0, safe=0, d_time_window=0, d_after_no_alarm=0, d_no_observed=0, safe_with_candidate=0)
for i, (sid, outcome) in enumerate(zip(order[20:], rows[20:]), start=21):
    name = f'{i:04d}_{sid}'
    seal = load(root / 'shots' / f'{name}_seal.json')
    curve_path = root / 'shots' / f'{name}.npz'
    assert seal['shot'] == sid and seal['index'] == i and sha(curve_path) == seal['sha256']
    with np.load(curve_path, allow_pickle=False) as z:
        times, cold = z['times'], z['cold']
    assert times.ndim == cold.ndim == 1 and len(times) == len(cold)
    assert np.isfinite(times).all() and np.isfinite(cold).all() and (np.diff(times) > 0).all()
    clocks = [r['clock'] for r in seal['updates']['feedback']]
    assert clocks == list(range(256, len(times), 128))
    hit = np.flatnonzero(cold >= threshold)
    first = int(hit[0]) if len(hit) else len(times)
    path = io23.path_for('EAST', sid)
    assert sha(path) == hashes[str(sid)]
    with h5py.File(path, 'r') as f:
        disruptive = bool(np.asarray(f['meta/IsDisrupt'][()]).reshape(-1)[0])
        assert disruptive == outcome['is_disrupt']
        masks = np.zeros((len(clocks), 2, 23), dtype=bool)
        for j, channel in enumerate(io23.RAW23):
            if not active[j]:
                continue
            values = np.asarray(f['data'][channel][:], dtype=np.float32)
            assert len(values) == len(times)
            available = np.isfinite(values)
            if j == 1:
                available &= ~np.isclose(values, -10.0, rtol=0.0, atol=0.01)
            for k, c in enumerate(clocks):
                masks[k, 0, j] = available[c-51:c-25].all()
                masks[k, 1, j] = available[c-26:c].all()
        assert int(masks.sum()) * 25 == seal['updates']['local_readout']['observed']
        if disruptive:
            counts['disruptive'] += 1
            down = float(np.asarray(f['meta/DownTime'][()]).reshape(-1)[0])
            eligible = [c for c in clocks if np.round((down - times[c]) * 1000.0, 9) > 30.0 and (down - times[c]) * 1000.0 <= 125.0]
            counts['d_time_window'] += bool(eligible)
            candidates = [c for c in eligible if first >= c]
            counts['d_after_no_alarm'] += bool(candidates)
            candidates = candidates[-1:]
        else:
            counts['safe'] += 1
            candidates = [c for c in clocks if first >= c]
        index = {c: k for k, c in enumerate(clocks)}
        kept = [(sid, c, float(cold[c-1]), masks[index[c]].tobytes())
                for c in candidates if masks[index[c]].any()]
        if disruptive and candidates:
            if kept:
                positives.append(kept[0])
            else:
                counts['d_no_observed'] += 1
        elif not disruptive and kept:
            counts['safe_with_candidate'] += 1
            negatives.extend(kept)
used, pairs = set(), []
for dsid, dc, dp, dm in sorted(positives):
    choices = [(abs(sp-dp), abs(sc-dc), ssid, sc)
               for ssid, sc, sp, sm in negatives
               if ssid not in used and sm == dm and abs(sc-dc) <= 128 and abs(sp-dp) <= 0.05]
    if choices:
        gap, clock_gap, ssid, sc = min(choices)
        used.add(ssid)
        pairs.append((dsid, dc, ssid, sc, round(gap, 8), clock_gap))
status = {r['shot']: r['timely'] for r in rows[20:] if r['is_disrupt']}
print('counts:', counts)
print('eligible_disruptive:', len(positives), 'of which cold_not_timely:', sum(not status[p[0]] for p in positives))
print('matched_pairs:', len(pairs), 'of which cold_not_timely:', sum(not status[p[0]] for p in pairs))
print('pairs (D, D_clock, S, S_clock, probability_gap, clock_gap):')
for pair in pairs:
    print(pair)
print('unmatched_disruptive:', [p[0] for p in sorted(positives) if p[0] not in {x[0] for x in pairs}])
print('checks: 340 old cold-curve seals and original HDF5 hashes; 340 observed-mask totals; fixed prior hash; no new-run files')
PY
```

</details>

## 原始文献核读范围（2 篇，非系统综述）

1. [Sun et al., ICML 2020, *Test-Time Training with Self-Supervision*](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf)：实际查看原文第 4 节图 4 和定理 1（PDF 第 7–8 页）。其正梯度内积结论有光滑、凸、有界梯度及步长条件；图像任务的经验关系不能保证这里非凸、延迟、整炮首次报警的表现。
2. [Du et al., arXiv:1812.02224v2, *Adapting Auxiliary Losses Using Gradient Similarity*](https://arxiv.org/html/1812.02224)：实际查看原文第 3 节命题 1–2、算法 1 和第 6 节局限。该方法使用主任务梯度并明确提醒局部相似度与噪声的限制；本炮 EAST 结局梯度在线不可得，不能照搬为门控。两篇均取得原文相关部分；没有声称通读全文或验证本文实验。
