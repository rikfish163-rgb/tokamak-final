# R2d：EAST FP-only 跨炮 Consolidated Episodic 设计

> 状态：DESIGN_ONLY / DIAGNOSTIC_ONLY_TEST_DRIVEN_EXPLORATION  
> 工作树：/media/hetaisheng/044A81D94A81C83E/Fuison-JTEXT-EAST-cmma-shadow-20260826  
> 本轮只写本文件；不改 Python、测试、R2/R2b/R2c 或既有产物。

## 1. 裁决与固定边界

裁决：R2d 可进入独立实现审查，但当前不能部署、发布或称为独立未来确认。采用 FP-only 跨炮 slot consolidation、三个 expert 的一次联合反传、固定 parent/threshold/weight/alarm、严格 seal/query/reveal/update 因果门。

固定主线：

~~~text
J-TEXT 静态 max(25ms,AdvancedTime)
  → EAST support110 静态迁移 fixed125
  → EAST balanced360 按炮序 predict→seal→fixed query→selected reveal→update
~~~

- J-TEXT 静态和 J-TEXT 在线完全不动；J-TEXT 在线永不进入 EAST。
- EAST support110 只提供 parent、scaler、threshold、guard、replay。
- balanced360 只按 roster 的 stream_index 排序；physical shot 仅作身份校验，不作排序或硬编码条件。
- 只使用 fixed query manifest 选中的 reveal；未查询结局不可读、不可推断、不可作 blocker。
- 不调报警状态机、阈值、mixture 权重、query 数量或权重更新；不做 intra-shot update、共享 LoRA。
- R2/R2b/R2c 保留，R2d 使用独立入口、schema、ledger 和产物目录。
- 首 120 的唯一运行门：至少一个 parent FP→candidate TN，harmful flip=0，TN→FP=0；失败即停止，不重调参。

## 2. 证据、根因与反证

已审计对象：run_cmoeam_r2.py、episodic_feature_adapter.py、ordered_trajectory_memory.py、causal_slot_activation.py、causal_rescue_ceiling.py，以及 full360_v1、cmoeam_r2c_first120_20260827 产物。

| 证据 | 结论边界 |
|---|---|
| full360 static/intra/inter/intra+inter：TPR30=.8333、FPR=.1278、J=.7056 | 固定权重旧神经路径没有改善 |
| full360 weight-only：.8556/.1056/.75 | 不能借用；R2d 禁止改权重 |
| R2 首 120 零增益，R2b 新增 2 FP | 单炮局部路径无效或有害 |
| R2c：7 个 probation、29 次验证全中性、active=0 | 安全门有效，但 singleton slot 未形成跨炮功能证据 |
| R2c：148 shadow candidate，最大 effect 约 1.99，实际部署 adapter application=0 | 有局部 actuation，effect 不等于 FP→TN |
| causal_rescue_ceiling_v1：cap2 strict=8（FP=5、MISS=3） | 只是必要乐观 ceiling，不是性能保证 |

根因：

1. 当前错误路径按炮新建 episode slot；每个 slot 只有单炮证据、独立 memory 和独立三个 adapter，证据碎片化。
2. retrieval 在固定物理/风险轨迹空间，adapter 只在单炮 Transformer hidden endpoint 训练；路由相似性与优化目标不一致。
3. 当前保存 campaign block 但 route 未把跨 block 多样性作为硬激活门。
4. shadow effect/routed count 只说明 logits 可变，不能说明固定报警器真的把 FP 变成 TN，也不能排除正确样本被翻坏。
5. R2c 局部 fit 可接受但未来全中性，反证不是“没有候选”或 parent 重建错误，而是局部 actionability 没有跨炮泛化。

## 3. R2d 架构

选择一个 ConsolidatedFPSlot：

- direction 固定为 FP；FN、MISS、TARDY 不进入 R2d 训练或合并。
- FP source 必须是 selected、已 seal、已 reveal，且 parent group=FP、真实 outcome=safe。
- 新 source 先尝试加入兼容 cluster；只有无兼容 cluster 才建立 probation cluster。
- active 前至少需要 contract 规定数量的 source，且来自至少两个不同 block；block 由 manifest 的 block_partition_spec 从 stream_index 派生，不写物理炮号。
- cluster 保存 source references、blockers、固定轨迹索引和三个 expert 的联合 adapter；不复制、平均或覆盖原始 source。
- active 从 source reveal 之后的下一炮开始；不得改写当前炮已 seal 预测。
- future neutral 保持 active；future harmful 立即 retire，retired 不原地重激活。

~~~mermaid
flowchart LR
 A["label-free prefix"] --> B["fixed parent/static"]
 B --> C["consolidated FP route"]
 C --> D["predict→seal"]
 D --> E["fixed query"]
 E -->|unselected| F["discard; no reveal/update"]
 E -->|selected| G["reveal"]
 G --> H["append FP source/blocker"]
 H --> I["merge→joint fit→compatibility gate"]
 I --> J["active from next shot"]
 J --> A
~~~

## 4. 状态机与因果边界

状态：

~~~text
UNSEEN → FP_PROBATION → CONSOLIDATION_PENDING → JOINT_FIT_READY
       → ACTIVE_NEXT → ACTIVE_VALIDATING → RETIRED
~~~

- singleton 只能 probation/shadow，不能 active。
- 合并失败保留独立 probation，不强制最近邻合并。
- fit 失败保留旧 active version；candidate 不部署。
- ACTIVE_NEXT 只在下一炮生效；同炮不反馈。
- correct→error 或 TN→FP 后，slot 对后续炮原子 retire；保留所有历史 receipt。
- R2d 使用独立 ledger；不改 causal_slot_activation.py，也不把 R2c 的 future-beneficial 状态含义重用成 R2d active。

| 阶段 | 允许 | 禁止 |
|---|---|---|
| prefix/predict/seal | 当前 label-free prefix、parent、严格过去 source、active slot | 当前 outcome、future outcome、最终评估表 |
| query | 固定 manifest 的 selected 判断 | 按 effect 临时增 query |
| selected reveal | 本 query outcome | 未选中炮 outcome |
| update | 已公开 source/blocker、support110 guard/replay | 同炮回写、future label |
| future route | source_stream_index < current_stream_index | current/future source |
| final audit | 全量 outcome 只作离线统计 | 反向改变运行状态 |

pre-reveal 只能记录 finite、directional、cap、route、effect；beneficial/harmful 必须在 selected reveal 后，用 parent 与 candidate 的固定报警输出比较。

## 5. Slot 合并、联合反传和 actuation 公式

### 5.1 合并与路由

沿用 ordered_trajectory_memory.py 的固定 offsets、availability、velocity、endpoint 和物理/风险距离 D，不把 hidden 或 outcome 拼入距离。两个 cluster A/B 的 source 集合为 S_A/S_B：

\[
D_{\leftrightarrow}(A,B)=\frac12[
|S_A|^{-1}\sum_{r\in S_A}\min_{u\in S_B}D(r,u)+
|S_B|^{-1}\sum_{u\in S_B}\min_{r\in S_A}D(u,r)]
\]

对 source 集合 S 和 blocker 集合 B：

\[
D_S(q)=\min_{r\in S}D(q,r),\quad D_B(q)=\min_{b\in B}D(q,b),\quad
G(q)=D_B(q)-D_S(q)
\]

\[
\mathcal M(A,B)=
1[dir=FP]1[parent/path/alarm/schema\ hashes\ equal]
1[query\ ids\ disjoint]1[distinct\ blocks\ge n_{block,min}]
1[D_{\leftrightarrow}\le R_{merge}]1[G_{merge}\ge G_{min}]
\]

所有阈值在 run contract 启动前锁定，不能看首 120 后调整。slot ID 为父本 hash、排序后的 query IDs、path hashes 和 schema 的内容 hash，不含物理炮号。

未来 route 还必须满足：过去 source 支持数、distinct blocks、固定 radius、blocker gap、anchor/cooldown、reliability 和 slot margin；任一不满足即 abstain，不叠加多个未经证明的 adapter。

### 5.2 联合 adapter 与反传

对 expert e：

\[
h'_{e,r,t}=h_{e,r,t}+\delta_f\tanh((h_{e,r,t}^{T}d_e)u_e),\quad
z^\Theta_{e,r,t}=f^P_e(h'_{e,r,t})
\]

三个 expert 的参数一次构成 Θ=(θ1,θ2,θ3)，每个保留 128 个 rank-1 参数；parent encoder、risk head、threshold、weight 固定。

\[
M^\Theta_{r,t}=\sum_e w_e(z^\Theta_{e,r,t}-\tau_e),\quad
a_{r,t}=M^0_{r,t}-M^\Theta_{r,t}
\]

对全部 consolidated source S 和 support110 replay/guard R 使用一个固定目标：

\[
\begin{aligned}
L(\Theta)=&
\frac{1}{\sum_{r\in S}|I(r)|}\sum_{r\in S,t\in I(r)}
[\sum_e\lambda_e\rho_{target}(z^\Theta_{e,r,t},safe,\tau_e,\gamma_e)
+\lambda_m\rho_{mix}(M^\Theta_{r,t},M^{safe}_{r,t})]\\
&+\lambda_R\frac{1}{\sum_{v\in R}|I(v)|}
\sum_{v\in R,t\in I(v)}\rho_{replay}(z^\Theta_{e,v,t},z^0_{e,v,t})
+\lambda_G\rho_{guard}(\Theta)
\end{aligned}
\]

\[
\Theta_{k+1}=\Pi_{\mathcal C}(\Theta_k-\eta\nabla_\Theta L(\Theta_k))
\]

这不是三个 per-shot trainer 的顺序调用，也不是 factor 平均；任一 expert 非 finite、超 cap 或 guard 失败，三个 adapter、slot version、registry 一起回滚。

### 5.3 Actuation compatibility

\[
\Delta z_{e,r,t}=z^\Theta_{e,r,t}-z^0_{e,r,t}\le0,\quad
0\le a_{r,t}\le2
\]

\[
B=\#\{G^0=FP,G^\Theta=TN\},\quad
H=\#\{G^0\in\{TN,TP\},G^\Theta\in\{FP,FN,TARDY\}\},\quad
N=\#\{G^0=TN,G^\Theta=FP\}
\]

\[
C_{act}=1[\max a\ge\delta_{min}]1[\max a\le2]
1[\forall\Delta z\le0]1[H=0]1[N=0]1[B\ge1]
\]

\[
C_{compat}=C_{merge}\cdot C_{route}\cdot C_{fit}\cdot C_{act}
\]

support/guard 上的 C_act 只代表候选可部署；首 120 的真实成功仍必须由后续 selected reveal 重新计数。

## 6. 数据结构与 API

contract 至少包含：

~~~text
R2DContract(schema_version, parent/scaler/representation/risk_head hashes,
 thresholds, fixed_weights=[0.3,0.1,0.6], alarm hash,
 support110 guard/replay hashes, roster/query hashes,
 path schema/offsets, block_partition_spec, merge/route/gap/cap/actionability,
 optimizer/seed)
~~~

核心记录：

~~~text
FPSourceRecord(source_id, query_id, stream_index, physical_identity,
 parent_group=FP, revealed_safe, selected, sealed, parent_contract_hash,
 path_hash, availability_hash, block_id, receipt_hash)

ConsolidatedFPSlot(slot_id, version, status, source_ids_sorted,
 distinct_block_ids, blocker_ids, route_index_hash,
 adapter_hash_by_expert, base_hash_by_expert, fit/guard hashes,
 lineage, activated_from_stream_index, retired_at)

RouteReceipt(current_stream_index, slot_version, source_ids,
 support_count, block_count, distances, blocker_gap,
 abstain_reason, outcome_free=true)

FitReceipt(source_ids, block_ids, pre/post adapter hashes,
 parent/base hashes, optimizer/steps/seed, finite, cap,
 guard_harm, actionable_delta, functional_B/H/N, accepted, rejection_code)

ValidationReceipt(query_id, baseline_group, candidate_group,
 routed_count, beneficial, harmful, tn_to_fp, status_before/after)
~~~

API：

~~~text
observe_prefix(identity, stream_index, label_free_signal) -> SealedPrefix
predict_and_seal(prefix, parent, active_versions) -> SealedPrediction
fixed_query_decide(sealed, fixed_manifest) -> QueryDecision
reveal_selected(query_decision, outcome) -> RevealedOutcome
append_fp_source(revealed, baseline_prediction) -> SourceReceipt
merge_fp_source(source, existing_versions, contract) -> MergeReceipt
fit_joint_fp_slot(candidate, source_refs, guard_refs, replay_refs) -> FitReceipt
route_future(sealed_prefix, active_slot) -> OutcomeFreeRouteReceipt
validate_after_reveal(sealed, query, reveal, candidate, parent) -> ValidationReceipt
rollback(slot_id, from_version, to_version, reason) -> RollbackReceipt
~~~

所有 API 必须幂等、append-only、拒绝重复 identity。拒绝码至少包括 causal_violation、contract_mismatch、direction_not_fp、insufficient_distinct_blocks、merge_distance_exceeded、blocker_gap_failed、actionability_floor_not_reached、directional_cap_failed、guard_harm、non_finite_gradient、base_hash_changed、functional_gate_failed。

## 7. TDD 矩阵

本轮只写设计，以下均为待实现/待运行，不是已通过。

| ID | 断言 | 预期 |
|---|---|---|
| R2D-01 | 本轮 git 边界 | 仅本 Markdown 新增，既有未跟踪文件不变 |
| R2D-02 | parent/head/threshold/weight/alarm hash 改变 | contract_mismatch，fail closed |
| R2D-03 | J-TEXT online 或非 support110 数据进入 | domain gate 拒绝 |
| R2D-04 | pre-reveal payload 含 down_time/is_disrupt/group | causal_violation，不落模型 |
| R2D-05 | unselected reveal 或临时加 query | 拒绝，manifest/query 不变 |
| R2D-06 | physical shot 乱序/改名，stream_index 合法 | 仍按 stream_index；identity 不匹配则拒绝 |
| R2D-07 | same block、duplicate、MISS/FN/TARDY、未 reveal source | 不合并、不 active |
| R2D-08 | 两个兼容 FP、不同 block、距离/gap 合法 | 确定性 consolidated slot |
| R2D-09 | source 输入置换 | ID、source set、结果 deterministic |
| R2D-10 | 联合 fit 遮蔽任一 source | loss/gradient 必须依赖全部 source |
| R2D-11 | 任一 expert NaN/cap/guard harm | 三 expert 和 registry 原子回滚 |
| R2D-12 | fit 后父本/head/weight 比较 | 全不变，仅 adapter 可变 |
| R2D-13 | route 带 current/future source | 严格过滤并 abstain |
| R2D-14 | pre-reveal 大 effect | 只记 effect，不计 beneficial、不改 query |
| R2D-15 | future correct→error 或 TN→FP | subsequent route retire，历史保留 |
| R2D-16 | first120 | FP→TN≥1、harmful=0、TN→FP=0 才 PASS |
| R2D-17 | R2/R2b/R2c 回归审计 | 入口、artifact、结论不被修改 |
| R2D-18 | first120 未通过后请求 full360 | 拒绝扩展，不调参、不补读未选 outcome |
| R2D-19 | summary provenance | 写明 TEST_DRIVEN_EXPLORATION、independent_future_confirmation=false |
| R2D-20 | 未触碰未来流复验 | 只有该阶段才可提高科学声明级别 |

首 120 是 run contract 的 cohort 语义，不是路由代码可利用的炮号常量。

## 8. 晋级、回退、复用与新建

晋级：selected FP reveal → compatible merge → distinct-block/gap gate → joint fit → fixed guard/cap/actionability/functional gate → candidate version → 下一炮 ACTIVE_NEXT。

回退：fit rejection 保持旧 active；future harmful 原子切回旧安全 version 并 retire 当前 slot；不删除日志、不改 seal、不改 parent、不改权重、不读未查询结局。neutral 保持 active，不计 gain。

复用：

- run_cmoeam_r2.py 的 lifecycle、OutcomeAccessGate、fixed manifest、sealed neutral logits、receipt 风格；
- episodic_feature_adapter.py 的 rank-1 bounded adapter、SealedAdapterBatch、target-v2、cap/guard projection；
- ordered_trajectory_memory.py 的 fixed offsets、物理/风险距离、严格过去过滤；
- causal_slot_activation.py 的 selected/sealed/revealed 检查、chronology、唯一性、retired 审计思想，但不改其状态语义；
- causal_rescue_ceiling.py 的 FP 方向、cap2、strict blocker/paired gate；
- support110 parent/scaler/guard/replay 和既有 static seal/query/roster。

新建设计对象（本轮不创建）：

- 独立 run_cmoeam_r2d.py；
- R2DSlotLedger、ConsolidatedFPSlot、FPConsolidator；
- JointFPAdapterTrainer、ActuationCompatibilityGate；
- merge/fit/route/validation/rollback receipts；
- 独立 cmoeam_r2d_fp_consolidated_first120_<run_id> 产物目录。

最终裁决：R2d 设计 GO、实现前部署 NO-GO。R2c 的 7 slot/29 neutral 和 ceiling 的 strict=8 都只能作为设计依据；在首 120 固定门未通过前，不得声称 R2d 有在线增益或未来确认。

