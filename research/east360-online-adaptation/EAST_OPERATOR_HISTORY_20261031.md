# EAST context-operator history and gradient-scope audit — 2026-10-31

**Scope and conclusion.** In the bounded WT/MAIN code, run-artifact and prior-audit search described below, I did **not find a completed experiment that exactly matches** training only the current-order `context8` operator on its revealed cold20 while keeping the same source, fitted slow state and memory output fixed, and differentiating the supervised risk through the actual causal trace. This is not a proof that no other archive or experiment exists, and it is not a novelty or performance claim. Closely related source-domain training, target-supervised memory training with a different memory, and target meta-learning have already been done.

The proposed parent is the completed development run [current_order_context360_v2](probe_outputs/current_order_context360_v2/completed.json): current-only order, source `c7270c1e…`, context8 source memory `7cef5d4e…`, `full` readout, progressive feedback, `disruptive_positive_only` target loss, threshold `0.7258946895599365`, and 20 warmup shots. Its first 20 are **18 disruptive / 2 safe**; the two safe shots are **83646 and 87484**. The 20 were predicted and sealed before reveal; the input map and neural cold state were fitted after shot 20. The 20 warmup predictions remain source predictions in every arm, not retrospective predictions by the fitted cold model. See the [protocol](probe_outputs/current_order_context360_v2/protocol.json), [runner](experiments/risk_aligned_online/lifecycle_stream_20261025/run.py#L154-L192), and [per-shot decisions](probe_outputs/current_order_context360_v2/per_shot.json).

The corresponding cold state SHA-256 is `31ccf7a12567d3af4d4df93d41894339a81077e89d6d47aebcc24306673173a7`; the map is `7056d394be54463f6891c166e47de3918fb0d4c38a0585aad1341bb274725ff2`; the order is `ad44224b7f760eed43305420200c790ee893dc5511dd2f0c0bd280d1a8d5e5c9`. These hashes were recomputed here. The stored [independent audit](probe_outputs/current_order_context360_v2/independent_audit.json) reports 360 verified prediction hashes and a verified code archive, but explicitly does not establish independent future validation or real-time latency.

## What exists already

| Completed or observed item | What actually trained or ran | Why it is not the exact proposed experiment |
|---|---|---|
| Current lifecycle cold fit | [Lifecycle learning](experiments/risk_aligned_online/lifecycle_stream_20261025/learning.py#L109-L126) uses Adam on **only `slow.weight` and `memory.output.weight`**. The 20-shot receipt records 20 steps and nonzero retained changes to both. [Lifecycle memory_delta](experiments/risk_aligned_online/lifecycle_stream_20261025/model.py#L41-L48) is under `no_grad` and detaches the endpoint delta; [cache_example](experiments/risk_aligned_online/lifecycle_stream_20261025/learning.py#L14-L24) also caches detached delta. | This is the actual fixed parent to compare against, not target-trained key/kernel/rate. The old fit used mixed fast/off modes, target plus source replay, and can not be equated to an unspecified new target-only objective. |
| `context8_source_service_20261028` | Source [training](experiments/risk_aligned_online/context_memory_20261028/run.py#L75-L84) freezes slow and trains the **entire memory**, including output, through task risk on J-TEXT TRAIN1353 for two epochs; its [completion](experiments/risk_aligned_online/runs/context8_source_service_20261028/complete.json) records no target or test signals read. Source `detached` means the reference branch of a direction objective is stopped, not that the main causal trace is detached: see [loss_from_example](experiments/risk_aligned_online/direction_teaching_20261026/learning.py#L55-L93). | Same family and real gradient through `trace -> endpoint_corrections -> risk`, but source labels, different trainable scope and no current-order EAST cold20 fit. The old source result is important prior art, not proof of target adaptation. |
| `trajectory_pair_v1` | The [trajectory key](experiments/risk_aligned_online/trajectory_keys_20261028/model.py#L14-L34) adds observed patch changes and pair masks, making a 92-input rather than 46-input key. The [two-epoch source pair](experiments/risk_aligned_online/trajectory_keys_20261028/run.py#L90-L101) trains all memory parameters with scaled forecast plus risk-direction objective. Its [completion](probe_outputs/trajectory_pair_v1/complete.json) records zero EAST records and a failed source mechanism gate. | Different input, scaled memory/objective, trainable output and source-only data; no current-order target-operator result. |
| `expected_dynamics` / `expected_residual_pair_v1` | A separately source-trained expectation predictor is frozen; [ExpectedMemory](experiments/risk_aligned_online/expected_dynamics_20261028/memory.py#L11-L32) subtracts its previously issued expected change. The [pair](experiments/risk_aligned_online/expected_dynamics_20261028/pair.py#L90-L116) trains key, kernel, rate **and output** using source risk plus forecast loss. Its [completion](probe_outputs/expected_residual_pair_v1/complete.json) says zero EAST records and source gate ineligible. The separate [EAST forecast-only path](experiments/risk_aligned_online/expected_dynamics_20261028/east_forecast.py#L44-L49) specifies zero optimizer steps. | Neither the residual mechanism nor its training subset/data/objective matches the proposed plain context8 target fit. EAST forecasting is not EAST risk-operator training. |
| `delayed_recurrent_memory_cal10_v2_20261025` | A completed target-history screen actually trained a **7,093-parameter DiagnosticMemory**, including its output, through trace and risk on older EAST preparation shots, with a shot-bag risk and forecast auxiliary objective. The [runner](experiments/risk_aligned_online/delayed_recurrent_memory_20261025/run.py#L74-L95) and [training path](experiments/risk_aligned_online/delayed_recurrent_memory_20261025/learning.py#L60-L96) show this. The [completion](experiments/risk_aligned_online/runs/delayed_recurrent_memory_cal10_v2_20261025/completed.json) reports its historical gate failed and no eligibility for 360. | Strong counterexample to a blanket claim that target-supervised memory training is new. It has a different matrix/prior state, 40-sample first effect, parent, threshold, shot histories and optimizer scope. |
| `target_feedback15_holdout5_test360_20261023` | Completed historical EAST **target meta-learning**: [15-shot fitting](experiments/risk_aligned_online/target_feedback_meta_20261023/run.py#L35-L71) updates only the auxiliary forecast head through temporary encoder steps and risk on the next block. The [protocol](experiments/risk_aligned_online/runs/target_feedback15_holdout5_test360_20261023/protocol.json) says head warmup and map had used all 20; it is not a fully independent five-shot holdout. [Saved full360 results](experiments/risk_aligned_online/runs/target_feedback15_holdout5_test360_20261023/completed.json) include static 117/27, point-meta 119/39 and maximum-meta 116/34 TP/FP. | Same broad idea of target risk teaching an update rule, but not context8 or the current cold20. These are historical development results, not estimates for the present proposal. |
| `trajectory_memory360_v2_20261024` | Historical target gradients are routed by a signal-trajectory distance and applied to an encoder: [implementation](experiments/risk_aligned_online/trajectory_memory_20261024/learning.py#L107-L146), [completed result](experiments/risk_aligned_online/runs/trajectory_memory360_v2_20261024/completed.json). | This is a gradient-retrieval bank, not the context8 recurrent operator. Same qualified source alone does not make its parent, threshold, preparation or intervention matched. |

A new [east_context_operator_20261031/learning.py](experiments/risk_aligned_online/east_context_operator_20261031/learning.py) appeared while this audit was in progress. At 04:36:58 local it contained the intended freeze selection and differentiable trace-to-risk helper, SHA-256 `0196f46ad654d23a93655167621051a8ad2b3aeb448737cfb29a1366053bbeda`; no corresponding `probe_outputs/*operator*` completion directory was present in the inspected listing. This is **in-progress code, not a completed target training run**; it may change after this observation.

Search boundary: inspected WT `experiments/risk_aligned_online` code and selected tests, WT `probe_outputs`, MAIN `experiments/risk_aligned_online/runs` (the WT runs path is a symlink to MAIN), the named current-order and historical reports, and specific protocols/completions above. This was not an exhaustive search of all Git objects, deleted runs, external machines, other worktrees, or every differently named nested artifact. A partially completed broad order-hash scan found the current v1/v2 and manifest artifacts; it was not an exhaustive run census.

## Gradient path, shape and cache conditions

The exact current operator parameter layout is [declared here](experiments/risk_aligned_online/context_memory_20261028/checkpoint.py#L17-L21): key weight `8×46` = 368, key bias `8`, kernel `9×4×23×8` = 6,624, rate scalar `1`; total **7,001**. The frozen output is `64×92` = 5,888, and the frozen slow matrix is `64×64` = 4,096. Eight is the *per-shot state* dimension, not the number of learned operator parameters.

For actual progressive context8, `patch_signal` forms complete 8-sample, 23-channel patches; [ContextMemory.trace](experiments/risk_aligned_online/context_memory_20261028/model.py#L52-L111) constructs keys/bases and updates a fresh 8-vector using only available arrived patch pairs. The `writes` detach at line 75 applies to the Boolean availability schedule, not learned tensors. In **memory** mode the learned path remains `key/kernel/rate -> write(s) -> trace['delta'] -> endpoint_corrections -> frozen memory.output -> slow/second frozen Transformer block -> frozen risk head`. The detached reference norm applies to control modes, not normal `memory`. `full` decomposition is the identity. [Endpoint mapping](experiments/risk_aligned_online/delayed_recurrent_memory_20261025/model.py#L34-L38) uses `floor(endpoint/8)-1`; [risk](experiments/risk_aligned_online/delayed_recurrent_memory_20261025/model.py#L122-L131) requires one 92-vector per window. Frozen parameters must still permit autograd **through** their operations: wrapping risk in `no_grad`, using the lifecycle `memory_delta`, detaching/caching its delta, or merely changing the optimizer list while leaving `requires_grad=False` would block the intended operator gradients.

Caching `h` and `valid` is gradient-safe **for this parameter scope**, because [tokens](experiments/risk_aligned_online/internal_fast_window_20261023/model.py#L77-L91) ends at the frozen first Transformer block, before both slow and memory insertion. It is conditional on the same fitted map, frozen source/scaler, windows and masks, dtype/backend/eval semantics; do not reuse cached post-adaptation tokens, logits, or trace deltas. Recompute the trace after each operator update. Endpoint indices must also be validated against the actual shot length; patch count alone cannot identify the length of a partial final patch.

Before the latest instruction to stop experimentation, I performed one **CPU-only, in-memory synthetic connectivity check**, loading the actual hash-checked source, source-context8 and current cold checkpoints but using synthetic windows and patches. The active adapter gave `h=(4,17,64)`, `valid=(4,17)`, `means=(20,23)`, `delta=(4,92)` and 7,001 selected parameters. Cached-versus-fresh-token logits agreed exactly in that single check. Risk backward gave finite nonzero norms for key weight `2.6603e-4`, key bias `4.4163e-5`, kernel `1.3836e-4` and rate `1.0683e-5`; no frozen parameter had a gradient. The loaded cold output norm was `2.8051`, so it is not the all-zero initialization in this check. This is shape/connectivity evidence only: synthetic patches and windows were not a real paired EAST trajectory, and nonzero gradients, causality, numerical equivalence and utility are not guaranteed for all real shots, masks or backends. No optimizer step or model artifact was produced.

## Available cold20 and endpoint sampling

The existing [sampler](experiments/risk_aligned_online/lifecycle_stream_20261025/protocol.py#L42-L56) first restricts valid endpoints to zero-based index **≥127** and, for disruptive shots, lead strictly **>30 ms**. It unions up to 32 evenly spaced, 16 highest source-score, and 16 evenly spaced positive endpoints, removing duplicates. The current `disruptive_positive_only` [loss](experiments/risk_aligned_online/lifecycle_stream_20261025/learning.py#L69-L96) subsequently drops selected negative endpoints on disruptive shots; all selected safe endpoints remain. Existing population loss weights classes equally and shots equally within a class: with 18D/2S, each safe shot receives nine times the target population weight of each D shot, before window averaging. The exact future operator objective/budget must be stated separately; these are the current lifecycle settings, not an assertion about an unrun experiment.

I reconstructed the existing selection from the **completed and hash-verified** first-20 sealed NPZ files' `times` and `source` arrays, joining the already-saved [authoritative outcome records](probe_outputs/anchored_full360_diagnostic_v1/authoritative_outcomes.json) and checking each flag against current `per_shot.json`. No raw HDF5 was read in this reconstruction. The following counts are reconstructed from saved artifacts and current hash-matched sampler code, not a saved training manifest:

| Rank | Shot | Class | Unique selected | Used by current loss |
|---:|---:|:---:|---:|---:|
| 1 | 71304 | D | 62 | 21 |
| 2 | 74087 | D | 63 | 16 |
| 3 | 71265 | D | 63 | 16 |
| 4 | 71370 | D | 63 | 16 |
| 5 | 71415 | D | 60 | 30 |
| 6 | 69454 | D | 62 | 18 |
| 7 | 87494 | D | 63 | 16 |
| 8 | 79712 | D | 60 | 31 |
| 9 | 79604 | D | 62 | 16 |
| 10 | 90874 | D | 63 | 16 |
| 11 | 69765 | D | 63 | 16 |
| 12 | 78789 | D | 63 | 16 |
| 13 | 69458 | D | 59 | 30 |
| 14 | 83646 | S | 48 | 48 |
| 15 | 83346 | D | 63 | 19 |
| 16 | 89881 | D | 63 | 16 |
| 17 | 87484 | S | 48 | 48 |
| 18 | 87508 | D | 62 | 19 |
| 19 | 87487 | D | 59 | 29 |
| 20 | 87537 | D | 61 | 16 |

Total **1,210 selected**, of which **453 contribute** under that loss: **357 D-positive** and **96 safe**. Every selected shot has earliest selected endpoint 127; both safe shots use endpoint 127. These are correlated windows, not 453 independent discharges. The current cold parent and map have already used all 20, so holding out a shot only from a later operator fit would not make it independent of the parent; a genuinely held-shot assessment would need all target-fitted components refit within each fold. With two safe trajectories, even a two-fold class-stratified split leaves just one safe fit shot per fold.

## Current cold false alarms and reachable timing

Only after verifying `completed.json` status, I read the 180 safe-shot sealed NPZ files, recomputed each SHA-256 against its seal, and reproduced all 180 saved cold alarm decisions using `score >= 0.7258946895599365`. All 42 false alarms occurred in the suffix after cold fitting. Seven first crossed at zero-based index **<16**; 16 first crossed at **<128**; specifically 15 were **<127** and one was exactly **127**. The earlier-than-128 cases, in stream-rank order, are:

| Rank | Shot | First crossing index |
|---:|---:|---:|
| 101 | 85320 | 15 |
| 105 | 70258 | 127 |
| 106 | 60239 | 15 |
| 107 | 85316 | 15 |
| 108 | 75226 | 109 |
| 132 | 72511 | 22 |
| 134 | 72419 | 21 |
| 217 | 90740 | 63 |
| 248 | 70217 | 119 |
| 255 | 64562 | 29 |
| 284 | 67900 | 68 |
| 293 | 87363 | 7 |
| 295 | 93983 | 5 |
| 303 | 85422 | 5 |
| 310 | 93306 | 15 |
| 356 | 75135 | 87 |

At indices `<16`, context8's initial context yields zero correction: the first possible progressive feedback effect is index 16, conditional on an observed pair. With the same fixed source, map, slow, output, threshold, and no other intervention, operator training alone therefore cannot change those seven already-issued first crossings. Indices 16–126 are potentially affectable by an active memory path, but the existing cold20 endpoint sampler directly supervises none of them; index 127 is sampled on both safe training shots. This timing observation neither predicts how many later false alarms would be removed nor licenses a different alarm policy.

The completed current-order development baseline records cold **134 TP / 42 FP**, intra **133 / 42**, over 180D/180S including startup; intra acceptance is false. A same-parent comparison to unchanged `intra` is needed to isolate changing the operator, while `cold` is the no-fast-reference. All such comparisons remain on a development-exposed stream and would not establish independent confirmation. The cold20 was also used to fit the current slow/output parent, so no claim of held-shot independence, scientific novelty, or gain follows from this audit.

## Verification and limits

- Read both applicable root `AGENTS.md` files. This audit was delegated by the main agent to `gpt-6.1-terra` with `max` reasoning; the auditor did not launch further children. The spawn interface did not expose a service-tier argument, so fast/priority was not set or claimed. This routing statement was corrected by the main agent against its actual delegation record.
- The seven key live files for the recorded current lifecycle path (runner, lifecycle model/learning/protocol, context model, delayed memory model, first-block adapter) each matched their SHA-256 recorded in the completed run's protocol. This establishes the inspected code matches those recorded bytes, not that every possible caller is equivalent.
- Recomputed three parent/order/map hashes, reconstructed the 20-shot sampler, checked 180 safe prediction seals and the 42 saved false-alarm decisions, and performed the one explicitly synthetic CPU connectivity check described above. No new GPU work, raw-data modification, training run, code modification, or external action was performed by this audit. The sole file written is this report.
- Independent per-parameter gradients on real cold20 data, full-stream runtime/memory cost, early-prefix real-data parity, and prospective performance were **not** verified here. The saved prior [current-order verification](CURRENT_ORDER_VERIFY_20261031.json) reports its own historical GPU checks; those are not presented as work performed in this audit.
