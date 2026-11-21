# Conditional-likelihood cold20 independent audit — 2026-10-30

## Verdict and scope

The archived **v2 neural result is reproducible from its saved logits and outcome times**: all 17 timely-danger shots worsen, all 3 safe shots improve, and the predeclared direction screen fails. The 2,707 real-arm count means **2,707 nonzero, independently reset temporary Q/K episode states**, not 2,707 persistent parameter updates or an operational stream. V1 and v2 have exactly identical sealed and neural NPZ contents in all five folds; v1's reported failure is consistent with the safe-label shape error in the archived evaluator.

There is one important definition mismatch: v2 groups each **subsequent prediction time** as timely or not, while the plan also describes an eligibility rule on the **original endpoint and feedback-arrival time**. Of v2's 1,615 timely-danger future points, 638 from 34 episodes have an original endpoint more than 125 ms from disruption. An exploratory intersection with the plan's endpoint rule still produces 17/17 worsening and a failed screen; it is not the published gate and should not be presented as a replacement preregistered result.

This is an independent CPU-only inspection of saved arrays, outcomes, fold metadata, and code snapshots. The audit did not call the original evaluator, load the neural model, rerun the experiment, access raw shots, use a GPU, or modify the original code/results. Density-model **provenance** was inspected, not density calibration or literature; that diagnosis belongs to the main task. The results-analysis workflow was used to separate descriptive evidence from unsupported generalization; targeted TDD tests were written and run before the audit implementation and its boundary extension.

## Exact archive and v1/v2 checks

For every file below, `audit_v2.py` compares NPZ key sets, each array's shape, dtype, and exact values, and full-file SHA-256 between v1 and v2. It also matches each archive SHA against both corresponding `sealed.json`/`neural_seal.json` declarations. There are 11 sealed arrays and 17 neural arrays per fold. The SHA shown is the same for v1 and v2.

| Fold | Sealed rows | Sealed SHA-256 | Neural episodes | Neural SHA-256 |
| --- | ---: | --- | ---: | --- |
| 0 | 383 | `7420af6b96efc098b536f112b1dc7f6e7af31f47169bdf5f37f40ff8c1134aa9` | 379 | `eb89da44069467ebc29fbb8c0a25341aae5bcef46b96d9936f56bae44b8e0f6f` |
| 1 | 553 | `faa51f07df9e9201ef33416791e379e559594f0941729ef03a0155ef1c970f19` | 549 | `459b56432b9db792747e88afb85ae7bcbfa684ac5413d6e4c7b19a81b20e364f` |
| 2 | 445 | `6a24aa6677bbc5ce1703b21b533f525e5347272491a51eb390635270004b7bf9` | 441 | `e7e7869d8589689dd7f1abef94cdca4378f5f4050eb175cec206237ead4bddf2` |
| 3 | 673 | `7c0fead50aa4a03e1c06175ab6374d04157687f33caae8b9d12385c1f3702704` | 669 | `37ed5882d012aaf64020c3b53754069e135039c7dc6c6f80e1ab7091412dad68` |
| 4 | 673 | `16e482daa79e75af7fe22e39404af0f5218d3b350dc9039e87e38022cb1d70dc` | 669 | `d71261f0f3dd28aa307207a35756d09dd18c0f506485ab1052c6a5a6b647d002` |

The eight observation arrays also agree exactly and byte-for-byte: 2,727 rows, SHA-256 `8e21a7e6a7716ec00578b0ed8232a6c232de1bfca97b57d477ca417da15c4e1d`. V1/v2 outcome JSON data agree, and the ten corresponding real/misplaced density-model files also have matching SHA-256 values; their individual hashes are in `probe_outputs/conditional_likelihood_cold20_audit/audit_final.json`.

The two archived code ZIPs have identical member names. Exactly two members differ: the experiment `run.py` and its run test. Their run-file diff adds `evaluation_labels`, flattening and restoring the future-time shape, and switches the neural evaluator to that helper; the test diff adds a safe/disruptive shape regression. Archived `neural.py` and `density.py` are byte-identical between versions. V1 `failed.json` reports a broadcast error `(141,)` against `(141,25)` after all five neural seals. Fold 0's first safe shot 70404 has 141 episodes, and the audit test independently reproduces that NumPy broadcasting failure without calling the original evaluator. This supports an evaluation-interface failure, not a successful v1 result; it does not by itself verify the correctness of every upstream computation.

**Moving-source provenance:** during this audit the live `experiments/risk_aligned_online/conditional_likelihood_20261029/run.py` was changed separately to add a cross-fit option. At inspection it had SHA-256 `40f5af01b89a460b428e9978645eb8e05ee651196dbee31957f0517aad72ea23`, not the v2 archived run SHA-256 `8a20fbd96fbb4260cc82ebd914f201ae4a3f3b5191de6d1f178e8439b0cd0bae`. This audit did not make that change. Interpret v1/v2 using their archived snapshots, not the moving live file.

## Independent future-BCE calculation

The audit loads each `neural.npz` and `outcomes.json` directly. It independently constructs labels and masks: for disruptive shots, `lead_ms = (down - future_time) * 1000`, valid if `lead_ms >= -1e-6`, positive if valid and `lead_ms <= 125 + 1e-6`; timely positive if the lead rounded to 9 decimal places is `>30` ms, and late otherwise. Early-normal points are valid negatives. Safe shots have zero labels and all saved future points valid. For each shot/arm/group it computes stable `logaddexp(0,z) - y*z` before and after; it averages pointwise delta within the shot, then averages those per-shot means with equal shot weights. No original evaluation or grouping function is called.

The audit recreates all **162** published per-shot records, including windows, timepoints, before/after/delta BCE and logit-change statistics; maximum absolute numerical difference from `neural_per_shot.json` is **0.0**. Every published summary field and the gate/reasons match `completed.json` within the audit's `1e-12` assertion. Full records are in `per_shot.json`; the final machine-readable summary is `audit_final.json`.

| Group / arm | Shots | Episode windows | Future points | Mean shot delta BCE | Improved / worsened |
| --- | ---: | ---: | ---: | ---: | ---: |
| Timely danger / real | 17 | 80 | 1,615 | +0.04758696792417115 | 0 / 17 |
| Timely danger / misplaced | 17 | 80 | 1,615 | -0.15731325165834462 | 11 / 6 |
| Safe / real | 3 | 527 | 13,175 | -0.03644510721200444 | 3 / 0 |
| Safe / misplaced | 3 | 527 | 13,175 | -0.03168387522850369 | 2 / 1 |
| Early normal / real | 17 | 2,112 | 52,587 | -0.021309601160193776 | 17 / 0 |
| Late danger / real | 17 | 22 | 298 | +0.0475488611669677 | 0 / 17 |

Negative delta is improvement. The 25 future points of one episode may span multiple groups, so group *window* counts are not mutually exclusive; future points are. The actual screen requires at least two shots in each of safe and timely danger, a negative real mean, at least half the shots improved, and a real mean strictly below the misplaced mean in **both** groups. Safe meets these conditions. Timely danger fails all directional expectations: its real mean is positive, 0/17 improve, and real is worse than misplaced. Thus `eligible_for_full_stream = false`, reason `danger_timely`.

### Origin-time sensitivity, kept separate from the published gate

The plan's endpoint criterion is an origin lead of at most 125 ms and an arrival lead strictly greater than 30 ms. For each archived timely future point the audit also joins the original sealed endpoint and arrival, then intersects the future-point mask with that criterion:

- **34/80 episodes**, comprising **638/1,615 points**, have origin lead `>125 ms`; these appear in all 17 disruptive shots. The remaining **46 episodes / 977 points** meet the origin/arrival criterion; zero timely points fail for some other origin/arrival reason.
- On that restricted set, real danger mean delta is **+0.04660445150330624**, with **0 improved / 17 worsened**; misplaced is **-0.17116153342686913**, with **11 improved / 6 worsened**. The safe figures do not change. The exploratory restricted screen also fails.

The plan specifies endpoint eligibility for the main statistic and separately discusses subsequent-prediction evaluation, so the precise intended finite-step interpretation deserves clarification. This sensitivity analysis makes the difference visible; it does not redefine the archived v2 result or claim a prespecified alternative test.

## Fold isolation, controls, update counts, and bounds

- Five folds each contain 16 fit and 4 held shots, with no overlap; each fold's union is the 20-outcome universe and each of the 20 shots is held exactly once. Sealed and neural files contain exactly the declared four held IDs in each fold. The 80 training-access entries and 20 evaluation-access entries match the declared ordered roles. Both versions' two seal JSONs per fold match the split, report zero held-outcome reads during scoring, and match their respective archive hashes. Collection metadata records zero outcome reads while encoding.
- The audit loads only the saved density-model metadata with `torch.load(..., map_location='cpu', weights_only=True)`. Across **5 folds × 2 arms × 23 channels × 2 classes = 460 class records**, every reported fitting shot ID is in that fold's fit set, no reported fitting ID is held, and `nshots` equals the unique saved ID count. This checks recorded provenance, not the unrecorded row-by-row execution history.
- All 2,707 complete neural episodes have nonzero saved state norm in both real and misplaced arms and zero in the identical arm. Their per-fold counts are **379, 549, 441, 669, 669**. All 67,675 saved future logits differ from baseline in each of real and misplaced; exactly zero differ in identical. Zero episodes in any arm have zero supported-channel counts. This is an exact comparison of saved arrays, not a separate neural forward pass.
- The identical control's two density columns are exactly equal to each other and to real class-0, its counts equal real counts, its saved evidence gradients, state norms, and changed flags are exactly zero, and its future-logit arrays equal baseline exactly. Every identical per-shot delta BCE is zero.
- Every saved norm is in **[0, 0.05] without tolerance**. Maxima: real **0.04891529306769371**, misplaced **0.04970547556877136**, identical **0**. Changed flags exactly match `norm > 0`. Using each neural episode's own saved origin logit and paired sealed densities, independently recomputed `p - posterior` agrees with saved gradients to max absolute error **1.7985612998927536e-14**; saved nonzero norms agree with `0.05 * abs(p - posterior)` to **1.0260059440936331e-08**, below the `1e-7` floating-point tolerance. Saved state vectors and Q/K gradients are not present, so their directions cannot be independently reconstructed from these artifacts.
- The sealed collection logit and neural origin logit are not generally bit-identical: **984/2,707** pairs are exact and max absolute difference is **4.76837158203125e-06**. The audit uses the neural origin logit for the feedback check, not the separately batched sealed logit. The archived routines use different batch sizes (32 versus 8), but this audit does not establish the precise cause of the small difference.

## Timing, exclusion, and leakage boundaries

Source inspection of the archived code shows 128-point input windows ending at each origin, H25 target samples at `end+1 ... end+25`, and later neural predictions at `end+26 ... end+50`; the temporary state is formed before the later `_predict` calls. Each episode is reset rather than carried to the next endpoint. The audit does not run that code or prove its execution chronology from wall-clock instrumentation.

Independently on the saved arrays, every neural `(shot, end)` is joined to its sealed endpoint and the next sealed endpoint exactly 25 samples later. The current arrival time equals the next origin time exactly, all 25 future times are strictly increasing and strictly later than arrival, and the final future time equals the next endpoint's saved arrival time exactly. Across all episodes, origin-to-arrival is 25 ms, first future time is 1 ms after arrival, and each next time advances 1 ms; the maximum absolute saved-cadence error is **2.3306009899748403e-15 seconds**. The source constructs offsets `+26 ... +50`; the NPZ stores the times, not every original raw index, so exact raw-index provenance is a source/metadata claim rather than independently replayed raw-data evidence.

There are **2,727 sealed endpoints** and **2,707** complete subsequent-H25 episodes: exactly **one incomplete last endpoint per shot**, 20 total, matching the neural seal's per-shot exclusion records in both versions. All **67,675/67,675** saved future points are valid under the recorded outcome-time convention; there are **0 post-terminal future points**, **0** episodes with no or partially valid future points, and **0** episodes whose feedback arrival is post-terminal. Safe outcomes have no disruption time, so their validity follows the archived safe-shot convention rather than an independently checked terminal timestamp.

The fold-level evidence supports a held-out **feedback component** only. The fixed parent, input map and persistent state had already used all 20 shots, as explicitly stated in the experiment plan/protocol; this is not leakage-free transfer cross-validation or an independent test. Held shots' outcomes are legitimate fit labels in *other* folds and can be read earlier in the same process; `evaluation_after_all_seals` applies to the final evaluation reads, not a claim that no outcome for any held shot was ever read before all folds were sealed. The scoring patch denies `io23.read_outcome` inside scoring, and saved model metadata exclude held IDs from that fold, but logs and SHA seals are self-reported integrity evidence, not signed proofs of time order or all possible accesses. Full trajectories were loaded before this offline diagnostic; saved times and source guards do not establish a live scheduler or independently prove all future-input isolation.

Other limits: the archived `completed.json` reports source weights unchanged, but this audit did not independently compare the original in-memory weights or saved Q/K vectors. The misplaced arm is a within-shot half-sequence label rotation, not an independently randomized holdout. The 17 disruptive/3 safe shots are a small development sample; no significance, independent confirmation, online alarm TPR/FPR, or continuous-stream benefit is established here.

## Reproduction, commands, and results

Working directory: `/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028`. The exact final targeted test command was:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -m pytest -c /dev/null --rootdir=. --noconftest -p no:cacheprovider -q tests/test_conditional_likelihood_audit.py
```

Observed final result: `10 passed in 2.99s`, exit 0. Earlier TDD runs failed first because `audit_v2.py` did not exist, then because the origin-boundary function/extra count was not yet implemented; subsequent targeted runs passed. No coverage percentage, full repository suite, GPU experiment, or original evaluator run is claimed.

The audit outputs were created by these successful, CPU-only commands, each exit 0:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python experiments/risk_aligned_online/conditional_likelihood_20261029/audit_v2.py --write
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python experiments/risk_aligned_online/conditional_likelihood_20261029/audit_v2.py --write-boundary
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python experiments/risk_aligned_online/conditional_likelihood_20261029/audit_v2.py --write-final
```

`audit.json` is the initial summary; `origin_boundary.json` is the separately saved sensitivity; `audit_final.json` is the complete final summary; `per_shot.json` contains all 162 independently recalculated records. The output options deliberately refuse to overwrite existing files. To repeat the full computation without writing, omit the output option. Additional read-only checks used `sha256sum` on the NPZ/model files, `diff -u` on code extracted from the two ZIP snapshots, and `rg`/`nl` to inspect the original label, target, window, plan, and timing contracts.
