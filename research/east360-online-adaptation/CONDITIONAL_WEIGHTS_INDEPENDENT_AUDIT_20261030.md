# Conditional-weights independent CPU audit — 2026-10-30

## Verdict and scope

Saved-array arithmetic, declared gates, fold/draw isolation and hash checks pass; **neither update family passes the candidate gate**. Direct has 3 failed conditions; representation has 5. No concrete numerical-summary or sampling-isolation bug was found within this audit. This is a cold20 component diagnostic, not an accepted adaptation method.

Only saved NPZ/JSON/tensor artifacts were read; CPU NumPy calculations and CPU `torch.load(weights_only=True)` were used. No experiment module or summary function was imported, and no forward pass, backward pass, training, GPU work, raw-signal/outcome access, deletion or neural replay was performed. Main owns neural replay. The sole new file is this report. The results-analysis workflow was used to separate independently reproduced measurements from reported training evidence.

## Independent arithmetic

For each shot (s), group (g), arm (a):
`BCE(z,y) = logaddexp(0, (1-2y)*float64(z))`;
`delta[s,g,a] = mean_windows(BCE(z_a,y)) - mean_windows(BCE(z_cold,y))`;
`group_delta[g,a] = mean_shots(delta[s,g,a])`.
Labels are 0 for safe/early and 1 for timely danger/late. Improvement/worsening uses delta < −1e−6 / > +1e−6. These are equal-shot means, not pooled-window means; no supported-row filtering was needed.

Coverage: **40 NPZ files / 760 arrays**, 2,707 opportunities per family, 21,656 arm-window BCE calculations, 432 shot/group/arm rows and 32 arm/group aggregates. All arrays have expected lengths and finite values. Records contain 527 safe, 63 timely-danger, 2,095 early and 22 late windows. Maximum absolute difference from saved per-shot fields is **8.88e−16**, and from group-summary fields **5.55e−16**. The two families' cold logits and support masks are exactly equal across all 2,707 rows.

| Family / arm | Safe mean ΔBCE (3 shots) | Timely-danger mean ΔBCE (17 shots) |
|---|---:|---:|
| direct / real | -0.1267458498 | -0.2575152152 |
| direct / reverse | -0.1295936455 | -0.2520739932 |
| direct / mean_feedback | -0.1288361718 | -0.2873886257 |
| direct / constant | 0.1076190760 | -0.2352253788 |
| representation / real | -0.1406170350 | -0.2915995004 |
| representation / reverse | -0.1406780208 | -0.2724143829 |
| representation / mean_feedback | -0.1428419887 | -0.3136522115 |
| representation / constant | 0.1605889777 | -0.3316396594 |

Both real arms improve **2/3 safe and 14/17 timely-danger shots**. The same exceptions worsen: safe 70404 (direct +0.11041625; representation +0.08136937), danger 70327 (+1.55714845; +1.55524068), 70861 (+0.03822994; +0.03459056), and 71370 (+0.00893983; +0.07809191). Direct has less mean BCE reduction than representation in both primary groups; this does not isolate parameterization as the cause, given unequal controller size (29,100 vs 14,316 parameters, independently counted in all six saved models).

Gate failures reproduced exactly:

- Direct: real does not beat reverse or mean-feedback on safe shots, and does not beat mean-feedback on danger shots.
- Representation: those same three failures, plus danger does not beat constant, plus direction-specificity threshold failure.
- Direct real/reverse median saved cosine = **0.98716724**, with |logit difference| >1e−4 in **2,691/2,707** cases; representation = **0.99521911**, **2,640/2,707**. Thus direct passes the declared sensitivity check, but that is not evidence of useful feedback.

The gate was independently reconstructed, not called from run code. Evidence: [declared gate](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_weights_20261030/run.py:52), [direct saved gate](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/conditional_weights_cold20_v1/complete.json:162), [representation saved gate](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/conditional_weights_cold20_v1/complete.json:328).

Early label-0 mean ΔBCE is +0.08279747 / +0.07881891 (direct/representation; 15/17 worsen). **This is point-label objective disagreement, not established operational harm.** Early means lead>125, whereas the current timely-alarm definition permits lead>30 without a 125 upper bound (>3000 reported separately). This audit does not evaluate that full-shot criterion. Late label-1 mean ΔBCE is −0.02836507 / −0.03259150; neither late nor early component BCE establishes alarm utility.

## Training versus held response

The fit columns below are read from hash-verified [train_groups.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/conditional_weights_cold20_v1/train_groups.json:1); held columns are independently recomputed from NPZ. **Fit predictions were not saved per window**, so fit BCE itself cannot be independently recomputed without the out-of-scope replay ([saving boundary](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_weights_20261030/run.py:252)).

| Family / fold | Fit safe Δ | Held safe Δ | Fit danger Δ | Held danger Δ |
|---|---:|---:|---:|---:|
| direct / 0 | -0.256219 | 0.110416 | -0.864930 | -0.501576 |
| representation / 0 | -0.256532 | 0.081369 | -0.823919 | -0.514265 |
| direct / 1 | -0.096890 | -0.315241 | -0.830459 | -0.307075 |
| representation / 1 | -0.097024 | -0.322486 | -0.797496 | -0.301697 |
| direct / 2 | -0.170313 | -0.175413 | -0.606944 | 0.094829 |
| representation / 2 | -0.169600 | -0.180734 | -0.578834 | -0.012285 |

All fit safe/danger shots improve in the six reported final-training summaries. All six training histories also have lower last-32 than first-32 mean minibatch loss; these are different batches, not a fixed-set loss comparison. This is evidence against complete failure to fit the teaching objective, not proof of optimization convergence.

The more discriminating contrast is **real minus mean-feedback danger ΔBCE**:

- Direct, folds 0/1/2: fit −0.034808/−0.050408/−0.072279; held **+0.000784/+0.044451/+0.047287**.
- Representation: fit −0.108463/−0.088761/−0.125841; held **+0.006238/+0.005272/+0.061167**.

All six danger comparisons therefore reverse sign on held shots. This supports failure to transfer the fitted current-feedback advantage within this conditional-parent screen, even though aggregate primary BCE still improves. It is not purely a held-only failure: direct fold-2 safe already loses to mean-feedback during fitting by +0.00002421. Direct fold-2 held danger also has a positive mean delta (+0.094829), despite 3/5 individual improvements.

## Isolation, controls and integrity

- **Three folds verified:** fit/held counts 13/7, 13/7, 14/6; no within-fold overlap; held sets partition exactly the 20 IDs. Each fold holds one safe shot. Records/folds equal the pinned prior versions, with no duplicate shot/origin/query tuples and all clocks satisfying origin=127+25k, maturity=origin+25, query=origin+50.
- **All 1,536 saved steps / 36,864 draw positions verified:** each batch has 12 safe then 12 danger draws, exclusively from that fold's fit shots; zero held, early or late teaching draws. An independent NumPy seed-131 shot-then-window sampler reproduces all six draw sequences exactly; direct/representation use identical draws within each fold. [pool and draw implementation](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_weights_20261030/run.py:31)
- **All six model artifacts checked:** saved fold/kind and parameter counts match. Using only saved fit-shot contexts (1,830/1,548/2,036 rows), independent NumPy moments reproduce every saved scale/flag exactly after float32 conversion. Saved controls' equal-shot means agree within **1.52e−7**; both families' scales and means are identical per fold. This is a deterministic statistics check, not model fitting. Unsupervised scale/mean contexts include all fit-shot groups; supervised draws remain safe/danger-only. [fit-only scale and control mean](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_weights_20261030/run.py:239)
- **Controls/support:** all 2,707 rows are supported, confirmed against saved context fractions; blocked channel 10's feedback means/fractions are zero. Real/reverse nonzero flags agree with saved raw norms. All three controls match real-update norms within **1.49e−8**; all saved norms respect 0.1 within float32 error. Reverse permutes five complete feedback blocks; mean-feedback replaces only feedback; constant replaces the whole controller context with the fit-shot mean. [control definitions](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/context_vector_20261030/teaching.py:27)
- **Interpretation of controls:** these are norm-matched direction controls, not unconditional no-information baselines: control magnitude is inherited from the actual real-context update; representation updates also depend on the actual origin Jacobian. Mean absolute real/reverse logit differences are 0.07413314 / 0.05911054; real/mean-feedback differences are 0.09718102 / 0.08933324. Perturbations measurably affect outputs but do not establish beneficial feedback. Saved `raw_norm` means pre-control-norm-matching, **not pre-radius-projection**. [state construction](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_weights_20261030/run.py:70), [norm matching](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_weights_20261030/run.py:143), [direct radius projection](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/conditional_weights_20261030/learning.py:53)
- **Hashes:** 60/60 completion-manifest artifacts, 72/72 live code/plan/test files, 72/72 ZIP members (also exact member-set equality), 6/6 frozen parent references, 3/3 pinned prior-input files, and the source checkpoint all match. Source SHA256: `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`. This verifies on-disk immutability, not an independent replay of runtime weights. The completion manifest does not self-hash or cover subsequent main-process replay output.

## Limits / blockers

No blocker to the requested saved-array audit. Missing fit prediction arrays limit independent validation of training BCE; saved scalar cosines permit checking aggregation, not reconstructing directions. Neural generation, prefix invariance and runtime source invariance remain main-process checks, not claims of this report. No raw-file rescan or new test run was performed.

The shared cold parent has already seen all 20 shots; folds isolate only the newly fitted conditional component ([explicit protocol scope](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/conditional_weights_cold20_v1/protocol.json:41)). These are 3 safe / 17 disruptive development shots and one seed, with no statistical-significance or independent-confirmation claim. **Not a 360-shot evaluation, not TPR/FPR evidence, not an FP-rescue result, and not full-stream acceptance.** Direct is historical-supervision-trained amortized parameter generation, not an online unsupervised backward update.

