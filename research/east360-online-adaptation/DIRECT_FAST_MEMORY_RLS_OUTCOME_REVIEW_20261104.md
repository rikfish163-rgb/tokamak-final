# RLS outcome review: more timely alarms, with more safe-shot alarms

**Assessment: verified within the requested artifact-and-metadata scope; share
with the limitations below.** On the fixed 360-shot development stream and
the declared nine-decimal, strictly-greater-than-30-ms rule, real feedback has
128 timely disruptions and 36 safe-shot alarms versus native's 116 and 29.
The paired change is 13 timely gains, one timely loss, eight safe false alarms
added, and one removed. The full-stream FPR is exactly 20.0%, leaving no margin
below the plan's 20% ceiling. This is a retrospective development-cohort result,
not independent prospective validation.

The direct forecast diagnostic is more mixed: averaged equally across the 340
post-startup shots, real-feedback pre-update MSE is 1.25% below fixed `W0`, but
0.089% **above** the hold-update arm; hold has lower MSE on 230 of 340 shots.
Consequently the alarm improvement over hold does not demonstrate that real
feedback consistently forecasts the diagnostic targets better than hold.

## Evidence and independent checks

Sources are the completed run's
[completion record](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/direct_fast_memory_ridge360_20261104_v1/completed.json),
[protocol](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/direct_fast_memory_ridge360_20261104_v1/protocol.json),
[per-shot results](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/direct_fast_memory_ridge360_20261104_v1/per_shot.json),
[summary](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/direct_fast_memory_ridge360_20261104_v1/summary.json),
and [HDF5 v3 metadata](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/hdf5_outcome_authority_20261104_v3/hdf5_metadata.json).
The run reports 376.4927 s and completed status. This review used CPU-only,
read-only calculations on those files and the sealed baseline; it did not read
raw HDF5, run the GPU, or change producer files.

- Rehashed all **1,086 / 1,086** completed artifacts: no mismatch, missing
  artifact, or unlisted extra file apart from `completed.json` itself. The
  latter is not self-included in its own artifact manifest; its independently
  recorded SHA-256 is
  `faeeaa386e3d160b4da4a2582adda79d3a3b226029c0767fe187bcdd90e732f6`.
- The v3 metadata file matches its completed-manifest hash
  `286f838542f5186be61696485bd3952fda66bcacc72a32a27f72a96971ea79b1`;
  both files in that v3 artifact manifest passed. This verifies the supplied
  metadata snapshot, not a new read or rehash of raw HDF5.
- All **130** locally imported Python source hashes match both their archived
  `code.zip` entries and the current files. The pinned baseline completion
  hash matches, and all 360 baseline curve seals and curve hashes verify.
- Independently joined the 360 IDs to unique HDF5-v3 EAST records, rebuilt
  each sample clock as `start + arange(samples)/rate`, and obtained exact
  equality with all 360 stored clocks. Independently recomputed all
  **5 × 360 = 1,800** alarm/lead/timely records and the full/startup/suffix
  summaries: no discrepancy with `per_shot.json` or `summary.json`.
- Each arm's first 20 entire curves equals its native curve exactly; all four
  alternative curves equal native before zero-based sample 135 on **all 360**
  shots. Native itself exactly matches the sealed `map_after20` baseline on
  all 360 clocks and curves. Each arm has 24 first threshold crossings before
  sample 135; their identity is invariant because those prefixes are equal.
- `fit.json` records `update_rule="ridge"`, 512 connection-fit steps, 9,036
  signal pairs, and exactly the first 20 labels. The hash-verified code fits
  the connection on the actual ridge trajectories (`fit.py:93-97,113-147`).
  The completed producer's end assertions require the source, trained
  injection, `W0`, and constant control to remain unchanged after fitting
  (`run.py:246-247,259-272`). This is verification of the recorded code path
  and successful assertions, not a separately saved end-state comparison.

For scoring, the fixed threshold is `0.7258946895599365`. A disruptive shot's
first eligible crossing is restricted by the producer's preterminal tolerance
`(DownTime - time)*1000 >= -1e-6` ms; the lead in milliseconds is then rounded
to nine decimal places before testing **`lead > 30`**. Safe shots count any
threshold crossing as an FP. Outcomes and down-times come only from the v3
HDF5 metadata snapshot; the 360-shot order and membership were not changed.

## Fixed-threshold outcomes

Each full360 arm has 180 disruptive and 180 safe shots. TPR means timely
alarms divided by disruptive shots; FPR means alarms divided by safe shots.
`J = TPR − FPR`; `>3 s` counts timely disruptive first alarms more than
3,000 ms before `DownTime` and is not a separate operational-success filter.

| Arm | TP | FP | TPR | FPR | J | Timely first alarm >3 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 116 | 29 | 64.44% | 16.11% | 0.4833 | 16 |
| Real RLS | 128 | 36 | 71.11% | 20.00% | 0.5111 | 21 |
| Hold feedback | 118 | 35 | 65.56% | 19.44% | 0.4611 | 19 |
| Rotated feedback | 122 | 44 | 67.78% | 24.44% | 0.4333 | 21 |
| Constant injection | 116 | 28 | 64.44% | 15.56% | 0.4889 | 16 |

All arms share the startup20 result: 8 disruptive and 12 safe, with **7 TP and
6 FP**. For the suffix340, whose denominators are 172 disruptive and 168 safe,
the respective `(TP, FP)` pairs are native `(109,23)`, real `(121,30)`, hold
`(111,29)`, rotated `(115,38)`, and constant `(109,22)`. The principal full360
result includes the startup cost; suffix340 is not an independent cohort.

## Paired shots: real feedback versus controls

“Timely gained/lost” compares each disruptive shot's strict timely indicator.
“FP added/removed” compares each safe shot's alarm indicator. Changes are
paired within the same shot, not differences between unpaired group totals.

| Comparator | Timely gained | Timely lost | Safe FPs added | Safe FPs removed |
| --- | ---: | ---: | ---: | ---: |
| Native | 13 | 1 | 8 | 1 |
| Hold | 10 | 0 | 4 | 3 |
| Rotated | 7 | 1 | 2 | 10 |
| Constant | 13 | 1 | 9 | 1 |

### Real versus native: disruption gains and loss

Leads are first-alarm milliseconds before disruption after the declared
nine-decimal normalization. “No alarm” means no preterminal crossing. Seven
of the 13 gains turn a native miss into a timely crossing; six turn a late
crossing into a timely one. Four of the 13 real-arm gains are more than three
seconds early, so a timely count alone does not determine their operational
usefulness.

| Shot | Native lead | Real lead | Change |
| ---: | ---: | ---: | --- |
| 74087 | no alarm | 7639.111720 | gained |
| 79712 | 10.060344 | 74.060344 | gained |
| 69768 | no alarm | 3791.900000 | gained |
| 69791 | no alarm | 336.000000 | gained |
| 70320 | no alarm | 211.000000 | gained |
| 87861 | 6.000000 | 4683.000000 | gained |
| 65944 | no alarm | 567.366569 | gained |
| 73405 | 27.000000 | 41.000000 | gained |
| 73403 | 9.000000 | 6277.000000 | gained |
| 77782 | 21.000000 | 38.000000 | gained |
| 76956 | no alarm | 1486.000000 | gained |
| 74970 | 28.000000 | 100.000000 | gained |
| 75603 | no alarm | 51.000000 | gained |
| 64258 | 31.000000 | 29.000000 | lost |

Safe-shot FPs **added** versus native: `87689, 70258, 75226, 94363, 73323,
73790, 64562, 83438`. Safe-shot FP **removed**: `94395`.

For the other controls, the exact paired IDs are:

- **Versus hold:** timely gained `79712, 70320, 87861, 80600, 73403, 77782,
  72578, 76956, 62592, 74970`; timely lost none. Safe FPs added `87689,
  94363, 64562, 83438`; removed `94395, 83376, 85599`.
- **Versus rotated:** timely gained `79712, 70320, 80600, 77782, 76956,
  61638, 62592`; timely lost `86155`. Safe FPs added `75226, 64562`; removed
  `70392, 76861, 83376, 94402, 62394, 93306, 85599, 87847, 89570, 75135`.
- **Versus constant:** timely gained `74087, 79712, 69768, 69791, 70320,
  87861, 65944, 73405, 73403, 77782, 76956, 74970, 75603`; timely lost
  `64258`. Safe FPs added `87689, 70258, 75226, 94363, 73323, 73790, 64562,
  83438, 67900`; removed `94395`.

## Nine-decimal deadline boundary: shot 71013

The earlier code audit noted that rounding before the deadline comparison can
in principle differ from a comparison on the binary floating-point value.
One such difference actually occurs in this run, and the requested **current
nominal convention** is the nine-decimal one:

- HDF5-v3 metadata: `DownTime=7.256 s`, start `3.0 s`, rate `1000 Hz`.
- Native first crossing: index `4226`, timestamp `7.226 s`. Binary float
  subtraction gives `30.00000000000025 ms`; normalized to nine decimals it is
  exactly `30.0 ms`, so native is **not** timely under the declared rule.
- Real's first crossing is at `7.231 s`, giving `25.0 ms`, also not timely.

Comparing the *unrounded binary floats* to 30 would change only this one
arm/shot among the 1,800 checks, making native's TP count 117 instead of 116
and adding a real-versus-native timely loss. That is a sensitivity to floating
representation, not evidence that the nominal 30.000000000 ms lead exceeded
the deadline. All results in this review use the declared normalized rule;
no post-hoc threshold was selected.

## Direct forecast error: shot-weighted, pre-update

For each suffix shot, only update rows with positive target support and, for
disruptive shots, an arrival timestamp no later than HDF5-v3 `DownTime` were
eligible. Within a shot, the mean is weighted by the row's observed target
count; the table then gives each eligible shot equal weight. All 340 suffix
shots contributed: **43,500** eligible update rows and **7,244,614** observed
target elements; no scheduled row had zero support or was post-terminal under
this criterion. These are producer-logged aggregate errors in source-asinh
coordinates, not errors recomputed from raw signals. The logged pre-update
forecast is reconstructed when its eight-point target arrives; with 32-point
updates there is no intervening update since its origin. It is not a separately
archived forecast emitted at origin time.

| Direct diagnostic predictor | Mean shot MSE | Median shot MSE | Shots where real MSE is lower / higher |
| --- | ---: | ---: | ---: |
| Real RLS feedback | 0.02828196 | 0.01674579 | — |
| Fixed `W0` predictor | 0.02863938 | 0.01754848 | 241 / 99 |
| Zero-increment persistence | 0.02941621 | 0.01711767 | 195 / 145 |
| Hold-update arm | 0.02825689 | 0.01607764 | 110 / 230 |
| Rotated-update arm | 0.02906386 | 0.01756513 | 265 / 75 |

No comparisons tied. Relative to each comparison's mean, real MSE is 1.25%
lower than fixed `W0`, 3.86% lower than zero-increment persistence, **0.089%
higher than the hold-update arm**, and 2.69% lower than the rotated-update
arm. `hold_mse` is the zero-increment persistence predictor, whereas the
hold-update arm has its own evolving weights and a separate pre-update MSE;
they are not interchangeable. Weighting update rows equally within each shot
instead gives real `0.02828192` and hold-update `0.02825684`, preserving the
same ordering.

In the 172 disruptive suffix shots, mean real MSE is `0.03534088`, versus
`0.03529945` fixed and `0.03483619` hold-update; real beats hold on 52 shots
and loses on 120. In the 168 safe shots, mean real MSE is `0.02105497`, versus
`0.02182073` fixed and `0.02152093` hold-update; despite the lower mean, real
beats hold on only 58 safe shots and loses on 110. For all 340, the median
paired `(real − hold)` shot MSE is `+0.00006991` while the mean is
`+0.00002507`; the safe-shot mean and median have opposite signs. These
shot-level asymmetries caution against treating a small change in overall mean
as a uniform improvement.

## Interpretation limits

- The RLS weight is projected to a Frobenius radius of 2; after projection it
  need not be the unconstrained ridge posterior. Logged displacement is from
  `W0`, not per-step movement. At or near the radius (`>=1.99999`) were 11,740
  of 43,500 real rows, 132 hold rows, and 13,222 rotated rows. These are
  boundary-occupancy counts, not directly logged counts of projection events.
- Rotated feedback rolls both values **and availability masks** between
  channels (`timeline.py:74-79`); the comparison changes channel support and
  value distributions as well as correspondence. The constant arm is a
  fixed first-20-derived internal injection, not a matched time-varying
  placebo. Real wins the stated alarm comparisons here, but these controls do
  not alone identify a physical mechanism.
- The connection was supervised on ended first-20 outcome labels and then
  frozen; intra-shot RLS uses arrived diagnostic signals without current-shot
  outcome labels. It is an internal token intervention, not an update to the
  source Transformer's parameters or a change to the fixed threshold. Reduced
  auxiliary MSE, nonzero fast weights, and lower in-sample post-update error do
  not by themselves establish better alarms.
- MSE values above are independently reaggregated from hashed update logs,
  with row schedules and baselines cross-checked across arms. The raw targets
  and individual channel/horizon residuals were not independently reread, so
  this review does not independently recalculate each logged target or MSE.
  Post-update MSE uses the same target as the update and is not held-out
  forecast performance.
- The source, input map, order, and 360-shot cohort have participated in
  development. The fixed-threshold, paired comparisons are descriptive; they
  do not establish prospective generalization, clinical readiness, or a
  universal physical relationship.
