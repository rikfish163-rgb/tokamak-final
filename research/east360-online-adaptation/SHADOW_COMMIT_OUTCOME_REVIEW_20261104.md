# Shadow commit: lower auxiliary error did not establish a policy gain

**Assessment: verified within the saved-receipt and metadata scope; the
prespecified policy test did not achieve the goal.** With the guard-trained
connection, guard yields **123 TP / 36 FP**, versus delay's **125 / 38**:
both have full360 `J = 0.483333`. Guard loses two timely disruptions, adds
one safe false alarm, and removes three. With the delay-trained connection,
guard yields **124 / 34**, versus delay's **126 / 37**, an approximately
`+0.005556` change in J but still two fewer timely disruptions. These are
descriptive results on the development cohort, not an independent validation.

The issued **active** diagnostic forecasts have lower shot-averaged MSE under
guard on 213 of 340 suffix shots and a 0.795% lower mean than delay. That
small auxiliary advantage did not translate reliably into better paired alarm
outcomes: guard lost both disruption decisions that changed at either fixed
connection. No post-hoc MSE threshold or threshold sweep was performed.

## Inputs, integrity, and scoring

The reviewed evidence is the completed run's
[completion record](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/shadow_commit360_20261104_v1/completed.json),
[protocol](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/shadow_commit360_20261104_v1/protocol.json),
[per-shot scores](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/shadow_commit360_20261104_v1/per_shot.json),
[summary](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/shadow_commit360_20261104_v1/summary.json),
sealed curves, forecast receipts, update histories, and the
[HDF5 v3 metadata snapshot](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/hdf5_outcome_authority_20261104_v3/hdf5_metadata.json).
The recorded elapsed time is `600.3279 s`. This review used CPU-only, read-only
computations; it did not read raw HDF5, use the GPU, modify producer code, or
rerun the experiment.

- Independently rehashed **1,429 / 1,429** completed artifacts: no missing,
  extra, or mismatching artifact, excluding the completion record itself from
  its own manifest. The completion record's SHA-256 is
  `2084a181db6bcf7a6614a0b5b08368df6a969b6ae055e5ab0b573bebd4d5b429`.
  All **135** recorded local source hashes match both `code.zip` and current
  source files. The two v3 metadata-manifest artifacts also matched; this is
  not a fresh raw-HDF5 verification.
- Independently reconstructed all 360 clocks from v3 `start`, `rate`, and
  `samples`; all match the saved clocks exactly. Recomputed **7 × 360 = 2,520**
  alarm, lead, and timely records and all three summary partitions with no
  discrepancy. Native matches the sealed `map_after20` baseline curve on all
  360 shots. All seven arms match native for every startup20 curve and for
  samples `0:167` on every shot.
- The threshold is fixed at `0.7258946895599365`. As in the producer, first
  preterminal alarm lead is rounded to nine decimal places in milliseconds,
  then tested with **`lead > 30`**. HDF5-v3 `IsDisrupt` and `DownTime` are the
  outcome/time authority. The known boundary case is native shot `71013`:
  `7.256 − 7.226 s` is nominally 30 ms; binary subtraction gives
  `30.00000000000025 ms`, while the declared nine-decimal rule gives `30.0`
  and therefore not timely. No other arm/shot changed under that raw-float
  sensitivity check; all results below use the declared normalized rule.
- The completed code asserts frozen source, connections and initial `W0`
  after the first-20 fit, and the completion record reports those assertions
  passed. The first-20 label IDs match the run and fit records, and no suffix
  training is recorded. The [integration audit](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/SHADOW_COMMIT_INTEGRATION_AUDIT_20261104.md)
  describes the code path and its remaining limitations.

## Fixed-threshold outcomes and paired policy changes

Full360 has 180 disruptive and 180 safe shots. `TPR` is timely disruptions /
180, `FPR` is safe-shot alarms / 180, and `J=TPR−FPR`. `>3 s` counts very
early timely first alarms; the main score does not exclude them.

| Arm | TP | FP | TPR | FPR | J | Timely first alarm >3 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 116 | 29 | 64.44% | 16.11% | 0.483333 | 16 |
| Guard connection × guard | 123 | 36 | 68.33% | 20.00% | 0.483333 | 19 |
| Guard connection × delay | 125 | 38 | 69.44% | 21.11% | 0.483333 | 20 |
| Delay connection × guard | 124 | 34 | 68.89% | 18.89% | 0.500000 | 19 |
| Delay connection × delay | 126 | 37 | 70.00% | 20.56% | 0.494444 | 20 |
| Guard connection × hold | 120 | 35 | 66.67% | 19.44% | 0.472222 | 20 |
| Guard connection × reverse | 118 | 37 | 65.56% | 20.56% | 0.450000 | 17 |

All arms have the same startup20 result: 7 TP among 8 disruptive shots and
6 FP among 12 safe shots. Suffix340 has 172 disruptive and 168 safe shots.
The full360 result includes startup cost. Guard's 20.00% FPR with the
guard-trained connection is exactly at the plan's ceiling, not below it.

Holding each connection fixed, the exact paired differences are:

| Connection | Timely gained by guard | Timely lost by guard | Safe FP added by guard | Safe FPs removed by guard | ΔJ guard − delay |
| --- | --- | --- | --- | --- | ---: |
| Guard-trained | none | `87861, 76956` | `94395` | `70392, 94363, 64562` | `0.000000` |
| Delay-trained | none | `87861, 76956` | `94395` | `87689, 94363, 64562, 89570` | `+0.005556` |

For comparison, guard-connection × guard versus native gains timely shots
`74087, 79712, 69768, 69791, 70320, 65944, 71086, 73405, 73403, 74970,
75603`, loses `63220, 64258, 62592, 85683`, adds safe FPs `87689, 70258,
75226, 73790, 83376, 85599, 87847`, and removes none. The equal J reflects
the net `+7 TP` and `+7 FP`, not identical individual outcomes.

## Actual issued-forecast and commit receipts

For each of the four traces on each suffix shot, the independent audit read
the saved issue-time candidate and active forecasts, actual released targets,
and masks. It recomputed float64 masked MSEs and the exact strict guard or
supported-row delay decision, rather than trusting the recorded `accepted`
field. Candidate and active MSEs agreed with logged values to maximum absolute
roundoff `4.45e-16`; **0 / 174,000** released decisions disagreed.

| Trace | Issued | Released / supported | Pending | Accepted | Rejected | Actual weight changes recorded and receipt-consistent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real feedback, guard | 43,667 | 43,500 | 167 | 22,626 | 20,874 | 22,626 |
| Real feedback, delay | 43,667 | 43,500 | 167 | 43,500 | 0 | 43,160 |
| Hold feedback, guard | 43,667 | 43,500 | 167 | 24,714 | 18,786 | 24,714 |
| Reverse feedback, guard | 43,667 | 43,500 | 167 | 23,627 | 19,873 | 23,627 |

Every released row had support; there were no zero-support releases. Each
issue index followed `127 + 32k`, each release was exactly `q+8`, and each
pending entry had `released_at=-1` with a zero target placeholder and
all-false target mask. The 340 accepted-but-unchanged delay events are the
first `W0` adoptions, one per suffix shot.

All real-guard versus real-delay **candidate forecast arrays and candidate
versions** matched exactly, including pending entries. Targets and target
masks matched exactly across all four traces. In particular, reverse feedback
uses the same observed availability as real feedback; the hashed source
negates the feedback target only after validating on the actual target. This
is a mask-matched wrong-value control, unlike the earlier channel rotation.

For accepted events in this run, every recorded weight change had differing
candidate/active forecast vectors; the remaining accepted-but-unchanged cases
had the same candidate and active version. Version transitions and all
recorded change flags were consistent with the audited source, with no
ambiguous accepted equal-forecast/different-version events. Receipts do **not**
store the full candidate weight or covariance, so this reconstruction relies
on the hash-verified version/update semantics; it is not an independent
per-event hash of the weight matrices. Receipt issue times are logical replay
times, not separate wall-clock seals at each sample.

## Pre-commit active forecast error versus risk

This analysis uses the active forecast **saved at issue**, before the
corresponding commit. For each suffix shot it includes supported releases
whose arrival is no later than the HDF5 disruption time, weights each row by
its observed target-element count, and then weights each shot equally. All
340 suffix shots contributed: **43,500** rows and **7,244,614** observed
target elements per real-feedback trace; none were excluded as unsupported or
post-terminal. The values are recalculated from saved forecast/target/mask
arrays in source-asinh coordinates, not from raw signals.

| Group | Shots | Guard mean active MSE | Delay mean active MSE | Guard lower / delay lower |
| --- | ---: | ---: | ---: | ---: |
| All suffix | 340 | 0.02806929 | 0.02829410 | 213 / 127 |
| Disruptive | 172 | 0.03507085 | 0.03534499 | 108 / 64 |
| Safe | 168 | 0.02090102 | 0.02107534 | 105 / 63 |

There were no tied shot means. Overall the mean difference is
`−0.0002248162`, or **−0.795%** relative to delay's mean; the median paired
difference is `−0.0000140942`. Candidate forecast errors match between real
guard and real delay because their saved candidate sequences and masks match;
the active states differ due to the commit policy.

The paired risk changes are too sparse and mixed to support an inference that
lower auxiliary MSE reliably causes better alarms:

- Among 172 disruptive suffix shots, guard's active MSE was lower on 108 and
  delay's on 64. The only two timely decisions changed were **both losses for
  guard**: `76956` had lower guard MSE, while `87861` had lower delay MSE.
  For `87861`, delay's first alarm was `4732 ms` early, which is counted as
  timely by the fixed rule but is also more than 3 seconds early.
- Among 168 safe suffix shots, guard's active MSE was lower on 105 and delay's
  on 63. Under the guard-trained connection, the guard-lower group contains
  three removed FPs, one added FP, and 101 unchanged shots; all 63
  delay-lower shots are unchanged. Under the delay-trained connection it
  contains four removed, one added, and 100 unchanged; again all 63
  delay-lower shots are unchanged.
- As descriptive associations only, using `x = guard shot MSE − delay shot
  MSE` and `y = guard − delay` in the timely indicator for disruptive shots
  gives Pearson `r = −0.0136` and Spearman `ρ = 0.0022` for either connection.
  Using `y = guard − delay` in the alarm indicator for safe shots gives
  Pearson/Spearman `0.0186 / 0.1137` for the guard-trained connection and
  `0.0009 / 0.1182` for the delay-trained connection. Only 2 disruptive and
  4 or 5 safe outcomes changed, respectively; these coefficients are not
  stable inferential evidence. For safe shots a positive `y` means an added
  false alarm, not a benefit.

Shot-level MSE summarizes the whole preterminal trajectory and can include
releases **after** a first alarm. Therefore this association is retrospective
and is not a claim that a future diagnostic error predicted an earlier alarm.
Accepted validation-window improvements are also selected by the guard rule
and cannot alone demonstrate next-window or risk improvement.

## The three earlier RLS false alarms

These three HDF5-v3 safe shots were new false alarms of the previous immediate
RLS real arm at zero-based sample `135` (`3.135 s`). The table shows each
first threshold crossing in the current run; `none` means no crossing over the
saved shot. The old RLS model used a separately trained connection and is a
historical reference, not the same-connection delay control.

| Shot | Old immediate RLS | Guard-trained × guard | Guard-trained × delay | Delay-trained × guard | Delay-trained × delay |
| ---: | --- | --- | --- | --- | --- |
| 87689 | `135` / 3.135 s | `1249` / 4.249 s | `167` / 3.167 s | none | `167` / 3.167 s |
| 70258 | `135` / 3.135 s | `167` / 3.167 s | `167` / 3.167 s | `167` / 3.167 s | `167` / 3.167 s |
| 64562 | `135` / 3.135 s | none | `167` / 3.167 s | none | `167` / 3.167 s |

Thus `guard_guard` removes the old `64562` FP, but `70258` is only delayed by
32 samples and `87689` reappears 1,114 samples later. The delay-only arms
retain all three at sample 167. `delay_guard` has no crossing on `87689` or
`64562` but still crosses on `70258`. Native has no crossing on any of them.

## Limits and known reporting issue

- The already-reported producer precision inconsistency remains unchanged:
  `static_mse` subtracts in float32 before conversion to float64, while the
  decision's candidate/active errors convert before subtraction. No claim in
  this review rests on a tiny `static_mse` advantage. The active-forecast
  comparisons above were independently recalculated from saved arrays.
- Reverse and hold trajectories still use the **real** matured target for
  guard validation; they are wrong-feedback controls conditional on real
  validation, not controls with all actual diagnostic access removed.
- The 360-shot order, numerical preparation, model and protocol have been
  developed using this cohort. Whole-shot file reads and logical causal replay
  do not establish real-time acquisition, latency, prospective performance,
  clinical readiness, or a physical mechanism. Forecast improvement and
  nonzero weight changes are not substitutes for paired alarm outcomes.
