# Sealed H25 sequential-fit diagnostic: fixed protocol

Fixed before running the new diagnostic. This is a retrospective, CPU-only
secondary analysis of the completed development experiment, not an online
method, a risk/alarm gate, new training, or an independent confirmation.

## Frozen inputs and preconditions

- `FACTORIAL_TASK_INTERFACE_REVIEW.md` SHA-256:
  `54028b334ba70070a326d36f7c0ba7ae80ed93d7a01978801ce43c1b175b5164`.
- `probe_outputs/local_factorial_east360_v1/complete.json` SHA-256:
  `8898224f755c1129411de2942789f32c0a9e979edc7f418f20b92593a23d56e2`.
- Its existing `audit.json` SHA-256:
  `7cf661f42b89d2454a757521bd891ca0364af5362b97197cd159ed87e4e02fca`;
  require `complete`, the above completion hash, matching audit-code hash,
  all first-passing checks, and the recorded full-360 audit scope. Do not
  rerun or replace the original audit. The original neural replay covered
  selected changed shots plus fixed positions, not all 360 neural replays.
- Independently verify the completion's hashes for protocol, per-shot rows,
  receipt list and auxiliary artifacts; verify the fixed prior, source input
  hash list and input map against the sealed protocol. For each suffix shot,
  verify original HDF5 and curve hashes, its seal against the sealed aggregate
  receipt, order, clocks, outcome, cold decision and raw masks. A failed seal,
  mask, required arm, nonfinite/missing observed loss or other contract check
  makes this analysis **insufficient evidence**, not a reason to select a
  different sample.

## Fixed retrospective sample and matching

Use the last 340 of the 360 ordered shots after cold20. For each disruption,
among feedback clocks `256, 384, ...` before the end of the recorded curve,
select the **last** with `np.round((down-time[clock])*1000, 9) > 30`, raw
`(down-time[clock])*1000 <= 125`, and no cold threshold crossing before the
clock. Equality at the lower bound is ineligible; the upper bound is not
rounded. The selected time uses the outcome and cannot be an online clock.
If that last clock has no observed target, exclude the shot without trying an
earlier clock. List no-clock, prior-alarm, no-target and unmatched exclusions.

For safe shots consider all feedback clocks before any cold crossing with at
least one observed target. Reconstruct each complete 2 x 23 channel mask from
the raw arrived prefix and fixed prior: CIII values within -10 +/- 0.01 are
unavailable; inactive prior channels (including MNM) are excluded; each of
`[clock-51:clock-25]` and `[clock-26:clock]` must be entirely finite and
available per channel. Verify fixed input-map finiteness and representability.
For each feedback and each of the nine factorial arms (and prior_true), verify
the exact common mask contract established by the archived implementation and
that `observed == 25 * mask.sum()`; each observed `loss_before` must exist and
be finite. Verify no-observation records separately. The receipt contains only
counts, so equal counts alone are not treated as proof of equal masks: equality
also depends on the pinned production implementation's single shared `valid`
mask and loss indexing, plus independently reconstructed channel identities.

Process eligible disruptive shot IDs ascending. Among currently unused safe
shots and all their candidate clocks require exact equality of both 23-channel
mask rows, absolute clock difference <=128 **samples**, and absolute cold
probability difference at index `clock-1` <=0.05. Choose the lexicographic
minimum `(probability gap, clock gap, safe shot ID, safe clock)`. Use each safe
shot at most once. Never select on losses, q, acceptance, arm outcomes, or any
effect. The old feasibility figures 45/41/29 are reference observations, not
targets or assertions; report the newly calculated values and any differences.

## Estimands and decision

Read all nine sealed pre-update losses `L[history,target]` at each matched
point. For each history H=true, reverse, hold, compute per point
`q_H = L[H,reverse] - L[H,true]` and the separate target-zero contrast
`r_H = L[H,hold] - L[H,true]`. This latter is NOT a hold-last prediction
baseline comparison. For either measure report the difference between the
means for matched disruptive and safe observations and its interval.

Generate exactly one `numpy.random.default_rng(131).integers(0, n,
size=(2000,n))` paired index matrix; use it for all histories, both measures,
and all three screening contrasts without rematching. Report NumPy linear
percentile 95% intervals, with no multiple-testing or confirmatory claim.
The three fixed screening contrasts are `D_true`, `abs(D_true)-abs(D_reverse)`
and `abs(D_true)-abs(D_hold)`, where each D is the mean of pair differences;
recalculate the absolute-difference contrasts within each bootstrap replicate.

Only when n>=20, the `D_true` interval excludes zero, and both absolute-gap
intervals have strictly positive lower bounds, label this specific screen
supported. With n>=20 but any screening condition unmet, stop this scalar
route; n<20 or failed seal/mask/feedback verification means insufficient
evidence. Do not change clocks, thresholds, losses, matching, or steps in
response to results. Report every pair, exclusions, all three D intervals,
all target-zero controls, and per-arm acceptance/abstention distributions.

## Interpretation limits

The selected points use labels after the fact. The bootstrap describes only
sampling variation of this fixed matched set, not rematching, development
selection, whole-stream significance, or prospective generalization. Matching
cold probability and clock is not matching the complete prefix. No inference
of conditional information beyond the prefix, physical causality, deployable
score, or alarm benefit is permitted. A null screen does not rule out other
representations or diagnostics.
