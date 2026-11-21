# Direct fast-memory integration audit — 2026-11-04

## Scope and conclusion

Read-only review of the frozen producer, plan, supporting input/scoring code,
and CPU test source. No raw shot files, GPU, network, or per-outcome live-run
outputs were accessed. The only live-run artifact inspected was the protocol's
code/plan hash metadata. No live producer, plan, or running experiment was
modified. The user reported 62 CPU tests and a CUDA real-update smoke test;
those results were not independently rerun in this audit.

**Bottom line:** the inspected replay is causally consistent at the level of
its pointwise model inputs and matured signal targets, conditional on the
existing fixed map and already-ended prefix. Its `pre_update_mse` is an
arrival-time reconstruction of the direct forecast that *would* have been
available eight samples earlier, not proof that such a forecast was separately
issued or stored then. The real/hold/rotated/constant arms are useful but do
not isolate every causal explanation for a difference. One inherited,
extremely narrow strict-deadline rounding defect is confirmed below; whether
it affects any actual shot is unknown.

References below are relative to this worktree. `timeline.py`, `memory.py`,
`model.py`, `fit.py`, and `run.py` refer to
`experiments/risk_aligned_online/direct_fast_memory_20261104/`.

## Confirmed defect: strict 30 ms comparison uses a rounded value

`experiments/risk_aligned_online/hdf5_score_recheck_20261103.py:25-33`
rounds the first-alarm lead to nine decimal places in **milliseconds**, then
uses that rounded value for `lead > 30`. This is not exactly the contractual
strict comparison on the unrounded timestamps. An independent CPU-only
synthetic check gave:

```text
unrounded lead = 30.0000000004000 ms  -> strictly > 30: True
stored lead    = 30.0 ms              -> reported timely: False
```

The discrepancy is confined to an approximately 0.5 picosecond band above
the deadline, subject to floating-point representation. It is an inherited
scorer defect, not evidence that any actual result is affected. After the run
is complete and outcomes may be examined, compare the original classification
against the unrounded `(down - first_alarm_time) * 1000 > 30` for every arm and
shot; report affected counts and any metric sensitivity. Do not change the
producer or silently replace the frozen outputs during the live run.

## Causal keys, target arrival, and forecast provenance

1. `simple_data.py:41-48` constructs each 128-point window through its own
   endpoint, with left padding; `timeline.py:11-28` constructs a key from
   each such window. `model.py:13-19` applies fixed source asinh coordinates,
   and `memory.py:68-85` uses only the masked last value, the two trailing
   eight-point means, and trailing availability. The input sentinel rule is
   pointwise (`causal_input.py:87-133`); the fixed post-prefix map is applied
   pointwise without changing masks (`east_evaluation.py:62-80`,
   `alignment.py:140-156`). Thus precomputing all keys does not by itself make
   an earlier key depend on later values. `test_direct_fast_memory_timeline_20261104.py:41-48,60-66`
   explicitly checks truncation invariance and window endpoints.
2. At arrival `t=135+32k`, `timeline.py:31-47,65-67` uses origin `q=t-8`,
   its 128-point history, and actual samples `q+1` through `q+8=t`. The eight
   *time positions* have arrived; individual target elements can still be
   unavailable. `memory.py:88-107` uses future-minus-origin-last in the 21
   continuous coordinates, excluding indices 2 and 10, and requires both
   corresponding masks. It does not use a synthetic target for the real arm.
   The feedback test checks the exact eight actual samples
   (`test_direct_fast_memory_timeline_20261104.py:28-38`); the core tests
   check true-last, target order, masking, and spatial differences
   (`tests/test_direct_fast_memory_core_20261104.py:37-138,257-270`).
3. `timeline.py:69-90` calculates `before = W @ key` before the current
   update and records its error against the actual arrived target. For the
   first update the state is `W0`; thereafter the previous update is at
   `t-32`, earlier than `q=t-8`, and there is no intervening update. The key
   at `q` is explicitly checked against the cached key at `q`
   (`timeline.py:65-68`). Therefore this reconstruction is mathematically the
   same direct forecast available at `q`, under the current cadence and
   ordering. A separate CPU-only synthetic replay independently recorded the
   issue-time candidate and reproduced the logged MSE exactly at six update
   points for each of real, hold, and rotated arms; maximum difference was 0.
4. The producer does **not** store or emit a direct forecast at `q`: it
   calculates the diagnostic `before` at arrival and stores an aggregate
   error (`timeline.py:69-99`). The actual risk path uses
   `(W-W0) @ current_key`, not `W @ key` (`timeline.py:101-106`,
   `model.py:22-30`). Accordingly call this an **equivalent reconstructed
   pre-update auxiliary forecast error**, not an independently archived
   issued forecast or the original Transformer's forecast-head error.
5. The current target ends at `t`, and the updated state is used for risk at
   `t` (`timeline.py:76-81,101-106`). This is causal only under the stated
   event order: the current signal sample arrives, then the update occurs,
   then the risk for that sample is evaluated. It is not a
   predict-before-current-sample-arrival convention. Outcome labels are
   different: `run.py:140-173,180-185` blocks `read_outcome` until after the
   full curve is written and sealed. `io23.py:62-96` reads signal datasets,
   whereas `io23.py:99-116` separately reads the HDF5 outcome. The preceding
   whole-file integrity hash (`run.py:129-132`) does read file bytes, including
   metadata bytes, but does not parse an outcome or feed it to prediction.

This is an **offline causal replay**, not a real-time deployment test:
`run.py:140-150` loads the whole current signal record; `run.py:37-62` first
constructs whole-shot innovation trajectories and then computes risk curves
in batches. Prefix dependence is constrained as above, but actual online I/O,
latency, and emission timing are not demonstrated by this implementation.

## Reset, freeze, first 20, and fitting provenance

- For each call and arm, `timeline.py:60-63` sets the origin to the common
  `W0` and clones a fresh shot-local state. `memory.py:149-170` implements the
  bounded update, and `timeline.py:101-106` injects only state displacement.
  Reset and discard are tested at
  `test_direct_fast_memory_timeline_20261104.py:51-57`. A history row with no
  observed targets leaves the state unchanged; a scheduled row is therefore
  not automatically a nonzero update.
- `run.py:75-83` loads the fixed parent, sets evaluation mode and disables
  gradients. `model.py:22-30` rejects a trainable/training source, while the
  source applies gain/bias to pre-Transformer tokens
  (`simple23.py:107-117`). Only the injection's parameters enter the prefix
  optimizer (`fit.py:113-115,136-147`); it is frozen afterwards
  (`fit.py:159-160`). End-of-run checks compare source/injection digests,
  `W0`, and constant control and reject source gradients (`run.py:251-264`).
  These are code guarantees/checks to be confirmed on completion, not a claim
  that the still-running process has already passed its final assertions.
- Before shot 21, `initial is None`; every arm copies native predictions
  (`run.py:37-62,125-127,195-226`). Only after shot 20 is sealed and its HDF5
  outcome revealed are the first 20 ended shots passed to `fit_prefix`;
  new signal/outcome reads are blocked during that fit (`run.py:195-227`).
  The CPU integration test asserts this boundary and startup-arm equality
  (`test_direct_fast_memory_runner_20261104.py:18-29,110-137`). Full360,
  startup20, and suffix340 are separately summarized (`run.py:265-275`), so
  the first-20 cost is included in the principal full-stream result.
- The first 20 actual predictions use unmapped inputs; **after** they end,
  the fixed map calibrated from those same first 20 is applied to their
  signals for fitting the suffix model (`run.py:195-205`,
  `east_evaluation.py:62-80`, `DIRECT_FAST_MEMORY_PLAN_20261104.md:18-19`).
  That retrospective reprocessing is not a rewrite of their sealed startup
  scores. The map itself is a numerical preparation, not a certified unit
  conversion or evidence of a transferable physical relation.
- `fit.py:65-90` fits `W0` from direct diagnostic pairs; `fit.py:93-147`
  trains the injection using first-20 HDF5-outcome-derived risk labels and
  a threshold-centered BCE. Hence the method combines **supervised ended-shot
  connection fitting** with **unlabeled intra-shot diagnostic updates**; it is
  not wholly label-free learning. EAST labels use the fixed 125 ms window,
  with the <=30 ms positives excluded from connection fitting
  (`fit.py:99-107`, `io23.py:119-125`).

## Feedback controls and measurement limitations

- The real arm updates using real arrived targets. The hold arm uses zero
  future-minus-current increments with the actual availability mask. The
  rotated arm rolls both target **and mask** by one of the 21 output channels
  (`timeline.py:69-77`). This is a deliberately wrong-channel control, but
  it changes *which output channels receive observations* and reassigns values
  with potentially different channel distributions. Equal total mask counts
  do not imply equal per-channel support or update dynamics. Thus a real-vs-
  rotated gap is consistent with useful channel correspondence, but does not
  by itself isolate correct value correspondence from mask/distribution
  confounding. A future, separately prespecified study could add a matched-
  support, distribution-aware wrong-feedback control; do not alter this run.
- The constant control is the mean of the 20 shot-wise mean real innovations
  (`fit.py:92-98,161-165`), applied after sample 135
  (`run.py:41-48`); it tests a fixed prefix-derived internal offset, not a
  magnitude- or availability-matched time-varying placebo. The discard arm
  is tested but not run as a separate full-stream arm; zero innovation is
  checked against native (`test_direct_fast_memory_model_20261104.py:26-32`).
- `post_update_mse` evaluates the *same* target just used to update W
  (`timeline.py:76-97`): it is an in-sample update diagnostic, not evidence
  of next-window generalization. Compare paired `pre_update_mse` against
  `static_mse` and `hold_mse` on the same shot/origin and actual target, then
  compare real against controls. Aggregate with explicit observation and shot
  weighting. `observed=0` is logged with zero MSE because of denominator
  clamping (`timeline.py:82-97`); exclude unsupported rows rather than
  treating zero as perfect prediction. The initial ridge `fit_mse` is also
  computed on its own fitting pairs (`fit.py:15-50`), not a held-out score.
  The stored update row contains only aggregate errors and a count, not
  per-horizon or per-channel residuals, so it cannot alone establish which
  diagnostics or horizons drove any aggregate difference.
- `fit.py:65-85` constructs diagnostic training pairs over the *whole ended
  shot* without censoring at `DownTime`. Similarly, update histories run to
  the end of each signal array (`timeline.py:65-66`), while risk scoring
  excludes points after disruption (`hdf5_score_recheck_20261103.py:25-33`).
  Thus training pairs or recorded diagnostic errors may include targets after
  disruption or windows crossing it; no assertion here establishes whether
  this occurs in the data. This is not future-outcome leakage into a suffix
  prediction, but post-terminal examples should not be called pre-disruption
  forecast evidence. After sealing, conservatively restrict a diagnostic row
  to one whose **arrival time** is no later than HDF5 `DownTime`; the saved
  aggregate row does not expose individual horizon errors for finer censoring.
- Logged `weight_displacement` is distance from `W0`, not the norm of that
  step (`timeline.py:98`), and rows count scheduled feedback, not necessarily
  actual nonzero weight changes. Do not claim a measured per-step update
  magnitude or a count of nonzero updates from those fields alone.

## Claims, scoring, and reproducibility

The learned projection changes pre-Transformer token gain/bias while the
source weights and risk head remain frozen. It is legitimate to describe an
**internal state-conditioned neural representation intervention**, but not
online gradient training of the Transformer's parameters or a new decision
threshold. `test_direct_fast_memory_runner_20261104.py:32-47` illustrates
why auxiliary weight movement alone cannot establish a risk gain: a zero
connection permits nonzero signal learning with identical risk curves.
Forecast MSE improvement, post-update error reduction, and nonzero weights are
not themselves evidence of timely alarm improvement or physical learning.

For decision claims, use the frozen same-shot/same-threshold full360
comparison, including startup, HDF5 outcomes and strict-deadline sensitivity;
report paired timely alarms gained/lost and safe false alarms gained/removed
against native and each control, as well as aggregate TP, FP, TPR, FPR and J.
`run.py:182-193,265-277` provides per-shot arm records and aggregate summaries
after completion; `timing_decomposition` defines the aggregates
(`source_anticipation_20261104/static_stream.py:36-78`). The same 360 cohort,
ordering, mapping, and architecture have participated in development
(`DIRECT_FAST_MEMORY_PLAN_20261104.md:3-5`); neither suffix340 nor full360 is
an independent prospective validation. No claim about the run's actual
performance is made in this audit.

At the time of this audit the live protocol's recorded package and plan hashes
matched the current files exactly:

| File | SHA-256 |
| --- | --- |
| `__init__.py` | `0f4772e0693357806c62d5f07cbc6c763a0d83cea516676f862dc680edd29fd5` |
| `model.py` | `234fbd07e4d995dc10d609e928def4d7ef7319e2d9f615d2f162936bc9942985` |
| `memory.py` | `4453e45cf85177ca736991b1728bdc4306eee77925c1ac085ba2c673d87c9fa7` |
| `timeline.py` | `b721e322c768cbe7082ede60b05ea66d37aa6e80ea32cc7b4cae6b965f18db7a` |
| `fit.py` | `55cf363bce45cd221dd5f3f73474fd5387d287ff5ccc6860523b9bed7ce677e9` |
| `run.py` | `ce20934d30eb31ab36daaea5ffbe13978a229e7606d206982ca3e5e4284da082` |
| `DIRECT_FAST_MEMORY_PLAN_20261104.md` | `14c8adf8efa8354933b3e2e2e0c31b0a75dc81c42851f026644427aec4a986af` |

This is a snapshot, not proof of future stability or of the complete runtime
environment. The producer also hashes imported local helpers and checks bound
files again at completion (`run.py:84-89,121-124,263-264`;
`context_pair.py:153-167`); that completion check remains pending here.
