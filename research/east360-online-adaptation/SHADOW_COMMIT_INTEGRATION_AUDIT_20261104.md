# Shadow-commit integration audit — 2026-11-04

## Assessment

Within the inspected code and CPU synthetic checks, the issue/release ordering
matches the plan: the first window validates `W0` before training the shadow;
the next window validates the previously trained candidate before it can be
committed. The 2 × 2 connection/policy crossing is correctly wired, and both
connections use the same ended-20 records and draw schedule.

**One confirmed reporting-precision defect:** the logged fixed-`W0` MSE
subtracts in float32 before casting to float64, while candidate and active
MSEs cast before subtraction. This can make a fixed and active forecast report
different errors even when their predictions are identical. It does **not**
change the guard/delay decision, which compares candidate and active errors
computed together in float64. No producer or plan file was changed for this
audit; see the reproduced example below before drawing fine-grained forecast
conclusions.

This is a code/synthetic audit, not evidence that the 360-shot experiment has
run or improved alarms. No GPU, raw shot files, or network were used. The
main agent's reported test run was not independently rerun here.

## Confirmed issue: fixed-predictor MSE arithmetic differs

At `experiments/risk_aligned_online/shadow_commit_20261104/timeline.py:107-123`,
`static_mse` evaluates `(static - target)` in the original float32 dtype and
only then calls `.double()`. By contrast,
`shadow_commit_20261104/decision.py:46-55` casts each input to float64 before
subtraction for `candidate_mse` and `active_mse`. `static` is also produced
by `key @ initial.T`, whereas the issued forecasts use batched matrix
multiplication (`timeline.py:31-33,108`).

An independent one-observation CPU synthetic trace, with `q=127`, `t=135`,
identical `W0` candidate and active predictions of `1.0`, and an observed
float32 target near `0.1`, produced:

```text
candidate_mse = active_mse = 0.8099999973177909
static_mse                 = 0.8099999570846563
absolute difference        = 0.0000000402331346
guard accepted             = False
```

This is a reproducible diagnostic inconsistency, not evidence of an incorrect
commit or of a material change in the forthcoming real run. It matters when
interpreting near-ties or counting tiny logged advantages over fixed `W0`.
For an exact fixed-versus-active error comparison, a future authorized
post-run recomputation should use the same prediction operation and float64
residual arithmetic for both, with the same target and mask. Do not silently
replace frozen artifacts or modify the producer during the run. Saved receipts
contain candidate/active predictions and targets, but not the key or fixed
prediction, so they alone cannot fully recalculate the fixed baseline.

## Issue, release, and pending behavior

Relevant code: `shadow_commit_20261104/timeline.py:14-34,37-144`;
`direct_fast_memory_20261104/timeline.py:12-48`;
`shadow_commit_20261104/decision.py:42-61`.

| Event | State and information actually used |
| --- | --- |
| `q=127` | `shadow=active=W0`; `issue()` clones the shadow weight and saves both forecasts and versions using only `key[127]`. No target is passed to `issue()`. |
| `t=135` | Actual samples `128:136` have arrived. Both issued forecasts are scored on their same actual mask. Guard rejects the equal forecasts; delay accepts if supported, but adopting the identical `W0` does not change active parameters. Only **after** this decision is the shadow trained on those arrived values. |
| `q=159` | The candidate is the post-135 shadow state, denoted `W1`; active remains `W0`. Both forecasts and the candidate-weight snapshot are made before samples `160:168` are consumed. If the prior row was unsupported or the step made no change, `W1` may equal `W0`. |
| `t=167` | The `q=159` forecasts are validated on actual `160:168`. Guard adopts the issued candidate only if its supported MSE is strictly lower; delay adopts it whenever support exists. Only afterwards is the shadow trained on this second target. |

The validation target is therefore not used to train the candidate being
validated. Since origins are 32 samples apart and arrival is eight samples
later, there is no intervening commit between an issue and its release. The
current sample at `t` is allowed to arrive before the update and the risk
readout at `t`; this is not a prediction made before the current sample.
`parameter_changed` checks actual weight equality, so it correctly distinguishes
delay's first acceptance of `W0` from a real state change. `shadow_version`
increments on supported feedback (`timeline.py:100-106`), not necessarily
only when the weight has numerically changed; use `shadow_step_norm` or
`parameter_changed` for movement claims.

For a final origin without all eight future positions, `arrival >= n` retains
the forecast with `released_at=-1`, zero target placeholders and an all-false
target mask (`timeline.py:64-77,145-153`). It does not train or commit on a
fabricated target. A released window with no available target has a real
release time but zero support, also rejects, and must not be counted as
zero-error performance; distinguish it from pending by `released_at`.

An independent CPU-only synthetic check reconstructed `W1` from the first
real target and confirmed the saved `q=159` candidate prediction equals
`W1 @ key[159]`. With a matching second target both policies committed at 167;
with an opposite second target guard rejected while delay committed. Changing
that future target did not alter the saved `q=159` candidate forecast. In the
opposite-target case the guard/delay candidate forecasts and target masks were
identical, as expected because the shadow update is independent of active
commit decisions.

## Prefix stability and receipt limits

The readout fills `[arrival, min(arrival+32,n))` using a fixed 32-row key
batch and discards padded outputs (`timeline.py:124-144`). At `t=135` the
active state is still `W0`, so the first possible nonzero innovation is at
`t=167`, conditional on an actual committed weight change. The no-bias
injection makes zero innovation equal to the native source path
(`direct_fast_memory_20261104/model.py:7-10,22-30`).

An independent synthetic CPU check compared 20 truncated lengths against the
corresponding full trace, including lengths immediately before/after 135 and
167 and unfinished targets; all prefixes were bitwise equal. Existing test
source also covers prefix extension, pending tails, snapshot copies, reset,
and missing support (`tests/test_shadow_commit_timeline_20261104.py:19-82`).
These are CPU checks on specific inputs, not proof of bitwise CUDA behavior or
real-time throughput.

“Issued” means computed and copied in the simulation before the code accesses
that window's target. `run.py:291-315` writes and seals the forecast receipt
after replaying the whole shot, not with an independent wall-clock seal at
each `q`. The receipt contains issued/released indices, versions, both
predictions, targets, and masks, but not the actual candidate weight snapshot,
key, or covariance (`timeline.py:14-21,64-74,129-153`). Thus a later reader
can recompute the decision from saved forecasts and released targets, but
cannot independently prove full weight lineage from the receipt alone.

## Crossed connections and scientific controls

`shadow_commit_20261104/run.py:37-50,88-118` maps the two trained connections
and two **real-feedback** test trajectories as follows:

| Connection trained on | Guard policy at test | Delay-only policy at test |
| --- | --- | --- |
| Guard trajectory | `guard_guard` | `guard_delay` |
| Delay trajectory | `delay_guard` | `delay_delay` |

Comparing a row holds the connection fixed while changing policy; comparing a
column holds the real-feedback trajectory/policy fixed while changing the
connection. The shadow itself updates from the same real target regardless of
whether active commits (`timeline.py:90-103`), so real guard and real delay
have the same candidate sequence under the same deterministic inputs. The
active sequences can differ. Interaction between connection and policy should
still be reported rather than collapsed into one universal guard effect.

`run.py:121-158` regenerates the same first-20 records and requires exact list
equality with the pinned RLS reference, constructs **one** schedule, and
passes that same records/schedule object to both connection fits. The schedule
uses seed 131, 512 steps, batch size 128, and positive/safe/earlier counts
64/32/32 (`cold20_compare_20261104/data.py:65-93`). Both `Injection` weights
start at zero and use the same Adam rate, threshold-centered BCE, penalty,
and clipping (`direct_fast_memory_20261104/model.py:7-10`,
`shadow_commit_20261104/training.py:44-76`); their inputs differ by the
guard versus delay innovation trajectories by design.

Using the pinned first-20 record list, an independent CPU reconstruction of
the prior RLS sampling loop matched all 512 new schedules exactly, including
all 128 indices and 64/32/32 group counts at each step. The SHA-256 of this
schedule serialized as sorted-key, compact JSON was
`8b45cf062c12522e4cd3addda7e43703329b7d08ce173a4ed6dcbb9d45e9762e`;
this is a canonicalized-content hash, not the byte hash of a future output
file.

`guard_hold` and `guard_reverse` use the **guard-trained connection and guard
commit policy**. The shadow's feedback is respectively zero increment or the
negated increment with the same availability mask, but validation still uses
the real matured target (`run.py:37-50`, `timeline.py:83-102`). These controls
test the feedback update conditional on real-target validation; they are not
controls with all access to diagnostic feedback removed, and they are not a
full wrong-feedback × both-connections crossing. The previous immediate-RLS
run used a separately trained connection and is a historical comparator, not
the same-connection delay control.

Guard selects using the current validation error, so better MSE on **accepted
validation windows** is partly guaranteed by the selection rule. It does not
establish improvement on later windows, risk predictions, or alarms. Compare
paired guard/delay outcomes at the same connection, retain unsupported and
pending counts, and report full360 including startup. The rule deliberately
accepts/rejects with as little as one observed element; no extra support or
gain threshold has been introduced by this audit. The 360-shot cohort and plan
are development evidence, not independent prospective validation.

## Pinned `W0`, frozen source, and outcome boundary

- The actual `load_initial` call is inside `fit_pair`, which is reached only
  after shot 20's curve is sealed and outcome revealed (`run.py:121-125,
  286-327,337-369`). It checks the pinned prior completion SHA, verifies the
  prior `fitted.pt`, `fit.json`, and `protocol.json` against that completion,
  and requires the same 20 IDs, source, RLS rule, and zero suffix training
  (`run.py:54-85`). The earlier first-20 curves are not rewritten.
- Independently, the pinned completion SHA matched
  `faeeaa386e3d160b4da4a2582adda79d3a3b226029c0767fe187bcdd90e732f6`,
  its three referenced artifact hashes matched, and a CPU call to
  `load_initial` with the pinned 20 IDs returned finite float32 `W0` of shape
  `(168,70)` and 9,004 matched training records. The SHA-256 of contiguous W0
  tensor bytes was `b6b6a3302f21510839020ccf538fcdb0dca48615ebd510032ae5f56a348e422c`.
  This read the prior checkpoint and first-20 metadata, not raw shot data.
- The runner loads the qualified source in evaluation mode with gradients
  disabled and verifies its digest and threshold against the baseline
  (`run.py:198-208`). Both fitted connections are frozen after fitting
  (`training.py:91-93`). Completion requires unchanged source/connection
  digests and `W0`, no source/connection gradients, and unchanged bound inputs
  and code (`run.py:383-399`). These final assertions cannot be claimed as
  passed until the requested run actually completes.
- During a shot, `read_outcome` is patched to raise until the curve, forecast
  receipt, update history, and seal have been written (`run.py:266-327`).
  `io23.read_signals` reads signal datasets; the later `read_outcome` reads
  HDF5 `meta/IsDisrupt` and `meta/DownTime` (`io23.py:62-96,99-116`). Only
  already-ended first-20 outcomes enter connection fitting. A whole-file
  integrity SHA is computed before prediction (`run.py:255-258`), so file
  bytes including metadata are physically hashed, but the outcome is not
  parsed or supplied to the predictor at that point.
- The implementation reads the whole current shot and precomputes causal
  windows before replay (`run.py:272-275`,
  `direct_fast_memory_20261104/timeline.py:22-29`). Pointwise window construction
  itself ends at each query (`simple_data.py:41-48`). This supports an offline
  causal replay claim, not actual online acquisition, latency, or independent
  wall-clock issuance.

## Audited snapshot

The following SHA-256 values matched at the beginning and end of code review;
they identify the inspected snapshot, not a guarantee against later edits:

| File | SHA-256 |
| --- | --- |
| `decision.py` | `fc5d92accb62b58f0b1e8ec08ea5c3e00425d9799a94c88f37c520cc6455fd8d` |
| `timeline.py` | `73152b109c3d7696a5e21f7c3a8858d0d9336366e666ffffcdb110d713c502c8` |
| `training.py` | `182e3b2f1d48a4ed7404e4aa82206cdc3b9cefd376303c520822832010061ebb` |
| `run.py` | `2aecbb435a12a2182d5e2530096d0e7be700e5fa4b86228cca74b0fe254e4878` |
| `SHADOW_COMMIT_PLAN_20261104.md` | `58d1d1d28911e8242bda83abda4fdc399175a4d34afe1bfba51bd64bbfc7e7cb` |

The frozen decision test SHA remained
`5b21848471dff6b5b638645418a456d43738c4e33812d068400b6504058dac74`.
