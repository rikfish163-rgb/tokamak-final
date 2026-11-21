# Timed-token gradient teacher: bounded history and data audit

2026-11-05. This is a read-only research audit and a **proposal**, not an
implemented intervention or an experiment result. Main owns the separate
matched absolute-age support intervention. No raw HDF5 files, raw signal
arrays, GPU, training, inference replay, jobs, or external services were used
for this audit. The sole new file is this note.

## Bottom line

The best immediately identifiable teaching corpus is the locked J-TEXT
TRAIN episode archive, with already-ended EAST first-20 episodes as a small
target-domain supplement. Their **indices and old aggregate teachers are not
already a local norm1 teacher**. A future authorized implementation must
regenerate local risk gradients on the same absolute-time support as main's
intervention. Reusing the existing 256-vector or fitting a different latent
readout would not isolate this question.

The older `recent3`, `local_readout`, and `multi_token_context` studies are
real executed precedents, but none is the proposed fixed-direction,
matched-absolute-age norm1 comparison. The last of these did have matched
readout-routing controls; its same-norm tiled control matched its headline
outcome. That is evidence against attributing its result to its particular
per-token basis, not a test of the main intervention.

## Available ended-shot material

- [Source archive and protocol](probe_outputs/context_vector_source_v1/complete.json)
  contain 10,044 TRAIN episodes from 1,303 shots and 946 VAL episodes from
  145 shots. The locked roles contain 1,353 TRAIN and 160 VAL shots;
  therefore not all eligible role IDs contributed an episode. TRAIN groups
  are safe 9,293/1,163 shots, early-disruptive 555/124 shots, and
  danger 196/129 shots; VAL groups are 638/80, 191/51, and 117/64.
  Disruptive shots can occur in both early and danger groups, so distinct-shot
  group totals must not be added. I recounted the JSON rows and matched their
  hashes against the archive completion record.
- The source episode selector uses `o=127+25k`, original maturity `o+25`,
  and a query in `o+26..o+50`; it chooses the frozen source's **highest query
  score** among eligible points and caps episodes per group. Thus the records
  are outcome- and model-peak-selected offline examples, not unbiased online
  queries. See [source_episodes.py:95](experiments/risk_aligned_online/context_vector_20261030/source_episodes.py#L95)
  and [data.py:42](experiments/risk_aligned_online/source_event_tasks_20261102/data.py#L42).
  The hash inventory is archived evidence, not a fresh raw-file integrity
  check. Existing rows cannot by themselves supply fresh local gradients.
- The narrower, completed source-supported study has **1,410 episodes from
  1,299 TRAIN shots**: 1,163 safe, 122 early and 125 danger records. Its
  [saved fit](probe_outputs/source_supported_internal_update_20261104_v1/fit.npz)
  contains source features `1410 x 85` and normalized source gradients
  `1410 x 256`. The corresponding ended EAST20 fit has `60 x 256` teachers;
  the held EAST21–100 screen has 240 cases from 80 shots. These 256 coordinates
  are the **shared affine parameters of both final-layer norm1 and norm2**,
  and their gradients aggregate positions; slicing the first 128 entries does
  not recover a particular absolute-time token support. See
  [source_supported_internal_update.py:239](source_supported_internal_update_20261104.py#L239),
  [teacher computation:279](source_supported_internal_update_20261104.py#L279),
  and [internal_gradient_transport.py:189](internal_gradient_transport_20261104.py#L189).
- **Clock-name hazard:** that narrower producer overwrites each record's
  `maturity` with `query-24` for its 32-sample motion features; it is not the
  original `origin+25` H25 maturity. Joining all 1,410 saved records back to
  the original archive by `(shot, origin, query)` gives 1,225 with the new
  clock *before* the original H25 maturity, 77 equal, and 108 after. Treat
  them as separate named clocks. In particular, do not claim a complete H25
  observation at `query-24` when it has not arrived.
- Source views must use actual availability: the existing paired comparison
  used the same source examples with either MNM and CIII masked, or MNM
  masked while preserving actual source CIII. CIII must not be fabricated.
  Its local held80 dangerous-point result still worsened versus no update;
  see [the terminal report](SOURCE_VIEW_AND_FEEDBACK_DIRECTION_RESULT_20261105.md).
  Event A–F are optional nominal offline strata, not ordered stages, EAST
  observations, or substitutes for HDF outcome.

### Coverage order correction

[SOURCE_VIEW_COVERAGE_PLAN](SOURCE_VIEW_COVERAGE_PLAN_20261104.md) repeats the
historical “1 of first 20 / 290 of suffix 340 have CIII” observation from
[INPUT_PIPELINE_AUDIT](INPUT_PIPELINE_AUDIT_20261031.md), which concerns the
older `lifecycle_context8360_service` order, not the present stratified order.
The current stratified-order
[coverage.json](probe_outputs/east_mask_repair_static360_20261104_v1/coverage.json),
whose hash matches its completion record and whose 360 IDs match the source
static protocol in order, reports **15/20** first shots and **276/340** suffix
shots with any available CIII. Both orders total 291/360. On current indices
21–100 it is 59/80, and on 101–120 it is 18/20. MNM is unavailable for all
these shots. This is an order mismatch, not evidence that CIII availability
changed; these are whole-shot mask summaries, not per-episode support.

## Historical interventions, checked against their code and terminal records

| Study | Actual mechanism and result | What it does not establish |
| --- | --- | --- |
| `internal_dynamics` `recent3` | [model.py:17](experiments/risk_aligned_online/internal_dynamics_20261022/model.py#L17) adds a learned-basis `SiLU(LN(h)W)U/sqrt(8)` to every valid data token before the last block; CLS is not directly changed. `recent3` rebuilds the shot-local 8-by-64 `U` from zero using the last three matured H25 feedback windows ([stream.py:36](experiments/risk_aligned_online/internal_dynamics_20261022/stream.py#L36)). Its archived recount reports static 145 TP/94 FP, full-history M 141/92, and recent3 144/92, with 59,761 update opportunities and 178,203 recent3 gradient steps. | Recent **feedback-history** length, not where a fixed norm1 update acts; trained basis/decoder and U differ, and this old unmapped source lineage is not the current cold or mapped-source baseline. The verified recount is historical, not rerun here. [Terminal recount](../Fuison/outputs/source_event_rebuild_audit_20261022/internal_dynamics_summary.json). |
| `local_readout` | [learning.py:22](experiments/risk_aligned_online/local_readout_20261029/learning.py#L22) fits a per-channel 64-dimensional last-token-to-H25 latent diagnostic readout on older observations, then applies normalized first-block Q/K gradients from newer observations; fit and gradient targets are disjoint. The saved complete full360 is local 125/33 versus cold 125/32 ([complete](probe_outputs/local_readout_east360_v1/complete.json)). | “Local” refers to fitting the auxiliary readout within a shot, not selecting a recent absolute-time subset of final-layer norm1 activation. Its cold20 order and parent are older. |
| `multi_token_context` | A frozen c8 latent context is decoded with each window-relative token's basis via learned `O` (64 by 92); independent copies of `O` were fitted using the first 20 ended shots ([model.py:66](experiments/risk_aligned_online/multi_token_context_20261101/model.py#L66), [learning.py:91](experiments/risk_aligned_online/multi_token_context_20261101/learning.py#L91)). Cold was 125/35, per-token real 127/34, same-O last 125/35, and same-O norm-matched tiled 127/34. | It did compare same-O routing, including a tiled norm control, but not the same absolute-age support, update direction or norm1 intervention. Its operational gate was true and its attribution gate false; no promotion or independent confirmation ([completion](probe_outputs/multi_token_context_v1/completed.json), [verification](probe_outputs/multi_token_context_v1/verification.json)). |
| Arrival evidence | The scoped [prior-art audit](ARRIVAL_EVIDENCE_PRIOR_ART_20261030.md) describes five-point logistic increments and a separate H25 `p-r` feedback in 8,192 temporary Q/K coordinates. The H25 conditional-density screen reported danger mean delta BCE +0.047587, 0/17 improved, and safe −0.036445, 3/3 improved; it was reset-episode development, not a full-stream gain. | A classifier evidence increment is not a local norm1 gradient teacher; neither an auxiliary score nor a threshold adjustment demonstrates deployable intra-shot gain. |

The above result numbers come from different recorded cohorts/parents and
must not be compared as a single paired experiment. Current versus historical
first-20 membership differs, even though the fixed total shot membership is
shared.

## Minimal proposed teacher and held-shot design

This is a proposal for main to consider, not an instruction to execute or
duplicate its support implementation.

1. **Target.** Keep the qualified J-TEXT checkpoint, risk head, threshold,
   and the main intervention's exact support function fixed. At a future
   query `q=m+24`, define a hypothetical zero-initialized 128-dimensional
   final-layer norm1 affine displacement `(delta_gamma, delta_beta)` that
   acts only on main's already-defined absolute sample support anchored at
   maturity `m`. On an **ended fit shot only**, compute the true
   `grad_delta BCE(z_q, y_q)` at zero displacement, with the unchanged risk
   model, actual mask and same support. Normalize a nonzero target to unit
   length; record zero gradients and unsupported cases separately. This is a
   local teacher target, not a 256-parameter global gradient or a new output
   head. At `q=m+24`, window patch phase is aligned because 24 is three
   8-sample patches; support must still be matched by absolute sample bounds,
   not by blindly reusing a token index. Any other query phase needs main's
   exact overlap handling and its own tests.
2. **Inputs and fit.** At `m`, only use already available state, masks,
   native source logit, and measured ordered motion through `m`; the prior
   `64 state + 21 motion` features and fixed ridge 0.1 are a simple baseline.
   Fit a 128-output teacher on source TRAIN, optionally with ended EAST20 at
   a predeclared 1:1 source/EAST weight, with group/shot/case weighting and
   train-fold-only standardization. Source `max(25 ms, AdvancedTime)` and
   EAST fixed125 ms labels remain distinct. For a motion-specific candidate,
   use the fixed linear odd component `(g(s,v)-g(s,-v))/2`; zero/unsupported
   motion is an explicit no-op. Do not substitute a learned risk score,
   output offset, new latent readout, or query/future features for the
   teacher. Training uses ended-shot labels; live construction does not.
3. **Causality and controls.** Apply the predicted negative-gradient direction
   through main's *same* local support mechanism and fixed step as the
   comparison direction. Compare the same source/map/head, examples, mask,
   support, clocks, update budget and norm against no update, the fixed
   feedback direction, and a same-norm reversed or time-shuffled motion
   teacher. State-only and zero-motion controls distinguish state bias from
   incremental motion. Do not tune on held labels or use them to select a
   deployable policy. Freeze all arms before any held outcomes are revealed;
   report zeros, support, exact absolute bounds, and real/reverse overlap.
4. **Held shots.** First evaluate a source-TRAIN-trained teacher on locked
   source VAL as a held-**teacher** diagnostic, not a held-parent guarantee.
   For EAST, use the previously declared four 15-fit/5-held folds of the
   ended 20; fit both teacher and every target-fitted preprocessing component
   on each fold's 15, or run an explicitly unmapped-source version. The
   existing all20-fitted input map or cold parent makes an all20-held fold
   conditional, not full-pipeline independent. These offline folds can use
   later-in-order ended shots and are not a prequential simulation. Freeze the
   final fit on allowed source and first20 before applying it to later shots.
   For later-shot evaluation, use an outcome- and score-blind fixed query
   schedule, score only observations that actually arrive, seal predictions
   and controls, then reveal HDF outcomes for descriptive shot-level analysis.
   Report safe, disruptive-early, and timely-danger separately, not window
   counts as independent shots; distinguish ordinary BCE from
   threshold-centered margin loss and from whole-shot first-alarm TPR/FPR.

The old source archives' peak-query sampling is acceptable as *historical
offline training material* only with that selection bias disclosed; a fresh,
predeclared score-blind schedule would better match a live query distribution.
The old 1410/60 cached gradients cannot generate the specified local targets;
authorized future access and computation would be needed. In the source
archive H25 is fully observed only at its own `origin+25`, not at every
redefined `query-24` clock.

The two meanings of calibration should remain separate. The
[EAST decoder calibration review](EAST_DECODER_CALIBRATION_REVIEW_20261102.md)
fits a separate low-rank diagnostic decoder and warns that four folds with an
all20-exposed parent are conditional. The
[past20 threshold control](COLD20_CALIBRATION_PLAN_20261104.md) uses only
12 ended safe shots for decision-threshold calibration and is not a neural
gradient teacher. Neither demonstrates the proposed local direction, and
neither authorizes choosing a threshold from later-shot labels.

## Later development-shot candidates, selected without outcome data

Use the first 20 shots *after one-based stream index 100*, i.e. indices
101–120, in the current stratified order. This is a transparent index-only
choice, not outcome, alarm, risk-score, or efficacy selection. For each of
the 20, I checked: current ordered ID; a matching archived raw-HDF SHA in
both the source-static protocol and the stratified input inventory; the
actual saved qualified-source curve file against its seal and completion
record; the saved coverage curve against its seal; and the coverage seal's
reference hash against the source seal. All 20/20 passed. The sealed source
NPZ contains both `unmapped` and `map_after20` curves. These checks read
**saved artifacts only**; they do not freshly hash or open any HDF file.

| Index | EAST shot | Saved points | CIII available points |
| ---: | ---: | ---: | ---: |
| 101 | 60916 | 4,997 | 4,997 |
| 102 | 54165 | 3,197 | 3,197 |
| 103 | 75174 | 581 | 581 |
| 104 | 79354 | 4,697 | 4,697 |
| 105 | 83887 | 558 | 558 |
| 106 | 72984 | 5,398 | 5,398 |
| 107 | 74928 | 4,999 | 4,999 |
| 108 | 70258 | 4,999 | 0 |
| 109 | 60239 | 4,999 | 4,999 |
| 110 | 75226 | 4,999 | 4,999 |
| 111 | 81684 | 2,038 | 2,038 |
| 112 | 59886 | 4,999 | 4,999 |
| 113 | 74939 | 4,999 | 4,999 |
| 114 | 59935 | 2,998 | 2,998 |
| 115 | 74875 | 1,171 | 1,171 |
| 116 | 66116 | 5,742 | 0 |
| 117 | 62250 | 2,997 | 2,997 |
| 118 | 73405 | 18,729 | 18,729 |
| 119 | 73403 | 24,226 | 24,226 |
| 120 | 61629 | 1,407 | 1,407 |

All have at least 558 saved time points, so the proposed `m>=135, q=m+24`
schedule has at least one length-eligible query. The saved coverage only
describes CIII and MNM; it does **not** establish per-query common-channel
support, preterminal eligibility, or valid local gradient support. Treat
unsupported queries as no-op and report them; do not silently replace shots.
Across indices 101–360, 260/260 IDs are unique and have matching archived
raw-hash entries plus coverage entries; the minimum saved length is 558.

Source artifacts:
[source protocol](probe_outputs/qualified_source_static360_20261104_v1/protocol.json),
[source completion](probe_outputs/qualified_source_static360_20261104_v1/completed.json),
[raw-hash inventory](probe_outputs/stratified_cold20_context360_v1/input_hashes.json),
and [coverage](probe_outputs/east_mask_repair_static360_20261104_v1/coverage.json).
For example, [101 source seal](probe_outputs/qualified_source_static360_20261104_v1/shots/0101_60916_seal.json)
and [101 coverage seal](probe_outputs/east_mask_repair_static360_20261104_v1/shots/0101_60916_seal.json)
bind the saved curves. All these 360 shots, including 101–120, have already
appeared in earlier development and sealed-score analyses: this is a new
**teacher-specific development slice**, not untouched future validation.
The saved hashes are internal consistency evidence, not independent proof of
historical ordering or present raw-data integrity.

## Audit verification and access boundary

I read the target [AGENTS.md](AGENTS.md) and
[HDF5 outcome contract](HDF5_OUTCOME_CONTRACT_20261103.md), inspected the
named source/protocol/terminal artifacts, recounted the saved JSON records,
checked listed artifact hashes, and hashed the 40 saved candidate curve files.
No experiment, model replay, raw-file read, HDF5 access, training, GPU work,
or new test run was performed. `meta/IsDisrupt` and `meta/DownTime` remain the
only hard outcome authority when a later authorized evaluation reaches its
reveal boundary; shot-list classes and event labels must not override them.
No policy was selected from evaluation labels. Skill workflows informed the
separation of recorded observations, causal limitations, and proposals.
