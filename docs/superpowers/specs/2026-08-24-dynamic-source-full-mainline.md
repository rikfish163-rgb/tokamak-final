# Dynamic-label J-TEXT to EAST full-mainline specification

## 1. Research question and evidence boundary

This specification freezes the post-adjudication development candidate that the
user chose to rerun after inspecting the corrected four-arm label ablation.  It
defines the only allowed execution path for the next full-chain experiment; it
does not rewrite the historical ablation adjudication.

The selected label arm is `source_dynamic__east_fixed`:

\[
H^J_s = \max(25\ \mathrm{ms}, A_s), \qquad H^E_s = 125\ \mathrm{ms},
\]

where `A_s` is the J-TEXT `AdvancedTime` value for shot `s`.  Missing J-TEXT
values fall back to 25 ms.  `AdvancedTime` changes only the positive endpoint
horizon.  It never changes the shot outcome or terminal time.  The sole
authority for those fields is the source HDF5 file:

\[
D_s = \operatorname{meta/IsDisrupt}_s, \qquad
T^d_s = \operatorname{meta/DownTime}_s.
\]

The arm has the largest mean balanced360
`J = operational_TPR - FPR`, but the historical preregistered per-seed gate
failed because seed133 had `TPR@30 < 0.80`; fixed125 therefore remained the
historical admitted primary label.  The new machine-readable role is
`post_adjudication_user_selected_development_candidate`, with
`historical_admission=FAIL` and `confirmation_eligible=false`.  All current
TEST220 and balanced360 outputs retain
`TEST_DRIVEN_EXPLORATION` or `PREQUENTIAL_DIAGNOSTIC` status.  Only a new,
previously untouched chronological stream can provide confirmatory evidence.

## 2. Fixed data roles

| Cohort | Count | Role | Outcome access |
|---|---:|---|---|
| J-TEXT TRAIN1353 | 1353 | source fitting | allowed during source training |
| J-TEXT VAL160 | 160 | epoch and source-threshold selection | allowed only for selection |
| J-TEXT TEST220 | 220, balanced 110/110 | static source evaluation and ordered prequential diagnostic | after static seal; per-shot only after selected online query |
| EAST support110 | 110, HDF5-authoritative 15D/95N | target adaptation and target threshold | allowed during support fitting |
| EAST balanced360 | 360, 180D/180N | static and online development evaluation | after static seal; per-shot only after selected online query |

All rosters, source checkpoint, scaler, thresholds, label maps, configurations,
and prediction ledgers are hash-bound.  The source and target runners fail
closed when a required artifact is absent or when a hash/policy differs.

The TEST220 and balanced360 prediction phases consume ordered, label-free
rosters containing shot identifiers only.  Class-partitioned manifests and
objects containing outcome, terminal time, `AdvancedTime`, or equivalent label
fields are forbidden before the corresponding static seal.  A static seal is
complete only after every prediction row and every per-shot risk-trace digest
has been closed, flushed, fsynced, and covered by a seal manifest.  The lazy
outcome oracle may be constructed only after that receipt exists.

The source checkpoint also binds the exact causal input contract.  The old
whole-shot availability rule is prohibited because future suffix values can
change whether an early prefix is treated as available.  The selected contract
is:

```text
schema: jtext-rebuilt17-prefix-causal-v2
history_alignment: right
left_padding: constant_zero
window_points: 128
valid_lengths_required: true
valid_length_range: [1, 128]
patch_mask_rule: patch_end <= window_points - valid_length
availability: finite-and-prefix-causal-seen-nonzero-v1
preprocessing_sha256: required
```

The availability state at endpoint `k` may depend only on samples through
`k`.  Mutating any suffix after `k` must leave its features, valid length,
embedding, risk, query value, and prediction seal unchanged.  Existing dynamic
checkpoints without this contract are historical reproduction artifacts only
and are rejected by the selected runner.

## 3. Two isolated branches from one source parent

For each seed in `131, 133, 137`, train exactly one dynamic-label J-TEXT source
checkpoint and copy it into two independent branches.

### Branch A: formal cross-device development chain

```text
J-TEXT TRAIN1353 dynamic source
-> VAL160 checkpoint and threshold lock
-> static TEST220 seal/evaluation
-> EAST support110 adaptation with explicit fixed125 labels
-> shared hazard and EAST-private parent construction
-> static balanced360 complete prediction seal
-> post-seal outcome join and static metrics
-> reset the identical migrated parent
-> ordered EAST online-active balanced360
```

No J-TEXT TEST220 online-updated state may enter this branch.

### Branch B: J-TEXT TEST220 online-active analogue

```text
same locked dynamic source checkpoint
-> static TEST220 complete prediction seal
-> reset the identical source parent
-> ordered TEST220 predict/seal/query/reveal/update replay
```

The final online-updated J-TEXT state is diagnostic and must not become an
EAST transfer parent.  For this goal, even a separately named
`JTEXT_ONLINE_PRECONDITIONED_TRANSFER_DIAGNOSTIC` arm is prohibited: every EAST
run starts from the pre-online source checkpoint sealed immediately after
TRAIN1353/VAL160 training.  TEST220 online residuals, optimizer state,
normalization state, thresholds, selector/value memory, replay, and query
receipts remain confined to Branch B.

## 4. Mandatory lineage and fail-closed behavior

The selected production entry point must reject all of the following:

- a source checkpoint whose `training_label.policy` is not `advanced_floor`;
- a missing source checkpoint or a silent fallback to an archived fixed source;
- target adaptation invoked with `source_checkpoint=None`;
- a target parent whose embedded source SHA differs from the selected source;
- an EAST parent whose target label policy is not explicitly `fixed125`;
- a parent schema that lacks source/target label contracts;
- scaler, roster, or threshold hashes that differ across static and online reset;
- reading EAST or J-TEXT test outcome fields before the complete static seal;
- using an unqueried online shot to construct a supervised target;
- changing a prediction after its endpoint or shot seal.
- a label-bearing TEST220 or balanced360 roster before the static seal;
- a threshold without a hash-bound selection receipt or outside `0 < t < 1`;
- a target preprocessing, physics scale, reconstruction constant, padding, or
  state digest that differs between static and online reset;
- a static risk trace whose recomputed digest differs from the seal manifest;
- a selected-path metadata fallback for missing `SampleRate` or `StartTime`
  unless the fallback rule and its invocation are explicitly hash-bound.

The selected parent schema is `dynamic-source-east-fixed-parent-v1`.  It must
contain at least:

```text
schema_version
seed
selection_role
historical_admission
confirmation_eligible
corrected_ablation_report_sha256
selection_metric
historical_failure_reason
source_checkpoint_sha256
source_label_policy
source_label_map_sha256
source_split_sha256
source_scaler_sha256
source_input_contract_sha256
east_target_label_policy
east_target_horizon_ms
east_support_roster_sha256
east_support_data_manifest_sha256
east_support_outcome_manifest_sha256
east_threshold_receipt_sha256
east_threshold
target_scaler_sha256
physics_scales_sha256
reconstruction_constants_sha256
target_encoder_sha256
target_head_sha256
shared_parent_sha256
private_parent_sha256
claim_status
```

The source VAL threshold and EAST support threshold are distinct decisions and
have distinct receipts.  The EAST receipt binds the actual recomputed support
roster, support predictions, candidate rule, target TPR, selected finite
threshold, and all support data/outcome manifest digests.  State hashes use one
canonical byte codec shared by the parent writer, selected validator, and
unified loader; embedded hashes are never trusted without recomputing them from
the decoded state.

The selected state machine is:

```text
source-train -> jtext-static -> east-parent -> east-static -> east-online
```

`jtext-static` and `east-static` produce their complete outcome-free seals
before their first outcome read.  `east-parent` is constructed from a typed
support110 result only; it cannot call a loader that includes balanced360,
consume A1 balanced360 replay artifacts, or inherit a historical threshold.
Every resume receipt binds the code/config plus the individual source split,
label map, support roster, test roster, and HDF5-manifest digests.

Legacy schemas remain readable only by legacy replay commands; the selected
runner never silently converts them.

## 5. Causal J-TEXT online protocol

The TEST220 roster is ordered deterministically by the recorded shot sequence.
Because current role ranges overlap and this cohort has already been inspected,
the claim is `JTEXT_TEST220_ORDERED_PREQUENTIAL_DIAGNOSTIC`, not an independent
future-stream claim.

For every shot and every causal endpoint:

```text
prefix input
-> risk prediction
-> immutable endpoint seal
-> optional label-free intra-shot update using later observed prefixes
-> later endpoint prediction
-> immutable shot seal
-> label-free query decision
-> selected reveal or unqueried discard
-> post-reveal temporal target
-> guarded replay/update or rollback
-> next shot
```

The feature-only loader reads the 14 raw signals needed to construct rebuilt17
and sample-rate metadata but not
`meta/IsDisrupt`, `meta/DownTime`, or `AdvancedTime`.  A lazy oracle may read
the HDF5 outcome only after the shot seal and only for a selected query.
`AdvancedTime` may be read only after the same reveal receipt and only for a
disruptive selected shot.

The source encoder and original source head remain bitwise frozen.  Online
updates are limited to a zero-initialized, bounded J-TEXT residual/controller:

- `fast` parameters reset from persistent state at every shot;
- intra-shot updates affect later endpoints only and read no outcome;
- `persistent` parameters change only after selected reveal;
- replay contains only previously revealed examples;
- model and optimizer are restored together when a guard rejects an update.

The J-TEXT controller is a dedicated transaction boundary, not a thin alias of
the EAST two-timescale controller.  Its intra batch is a field-whitelist type
that cannot carry outcome, terminal, TTD, AdvancedTime, or horizon fields.  A
loss receipt records `enabled`, coefficient, raw loss, and weighted loss for
each term.  Inter update checks loss, gradients, parameters, and optimizer state
for finite values and restores model plus optimizer on any failure.  Complete
resume state is written only at a finalized shot boundary and includes model,
optimizer, replay, selector, value memory, cursor, RNG state, and all hashes.

The first implementation uses the locked rebuilt17 path.  EAST-private raw8,
EAST support normalization, and EAST physics thresholds are forbidden in the
J-TEXT primary arm.

## 6. Online losses and query value

The enabled intra-shot loss is reported term by term:

\[
\mathcal L_{intra} =
\lambda_{next}\mathcal L_{next}
+ \lambda_{teach}\mathcal L_{teacher}
+ \lambda_{anchor}\mathcal L_{risk-anchor}
+ \lambda_{phys}\mathcal L_{physics}.
\]

Every non-zero coefficient must appear in the effective configuration and
update receipt.  A zero-weight or shadow term is described as disabled and is
not credited for an observed gain.

The query score is implemented by a J-TEXT-specific immutable value object and
is label-free:

\[
V_s = w_s U_s + (1-w_s) C_s + \eta N_s,
\]

where `U_s` is sealed predictive uncertainty, `C_s` is a causal contradiction
score derived from audited J-TEXT rebuilt17 proxies, and `N_s` is a bounded
novelty/error-memory distance computed only from previously revealed shots.
The projected weight `w_s`, the novelty coefficient `eta`, and all scales are
selected on TRAIN/VAL simulations and frozen before TEST220.

The primary TEST220 budget is 20%: 20-shot blocks with four selected shots,
giving exactly 44 queries over 220 shots.  Diagnostic budget sensitivity uses
10%, 20%, and 30%.  Active and random controls match block-wise query counts,
optimizer attempts, sampled windows, replay size, and reported FLOPs.

## 7. Required experimental arms

Every arm starts from the same seed-specific parent and frozen threshold:

| Arm | Intra update | Query/reveal | Persistent inter update |
|---|---:|---:|---:|
| `static` | no | no | no |
| `intra_only` | yes | no | no |
| `inter_random` | no | block-matched random | yes |
| `inter_active` | no | active | yes |
| `full_random` | yes | block-matched random | yes |
| `full_active` | yes | active | yes |
| `frontload_equal_labels` | no online query | same revealed-label count before evaluation suffix | fitted before suffix |
| `equal_compute_random` | matched to active | block-matched random | matched attempts/windows/FLOPs |

The primary selector estimand is `full_active - full_random`; the online
adaptation estimand is `full_active - static`; the intra/inter decomposition is
reported with `intra_only`, `inter_active`, and `full_active`.  Front-load is
evaluated only on shots it did not train on.

## 8. Metrics and abnormal-run gates

For J-TEXT, an operational true positive requires first accepted alarm lead
strictly greater than 10 ms.  For EAST it requires lead at least 30 ms.  Leads
greater than 3000 ms remain operational true positives but are separately
reported as very-early alarms.

For each device and arm report:

```text
raw TP/FN and raw TPR
TARDY, operational TP/FN and operational TPR
FPR
J = operational TPR - FPR
AUC
median and IQR of raw-alarm lead
median and IQR of operational-TP lead
very-early alarm rate
TP/FN/TARDY/FP/TN counts
four chronological blocks and worst block
query count, update attempts, accepted updates, rollbacks, windows, FLOPs
```

Any full arm with `TPR < 0.60` or `FPR > 0.40` stops before hyperparameter
tuning and triggers a label, threshold, alarm-index, scaler, order, and output
audit.

The result schema must not reuse the ambiguous historical field
`median_lead_ms`.  It writes `raw_alarm_lead_ms` and
`operational_tp_lead_ms` separately.  For two arms, an arm-wise median
difference is composition-sensitive and is not called paired.  A paired timing
delta is calculated only on the common set of shots that are operational true
positives under both arms, and reports that common-shot count.

## 9. Statistical interpretation

- Show every seed; do not treat three seeds as independent physical shots.
- Use shot-paired binary differences for static versus online TPR/FPR/J
  comparisons after verifying shot-order, truth-join, and evaluation-mask SHA.
- Use per-seed Wilson intervals for descriptive binomial raw TPR, operational
  TPR, and FPR uncertainty; never pool identical physical shots over seeds.
- Use chronological block bootstrap only as a descriptive adaptive-stream
  sensitivity analysis.
- Do not attach confirmatory p-values to the arm selected on balanced360.
- If multiple primary contrasts are tested, apply Holm correction.
- A mechanism claim requires active to beat the strongest matched random or
  shuffled control in every seed, not only in the mean.

Current development cohorts emit no confirmatory p-values; their multiplicity
mode is `disabled_post_selection_development`.  Holm is implemented and tested
for a future preregistered stream only.

The new front-load, random, and equal-compute controls must be executed from the
new parent and write their own complete ledgers.  Historical controls may be
reported as historical falsification context but cannot satisfy a new arm's
completion receipt.  A primary front-load control freezes its suffix start
before any TEST outcome read, obtains labels only from the prefix using a
label-free rule, fits once, and evaluates all arms on the identical suffix
mask.  A control that moves labels selected from the future suffix into the
prefix is named `frontload_same_future_labels_oracle` and remains a separate
post-selection diagnostic.

## 10. Evidence, figures, report, and acceptance

Every run writes effective config, command, environment, git state, data and
roster hashes, label contract, parent hashes, static prediction seal,
post-seal outcome join, online prediction/query/reveal/update ledgers, per-shot
risk traces, metrics, and an explicit claim status.

Quantitative figures are generated from these artifacts with Matplotlib in
PDF/SVG and 600-dpi PNG.  They use an accessible palette plus markers and line
styles that remain distinguishable in grayscale.  Image generation is allowed
only for a clearly marked non-data conceptual graphic without quantitative
curves, axes, values, or model outputs.  Scientific text, arrows, event order,
cohort counts, and policy labels are deterministic vector overlays.  The
legacy fixed125/paper-v3c report and figures remain an explicitly named archive;
they are not edited or renamed into the selected dynamic-source package.

Every quantitative panel has its own lineage entry, including claim status,
cohort and physical-shot count, model-seed count, statistic, uncertainty
definition, artifact schema/path/digest, row selector, parent digest, label
policy, and generator/config/caption/output digests.  No generator may fall
back to a legacy or external run directory.  Single-shot outcome annotations
come from the post-seal join, never from hard-coded plotting literals.

The Chinese report is authored from one Markdown source and compiled into
standard LaTeX/PDF.  It includes a symbol glossary, formula-to-code map,
effective-configuration table, complete loss and value-function derivations,
data-role table, per-seed and per-block results, all controls, limitations,
artifact hashes, and simulated plasma-physics and machine-learning reviewer
responses.

The implementation is accepted only when:

1. targeted contract tests first fail and then pass;
2. the full repository test suite passes with the external data root;
3. smoke runs use `rl_env` and the RTX 5070;
4. all three seeds complete for the selected EAST chain and J-TEXT online
   matrix, or every missing arm has a machine-readable failure reason;
5. source encoder/head hashes are bitwise unchanged during online stages;
6. static and online initial parents match bitwise;
7. no outcome access occurs before the registered seal;
8. report tables and figures are regenerated from hash-bound evidence;
9. five independent `gpt-5.6-sol` max-reasoning review lanes complete and all
   load-bearing findings are fixed or explicitly ruled on;
10. current-cohort claims remain diagnostic until a new future stream exists.
