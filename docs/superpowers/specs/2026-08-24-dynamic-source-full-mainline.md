# Dynamic-label J-TEXT to EAST full-mainline specification

## 1. Research question and evidence boundary

This specification freezes the development candidate selected by the corrected
four-arm label ablation and defines the only allowed execution path for the
next full-chain experiment.

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

The arm was selected by mean balanced360
`J = operational_TPR - FPR`; this is a development selection, not independent
confirmation.  All current TEST220 and balanced360 outputs retain
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

The final online-updated J-TEXT state is diagnostic and must not become the
formal EAST transfer parent.  A separate
`JTEXT_ONLINE_PRECONDITIONED_TRANSFER_DIAGNOSTIC` arm may be run only after the
isolated primary chain exists and only with a no-J-TEXT-online matched control.

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

The selected parent schema is `dynamic-source-east-fixed-parent-v1`.  It must
contain at least:

```text
schema_version
seed
source_checkpoint_sha256
source_label_policy
source_label_map_sha256
source_split_sha256
source_scaler_sha256
east_target_label_policy
east_target_horizon_ms
east_support_roster_sha256
east_threshold
target_encoder_sha256
target_head_sha256
shared_parent_sha256
private_parent_sha256
claim_status
```

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

The feature-only loader reads signal datasets and sample-rate metadata but not
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

The query score is label-free:

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
raw TPR
operational TPR
FPR
J = operational TPR - FPR
AUC
median and IQR of operational TP lead
very-early alarm rate
TP/FN/TARDY/FP/TN counts
four chronological blocks and worst block
query count, update attempts, accepted updates, rollbacks, windows, FLOPs
```

Any full arm with `TPR < 0.60` or `FPR > 0.40` stops before hyperparameter
tuning and triggers a label, threshold, alarm-index, scaler, order, and output
audit.

## 9. Statistical interpretation

- Show every seed; do not treat three seeds as independent physical shots.
- Use shot-paired differences for static versus online comparisons.
- Use Wilson intervals for descriptive binomial TPR/FPR uncertainty.
- Use chronological block bootstrap only as a descriptive adaptive-stream
  sensitivity analysis.
- Do not attach confirmatory p-values to the arm selected on balanced360.
- If multiple primary contrasts are tested, apply Holm correction.
- A mechanism claim requires active to beat the strongest matched random or
  shuffled control in every seed, not only in the mean.

## 10. Evidence, figures, report, and acceptance

Every run writes effective config, command, environment, git state, data and
roster hashes, label contract, parent hashes, static prediction seal,
post-seal outcome join, online prediction/query/reveal/update ledgers, per-shot
risk traces, metrics, and an explicit claim status.

Quantitative figures are generated from these artifacts with Matplotlib in
PDF/SVG and 600-dpi PNG.  They use an accessible palette plus markers and line
styles that remain distinguishable in grayscale.  Image generation is allowed
only for a clearly marked non-data conceptual graphic.

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
