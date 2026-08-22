# Paper-grade J-TEXT → EAST three-stage package

This directory is the auditable manuscript/evidence bundle for the same-lineage chain:

```text
J-TEXT causal source model
→ EAST support110 few-shot transfer
→ EAST balanced360 static evaluation
→ reset to the same transferred parents
→ chronological online active adaptation
```

The package does **not** call balanced360 an independent final test. It was used for architecture and 972-candidate parameter development; all EAST claims are `TEST_DRIVEN_EXPLORATION`. J-TEXT TEST220 is held out within the executed lineage, but it is a 110/110 paper131-count-aligned comparator, not a strict chronological or prevalence-representative test. The successful primary source lineage also fails the registered padding/context shortcut gate.

## Headline metrics

Three-seed means from one traceable checkpoint lineage:

| Stage | Formal lead rule | Operational TPR | FPR | AUC | Operational median lead |
|---|---|---:|---:|---:|---:|
| J-TEXT TEST220 | >10 ms | 0.8515 | 0.2121 | 0.9219 | 116.6 ms |
| EAST static balanced360 | ≥30 ms | 0.8407 | 0.1667 | — | 242.0 ms |
| EAST online balanced360 | ≥30 ms | 0.8481 | 0.1519 | — | 229.5 ms |

The formal TPR counts only lead-qualified alarms; an EAST alarm with lead below
30 ms is an operational FN.  Alarms earlier than 3 s remain eligible under the
registered contract and are audited separately.  Raw alarm occurrence is kept
inside machine evidence only as a diagnostic reproduction field, not promoted
as a second result.  Every J-TEXT alarm ledger now stores its causal endpoint
and `alarm_time_s`; the evidence builder recomputes lead against HDF5
`/meta/DownTime` and fails closed if the archived and recomputed values differ.

Paired circular moving-block resampling (block 20) gives online-minus-static
operational TPR median `+0.00717` (95% descriptive interval
`[+0.00171, +0.01500]`), FPR median `-0.01389`
(`[-0.03521, -0.00193]`), and operational median lead `-11.66 ms`
(`[-69.01, +4.00] ms`). Block sizes 10/20/30/45 are retained as sensitivity
checks. These are unadjusted, post-selection summaries of the observed ledger;
the learner was not rerun inside resamples and they are not future-campaign
confidence intervals. The older iid stratified result remains secondary only.

## EAST-only baselines

Using only the same support110 labels, a randomly initialized Transformer with
an EAST-local scaler and the full target-side pipeline obtains equal-seed mean
TPR@30 `0.7963`, FPR `0.1444`, and TPR@30-FPR `0.6519`. The transferred static
model obtains `0.8407`, `0.1667`, and `0.6741`, respectively: transfer adds
`+0.0444` TPR and `+0.0222` FPR, for a net `+0.0222` on the combined objective
and `+76.5 ms` median lead. A random model using only the source scaler falls
to TPR@30 `0.7315`, exposing a normalization mismatch. These controls use the
same target window and configured optimizer-step budget, but not equal
trainable capacity or FLOPs, and balanced360 remains development evidence.

The alarm executed on the development stream is a causal two-route state
machine rather than one scalar threshold; it has not undergone tokamak-side
deployment acceptance. `high_threshold`, `physics_weight`, `trend_weight`, and
`early_threshold` are projected and updated only after a selected shot's label
is revealed.  The frozen ledgers contain 38/30/40 real parameter-changing
updates for seeds 131/133/137.  The downstream episode gate may delay or veto
an alarm but cannot create or advance one.

## Same-label-access ablation

The exact labels queried by each online seed were also front-loaded into an
ordinary target refit, and those training shots were removed from that seed's
evaluation complement. The support-only reconstruction first passed exact
parent-hash, threshold, and prediction fidelity. Across the three disjoint
complements, front-loading changed TPR@30 by `+0.0641` and FPR by `+0.1673`,
so `TPR@30-FPR` fell by `0.1032` in every seed. Causal online exceeded the
front-loaded control by `0.1163` on the same objective in every seed. Three
label-agnostic, block-count-matched random front-load replicates per seed show
the same sensitivity/FPR displacement.

Under the registered ordinary target-refit rule, this rejects the narrow
"maybe it only saw more labels" alternative: label access alone does not
reproduce the online operating point. A separate causal random-query replay now
matches each seed's label count, per-20-shot counts, parent, runner and
configured optimizer budget with three random schedules per seed. Active
TPR@30-FPR exceeds the within-seed random mean in all three seeds (equal-seed
mean difference `+0.0160`). This remains descriptive: the active selector
triggers 2.21 times as many realized inter-shot update attempts, the exact
one-sided three-sign probability is 0.125, and balanced360 is the development
stream rather than prospective confirmation.

An additional selector-roster screen removes the realized window and
optimizer-attempt-count mediator under a separate operator.
It replays the archived active identities and three label-free random rosters
per seed through a separate fixed assimilation operator: every query receives
exactly 16 current windows, 16 immutable support110 replay windows, two shared
optimizer attempts and two private attempts; intra-shot, alarm, episode and
memory learners are frozen. Guard acceptance and committed parameter changes
can still differ. Active-minus-random-mean `TPR@30-FPR` is `-0.00185`, `0`, and
`-0.00556` for seeds 131/133/137. The post-development, declared all-seed
screening rule therefore fails. This does not prove equivalence or a universal
null and does not estimate the adaptive selector-policy effect. It shows only
that archived-roster superiority is unsupported under this fixed operator; the
full online gain remains a coupled treatment that this screen cannot isolate.

## Contents

```text
evidence/
  three_stage_results.json          machine-readable adjudication
  table_*.csv                       flat result tables
  required_hdf5_content_manifest.json
                                    portable SHA-256 manifest for 2,203 HDF5 files
  source_context_shortcut_audit.json
  equal_label_budget_frontload_v2/  selected/random front-load counterfactual
  causal_random_query_online_v1/    3 seeds × 3 causal random-query replays
  equal_compute_selector_control_v1/
                                    fixed-window/step selector-roster screen
  east_target_only_v1/              strict EAST-only and scaler controls
  source_lineage/                   source checkpoints and TEST220 ledgers
  primary_online_artifacts/         compact static/online/query/reveal ledgers
  case_study_seed131_shot73790.npz self-contained figure-4 trace
figures/
  fig0...fig11.{pdf,svg,png}        vector + 600 dpi publication figures
  figure_manifest.json              input/output SHA-256 ledger
manuscript/
  main.tex / main.pdf               English main manuscript
  supplement.tex / supplement.pdf   detailed supplementary information
  references.bib
  tables/                            evidence-generated LaTeX tables + hashes
scripts/
  build_evidence.py                 recompute labels, metrics, block resampling, audits
  reproduce_primary_chain.py        exact frozen source/target/online command plan
  generate_latex_tables.py
  generate_figures.py
  validate_paper_package.py
  run_causal_random_query_baseline.py
  run_east_target_only_suite.py
  archive_causal_random_query_results.py
  archive_east_target_only_results.py
  train_source_padfix.py             shortcut-repair ablation
  evaluate_locked_source_padfix.py
TECHNICAL_REPORT_zh.md               formal Chinese technical report with derivations and evidence grading
TECHNICAL_REPORT_zh.pdf              rendered 30-page A4 Chinese technical review copy
NATURE_PORTFOLIO_READINESS.md        submission-readiness gate and blockers
FIVE_WAY_AUDIT.md                    independent five-lane verdict and fixes
```

## Data contract

Raw data are external. From the repository root, provide:

```text
data/J-TEXT/unified_hdf5/<shot//100*100>/<shot>.hdf5
data/EAST/unified_hdf5/<shot//100*100>/<shot>.hdf5
```

Alternatively, keep the raw directory outside the checkout and set
`FUSION_DATA_ROOT=/absolute/path/to/data` for evidence rebuilding, tests, and
full retraining. This is the layout used by the clean local release checkout.

Only HDF5 `/meta/IsDisrupt` and `/meta/DownTime` define outcomes and disruption times. J-TEXT TRAIN/VAL/TEST contain 188/80/110 disruptions and are zero-overlap, but their role ranges are not chronological; balanced VAL160/TEST220 are count-aligned comparators. The EAST support roster contains 110 shots and is **15 disruptive / 95 safe by HDF5**, although the legacy JSON headings are 10/100. The same support roster fits target components and derives the target threshold; there is no independent EAST validation. Support110 and balanced360 are disjoint.

## Rebuild paper evidence

Use the project’s CUDA-capable `rl_env`:

```bash
conda activate rl_env
export FUSION_DATA_ROOT=/absolute/path/to/data  # omit when using ./data
python experiments/transfer_methods_2026/paper_grade_package/scripts/build_evidence.py
python experiments/transfer_methods_2026/build_equal_label_budget_report.py
python experiments/transfer_methods_2026/paper_grade_package/scripts/run_equal_compute_selector_control.py \
  --output-root experiments/transfer_methods_2026/runs/equal_compute_selector_control_v1 \
  --seeds 131 133 137 --replicates 3 --stage run
python experiments/transfer_methods_2026/paper_grade_package/scripts/archive_equal_compute_selector_results.py \
  --source-root experiments/transfer_methods_2026/runs/equal_compute_selector_control_v1 \
  --destination-root experiments/transfer_methods_2026/paper_grade_package/evidence/equal_compute_selector_control_v1 \
  --replace
python experiments/transfer_methods_2026/paper_grade_package/scripts/build_equal_compute_selector_report.py \
  --root experiments/transfer_methods_2026/paper_grade_package/evidence/equal_compute_selector_control_v1 \
  --data-root "$FUSION_DATA_ROOT" \
  --out experiments/transfer_methods_2026/paper_grade_package/evidence/equal_compute_selector_control_v1/report.json
python experiments/transfer_methods_2026/paper_grade_package/scripts/build_equal_compute_code_manifest.py
python experiments/transfer_methods_2026/paper_grade_package/scripts/build_equal_compute_artifact_manifest.py
python experiments/transfer_methods_2026/paper_grade_package/scripts/generate_latex_tables.py
python experiments/transfer_methods_2026/paper_grade_package/scripts/generate_figures.py
pytest -q tests/experiments/test_paper_grade_evidence.py \
  tests/experiments/test_jtext_source_padfix.py \
  tests/experiments/test_equal_label_budget_ablation.py
```

The archive step validates the complete 3-seed x 4-arm bundle, copies it through
a temporary directory, and preserves the previous release as a recovery backup
until the replacement succeeds.  The report and content manifest are then built
from the released directory itself, so regenerated metrics cannot be paired
with stale package-local ledgers or checkpoints.

The fixed-shape control consists of 12 RTX 5070 runs and takes roughly 26
minutes on the recorded machine. `--stage plan` writes and audits commands
without running them. Its code manifest is explicitly a post-run release
snapshot, not proof of an execution-time code seal.

Compile and validate:

```bash
cd experiments/transfer_methods_2026/paper_grade_package/manuscript
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error supplement.tex
cd ../../../..
python experiments/transfer_methods_2026/paper_grade_package/scripts/validate_paper_package.py
```

The validator currently checks 128 package invariants, including source and
target-parent lineage, Q30's recoverable identity boundary, primary causal
ledger relations, compact-ledger decompression hashes, figure/table input and
output manifests, text-artifact portability, and the portable 2,203-file HDF5
content manifest. A 128/128 result
means the released development evidence is internally complete and mutually
consistent; it does **not** upgrade the scientific status to independent
confirmation.

The evidence rebuild first reads the immutable, package-local source and online ledgers. Large `updates.jsonl` ledgers are stored as gzip archives and are verified against the SHA-256 of their decompressed original bytes. If the compact archive is absent in a development checkout, the scripts fall back to `experiments/transfer_methods_2026/online_episode_value_v3_runs/`. Full model retraining requires the experiment modules plus `data/`; paper metric regeneration does not retrain or change thresholds.

To inspect the exact frozen model commands without running them:

```bash
python experiments/transfer_methods_2026/paper_grade_package/scripts/reproduce_primary_chain.py \
  --seed 131 --stage plan --output-root results/reproduced-v3c
```

To execute one full seed from the bundled hash-bound source checkpoint and the
fidelity-verified archived EAST target parent, use `--stage all`. Add
`--retrain-source` to regenerate the J-TEXT parent from HDF5; the archived
target-parent loader then fails unless the regenerated source hash matches its
registered lineage. The wrapper fails if a stage output already exists and
preserves the `TEST_DRIVEN_EXPLORATION` status; it does not turn balanced360
into an independent confirmation set.

The three post-support target-parent model states are bundled with exact
encoder/head, shared-adapter, EAST-private-adapter and threshold fidelity
records. Historical target optimizer states are not bundled. Cross-hardware
source retraining is not claimed to reproduce floating-point trajectories byte
for byte; every archived-parent replay is nevertheless required to reproduce
the package static prediction ledger exactly before online comparison.

## Scientific boundary

Verified:

- causal `predict → seal → query → selected reveal → later-shot update` ordering;
- prediction-time A1 feature loading does not open outcome metadata; selected
  outcomes are read lazily only after the current prediction is sealed;
- same source checkpoint lineage across all three stages;
- HDF5-authoritative labels and disjoint support/test rosters;
- the recorded online FPR reduction direction across seeds on the development stream;
- query error enrichment, negative controls, temporal blocks, early-alarm audit;
- exact-parent, query-label-disjoint selected/random front-load ablation;
- causal block-matched random-query online replay (three replicates per seed);
- a fixed-shape selector-roster control with exact realized windows and
  optimizer attempts, whose active-superiority screen fails;
- PADFIX passes source context gates but causes negative transfer under the frozen target policy.

Not verified:

- an independent, never-used EAST future stream;
- a strict chronological or prevalence-representative J-TEXT test;
- an independent target-validation cohort separate from support110;
- adaptive selector-policy superiority under matched realized compute;
- deployment latency, actuator integration, or campaign-general operation;
- a shortcut-free primary parent with preserved transfer performance;
- proof that the model learned all causal disruption mechanisms.

The only claim-changing next step is a fully frozen replay on newly acquired chronological EAST shots, with static/active/random branches and a target-validation rule fixed before acquisition. Do not retune this package on balanced360. This release is an auditable development manuscript, not yet a confirmatory Nature Portfolio submission.
