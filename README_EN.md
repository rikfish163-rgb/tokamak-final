# J-TEXT → EAST: causal transfer and chronological online adaptation for tokamak disruption warning

**Language:** [简体中文（默认）](README.md) · **English**

[Code-level method details (Chinese)](IMPLEMENTATION_DETAILS_CN.md) · [Reproduction and deployment](#quickstart) · [Paper-grade package](experiments/transfer_methods_2026/paper_grade_package/README.md) · [Data contract](DATA_CONTRACT.md) · [Chinese technical report PDF](experiments/transfer_methods_2026/paper_grade_package/TECHNICAL_REPORT_zh.pdf)

[![Repository contract](https://github.com/rikfish163-rgb/tokamak-final/actions/workflows/ci.yml/badge.svg)](https://github.com/rikfish163-rgb/tokamak-final/actions/workflows/ci.yml)
![Evidence boundary](https://img.shields.io/badge/evidence-development%20only-E69F00?style=flat-square)
![Prospective confirmation](https://img.shields.io/badge/prospective%20confirmation-pending-D55E00?style=flat-square)
![Release lineage](https://img.shields.io/badge/release-paper%20v3c%20archived--parent-2F80ED?style=flat-square)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&logoColor=white)
![HDF5](https://img.shields.io/badge/HDF5-authoritative%20labels-00599C?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Linux%20GPU-555555?style=flat-square&logo=linux&logoColor=white)

![Model](https://img.shields.io/badge/model-causal%20Transformer-0072B2?style=flat-square)
![Transfer](https://img.shields.io/badge/transfer-J--TEXT%20%E2%86%92%20EAST-009E73?style=flat-square)
![Online learning](https://img.shields.io/badge/online-chronological%20active%20learning-8E6C8A?style=flat-square)
![Reproducibility](https://img.shields.io/badge/reproducibility-hash--bound-2F80ED?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-0072B2?style=flat-square)

> A reproducibility-first research package for causal J-TEXT → EAST transfer and chronological online adaptation, with HDF5-authoritative labels and sealed query/reveal ledgers.
>
> Evidence ceiling: `DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION`. `balanced360` is a method-development stream, not an independent final test or deployment confirmation.

## Table of contents

- [Scope and status](#scope-and-status)
- [Architecture and causal protocol](#architecture-and-causal-protocol)
- [Three-stage results](#three-stage-results)
- [Quickstart](#quickstart)
- [Data contract](#data-contract)
- [Reproducibility contract](#reproducibility-contract)
- [Evidence map](#evidence-map)
- [Completed development controls](#completed-development-controls)
- [Prospective confirmation seal](#prospective-confirmation-seal)
- [Limitations and claim boundary](#limitations-and-claim-boundary)
- [Repository layout](#repository-layout)
- [Citation and license](#citation-and-license)

## Scope and status

This checkout packages one traceable chain:

```text
J-TEXT source fit
→ EAST support110 few-shot transfer
→ EAST balanced360 static evaluation
→ reset to the same transferred parent
→ chronological balanced360 online-active replay
```

The chain is designed for auditable method development. `balanced360` was used for architecture development and 972-candidate parameter screening, so its static and online metrics cannot be presented as an untouched test, an independent future campaign, or a deployment result. The current default CLI is the paper **v3c archived-parent replay** path; historical v8 is excluded from the final paper-lineage release contract.

All outcome and disruption-time labels come from HDF5 `/meta/IsDisrupt` and `/meta/DownTime`. The paper-grade package, machine evidence, figures, and readiness audit are collected in [the paper-grade package](experiments/transfer_methods_2026/paper_grade_package/README.md).

### Two explicitly isolated experiment lineages

The repository keeps two scientifically different lines side by side; neither
entry point silently overwrites the other:

| Lineage | J-TEXT source label | Online stage | Allowed as an EAST transfer parent |
|---|---|---|---|
| `paper-v3c-archived-parent-replay` | fixed 125 ms (archived release line) | EAST V3C two-timescale adaptation | yes, under the archive contract |
| `jtext-adv25-online` | `max(25 ms, AdvancedTime)` | delayed-feedback TEST220 replay | no; J-TEXT online state is forbidden as an EAST parent |

The latest J-TEXT branch is in
[experiments/transfer_methods_2026/jtext_online/](experiments/transfer_methods_2026/jtext_online/).
It uses the three registered static Adv25 checkpoints and processes TEST220 in
shot order. Each complete block seals all predictions before the query budget
opens selected HDF5 outcomes. The online component updates only the convex
weights of frozen source experts, so it demonstrates delayed-feedback decision
adaptation rather than online Transformer representation learning. Its exact
command and field contract are documented in the package
[README](experiments/transfer_methods_2026/jtext_online/README.md).

The committed Adv25 reference summary contains only the budget-20 run that was
actually preserved; missing runs are not interpolated or invented. Its status
remains `TEST_DRIVEN_EXPLORATION` because TEST220 participated in method
development.

## Architecture and causal protocol

The following publication-style overview summarizes the executed architecture. The
tracked code and [implementation details](IMPLEMENTATION_DETAILS_CN.md) remain the
authoritative specification for losses, active versus shadow branches, and update
order:

![J-TEXT source, EAST few-shot transfer, and online-active architecture](docs/figures/jtext_east_mainline_architecture_v2.png)

The separate roster and information-access protocol is available as
[Figure 1](experiments/transfer_methods_2026/paper_grade_package/figures/fig1_protocol_and_causal_contract.svg).

The online contract is:

```text
predict → seal → query → selected outcome reveal → later-shot update
```

The current-shot outcome is not available to its own prediction. Static evaluation is sealed before the online branch is reset to the identical transferred parent, so the two EAST branches remain phase-separated within the recorded lineage.

## Three-stage results

These are the maintained three-seed means for the current paper-grade lineage. The formal TPR counts only alarms meeting the stage-specific lead rule; an EAST alarm with lead `<30 ms` is an operational false negative. AUC is reported only where it is part of the current stage table.

| Stage | Evaluation role | Operational TPR | FPR | AUC | Operational median lead |
|---|---|---:|---:|---:|---:|
| J-TEXT source | non-temporal TEST220 comparator, lead `>10 ms` | 0.8515 | 0.2121 | 0.9219 | 116.6 ms |
| EAST static | balanced360 after support110 transfer, lead `≥30 ms` | 0.8407 | 0.1667 | — | 242.0 ms |
| EAST online active | reset to the same transferred parent, chronological balanced360 replay, lead `≥30 ms` | 0.8481 | 0.1519 | — | 229.5 ms |

The table is evidence from a development stream, not a claim of prospective generalization. Alarms earlier than 3 s remain eligible under the registered contract and are audited separately; raw alarm occurrence is not promoted as a second primary result.

## Quickstart

### 1. Environment and data

Use the project `rl_env` environment and provide the raw HDF5 directory either as `data/` in the checkout or through `FUSION_DATA_ROOT`:

For the CPU-only unit/audit surface used by public CI, use the dedicated locked
dependency group; it deliberately does not install the CUDA project runtime:

```bash
uv sync --frozen --only-group ci --no-install-project
```

GPU training/replay uses the maintained CUDA-capable `rl_env`. The repository
is the reproducibility artifact: it is not a standalone wheel. An installed
`fusion-reproduce` entry means an editable/source-checkout installation and
still requires the complete checkout (or an explicit `FUSION_PROJECT_ROOT`)
because the frozen scripts, checkpoints, split contracts and reference results
live outside the Python package. The release does not claim that the post-run
environment snapshot is an execution-time seal.

```bash
export FUSION_DATA_ROOT=/absolute/path/to/data

data/
├── J-TEXT/unified_hdf5/<shot//100*100>/<shot>.hdf5
└── EAST/unified_hdf5/<shot//100*100>/<shot>.hdf5
```

Raw HDF5 data are external to Git. Generated ledgers should preferably be written to an ext4 path; high-frequency `fsync` on some `ntfs3` mounts can block.

### 2. Inspect the frozen v3c chain without running training

The plan stage prints the exact source → support → static → reset → online command without executing the replay:

```bash
conda run -n rl_env python \
  experiments/transfer_methods_2026/paper_grade_package/scripts/reproduce_primary_chain.py \
  --seed 131 --stage plan --output-root results/reproduced-v3c
```

Use `--stage all` to execute one seed from the bundled hash-bound source checkpoint and fidelity-verified archived EAST target parent. The output directory must be new/empty.

### 3. Run the default reproducibility CLI

From a complete source checkout, `python -m reproducibility.cli reproduce` and
the editable-installed `fusion-reproduce reproduce` entry both use the paper
v3c archived-parent path and compare against
`results/reference/paper_v3c_multiseed_summary.json`:

```bash
conda run -n rl_env python -m reproducibility.cli reproduce \
  --data-root "$FUSION_DATA_ROOT" \
  --output results/reproduced-v3c \
  --seeds 131 133 137
```

For a contract-only audit or a GPU/reference smoke check:

```bash
conda run -n rl_env python -m reproducibility.cli audit \
  --data-root "$FUSION_DATA_ROOT" --full \
  --output results/generated/data_audit.json
conda run -n rl_env python -m reproducibility.cli smoke \
  --data-root "$FUSION_DATA_ROOT"
```

### 4. Rebuild and validate the paper evidence

These commands rebuild/check the package evidence; they require the data contract and the project environment:

```bash
conda run -n rl_env python \
  experiments/transfer_methods_2026/paper_grade_package/scripts/build_evidence.py
conda run -n rl_env python \
  experiments/transfer_methods_2026/paper_grade_package/scripts/validate_paper_package.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete contract, replay notes, and missing-confirmation boundary.

## Data contract

- Raw inputs are HDF5 files under `J-TEXT/unified_hdf5/` and `EAST/unified_hdf5/`; `FUSION_DATA_ROOT` may point outside the checkout.
- Only HDF5 `/meta/IsDisrupt` and `/meta/DownTime` define outcomes and disruption times. CSV/JSON files provide rosters, split metadata, or audit information; they do not override HDF5 truth.
- J-TEXT `TEST220` is a 110/110 count-aligned, non-temporal balanced comparator. It is not a strict chronological or prevalence-representative test set.
- EAST `support110` contains 110 shots and is 15 disruptive / 95 safe by HDF5. It is used both to fit target components and to select the target threshold; there is no independent EAST validation cohort.
- `support110` and `balanced360` are disjoint, but `balanced360` was still used for method development and candidate screening.

The full field, split, and roster contract is documented in [DATA_CONTRACT.md](DATA_CONTRACT.md) and the package-level account is in [the paper-grade README](experiments/transfer_methods_2026/paper_grade_package/README.md).

The Adv25 J-TEXT branch additionally reads
`shot_list/AdvancedTime_J-TEXT.json`. This file defines only the positive-window
upper edge for disruptive shots: missing values fall back to 25 ms and present
values use `max(25 ms, AdvancedTime)`. Entries for safe shots are ignored; HDF5
outcome fields remain authoritative.

## Reproducibility contract

The release is organized around the following checks:

1. The source and target stages use the recorded checkpoint lineage.
2. Static EAST predictions are completed and sealed before the online branch is reset to the same transferred parent.
3. Every online shot seals prediction and query decisions before any selected outcome is revealed.
4. The evidence builder recomputes labels and J-TEXT lead from HDF5 truth and archived alarm times, then fails closed on disagreement.
5. Large online ledgers are hash-checked against their decompressed bytes; committed summaries alone are not a successful reproduction.

The current release scope is in [RELEASE_SCOPE.md](RELEASE_SCOPE.md). Historical v8 artifacts are not part of the final paper-lineage default reproduction contract.

## Evidence map

| Question | Primary evidence | What it establishes |
|---|---|---|
| Is the online order causal? | [Fig. 1](experiments/transfer_methods_2026/paper_grade_package/figures/fig1_protocol_and_causal_contract.svg), [online artifacts](experiments/transfer_methods_2026/paper_grade_package/evidence/primary_online_artifacts/), [validator](experiments/transfer_methods_2026/paper_grade_package/scripts/validate_paper_package.py) | Recorded `predict → seal → query → reveal → later-shot update` ordering |
| Where do the three-stage means come from? | [three_stage_results.json](experiments/transfer_methods_2026/paper_grade_package/evidence/three_stage_results.json), [table_three_stage.csv](experiments/transfer_methods_2026/paper_grade_package/evidence/table_three_stage.csv) | Hash-bound machine summaries for the displayed metrics |
| Which labels and rosters are authoritative? | [DATA_CONTRACT.md](DATA_CONTRACT.md), [split manifests](config/splits/), HDF5 `/meta/*` | Data and label lineage; no CSV/JSON label override |
| Are the publication figures hash-accounted? | [figure_manifest.json](experiments/transfer_methods_2026/paper_grade_package/figures/figure_manifest.json) | Figure input/output SHA-256 ledger |
| Is the source representation shortcut-free? | [source_context_shortcut_audit.json](experiments/transfer_methods_2026/paper_grade_package/evidence/source_context_shortcut_audit.json), [Fig. 3](experiments/transfer_methods_2026/paper_grade_package/figures/fig3_source_shortcut_and_padfix_tradeoff.svg) | A documented shortcut failure and the PADFIX trade-off, not a shortcut-free claim |
| What controls exist for label access? | [equal-label front-load evidence](experiments/transfer_methods_2026/paper_grade_package/evidence/equal_label_budget_frontload_v2/), [Fig. 8](experiments/transfer_methods_2026/paper_grade_package/figures/fig8_equal_label_budget_frontload_ablation.svg) | A diagnostic selected-label/front-load comparison; not an equal-compute causal random-query result |
| Does active querying beat a causal random schedule? | [causal random-query evidence](experiments/transfer_methods_2026/paper_grade_package/evidence/causal_random_query_online_v1/), [Fig. 9](experiments/transfer_methods_2026/paper_grade_package/figures/fig9_active_vs_causal_random_query.svg) | Active utility exceeds the random mean in all three seeds on the development stream; realized update attempts differ, so selector-only equal-FLOP causality is not isolated |
| Does the archived active roster beat random when windows and optimizer attempts are fixed? | [fixed-shape roster evidence](experiments/transfer_methods_2026/paper_grade_package/evidence/equal_compute_selector_control_v1/), [Fig. 11](experiments/transfer_methods_2026/paper_grade_package/figures/fig11_equal_compute_selector_screen.svg) | No: active-minus-random is `-0.00185/0/-0.00556`. This new operator matches windows, shapes, scopes and attempts, not committed updates; it does not estimate the adaptive selector policy or prove equivalence |
| Does J-TEXT transfer add value beyond EAST-only fitting? | [target-only evidence](experiments/transfer_methods_2026/paper_grade_package/evidence/east_target_only_v1/), [Fig. 10](experiments/transfer_methods_2026/paper_grade_package/figures/fig10_target_only_vs_transfer.svg) | Transfer improves TPR/FPR utility and lead over strict EAST-only on balanced360, with a small FPR increase and unequal trainable capacity |
| Which exact raw files define this release? | [portable HDF5 content manifest](experiments/transfer_methods_2026/paper_grade_package/evidence/required_hdf5_content_manifest.json) | SHA-256 for all 2,203 required J-TEXT/EAST HDF5 files without embedding a workstation path |
| Is this ready for a confirmatory Nature Portfolio or deployment claim? | [Readiness audit](experiments/transfer_methods_2026/paper_grade_package/NATURE_PORTFOLIO_READINESS.md) | No: the current ceiling is an auditable development manuscript |
| Did the five independent review lanes pass? | [Five-way audit](experiments/transfer_methods_2026/paper_grade_package/FIVE_WAY_AUDIT.md) | Yes for local package integrity; all five independently fail Nature confirmatory readiness because new external evidence is still required |

## Completed development controls

The causal block-matched random-query replay is complete for seeds 131/133/137 with three random schedules per seed. Active minus random mean `TPR@30−FPR` is `+0.0204`, `+0.0093`, and `+0.0185`; the equal-seed mean is `+0.0160`. Active selection also causes about `2.21×` as many realized error-conditioned inter-shot update attempts. This is therefore evidence for the complete active mechanism on the development stream, not an equal-FLOP isolation of selector quality and not independent confirmation.

A separate post-development roster screen uses a new fixed-shape assimilation operator. Every queried shot receives 16 current windows, 16 immutable support windows, two shared optimizer attempts and two private attempts; intra-shot, alarm, episode and memory learners are frozen. The archived active roster does not beat the three block-count-matched random rosters in any seed (`-0.00185/0/-0.00556` utility differences). Guard acceptance still causes different committed parameter changes. This negative result rejects only archived-roster superiority under that operator; it neither estimates the adaptive selector-policy effect nor proves a null/equivalence result.

The strict EAST-only baseline is also complete. Its equal-seed mean `TPR@30−FPR` is `0.6519`, versus `0.6741` for J-TEXT→EAST transfer-static; the transferred model increases TPR and lead but also increases FPR. Because EAST-only updates a full randomly initialized encoder while transfer updates a restricted target path, this is a target-data sufficiency/control result rather than a capacity-matched architecture ablation.

The seed-131 low-budget **Q30** roster was first selected causally while the
balanced360 stream advanced, using label-free current/past query-value evidence,
a rolling cutoff, token budget and exploration. It was only **frozen after
selection** so `inter_only`, `intra+inter` and direct-front-load controls could
reuse the same label identities. It is neither an arbitrary predeclared cohort
nor a hindsight global top-30. The immutable identities and positions are in
[the Q30 manifest](experiments/transfer_methods_2026/paper_grade_package/evidence/fixed_query_low_budget30_seed131.json).
The original 360-row decision ledger was lost after a tmpfs reboot, so Q30 is a
development diagnostic rather than independently replayable selector evidence.
The primary v3c online runs are separate and queried `102/90/99` shots for seeds
`131/133/137`.

## Prospective confirmation seal

The missing experiment cannot be manufactured from balanced360, but its
pre-outcome contract is now executable. After a new target-validation cohort
has produced a committed threshold artifact, and before any confirmation-stream
outcome is opened:

1. copy
   [the prospective template](configs/prospective_east_confirmation.template.json)
   outside the checkout;
2. replace its git-SHA, target-validation, pre-sealed random-schedule and
   frozen-execution-plan placeholders;
3. provide separate, strictly increasing, shot-ID-only validation and
   confirmation rosters; and
4. create the seal outside the clean checkout:

```bash
conda run -n rl_env python -m reproducibility.cli prospective-seal \
  --spec /external/preregistration.json \
  --validation-roster /external/target-validation-shots.txt \
  --roster /external/future-confirmation-shots.txt \
  --output /external/future-confirmation.seal.json

conda run -n rl_env python -m reproducibility.cli prospective-verify \
  --seal /external/future-confirmation.seal.json \
  --spec /external/preregistration.json \
  --validation-roster /external/target-validation-shots.txt \
  --roster /external/future-confirmation-shots.txt
```

The seal fails closed on known-shot overlap, validation/confirmation overlap,
metadata-bearing rosters, dirty Git state, a mismatched commit, missing frozen
artifacts, or any later byte change. It does not claim that the unavailable
future experiment has run.

## Limitations and claim boundary

- `balanced360` informed architecture and 972-candidate selection. It is a method-development stream, not an untouched final test.
- J-TEXT `VAL160`/`TEST220` are non-temporal count-aligned balanced comparators, so they do not establish prospective source-domain generalization.
- `support110` is reused for target fitting and threshold selection; there is no independent EAST validation cohort.
- The primary source parent exhibits a padding/context shortcut and fails the registered shortcut gate. The PADFIX path passes its context checks but incurs negative transfer under the frozen target policy.
- No independent, never-used chronological EAST future stream is available. The causal random-query control is complete, but realized error-triggered compute differs and the stream remains development-selected.
- Deployment latency, actuator integration, fail-safe behavior, and campaign-general operation have not been accepted on a device.
- The repository is not presented as Nature Portfolio-ready confirmation and does not claim that the model learned every causal disruption mechanism.

The only claim-changing next step is a fully frozen replay on newly acquired chronological EAST shots, with target validation, static/active/no-label branches, and an equal-budget random-query branch fixed before outcomes are opened. Do not retune the current package on `balanced360`.

## Repository layout

```text
reproducibility/                                      # data audit and v3c replay CLI
experiments/transfer_methods_2026/paper_grade_package/ # manuscript, figures, ledgers, validators
experiments/transfer_methods_2026/jtext_online/      # isolated Adv25 J-TEXT online branch
IMPLEMENTATION_DETAILS_CN.md                           # exact losses, values and update order
config/mainline_code_manifest.json                    # machine-checked mainline inventory
config/splits/                                         # fixed J-TEXT/EAST split contracts
shot_list/AdvancedTime_J-TEXT.json                    # Adv25 positive-window metadata
splits/                                                # readable J-TEXT shot lists
results/reference/                                     # compact reference summaries
tests/                                                 # reproducibility and experiment contract tests
```

## Citation and license

Use [CITATION.cff](CITATION.cff) for the repository's citation metadata. The checkout currently declares the [MIT license](LICENSE). This README does not assert a DOI, add an author list, or upgrade the evidence to deployment or Nature Portfolio readiness; any final author, affiliation, version, and DOI metadata must come from the maintainers.
