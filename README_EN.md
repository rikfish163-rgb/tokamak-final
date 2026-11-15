# J-TEXT → EAST: causal transfer and chronological online adaptation for tokamak disruption warning

**Language:** [简体中文（默认）](README.md) · **English**

## Giving EAST more time before a disruption

EAST is a fusion experiment that confines hot plasma with magnetic fields. A discharge is called a *shot*. A disruption can impose large heat and mechanical loads on the machine. A warning therefore needs to arrive early enough to be useful, without triggering too often on non-disruptive shots.

We train a base model on J-TEXT, prepare it for EAST, and then adapt within each shot: **predict future diagnostic readings, wait for the measurements, and use the prediction error to improve later risk scores.** The current shot's outcome is not available during this adaptation.

> **360 archived development shots:** **146/180 timely detections (81.11%)** and **34/180 false alarms (18.89%)**. The first four steps add eight timely detections over the prepared static baseline; step 5 adds two more.  
> **2,593 additional archived shots:** the full method's false-positive rate is **27.56%**, above the 20% acceptance limit.

Research results through **2026-11-14**; README updated **2026-11-15**. [Detailed research directory (Chinese)](research/east360-online-adaptation/README.md)

## How it works

1. **Train on J-TEXT.** A risk model assesses disruption risk; an auxiliary forecaster estimates future diagnostic readings. Forecasts are not measurements and do not directly trigger alarms.
2. **Prepare on the first 20 EAST shots.** Their predictions are saved first. Once all 20 finish, align the density-ratio input `ne_nG`, fit the EAST scoring formula, and set the later threshold. All 20 shots remain in the full-stream score, with no retrospective re-prediction.
3. **Replay the remaining 340 shots.** Keep the base network and scoring formula frozen. Use only measurements already available to update temporary within-shot state; clear that state at the end of every shot.

Risk is scored at each 1 ms sample from the latest 128 samples of 23 diagnostic channels. Missing observations are marked, not filled from the future. Every 32 samples, the auxiliary model forecasts the next 25 samples; the corresponding error becomes usable only after all measurements arrive. The main method adds no Fourier transform; some magnetic statistics are already present in the supplied diagnostics.

The first 20 shots use the source threshold **0.7258947**. For each of the 12 non-disruptive preparation shots, fit on the other 19 and collect the held-out shot's maximum score. The subsequent main threshold, **0.5062586**, is just above the second-highest of those maxima. Every main comparison uses the same threshold within each stage. The stricter **0.6766528** is reported separately.

## What each step changes—and contributes

These are cumulatively enabled modules, not five stages that must finish before an alarm can fire.

| Added processing | Timely / 180 | False alarms / 180 | Net change from preceding row: timely / false |
|---|---:|---:|---:|
| 0 · Prepared static baseline; no within-shot correction | 136 (75.56%) | 32 (17.78%) | — |
| 1 · Let the score read recent forecast bias and error trends | 138 (76.67%) | 31 (17.22%) | +2 / −1 |
| 2 · Add error beyond the same shot's usual, condition-matched error | 141 (78.33%) | 34 (18.89%) | +3 / +3 |
| 3 · Correct temporal relationships in the observed input copy | 143 (79.44%) | 34 (18.89%) | +2 / 0 |
| 4 · Subtract persistent offset from valid input values | 144 (80.00%) | 34 (18.89%) | +1 / 0 |
| **5 · Use latest error to correct internal representations; wait 5 samples** | **146 (81.11%)** | **34 (18.89%)** | **+2 / 0** |

Steps 1–2 change what the scoring formula reads; step 3 changes input shape, and step 4 changes input level. **144 is already an adaptive comparison, not a static baseline.** Together, steps 1–4 add eight timely detections (+4.44 percentage points) and two false alarms (+1.11 points). Step 3's unchanged false-alarm count hides one removed and one newly added false-alarm shot.

Step 5 changes the input to the **first attention layer**, not the original readings or frozen network weights. The two-layer `Simple23` network divides 128 samples into 16 blocks of eight samples, with 64-dimensional internal representations. Fifteen adjacent block pairs × four diagnostic groups give **60 temporary coefficients**. Measured, signed errors across five forecast segments determine a correction direction. Equal-and-opposite changes to adjacent blocks alter their contrast, within an amplitude limit. The fixed downstream score can then rise or fall.

The new correction waits five extra samples and blends in over eight sampling intervals. Scoring continues throughout. Temporary coefficients reset at shot end.

Step 5 turns **70320 and 73405** from late into timely detections (+1.11 points), with the **same 34 false-alarm shots** as steps 1–4. The full method adds ten timely detections over static; only two are the independent increment from step 5.

## Real curves: an earlier warning and an early false alarm

![Real score traces for shots 70320 and 67900](research/east360-online-adaptation/outputs/east360_technical_20261114/two_shot_timelines.png)

Dashed grey: steps 1–4. Blue: add step 5. Horizontal dotted line: the shared alarm threshold. The left vertical line is disruption; the right one is step 5's first actual write.

**70320:** forecast at **3.639 s** for **3.640–3.664 s**; the full error becomes usable at **3.665 s**. After the five-sample wait, handoff starts at **3.670 s**, first mixes in at **3.671 s**, and completes at **3.678 s**. At **3.697 s**, the score changes from **0.47423 to 0.50769**, crossing **0.5062586**. The shot disrupts at **3.922 s**: first alarm moves from **3.910 s (12 ms lead)** to **3.697 s (225 ms lead)**, gaining **213 ms**. An equal-size correction from earlier error instead gives **0.4601** at that point.

The first crossing lasts only one sample. Under two-sample confirmation, this shot is no longer timely. The other newly timely shot, 73405, moves from **29 to 33 ms** lead; the earlier-error control also rescues it.

**67900:** a non-disruptive shot alarms at **3.176 s**, before step 5 first changes the model at **3.287 s**. Its roughly five-second recording is not short; the alarm occurs only 176 ms after recording starts at 3.000 s. The 3.127 s forecast lacks the required older reference. Forecasts at 3.159, 3.191 and 3.223 s accumulate usable history; the 3.255 s forecast first meets the shared latest/earlier-error eligibility conditions. Its measurements finish at 3.280 s, error is released at 3.281 s, and the delayed handoff first changes the model at 3.287 s. Later feedback cannot revoke an earlier alarm. Step 2 first causes a false alarm at 3.281 s; step 3 moves it to 3.176 s.

**Why wait five samples?** 70320 alarms at the same time with or without the wait. Non-disruptive shot 73743 falsely alarms at 4.698 s without waiting, but never alarms with the wait. Across all 360 shots, waiting changes **146 timely / 35 false → 146 / 34**, with identical timely-shot identities.

[Plot data](research/east360-online-adaptation/outputs/east360_technical_20261114/two_shot_curves.csv) · [Python / seaborn script](research/east360-online-adaptation/outputs/east360_technical_20261114/plot_two_shots.py) · [Vector PDF](research/east360-online-adaptation/outputs/east360_technical_20261114/two_shot_timelines.pdf) · [Internal measurements and changes](research/east360-online-adaptation/outputs/east360_row5_70320/70320_mechanism.pdf)

## Scoring, information access and balance

The 360-shot stream contains 180 disruptive and 180 non-disruptive shots. A first alarm is timely only with lead **strictly greater than 30 ms**; a non-disruptive shot with any alarm is a false positive. The acceptance target is at most **36 false alarms (20%)**.

Accuracy counting late alarms as wrong is **78.89% static → 80.56% steps 1–4 → 81.11% full**. The full method also has 22 late and 12 missed disruptive shots. If late alarms count as correct, all three versions have 87.22% accuracy, concealing the timing improvement.

Current predictions use only current and earlier measurements; full forecast targets must arrive before training on their error. Outcomes are revealed only after predictions are sealed and shots end. HDF5 `meta/IsDisrupt` is the sole outcome authority; `meta/DownTime` is a disruption time only for disruptive shots. EAST's 125 ms training-positive window is distinct from the >30 ms scoring criterion.

J-TEXT uses fixed 1,353-shot training, 160-shot validation and 220-shot test splits. Preparation fitting assigns half of point-loss weight to each class and equal weight to participating shots within each class, preventing long shots from dominating. The main method does not refit on later EAST outcomes.

Inter-shot adaptation is a **separate comparison**: after shot-end reveal, use one previously revealed disruptive and one non-disruptive shot, eight equally weighted points each, to update scores for later shots, with 16 reveals in total. Steps 1–4 plus inter-shot updates yield **145/38**, and adding step 5 yields **147/38**; both have 21.11% false positives and exceed the main limit.

The 360 shots have already been used for method selection. Their fixed replay order is not a certified acquisition-date order, and upstream diagnostic availability and field latency remain unverified. Causal access checks do not turn this development cohort into an independent test.

## Larger samples expose a limit

Using the original model and main threshold, without refitting on the 2,593-shot extension:

| Additional archived shots | Timely / 1,018 | False alarms / 1,575 |
|---|---:|---:|
| Prepared static | 695 (68.27%) | 367 (23.30%) |
| Steps 1–4 | 697 (68.47%) | 420 (26.67%) |
| Full method | 706 (69.35%) | 434 (27.56%) |

Step 5 adds nine timely detections and fourteen false alarms. Even static exceeds 20%, so the problem is not confined to the last step. Balancing to 833 shots per class still gives **230/833 false alarms (27.61%)**.

Back in the 360-shot study, six false alarms occur during preparation, seven before step 5's first write, and 21 after it becomes active; those 21 also alarm with step 5 disabled. Individual alarm times can nevertheless change.

Time-domain features yield **147/42 (23.33% false)**, Fourier bands **149/41 (22.78%)**, Fourier plus current confirmation **146/36 (20.00%)**, and three-diagnostic confirmation **147/39 (21.67%)**. None increases timely detections while retaining the 20% limit, so they are not included in the main method. Stricter-threshold and confirmation checks are kept separately in the [evidence appendix](research/east360-online-adaptation/EAST360_TECHNICAL_EVIDENCE_20261114.md).

Both cohorts are historical. Independent new EAST shots and measured end-to-end latency are still needed. The next validation should freeze the version and compare steps 1–4 with the full method on new shots in recording-only operation, before connecting alarms to machine control.

## Code, evidence and getting started

[Technical proposal (Chinese)](research/east360-online-adaptation/EAST360_TECHNICAL_PROPOSAL_20261114.md) · [Presentation script (Chinese)](research/east360-online-adaptation/EAST360_TECHNICAL_TALK_20261114.md) · [13-row scoring CSV](research/east360-online-adaptation/outputs/east360_row5_70320/EAST360_完整计分.csv) · [False-alarm cases](research/east360-online-adaptation/outputs/east360_row5_70320/EAST360_误报逐炮.csv) · [All 360 shot curves, about 64 MB](research/east360-online-adaptation/outputs/east360_row5_70320/EAST360_逐炮图谱.pdf)

```bash
git clone --branch research/east360-online-adaptation \
  https://github.com/rikfish163-rgb/tokamak-final.git
cd tokamak-final/research/east360-online-adaptation
# With NumPy, matplotlib and seaborn already available:
python outputs/east360_technical_20261114/plot_two_shots.py
```

Reading published results requires no GPU. In the existing project `rl_env`, CPU timing tests can be run with:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q -p no:cacheprovider \
  tests/test_commissioning_write_delay_20261112.py \
  tests/test_commissioning_group_carry_20261112.py
```

A full model replay also needs the original HDF5 data, weights and input caches, which are not uploaded. See [external assets](research/east360-online-adaptation/EXTERNAL_ASSETS.md) and [sync validation](research/east360-online-adaptation/SYNC_VALIDATION.md). Reading saved results is not a new model experiment. [MIT license](LICENSE)

The earlier support110/V3C route uses different splits, processing and thresholds. Its original documentation is retained separately below.

---

<details>
<summary>Historical route: original support110 / V3C repository documentation</summary>

## Historical repository route

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
IMPLEMENTATION_DETAILS_CN.md                           # exact losses, values and update order
config/mainline_code_manifest.json                    # machine-checked mainline inventory
config/splits/                                         # fixed J-TEXT/EAST split contracts
splits/                                                # readable J-TEXT shot lists
results/reference/                                     # compact reference summaries
tests/                                                 # reproducibility and experiment contract tests
```

## Citation and license

Use [CITATION.cff](CITATION.cff) for the repository's citation metadata. The checkout currently declares the [MIT license](LICENSE). This README does not assert a DOI, add an author list, or upgrade the evidence to deployment or Nature Portfolio readiness; any final author, affiliation, version, and DOI metadata must come from the maintainers.

</details>
