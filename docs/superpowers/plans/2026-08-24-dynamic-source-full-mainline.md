# Dynamic-source J-TEXT/EAST full-mainline implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the complete J-TEXT dynamic-label source to EAST static/online chain and add a causally sealed J-TEXT TEST220 online-active analogue with matched controls, evidence, publication figures, and a Chinese LaTeX report.

**Architecture:** A hash-bound selected-mainline contract drives a fail-closed EAST branch and an isolated J-TEXT online branch from the same dynamic source checkpoint.  The J-TEXT runner reuses causal query/reveal and two-timescale control concepts but has feature-only J-TEXT I/O, a rebuilt17 residual, device-specific metrics, and no EAST-private raw8 dependency.  Evidence builders consume immutable ledgers; figures and report never infer per-shot facts from aggregate metrics.

**Tech Stack:** Python 3.11, PyTorch 2.11/CUDA 12.8, h5py, NumPy, SciPy, scikit-learn, pytest, Matplotlib, Pandoc/LaTeX.

**Spec:** `docs/superpowers/specs/2026-08-24-dynamic-source-full-mainline.md`

## Global Constraints

- Use only `rl_env`; GPU experiments run on the local RTX 5070.
- Selected source label policy is exactly `advanced_floor`: J-TEXT `max(25 ms, AdvancedTime_ms)`, missing to 25 ms.
- EAST target and post-reveal policy is explicitly `fixed125`; EAST AdvancedTime is not read by the selected arm.
- HDF5 `meta/IsDisrupt` and `meta/DownTime` are the only outcome authority.
- J-TEXT TRAIN1353/VAL160/TEST220, EAST support110, and balanced360 roles and hashes are immutable.
- Source checkpoint paths are explicit; no archived fixed-source fallback and no bare encoder/head path are allowed in the selected runner.
- Static predictions are completely sealed before any test outcome read; online order is `predict -> seal -> query -> reveal -> update`.
- J-TEXT TEST220 online-updated weights never enter the formal EAST parent.
- J-TEXT operational TPR uses `lead > 10 ms`; EAST operational TPR uses `lead >= 30 ms`.
- Current TEST220 and balanced360 results remain diagnostic/test-driven evidence.
- Every production edit follows RED-GREEN-REFACTOR and every task receives a separate spec/quality review.
- All subagents for this goal use `gpt-5.6-sol` with `max` reasoning; Luna is prohibited.

---

### Task 1: Fail-closed selected-mainline contract and EAST lineage

**Files:**
- Create: `experiments/transfer_methods_2026/selected_mainline_contract.py`
- Create: `experiments/transfer_methods_2026/run_selected_full_mainline.py`
- Create: `experiments/transfer_methods_2026/paper_grade_package/configs/primary_chain_dynamic_source_v1.json`
- Modify: `experiments/transfer_methods_2026/grid_search.py`
- Modify: `experiments/transfer_methods_2026/run_online_active_a1.py`
- Modify: `experiments/transfer_methods_2026/run_dynamic_hazard_adapter.py`
- Modify: `experiments/transfer_methods_2026/run_unified_online_active.py`
- Test: `tests/experiments/test_selected_mainline_contract.py`
- Test: `tests/experiments/test_selected_mainline_runner.py`

**Interfaces:**
- Produces: `SelectedMainlineContract`, `load_selected_source_checkpoint(path, seed)`, `validate_selected_parent(payload, contract)`, and parent schema `dynamic-source-east-fixed-parent-v1`.
- Produces: CLI `run_selected_full_mainline.py --seed N --data-root PATH --output PATH --stage {source,east-parent,static,online,all}`.
- Consumes: existing source trainer, support-only A1, shared/private parent components, and unified online runner.

- [ ] **Step 1: Write failing contract tests**

  Add literal-fixture tests proving that a source with `training_label.policy != "advanced_floor"`, missing source path, mismatched SHA, legacy parent schema, or `source_checkpoint=None` is rejected.  Add a test that a valid payload with explicit `source_dynamic/east_fixed` metadata is accepted.

- [ ] **Step 2: Verify RED**

  Run:

  ```bash
  FUSION_DATA_ROOT=/media/hetaisheng/044A81D94A81C83E/Fuison/data \
    conda run -n rl_env python -m pytest -q \
    tests/experiments/test_selected_mainline_contract.py
  ```

  Expected: collection or assertion failure because the selected contract does not exist.

- [ ] **Step 3: Implement the minimal contract module**

  Implement immutable dataclasses with exact policy strings and SHA fields.  The selected source loader delegates to the existing checkpoint loader, then verifies policy, seed, split, scaler, label-map provenance, and `test_shards_read == 0`.  Do not change legacy replay defaults globally; the selected entry point passes `allow_fallback=False` and rejects a missing checkpoint.

- [ ] **Step 4: Write and verify failing no-test-read tests**

  Instrument the HDF5 outcome reader in tests and assert that source fitting, EAST support adaptation, shared adapter fitting, and private parent fitting complete without opening any balanced360 `/meta` outcome.  Assert the first outcome read occurs only after a 360-row static seal has been closed and hashed.

- [ ] **Step 5: Fix EAST selected path**

  Change the selected adapter path from `shot_alarm.train()` to the support-only entry point.  Pass `target_label_policy="fixed125"` and the dynamic source checkpoint explicitly.  Derive target threshold from support110.  Emit a new parent payload rather than relabeling a legacy payload.  Extend the unified validator to accept the new schema only after full contract validation.

- [ ] **Step 6: Implement orchestration and stage-resume receipts**

  The runner writes `run_contract.json`, `stage_receipts.jsonl`, effective command/config, source/target SHA fields, and claim status.  A completed stage may resume only when code/config/data/parent hashes match.

- [ ] **Step 7: Verify GREEN and regressions**

  Run the two new files plus existing source, AdvancedTime, A1, adapter, unified, and prospective-seal tests.  Confirm all pass with the external data root.

- [ ] **Step 8: Commit**

  ```bash
  git add experiments/transfer_methods_2026/selected_mainline_contract.py \
    experiments/transfer_methods_2026/run_selected_full_mainline.py \
    experiments/transfer_methods_2026/paper_grade_package/configs/primary_chain_dynamic_source_v1.json \
    experiments/transfer_methods_2026/grid_search.py \
    experiments/transfer_methods_2026/run_online_active_a1.py \
    experiments/transfer_methods_2026/run_dynamic_hazard_adapter.py \
    experiments/transfer_methods_2026/run_unified_online_active.py \
    tests/experiments/test_selected_mainline_contract.py \
    tests/experiments/test_selected_mainline_runner.py
  git commit -m "feat: bind dynamic source through EAST mainline"
  ```

### Task 2: J-TEXT signal-only I/O, ordered roster, and lazy outcome oracle

**Files:**
- Create: `experiments/transfer_methods_2026/jtext_online_io.py`
- Create: `config/splits/jtext_test220_ordered_v1.json`
- Test: `tests/experiments/test_jtext_online_io.py`

**Interfaces:**
- Produces: `JTextSignalShot`, `load_jtext_signal_only(data_root, shot_id)`, `JTextLazyOutcomeOracle`, `load_ordered_test220(root)`, and `build_postseal_jtext_target(...)`.
- Consumes: rebuilt17 feature construction, source checkpoint scaler/padding metadata, HDF5 files, split lists, and AdvancedTime mapping.

- [ ] **Step 1: Write failing feature-isolation tests**

  Create temporary complete HDF5 fixtures.  Assert the signal-only object contains samples, timebase, availability, and causal metadata but no outcome-bearing key.  Mutating `IsDisrupt`, `DownTime`, or AdvancedTime before reveal must not alter signal features, risk input, or query input.

- [ ] **Step 2: Verify RED**

  Run `conda run -n rl_env python -m pytest -q tests/experiments/test_jtext_online_io.py` and observe failure because the module is absent.

- [ ] **Step 3: Implement signal-only loading and checkpoint-bound windows**

  Reuse rebuilt17 transformations but separate signal reads from outcome reads.  Use the exact source-checkpoint padding and `valid_lengths` contract; do not call the EAST `grid_search.causal_window` implementation for J-TEXT.

- [ ] **Step 4: Write failing oracle/order tests**

  Assert exactly 220 unique roster rows, deterministic recorded order, zero overlap with TRAIN/VAL, roster SHA, and explicit diagnostic claim.  Assert reveal before shot seal fails, unselected reveal fails, duplicate reveal fails, and selected post-seal reveal succeeds.  AdvancedTime is inaccessible until that receipt exists.

- [ ] **Step 5: Implement lazy oracle and target builder**

  Return HDF5 outcome only after `seal_current` and `reveal_for_query`.  For selected disruptive shots, attach `max(25, AdvancedTime)` after reveal; for missing values attach 25; for safe shots attach no positive horizon.

- [ ] **Step 6: Verify GREEN and commit**

  Run the new tests plus existing data-contract, AdvancedTime, online-audit, and source-transformer tests, then commit the three files with message `feat: add sealed J-TEXT online data boundary`.

### Task 3: J-TEXT two-timescale online runner and matched controls

**Files:**
- Create: `experiments/transfer_methods_2026/jtext_online_adapter.py`
- Create: `experiments/transfer_methods_2026/run_jtext_online_active.py`
- Create: `experiments/transfer_methods_2026/paper_grade_package/configs/jtext_online_active_v1.json`
- Test: `tests/experiments/test_jtext_online_adapter.py`
- Test: `tests/experiments/test_jtext_online_runner.py`

**Interfaces:**
- Produces: `JTextOnlineResidual`, `JTextOnlineController`, arm names `static`, `intra_only`, `inter_random`, `inter_active`, `full_random`, `full_active`, `frontload_equal_labels`, `equal_compute_random`.
- Produces: CLI `run_jtext_online_active.py --source-checkpoint PATH --arm ARM --budget PERCENT --seed N --data-root PATH --output PATH`.
- Consumes: Task 1 source contract and Task 2 signal/oracle interfaces; existing query budget, replay, guard, rollback primitives where their contracts are device-neutral.

- [ ] **Step 1: Write failing residual invariance tests**

  Assert zero initialization reproduces frozen source logits bitwise, online parameter count is reported, source encoder/head hashes never change, bounded residual cannot exceed the configured logit shift, and resetting at shot start restores the persistent state.

- [ ] **Step 2: Verify RED, then implement the minimal residual/controller**

  Use rebuilt17 causal summaries and source risk as inputs.  Keep a fast copy for intra-shot updates and a persistent copy for post-reveal updates.  Snapshot model and optimizer before inter update and restore both on non-finite loss or guard rejection.

- [ ] **Step 3: Write failing causal event tests**

  Assert every selected shot ledger is ordered `predict < seal < query < reveal < update`, unselected shots have no outcome read and no persistent update, intra updates use `source_endpoint < observed_endpoint`, current predictions are immutable, and future-suffix mutation leaves prior receipts unchanged.

- [ ] **Step 4: Implement the runner and enabled losses**

  Build static seal first, reset the exact parent, then replay the ordered stream.  Record every enabled loss coefficient and term.  Disabled or zero-weight terms emit `enabled=false`.  Keep EAST raw8 and EAST thresholds unreachable from this code path.

- [ ] **Step 5: Write failing budget/comparator tests**

  For the 20% primary budget, assert 20-shot blocks, four active and four random queries per block, 44 total queries, identical sampled-window and optimizer-attempt caps, label-agnostic random manifests, and front-load evaluation excludes its training shots.

- [ ] **Step 6: Implement all arms and abnormal-run gate**

  Implement primary and 10/20/30% sensitivity configurations.  Stop an arm with a machine-readable audit receipt when full-arm TPR is below 0.60 or FPR exceeds 0.40; do not tune alignment or loss weights after that trigger.

- [ ] **Step 7: Verify GREEN, synthetic integration, and commit**

  Run the new tests plus online audit, two-timescale adapter, unified value, and prospective seal suites.  Run a tiny two-shot CPU integration that exercises `static -> full_active -> resume` and verifies byte-identical replay.  Commit with message `feat: add J-TEXT online active control matrix`.

### Task 4: Hash-bound evidence, statistics, and publication figures

**Files:**
- Create: `experiments/transfer_methods_2026/paper_grade_package/scripts/build_dynamic_mainline_evidence.py`
- Create: `experiments/transfer_methods_2026/paper_grade_package/scripts/generate_dynamic_mainline_figures.py`
- Create: `experiments/transfer_methods_2026/paper_grade_package/figures_dynamic/.gitkeep`
- Modify: `experiments/transfer_methods_2026/paper_grade_package/scripts/publication_style.py`
- Test: `tests/experiments/test_dynamic_mainline_evidence.py`
- Test: `tests/experiments/test_dynamic_mainline_figures.py`

**Interfaces:**
- Produces: `evidence/dynamic_mainline_v1/report.json`, per-shot post-seal joins, statistical tables, `figure_manifest_v2.json`, and figures F1-F5 in PDF/SVG/600-dpi PNG.
- Consumes: Task 1 and Task 3 immutable ledgers only.

- [ ] **Step 1: Write failing evidence-lineage tests**

  Assert joins require a complete seal SHA; prediction seals contain no outcome keys; raw and operational metrics are separate; all per-seed confusion counts recompute exactly; active/random resources match; missing per-shot source predictions block calibration/lead plots instead of falling back to aggregate metrics.

- [ ] **Step 2: Verify RED, then implement evidence builder**

  Compute Wilson intervals, shot-paired deltas, descriptive chronological blocks, operational lead distributions, very-early alarm rates, query/update counts, and explicit selection/claim status.  Never multiply seed count by shot count for uncertainty.

- [ ] **Step 3: Write failing render-contract tests**

  Assert every figure input hash is in the manifest; each output exists in PDF/SVG/PNG; PNG metadata is at least 600 dpi; minimum configured font is at least 6 pt; every quantitative panel identifies statistic, uncertainty, cohort, seed count, and claim status.

- [ ] **Step 4: Implement F1-F5**

  Generate: causal/data-role pipeline; AdvancedTime exposure and 2x2 label ablation; J-TEXT/EAST static-online paired metrics with blocks; query/update trace and enrichment; intra/inter/selector/front-load/equal-compute falsification controls.  Use real artifacts only, Okabe-Ito-safe colors, markers, line styles, and grayscale-distinguishable encodings.

- [ ] **Step 5: Verify figures and commit**

  Run evidence/figure tests, regenerate into a temporary directory, visually inspect raster previews, verify vector files parse, and commit scripts/tests/style plus the small manifest—not large generated run directories—with message `feat: build dynamic mainline publication evidence`.

### Task 5: Reader-first Chinese report, README, and package verifier

**Files:**
- Create: `experiments/transfer_methods_2026/paper_grade_package/DYNAMIC_MAINLINE_TECHNICAL_REPORT_zh.md`
- Create: `experiments/transfer_methods_2026/paper_grade_package/technical_report_dynamic/main.tex`
- Create: `experiments/transfer_methods_2026/paper_grade_package/scripts/build_dynamic_report.py`
- Create: `experiments/transfer_methods_2026/paper_grade_package/scripts/validate_dynamic_mainline_package.py`
- Modify: `README.md`
- Modify: `README_CN.md`
- Test: `tests/experiments/test_dynamic_mainline_report.py`
- Test: `tests/reproducibility/test_dynamic_mainline_package.py`

**Interfaces:**
- Produces: Chinese Markdown as the single semantic source, compiled LaTeX/PDF, reproducibility commands, formula-to-code index, artifact manifest, and machine-readable validation result.
- Consumes: Task 4 report/manifest/figures and Task 1/3 effective configs.

- [ ] **Step 1: Write failing package-verifier tests**

  Assert the report source covers every defined symbol, distinguishes training/operational/timely labels, names enabled versus zero-weight losses, explains query budget and update order, embeds all required tables/figures, cites real artifact hashes, and refuses missing or stale inputs.

- [ ] **Step 2: Verify RED, then implement report build and verifier**

  The builder converts the Markdown source into escaped LaTeX sections and compiles a standard PDF.  The verifier checks report/figure/evidence/config/code hashes, PDF page count and parseability, and the absence of absolute data paths from portable manifests.

- [ ] **Step 3: Write the report and deployment README**

  Structure the report as: evidence status; data/label ontology; preprocessing; source model; target transfer; alarm state machine; intra loss; query value; inter loss/replay/guard/rollback; controls; statistics; results; failure cases; limitations; reproducibility; symbol glossary; formula-to-code mapping; reviewer questions and responses.  README commands start from a clone plus external `data/`, create/use `rl_env`, run audit/smoke/full/report, and explain expected artifacts.

- [ ] **Step 4: Build, inspect, verify, and commit**

  Compile the PDF, render every page to images, inspect for clipping/unexplained symbols/tiny labels, run package tests and validator, and commit with message `docs: publish dynamic mainline technical package`.

### Task 6: Local GPU execution and five-lane final review

**Files:**
- Generate under ignored output root: `experiments/transfer_methods_2026/paper_grade_package/evidence/dynamic_mainline_v1/`
- Generate: `experiments/transfer_methods_2026/paper_grade_package/FIVE_SOL_MAX_REVIEW.md`

**Interfaces:**
- Consumes all completed code and configs.
- Produces smoke, three-seed/full-chain, J-TEXT online matrix, figures, PDF, review decisions, and final validation receipts.

- [ ] **Step 1: Run audit and one-seed smoke in `rl_env` on RTX 5070**

  Verify CUDA device, HDF5 access, cohort hashes, dynamic source policy, fixed EAST policy, source/parent hashes, and tiny end-to-end static/online runs.

- [ ] **Step 2: Run selected source/EAST chain for seeds 131/133/137**

  Run source training, source TEST220 static seal, support110 adaptation, EAST parent construction, balanced360 static seal, same-parent online replay, and all pre-registered controls.  Preserve all failures as receipts.

- [ ] **Step 3: Run J-TEXT TEST220 online matrix**

  Run the primary 20% budget for every arm and seed.  Run 10/30% sensitivity only after the primary arm passes causality and normality checks.  Do not retune on TEST220 outcomes.

- [ ] **Step 4: Rebuild evidence, figures, and PDF**

  Run the hash-bound builders from Tasks 4 and 5, then validate that every table and plotted point recomputes from ledgers.

- [ ] **Step 5: Dispatch five independent Sol Max reviewers**

  Review lanes are: plasma-physics validity; ML/statistical validity; causal leakage/reproducibility; code/config/evidence consistency; reader comprehension/LaTeX/figure quality.  Each reviewer receives paths to the spec, effective configs, evidence report, figure manifest, Markdown, PDF, and code-review package.  Findings include severity and exact evidence.

- [ ] **Step 6: Fix load-bearing findings and run final verification**

  Use one fix wave, one scoped re-review, then run the full test suite with `FUSION_DATA_ROOT`, package validator, PDF parse/render checks, git diff/status review, and frozen-output hash audit.  Do not mark the research result admitted merely because the package is reproducible.
