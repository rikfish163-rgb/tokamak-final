# Dynamic-source J-TEXT/EAST full-mainline implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the complete J-TEXT dynamic-label source to EAST static/online chain and add a causally sealed J-TEXT TEST220 online-active analogue with matched controls, evidence, publication figures, and a Chinese LaTeX report.

**Architecture:** A hash-bound selected-mainline contract drives a fail-closed EAST branch and an isolated J-TEXT online branch from the same dynamic source checkpoint.  The J-TEXT runner reuses causal query/reveal and two-timescale control concepts but has feature-only J-TEXT I/O, a rebuilt17 residual, device-specific metrics, and no EAST-private raw8 dependency.  Evidence builders consume immutable ledgers; figures and report never infer per-shot facts from aggregate metrics.

**Tech Stack:** Python 3.11, PyTorch 2.11/CUDA 12.8, h5py, NumPy, SciPy, scikit-learn, pytest, Matplotlib, Pandoc/LaTeX.

**Spec:** `docs/superpowers/specs/2026-08-24-dynamic-source-full-mainline.md`

## Global Constraints

- Use only `rl_env`; GPU experiments run on the local RTX 5070.
- Selected source label policy is exactly `advanced_floor`: J-TEXT `max(25 ms, AdvancedTime_ms)`, missing to 25 ms.  Its role is `post_adjudication_user_selected_development_candidate`; historical admission remains failed.
- EAST target and post-reveal policy is explicitly `fixed125`; EAST AdvancedTime is not read by the selected arm.
- HDF5 `meta/IsDisrupt` and `meta/DownTime` are the only outcome authority.
- J-TEXT TRAIN1353/VAL160/TEST220, EAST support110, and balanced360 roles and hashes are immutable.
- Source checkpoint paths are explicit; no archived fixed-source fallback and no bare encoder/head path are allowed in the selected runner.
- Static predictions are completely sealed before any test outcome read; online order is `predict -> seal -> query -> reveal -> update`.
- J-TEXT TEST220 online-updated weights never enter the formal EAST parent.
- J-TEXT operational TPR uses `lead > 10 ms`; EAST operational TPR uses `lead >= 30 ms`.
- Current TEST220 and balanced360 results remain diagnostic/test-driven evidence.
- Every production edit follows RED-GREEN-REFACTOR and every task receives a separate spec/quality review.
- Every load-bearing causal test records a behavior RED after the API is importable, a named mutation RED, and GREEN with the same test-file SHA.  A missing-module collection failure or a self-reported summary flag is not sufficient evidence.
- All subagents for this goal use `gpt-5.6-sol` with `max` reasoning; Luna is prohibited.

---

### Task 1: Fail-closed selected-mainline contract and EAST lineage

**Files:**
- Create: `experiments/transfer_methods_2026/selected_mainline_contract.py`
- Create: `experiments/transfer_methods_2026/run_selected_full_mainline.py`
- Create: `experiments/transfer_methods_2026/paper_grade_package/configs/primary_chain_dynamic_source_v1.json`
- Create: `config/splits/jtext_test220_label_free_roster.jsonl`
- Create: `config/splits/east_support110_label_free_roster.jsonl`
- Create: `config/splits/east_balanced360_label_free_roster.jsonl`
- Modify: `experiments/transfer_methods_2026/grid_search.py`
- Modify: `experiments/transfer_methods_2026/run_online_active_a1.py`
- Modify: `experiments/transfer_methods_2026/run_dynamic_hazard_adapter.py`
- Modify: `experiments/transfer_methods_2026/run_two_timescale_online.py`
- Modify: `experiments/transfer_methods_2026/run_unified_online_active.py`
- Modify: `experiments/transfer_methods_2026/shot_alarm.py`
- Modify: `config/mainline_code_manifest.json`
- Test: `tests/experiments/test_selected_mainline_contract.py`
- Test: `tests/experiments/test_selected_mainline_runner.py`

**Interfaces:**
- Produces: `SelectedMainlineContract`, `load_selected_source_checkpoint(path, seed)`, `validate_selected_parent(payload, contract)`, and parent schema `dynamic-source-east-fixed-parent-v1`.
- Produces: CLI `run_selected_full_mainline.py --seed N --data-root PATH --output PATH --stage {source-train,jtext-static,east-parent,east-static,east-online,all}`.
- Consumes: existing source trainer, support-only A1, shared/private parent components, and unified online runner.

- [ ] **Step 1: Write failing contract tests**

  Add literal-fixture tests proving that a source with `training_label.policy != "advanced_floor"`, missing source path, mismatched SHA, legacy parent schema, or `source_checkpoint=None` is rejected.  Add a test that a valid payload with explicit `source_dynamic/east_fixed` metadata is accepted.  Require the post-adjudication selection role, historical failure status, non-confirmatory status, corrected-ablation digest, selection metric, and historical failure reason.  Add a cross-module test proving that parent writer, selected validator, and unified loader compute the same canonical digest for the same real module.  Flip one bit in a real static/online tensor while retaining the embedded digest and assert selected startup fails before the first online ledger row; parameterize equivalent scaler, threshold, and roster drift.

- [ ] **Step 2: Verify RED**

  Run:

  ```bash
  FUSION_DATA_ROOT=/media/hetaisheng/044A81D94A81C83E/Fuison/data \
    conda run -n rl_env python -m pytest -q \
    tests/experiments/test_selected_mainline_contract.py
  ```

  Expected: collection or assertion failure because the selected contract does not exist.

- [ ] **Step 3: Implement the minimal contract module**

  Implement immutable dataclasses with exact policy strings and SHA fields.  The selected source loader delegates to the existing checkpoint loader, then verifies policy, seed, split, scaler, label-map provenance, and `test_shards_read == 0`.  Do not change legacy replay defaults globally; the selected entry point passes `allow_fallback=False` and rejects a missing checkpoint.  The parent writer, validator, and unified loader share one canonical state byte codec and recompute every embedded state digest.

  It must also require `input_contract.schema == "jtext-rebuilt17-prefix-causal-v2"` and validate the registered zero-padding, valid-length patch mask, prefix-causal availability, raw/feature lists, and preprocessing SHA.  Old dynamic checkpoints without this object are rejected.

- [ ] **Step 4: Write and verify failing no-test-read tests**

  Instrument every relevant boundary—not only one reader—including HDF5 `/meta` dataset access, `read_hdf5_outcome`, rebuilt outcome loaders, AdvancedTime access, and label-bearing roster parsing.  Assert that source fitting, EAST support adaptation, shared adapter fitting, and private parent fitting complete without opening any test outcome.  Assert the first TEST220 outcome read occurs only after a closed/fsynced 220-row seal with per-shot trace hashes, and the first balanced360 outcome read only after an equivalent 360-row seal.  A pre-seal roster may contain ordered shot IDs only.

- [ ] **Step 5: Fix EAST selected path**

  Change the selected adapter path from `shot_alarm.train()` to a typed support-only entry point.  Pass `target_label_policy="fixed125"` and the dynamic source checkpoint explicitly.  Derive target threshold from support110 and emit a receipt binding the actual recomputed support roster, support predictions/candidates, selection rule, target TPR, selected finite `0<t<1`, and support data/outcome manifests.  Bind source/target scaler, physics-scale, reconstruction-constant, padding/input-contract, encoder/head/shared/private hashes.  Emit a new parent payload rather than relabeling a legacy payload.  Extend the unified validator to accept the new schema only after full contract validation; selected mode never reads A1 balanced360 artifacts or a legacy sidecar.

- [ ] **Step 6: Implement orchestration and stage-resume receipts**

  The runner executes `source-train -> jtext-static -> east-parent -> east-static -> east-online` and writes `run_contract.json`, `stage_receipts.jsonl`, effective command/config, source/target SHA fields, and claim status.  `jtext-static` and `east-static` atomically seal prediction rows and every per-shot risk trace before outcome join.  A completed stage may resume only after recomputing current code closure, config, checkpoint, parent, split/label map, support/test rosters, threshold/scalers, preprocessing closure, each HDF5 manifest/content binding, and trace seal hashes.  Any one-byte drift raises a dedicated mismatch before rewriting or appending any prior artifact.  Missing `SampleRate`/`StartTime` fails closed unless an explicit invocation receipt is part of the contract.

- [ ] **Step 7: Verify GREEN and regressions**

  Run the two new files plus existing source, AdvancedTime, A1, adapter, unified, and prospective-seal tests.  Execute C-MUT-01 outcome-before-seal, C-MUT-03 actual-parent bit flip, and C-MUT-04 real two-shot resume drift gates; mock-only orchestration is not sufficient.  Confirm all pass with the external data root, the portable config contains no absolute `/media` path, and the selected contract is included in the reproducibility code-manifest closure.

- [ ] **Step 8: Commit**

  ```bash
  git add experiments/transfer_methods_2026/selected_mainline_contract.py \
    experiments/transfer_methods_2026/run_selected_full_mainline.py \
    experiments/transfer_methods_2026/paper_grade_package/configs/primary_chain_dynamic_source_v1.json \
    config/splits/jtext_test220_label_free_roster.jsonl \
    config/splits/east_support110_label_free_roster.jsonl \
    config/splits/east_balanced360_label_free_roster.jsonl \
    experiments/transfer_methods_2026/grid_search.py \
    experiments/transfer_methods_2026/run_online_active_a1.py \
    experiments/transfer_methods_2026/run_dynamic_hazard_adapter.py \
    experiments/transfer_methods_2026/run_two_timescale_online.py \
    experiments/transfer_methods_2026/run_unified_online_active.py \
    experiments/transfer_methods_2026/shot_alarm.py \
    config/mainline_code_manifest.json \
    tests/experiments/test_selected_mainline_contract.py \
    tests/experiments/test_selected_mainline_runner.py
  git commit -m "feat: bind dynamic source through EAST mainline"
  ```

### Task 2: J-TEXT signal-only I/O, ordered roster, and lazy outcome oracle

**Files:**
- Create: `experiments/transfer_methods_2026/jtext_online_io.py`
- Create: `config/splits/jtext_test220_ordered_v1.json`
- Modify: `experiments/transfer_methods_2026/rebuilt_harness.py`
- Modify: `experiments/transfer_methods_2026/train_jtext_source_transformer.py`
- Test: `tests/experiments/test_jtext_online_io.py`
- Test: `tests/experiments/test_jtext_prefix_causal_source.py`

**Interfaces:**
- Produces: `JTextSignalShot`, `load_jtext_signal_only(data_root, shot_id)`, `JTextLazyOutcomeOracle`, `load_ordered_test220(root)`, and `build_postseal_jtext_target(...)`.
- Consumes: rebuilt17 feature construction, source checkpoint scaler/padding metadata, HDF5 files, split lists, and AdvancedTime mapping.

- [ ] **Step 1: Write failing feature-isolation tests**

  Create temporary complete HDF5 fixtures.  Assert the signal-only object contains samples, timebase, availability, and causal metadata but no outcome-bearing key.  Mutating `IsDisrupt`, `DownTime`, or AdvancedTime before reveal must not alter signal features, risk input, or query input.

  Add a suffix-mutation test: keep a prefix fixed, change only later Mirnov and mode-number samples, and assert the prefix availability, rebuilt17 values, causal window, valid length, embedding, and risk are bitwise unchanged.  Assert the old whole-shot checkpoint is rejected.

  This is C-MUT-00: the same test must kill whole-shot availability, repeat-first padding, ignored `valid_lengths`, and a hand-filled arbitrary preprocessing SHA.  Add C-MUT-07 proving that HDF5 outcome remains authoritative: a safe shot with an AdvancedTime entry stays safe, a disruptive shot with missing AdvancedTime gets the 25 ms floor, and AdvancedTime never changes outcome or terminal time.

- [ ] **Step 2: Verify RED**

  Run `conda run -n rl_env python -m pytest -q tests/experiments/test_jtext_online_io.py` and observe failure because the module is absent.

- [ ] **Step 3: Implement prefix-causal source preprocessing and retrainable checkpoint contract**

  Replace whole-shot all-zero availability with finite prefix-causal seen-nonzero state, preserving unavailable values as zero after scaling.  Serialize the exact `jtext-rebuilt17-prefix-causal-v2` input contract and preprocessing SHA in source checkpoints.  Reuse rebuilt17 transformations but separate signal reads from outcome reads.  Use zero left padding plus `valid_lengths`; do not call the EAST `grid_search.causal_window` implementation for J-TEXT.

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

  Assert the controller rejects outcome-bearing intra fields, reports every loss as `enabled/weight/raw/weighted`, permits `replay=None` for the first selected shot, restores model and optimizer on non-finite loss/gradient/parameter/optimizer state, and writes resume checkpoints only after shot finalization.

- [ ] **Step 2: Verify RED, then implement the minimal residual/controller**

  Use rebuilt17 causal summaries and source risk as inputs.  Keep a fast copy for intra-shot updates and a persistent copy for post-reveal updates.  Snapshot model and optimizer before inter update and restore both on non-finite loss or guard rejection.

- [ ] **Step 3: Write failing causal event tests**

  Assert every selected shot ledger is ordered `predict < seal < query < reveal < update`, unselected shots have no outcome read and no persistent update, intra updates use `source_endpoint < observed_endpoint`, current predictions are immutable, and future-suffix mutation leaves prior receipts unchanged.

- [ ] **Step 4: Implement the runner and enabled losses**

  Build static seal first, reset the exact parent, then replay the ordered stream.  Record every enabled loss coefficient and term.  Disabled or zero-weight terms emit `enabled=false`.  Keep EAST raw8 and EAST thresholds unreachable from this code path.

- [ ] **Step 5: Write failing query-value and budget/comparator tests**

  Assert a dedicated J-TEXT value object computes exactly `V=wU+(1-w)C+eta*N`, has `N=0` before any prior reveal, reads novelty only from previously revealed phenotype memory, and never changes frozen `w`, `eta`, or scales on TEST220.  For the 20% primary budget, assert 20-shot blocks, four active and four random queries per block, 44 total queries, identical sampled-window and optimizer-attempt caps, label-agnostic random manifests, and front-load evaluation excludes its training shots.

- [ ] **Step 6: Implement all arms and abnormal-run gate**

  Implement primary and 10/20/30% sensitivity configurations.  Stop an arm with a machine-readable audit receipt when full-arm TPR is below 0.60 or FPR exceeds 0.40; do not tune alignment or loss weights after that trigger.

- [ ] **Step 7: Verify GREEN, synthetic integration, and commit**

  Run the new tests plus online audit, two-timescale adapter, unified value, and prospective seal suites.  Run a real minimal two-shot CPU integration that exercises `static -> full_active -> resume` and verifies byte-identical replay.  Execute C-MUT-02 by changing only the queried current shot's post-seal outcome/DownTime/AdvancedTime: every current-shot endpoint/shot/query artifact remains byte-identical while a later update/shot changes non-vacuously.  Execute C-MUT-05 on the realized 220-shot ledgers: active/random have exactly 4 queries per 20-shot block, 44 total, and matched optimizer attempts, current/replay/guard windows, replay size, and FLOP inputs.  Execute C-MUT-06 after Adam has optimizer state and prove rejected updates restore model and optimizer before an identical next update.  Commit with message `feat: add J-TEXT online active control matrix`.

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

  Compute per-seed Wilson intervals, shot-paired binary deltas after three hash-equality checks, descriptive chronological blocks, distinct raw and operational lead distributions, common-operational-TP paired timing deltas, very-early alarm rates with explicit denominators, query/update counts, and explicit post-adjudication selection/claim status.  Never multiply seed count by shot count for uncertainty.  Development artifacts emit no confirmatory p-values; implement and fixture-test Holm only for a future registered family.  Recompute realized query/block counts, optimizer attempts, current/replay/guard windows, replay size, FLOP estimator inputs, and reported FLOPs from individual receipts; any active/random mismatch suppresses the selector estimand rather than merely adding a limitation note.

- [ ] **Step 2a: Enforce new-control provenance**

  Reject historical equal-label/random/equal-compute evidence as completion for a new arm.  Freeze `suffix_start_index` before any TEST outcome access; require front-load training IDs to lie in that prefix and require all compared arms to share one suffix evaluation-mask SHA.  Name any future-label oracle control explicitly and keep it outside the primary causal contrast.

- [ ] **Step 3: Write failing render-contract tests**

  Assert every figure input hash is in the manifest; each output exists in PDF/SVG/PNG; PNG metadata is at least 600 dpi; minimum configured font is at least 6 pt; every quantitative panel identifies statistic, uncertainty, cohort, physical-shot count, model-seed count, claim status, artifact schema/path/hash, row selector, parent/label-policy hash, and generator/config/caption/output hashes.  Assert generators cannot fall back to legacy/external runs and single-shot labels are read from post-seal joins rather than hard-coded literals.

- [ ] **Step 4: Implement F1-F5**

  Generate: causal/data-role pipeline; AdvancedTime exposure and 2x2 label ablation; J-TEXT/EAST static-online paired metrics with blocks; query/update trace and enrichment; intra/inter/selector/front-load/equal-compute falsification controls.  Use real artifacts only for F2-F5, Okabe-Ito-safe colors, markers, line styles, and grayscale-distinguishable encodings.  F1 may be a clearly marked non-data concept layer, but all text, arrows, event ordering, counts, and policies are deterministic vector content.  Draw endpoint seal before intra-shot update and distinguish it from shot seal; do not connect J-TEXT and EAST metrics into a visual performance trajectory.

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

  Assert the report source covers every defined symbol, distinguishes shot outcome, temporal training target, raw alarm, operational TP, and TARDY, names enabled versus zero-weight/shadow losses, explains query budget and update order, embeds all required tables/figures, cites real artifact hashes, and refuses missing or stale inputs.  Add a machine-readable formula registry mapping `formula_id -> equation -> exact function/class -> effective config -> receipt field -> artifact/hash -> enabled state`, including prefix-causal preprocessing and patch-mask semantics.  Tests reject a query-uncertainty formula, teacher state, auxiliary head, or loss term that is absent/disabled in the effective implementation.

- [ ] **Step 2: Verify RED, then implement report build and verifier**

  The builder converts the Markdown source into escaped LaTeX sections and compiles a standard PDF.  Markdown/YAML front matter is the only semantic source for title, abstract, version, claim status, and prose; the TeX template contains layout only.  The verifier checks report/figure/evidence/config/code/builder hashes, PDF page count and parseability, PDF freshness and digest closure, and the absence of absolute data paths from portable manifests.

- [ ] **Step 3: Write the report and deployment README**

  Freeze the old fixed125/paper-v3c report as `LEGACY_PAPER_V3C_ARCHIVE`; create a separate dynamic report.  Structure it as one reader-first path: evidence/adjudication status; data roles and label ontology; symbol glossary before the first dense formula section; prefix-causal preprocessing and source model; isolated target/J-TEXT-online branches and parent lineage; alarm state machine; exact enabled intra loss; exact query value; inter loss/replay/guard/rollback; controls and estimands; statistics; per-seed results; failure/abnormal-run audit; limitations; reproducibility; formula registry; reviewer questions and responses.  Avoid a second parallel retelling of the methods.  README presents an explicit status matrix for legacy archive, selected development candidate, and unavailable future confirmation.  Commands start from a clone plus external `data/`, create/use `rl_env`, run audit/smoke/full/report, and explain expected artifacts.

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
