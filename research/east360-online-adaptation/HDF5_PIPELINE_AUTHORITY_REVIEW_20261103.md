# HDF5 shot-outcome authority: bounded pipeline code audit

Integration receipt, later on 2026-11-03: the main process subsequently changed
fresh event-cache extraction and the TRAIN/VAL context-cache loader to read HDF
outcomes, including shuffled-control strata. A fresh 410-shot metadata recheck
found zero changed cache outcomes or shuffle targets. The review below preserves
the pre-integration findings and original line references; it is not a claim
those two live code paths remain unfixed. The legacy operator sensitivity and
historical parent-provenance limits remain scoped caveats. Final integration
evidence is in `probe_outputs/event_context_hdf_authority_20261103_v2/` and the
MAIN report `docs/research/source_event_rebuild_20261023/HDF5结局统一与全盘复核_20261103.md`.

Audit date: 2026-11-03. Paths below are relative to the worktree root
`/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028`, unless
explicitly prefixed `MAIN`, which is
`/media/hetaisheng/044A81D94A81C83E/Fuison`.

## Bounded verdict

**Not all inspected executable analyses obey an HDF5-only outcome rule.** The
named current source trainer, cold-start producer, and functional/parameter
stream scoring paths derive their shot outcomes and disruption times from
`meta/IsDisrupt` and `meta/DownTime`. I found no shot-list or event-CSV outcome
being used as the primary disruption BCE target or primary streaming truth in
those particular paths. However:

1. `east_context_operator_20261031/audit.py:155-167` explicitly produces an
   **alternative false-positive/safe-count evaluation defined by shot-list
   membership**, rather than HDF5 outcome. This is a separate sensitivity
   output, not the primary HDF-based metric, but it is a confirmed non-HDF
   outcome metric and conflicts with the requested universal authority rule.
2. The event-family feature cache stores CSV `disruption_label` as a shot
   outcome, and active conditional-context training uses it to stratify the
   shuffled-family control. This is outcome-based **control stratification**,
   not the main disruption BCE target. It remains an authority violation if
   all outcome-dependent stratification must use HDF5. The pinned cache and
   derived heads require separate provenance-aware treatment; changing only
   the live CSV parser will not change an already-materialized cache.
3. Some executable historical preparation paths use shot-list classes for
   candidate selection or agreement gates. Those uses are distinguishable from
   assigning their supervised window labels, but can alter the training cohort
   or reject HDF-authoritative records.
4. The original qualified source checkpoint's exact training-time data reader
   is not established by the currently available provenance examined here.
   Current source cannot be retroactively assumed to describe that checkpoint.

This is a code-path audit, not an all-repository finding, disk-wide metadata
inventory, numerical outcome reconciliation, or claim that every historical
result must change. No GPU experiment was run and no source code was edited by
this audit.

## Primary path: what reads outcome versus what selects a cohort

- `experiments/risk_aligned_online/io23.py:99-116` reads `meta/IsDisrupt` and
  `meta/DownTime` directly from the per-shot HDF5 and currently rejects a
  non-scalar/non-binary flag and an invalid disruptive timestamp. A safe shot's
  returned disruption time is `None`. At inspection the file SHA-256 was
  `5a26ba79e9fbc7bd9480ef03fe914f83878d108b11257a9e88b3ea4f6f8020e8`.
  This file changed concurrently during the review: the current stricter reader
  must not be confused with older archived readers described below.
- `simple_data.py:32-38` passes that HDF flag and time to
  `io23.labels_at` (`io23.py:119-125`). `simple_train.py:67-79,83-99,167-173`
  samples the resulting endpoint `y` for BCE; its validation/test scoring
  carries `is_disrupt` and `down` from those same shot records
  (`simple_train.py:43-64,102-116,148-149,203-205`). The current
  `joint_meta.py:99-106,121-135` uses `simple_train.load_shots` for TRAIN/VAL;
  `fast_meta.py:77-99,125-127` forms episode pools from those `y` values.
- J-TEXT `AdvancedTime` still comes from a shot-list JSON
  (`io23.py:22,152-187`) and sets the prediction horizon
  `max(25 ms, AdvancedTime)` in `simple_data.py:35-37`. It is **not** the
  shot disruption flag or terminal disruption time. Thus the window target
  depends on an external horizon even though shot outcome/time are HDF-sourced.
  EAST uses a fixed 125 ms target horizon in the same function.
- The separate older `train_source.py:13,176-184,199-201` imports
  `data.py:6`, which imports `experiments.fast_online_idea.pipeline`. That
  module is absent from this worktree, so this legacy entry point is not
  directly runnable from this worktree alone. The corresponding MAIN file
  `experiments/fast_online_idea/pipeline.py:193-208` does read HDF outcomes,
  but that is not evidence of what an older checkpoint used or of which module
  an externally modified import path would select. This is distinct from the
  standalone `simple_train` path above.
- `io23.py:128-149,190-199` obtains split/cohort identities from manifests and
  concatenates the EAST manifest's class-named lists. That is cohort membership,
  not a per-shot truth assignment in this reader. `fixed_test_protocol.py:23-44`
  also requires 180+180 manifest members; `:58-64` scores supplied rows but
  rejects an HDF-revealed class count that differs from the fixed expected
  balance. It does not override a row's outcome. This fixed-count gate may need
  review if the authorized HDF reconciliation changes the cohort composition.

## Cold-20, warm-start, and current streams

- `dual_history_ttt_20261101/run.py:103-110` loads the hash-locked source via
  `redesign_20261004/prepare_prior.py:22-23,54-73`, constructs a context
  adapter, and restores `probe_outputs/stratified_cold20_context360_v1/models/cold.pt`.
  Current verified SHA-256 values are source
  `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`, cold
  `a06377a218a0f06def1eb3652cc091b5c514653681bffcf5dbfda7906f9db344`, and
  reference `completed.json`
  `ef6e2fccd56f7a8b7bc0f93c44315cf77028ee6e122d74758e5f0678baa937ef`.
  `functional_stream.py:116-134` additionally checks the cold hash and parent
  threshold against its previous reference and checks the input-map hash.
  `parameter_controls.py:107-123` uses that same loader.
- `LifecycleAdapter.restore` only restores `slow` and `fast_output`
  (`lifecycle_stream_20261025/model.py:62-68`); `load_model` alone does not
  reload the full original context-memory checkpoint. The named FFN/dual-history
  route uses the frozen source and `slow` in `lora_stream_20261102/core.py:31-75`
  and `dual_history_ttt_20261101/model.py:142-162`; do not describe this loader
  alone as restoring every part of the original context-memory system.
- Current `lifecycle_stream_20261025/run.py:173-214` seals each first-20
  prediction, then calls `simple_data.attach_outcome`, constructs examples and
  fits/saves the cold state. Its `learning.py:27-37,40-64` uses HDF-derived
  target `y` for both EAST examples and source replay. Event-family intervals
  select replay windows; the CSV shot-outcome metadata is discarded there.
  The saved cold-run `code_snapshot.zip` contains the same HDF-reading path:
  archived `io23.py:99-116` and `simple_data.py:32-38`; the archived
  `io23.py` SHA-256 `714b0339e3119cb39f02234ea9e2df7643a90cc303c127d70b110fa74170f72c`
  matches its protocol. This is source-code evidence for that archived step,
  not independent verification of every historical input value.
- `warmstart_alignment_20261022/run.py:110-114,118-147` builds the input map
  from source signals and the ended target prefix before attaching outcome.
  `alignment.py:37-54,128-142` enforces signal-only records and exact prefix
  IDs. The inspected saved map hash is
  `43c37ad981261e059ac8bf8d11d89324bad375dd7405decc194c51e53614aff2`;
  its `calibration_ids` equal the first 20 in the relevant recorded order.
  This is signal alignment, not shot-outcome derivation.
- `east_decoder_calibration_20261102/functional_stream.py:298-356` and
  `parameter_controls.py:180-239` reuse sealed warmup/reference curves,
  predict on signal-only inputs, seal curves, then read `io23.read_outcome` for
  `classify`. The `truth` variable in their prediction loops is **matured
  signal forecast feedback**, not shot-outcome truth:
  `dual_history_ttt_20261101/stream.py:12-32`. Prior saved curves, cold state,
  decoder and basis still have their own historical provenance; a fresh HDF
  scoring pass does not retroactively retrain them.

## Confirmed non-HDF outcome uses and agreement gates

### Event CSV to active shuffled-control training

`event_prefix_probe.py:64-79` parses CSV `disruption_label` as
`metadata[shot]['outcome']`; `:172-205` writes that value into the feature
cache. `:235-244` uses it for the whole-event shuffled-family control's
TRAIN outcome-by-shot-quartile strata. The current conditional trainer loads
the pinned MAIN cache at `event_conditional_feedback_20261102/context_fit.py:23-28,115-143`
and uses its `outcome` at `:157-164`; paired context training calls that
function at `source_event_tasks_20261102/context_pair.py:41-54`. Thus the
shuffled head's family targets can depend on the CSV outcome. This is not the
true family target, nor is it the disruption risk label.

Actual cache-to-CSV check, without opening any HDF metadata: all 10,583 TRAIN
rows (327 shots) and all 2,598 VAL rows (83 shots) had cache `outcome` equal to
the CSV `disruption_label`; this establishes the stored source, **not** whether
any label agrees or disagrees with HDF. The live event parser SHA-256
`bdc81ab2bd307ab96754e6b6cb48774b33059885d66d38fca1acc0a57a358b94`
matches `MAIN/outputs/event_prefix_probe_20261021/protocol.json`'s code hash.
The pinned feature cache is `MAIN/outputs/event_prefix_probe_20261021/features.npz`.

Separately, `source_event_tasks_20261102/trial.py:125-143` discards the CSV
metadata returned by `load_events`, attaching only event families. Its risk BCE
uses archived episode `record['label']` at `:249-258,305-322`; the archive
producer `context_vector_20261030/source_run.py:125-146` attaches HDF outcome,
and the archived producer and HDF reader are present in
`probe_outputs/context_vector_source_v1/code_snapshot.zip`. Their archived
`io23.py` hash also matches that run's protocol. `data.py:42-63,66-117`
validates/reuses those archived records without re-reading HDF; `trial.py:125-137`
checks current whole-file HDF hashes against the archived inventory. This is a
provenance chain, not an independent per-record outcome recomputation in the
loader. Do not conflate event family labels with disruption outcomes.

### Other bounded shot-list uses

- `east_context_operator_20261031/audit.py:53-61,91-103` reconstructs its
  primary results from HDF. At `:155-167`, however, `alternative` counts
  alarms and the safe denominator using `Non-disruption_EAST_Test.json`.
  The `agree` subset also filters on shot-list/HDF agreement. Both are labeled
  sensitivity analyses; only the first is a direct non-HDF outcome-based FP
  metric. This older audit also demands code-snapshot parity at `:38-41`, so a
  changed current reader can block rerunning it before that section is reached.
- `preparation_factors_20261023/run.py:73-82` creates the `labels` candidate
  class map directly from shot lists and uses it to select the class-aware
  training cohort (`design.py:8-18`). It obtains actual selected-shot training
  labels with `simple_data.attach_outcome` and asserts selected membership
  agrees with HDF (`run.py:95-102`); selection is nevertheless shot-list-based.
- `labeled_alignment_20261023/run.py:39-61` reads HDF outcomes but excludes
  candidates when HDF and shot-list membership disagree or membership is
  absent/ambiguous. Its fit reattaches HDF-derived labels at `:64-79`.
  `learning.py:32-46` repeats the agreement gate. This is an exclusion/selection
  effect, not proof of shot-list-derived BCE truth.
- `state_capacity_20261031/prepare.py:31-59` reads primary flag/time directly
  from HDF but checks the roster against old shot lists and records disagreement;
  it does not replace the flag. `event_pairing_20261024/run.py:33-59` similarly
  checks CSV/HDF agreement before using HDF `s['y']` as risk truth. Such gates
  can stop or select work on disagreement; do not silently treat them as HDF
  outcomes or as harmless without considering the intended protocol.
- Both `state_capacity_20261031/prepare.py:43-47` and
  `east_context_operator_20261031/audit.py:56-61` independently coerce HDF
  `IsDisrupt` to `bool`, bypassing the current stricter `io23.read_outcome`
  validation. This is a conditional malformed-metadata risk, not evidence that
  any actual shot has an invalid flag.

## Historical source-parent limitation

`runs/gradient_aligned512_20260929/source.pt` has the locked source hash above.
Its neighboring `protocol.json` records upstream checkpoint SHA-256
`2291666ac48203d4e50468ca5fc242648620e8599c28d436309b6e3aaef4f0ae`
and hashes for several code files, but does not bind `io23.py`. Its recorded
`simple_train.py` hash matches the current file; recorded `simple_data.py`,
`joint_meta.py`, and `fast_direction.py` hashes do not match the current files.
The artifact identity therefore does not prove which historical outcome reader
or exact data values trained the original source. Locate and verify the
matching old reader/code and its input provenance before claiming the parent
was trained under the new authority rule. The later cold and episode snapshots
provide narrower evidence for their own downstream steps only. Archived readers
used `bool(first IsDisrupt value)` and lacked the current strict binary check;
this observation does not establish that any archived input was malformed.

## Verification and limits

- Targeted CPU pytest run, with `CUDA_VISIBLE_DEVICES=''`, bytecode disabled,
  and pytest cache disabled: **20 passed in 4.12 s**. Selected tests covered
  `tests/test_io23.py` import and label boundaries, `test_simple_data.py` causal
  windows, `test_simple_train.py`, selected signal-only warm-start and event
  tests, `tests/test_source_event_tasks_data_20261102.py` archive/CSV loading,
  and the outcome-input rejection tests in
  `tests/test_east_functional_protection_20261102.py` and
  `tests/test_east_parameter_controls_20261102.py`. The `test_io23`,
  `test_simple_data`, `test_simple_train`, warm-start, and event-probe tests
  are under `experiments/risk_aligned_online/tests/`; the source-event and two
  stream-control files are under the worktree's top-level `tests/`.
- In-memory HDF5 checks against the **current** reader returned `(False, None)`
  for flag 0 and `(True, 3.0)` for flag 1; flags 2, NaN, a byte string, and
  multiple values, plus a disruptive NaN timestamp, raised `ValueError`.
- No GPU, model training, full replay, whole-repository audit, or raw HDF5
  outcome inventory was performed. The main task owns any reconciliation,
  patch, artifact migration, retraining, and final integration decision.
