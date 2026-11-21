# Source forecast fidelity sidecar: CPU review

[fidelity.py](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/fidelity.py) is ready for independent review and the main process's GPU run. It is bound to [SOURCE_FORECAST_FIDELITY_PLAN_20261104.md](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/SOURCE_FORECAST_FIDELITY_PLAN_20261104.md), not the earlier risk-VAL plan. **No actual fidelity result was generated.**

The CLI requires `--fit --design --output`. It verifies completion manifests, the qualified source, the fit-to-design binding and archived producer hashes; the fit protocol must explicitly declare `teaching_objective` as `cosine` or `risk_utility`. Older fits missing that field are deliberately rejected rather than silently classified. Output must be a new directory outside the input artifacts and raw data.

The fixed 946 archive-matched VAL coordinates and 145 used shots are checked before signal access. Prediction uses only `xq/aq`, zero fast offset, the frozen native head and three separate frozen heads, in native and no-MNM/CIII views. Records lacking a full `q+1…q+25` time horizon have `eligible=false`, the exact missing-point count, no prediction index and no prediction in the sealed NPZ. Complete-horizon rows with zero observed target scalars are separately marked unscorable, not assigned zero loss.

The forecast NPZ and hash seal are written and checked before the separate, VAL-only, hash-rechecked future-signal reader is invoked. Scoring uses the existing `targets` availability rule, including its reverse-symmetric support and exclusion of mode index 10; the reduced view also masks CIII/MNM. It applies source-coordinate `asinh` and float64 SmoothL1, then averages observed scalar errors within each window, windows within shot, and shots within overall/four strata. Each head is paired against the native head and a same-coordinate hold-last-value baseline. Results include per-coordinate rows, coverage, group summaries, code ZIP and completion hashes; before/after source and head state digests are saved and compared.

The reader may load whole ended shots before prefix slicing; determining complete-horizon eligibility also uses shot length. Thus the model input is prefix-only, but this is not literal streaming I/O or an independently prospective selection. These are zero-fast-state forecasts, not adaptive-forecast validation, TPR/FPR evidence, or a change to the existing risk gate.

## Verification and handoff

- Tests were written before implementation. The final bounded CPU run produced **15 passed**: 13 fidelity tests and 2 existing scoring tests; Ruff lint passed. Tests use synthetic arrays/files and the hash-locked source checkpoint, not raw shots or GPU.
- A synthetic 946-row/145-shot end-to-end run, with I/O and device access mocked, verifies seal-before-future-read, one complete versus 945 incomplete windows, output hashes, and frozen-state digest records. Separate tests cover native `targets` masks, held-value coordinates, per-shot macros, nonfinite forecasts, zero-observation masks, tampered seals, role/hash rejection and missing/unknown fit objectives. This does not replace real-file or GPU verification.
- Read-only provenance validation passed for the requested `risk_utility` fit: **122 archived producer hashes**, the fixed **946 records / 145 shots**, and the new plan binding. The historical undeclared-objective fit is not accepted by this CLI.
- Final implementation SHA-256: `c19db16b6cde67dcfc3e6301b81ab871ed83c57abfcb8df31f40ab46e785a654`. The main process should independently inspect this final version before running. The main process's subsequent `utility.py` fastpath correction and previously completed failed results were not modified or reinterpreted here.

No GPU, raw-shot access, fitting, shared training/evaluation changes, or empirical efficacy claim occurred in this sidecar.
