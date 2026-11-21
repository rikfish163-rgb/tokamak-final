# N=1 amplitude calibration: CPU implementation review

2026-11-04. The new [calibration module](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/n1_alignment_20261104/calibration.py)
and [tests](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_n1_alignment_calibration_20261104.py)
are frozen for the main process. This is a numerical input control, not
neural adaptation, physical-unit conversion, or evidence of alarm benefit.

## Contract

- `source_reference(train_ids)` requires a **list exactly matching** the
  canonical ordered 1,353 TRAIN IDs. It checks that each source HDF file
  exists, calls `io23.read_outcome('J-TEXT', sid)` for its authoritative HDF
  flag, and reads only `data/n=1 amplitude` for HDF-safe shots. It takes each
  safe shot's median over **all finite values**, retaining zeros, then the
  median across safe-shot medians. It returns that positive finite `value`,
  the source safe IDs, per-shot medians, full TRAIN IDs and feature name.
  Missing files/feature, absent finite values, no safe shots, or a nonpositive
  or nonfinite final reference fail explicitly. No source `ip` conversion is
  applied to amplitude, and no VAL/TEST or EAST signal is read by this API.
- `fit_gain(reference, past, truth, allowed_ids)` uses exactly 20 distinct
  integer IDs with exactly matching in-memory `past` and `truth` keys.
  `truth[sid]` is the already revealed `(bool, down)` HDF outcome pair;
  `past[sid]` contains `float32[N,23]` values and a matching boolean mask.
  At least two safe shots must each contain observed amplitude values. The
  target reference is the median of their per-shot medians, and the gain is
  `reference['value'] / target_reference`. Zeros are valid observations;
  missing or nonpositive/nonfinite target reference and invalid gain fail.
- All 20 input arrays, including those from disrupted shots, are checked for
  shape, dtype and finiteness **at observed positions**. Disrupted-shot
  values never enter the reference or gain. The function does not modify its
  inputs, perform I/O, refit a map, or update a model. In its result,
  `source_reference` is the scalar reference value; per-shot target medians
  are keyed by shot ID, and `physical_unit_conversion_claim` is `False`.

The caller must enforce the locked first-20 identities/order, obtain actual
EAST masks, and seal each shot before revealing its HDF truth. The helper
rejects IDs outside the supplied 20, but cannot itself prove that the supplied
20 are the locked prefix, that the masks are physically correct, or that
sealing occurred. The main runner owns those checks, map composition, baseline
curve comparisons, and all later evaluation.

## Verification and limits

- TDD initial RED: test collection failed with `ModuleNotFoundError` because
  the new module did not yet exist. A later targeted regression found that
  floating-point dictionary keys could alias integer IDs; the module now
  rejects non-integer keys and the regression passes.
- Target CPU run, with bytecode/cache disabled and CUDA hidden:
  **`29 passed in 1.46s`**. A combined run of these and the existing coordinate
  probe tests reported **`35 passed in 2.40s`**.
- Source tests use 1,353 synthetic TRAIN IDs and real **in-memory HDF**
  outcome datasets. A deliberately conflicting hypothetical shot-list class
  does not override `meta/IsDisrupt`. Tests also cover source safe-only reads,
  unequal record lengths/equal-shot medians, zeros, mask and dtype errors,
  nonfinite values, missing files, fewer than two target-safe shots,
  no-observation/nonpositive references, unexpected IDs, and no I/O in
  `fit_gain`. No fixture HDF file was written to disk.
- This sidecar **did not run the real 1,353-shot source scan**, read real EAST
  signals, run a full360 stream, use GPU, train, or edit a shared producer.
  The main runner must execute the actual full-TRAIN method and handle its
  explicit failure if the observed reference is not positive; the earlier
  source64 coordinate sample is not a substitute.
- Frozen helper SHA-256:
  `993894fc170d382f5b450947aab57dff5da68738f648bb1b00ed3aa0c12cfdde`.
  Frozen test SHA-256:
  `3c95ab7832bcd234d5d520d47a3a9a63784412338118f0ea2d31561a0a4cf9b2`.
  This sidecar did not create or modify `__init__.py` or the main-owned runner.

TDD informed the explicit cohort, outcome, numerical and no-I/O regression
checks; no result here establishes EAST performance.
