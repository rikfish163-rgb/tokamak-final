# Shadow-commit decision review — 2026-11-04

## Scope and contract

`experiments/risk_aligned_online/shadow_commit_20261104/decision.py` provides
one pure comparison function. It neither changes weights nor manages queues,
forecasts, timestamps, shots, or outcome labels. The caller supplies an actual
matured diagnostic target, its availability mask, and the candidate and active
forecasts to be compared.

All four tensors have shape `(B,168)` and share a device. The three value
tensors must have the same floating dtype; `mask` must be boolean. Values at
observed positions must be finite. Unobserved target and prediction values
may be NaN or infinity: they are replaced with zero before subtraction and
cannot influence the result. Nonfinite observed values, malformed inputs,
device/dtype mismatches, and unknown policies raise `ValueError`.

For each row `b`, let `n_b = sum_i mask[b,i]`. When `n_b > 0`, the returned
float64 errors are

```text
candidate_mse_b = sum_i mask[b,i] * (candidate[b,i] - target[b,i])^2 / n_b
active_mse_b    = sum_i mask[b,i] * (active[b,i]    - target[b,i])^2 / n_b
```

The implementation casts masked values to float64 before subtraction and
divides residuals by `sqrt(n_b)` before squaring. This computes the same mean
while avoiding premature overflow when a squared residual is not representable
but its mean is. A mathematically unrepresentable float64 MSE is rejected
rather than silently used for a decision. Ordinary floating-point rounding
still applies.

The returned tuple is `(accept, candidate_mse, active_mse, support)`, where
`accept` is boolean, errors are float64, and support is an integer tensor. All
outputs remain on the input device. Policy rules are exactly:

```text
guard: accept_b = (n_b > 0) AND (candidate_mse_b < active_mse_b)
delay: accept_b = (n_b > 0)
```

There is no gain margin or other cutoff. Guard rejects exact equality; delay
accepts equality or a worse candidate when support exists. When `n_b = 0`,
both policies reject and return zero errors only as logging placeholders: an
unsupported row is not a zero-error performance observation.

## Caller-owned safety boundary

The function cannot establish that either prediction was issued before its
target arrived, that both correspond to the same origin/horizon and active
snapshot, or that the target is actually mature. The caller must preserve
those invariants, prevent current-shot outcome labels from entering the
decision, and own the commit/reset/queue behavior. Supplying a target or
snapshot from the wrong time would not be detected by shape and finite-value
validation alone.

This is a comparison of auxiliary diagnostic MSE on one supplied target. A
guard acceptance is not a guarantee of improved later diagnostic forecasting
or of improved risk/alarm decisions; those require separate prequential
evaluation and appropriate controls.

## Verification

- TDD red: the test file first failed to collect because the new module did not
  exist. An additional numerical regression failed before the MSE computation
  was adjusted to avoid premature overflow.
- CPU-only combined verification with CUDA hidden, bytecode/cache writes
  disabled, and one CPU thread: **81 passed** across the decision tests and the
  existing direct-memory core tests. Cases cover exact policies, strict tiny
  improvements and ties, missing/nonfinite values, per-row independence,
  float64 arithmetic, zero support, gradients, validation, and compatibility
  with the existing direct-target output.
- Device validation was exercised with CPU and metadata-only `meta` mismatch
  cases. CUDA execution, real data, GPU timing, and end-to-end snapshot/queue
  integration were not tested here and remain with the integration owner.
