# Projected recursive ridge core review — 2026-11-04

## Scope

`experiments/risk_aligned_online/direct_fast_memory_20261104/rls.py` is a
functional, no-forgetting recursive ridge primitive for the existing direct
diagnostic memory shapes. It is classical RLS, not a novelty claim. It does
not read signals or outcomes, schedule observations, fit an injection, run a
stream, or establish an alarm benefit. The module reuses validation helpers
from the existing frozen `memory.py`; the two files must both be included in
any integration code manifest.

`initial_covariance(batch, reference)` creates a distinct identity matrix for
each batch row and each of 168 outputs: shape `(B,168,70,70)`, with the
reference tensor's floating dtype and device. Thus the initial inverse
precision is `P0=I` and the prior precision is also `I`. Empty batches are
supported. The caller must initialize a fresh `P0` and a fresh copy of common
`W0` for each shot and feedback arm.

## Exact step and loss

For batch row `b`, output `j`, key column `k`, inverse precision `P_j`,
observed mask `m_j`, target `y_j`, and row weight `w_j`, the step computes

```text
u_j       = P_j k
d_j       = 1 + k^T u_j
g_j       = m_j u_j / d_j
e_j       = m_j (y_j - w_j k)
w_trial_j = w_j + e_j g_j^T
P_trial_j = P_j - g_j u_j^T
```

For observed outputs, the returned inverse precision is
`(P_trial_j + P_trial_j^T)/2`; for unobserved outputs it is exactly the input
`P_j`. No forgetting factor is applied. All denominators, including those of
unobserved outputs, must be finite and strictly positive. All floating inputs,
including masked targets and the explicit origin, must be finite, share a
dtype, and be on the same device. Masks must be matching boolean tensors.

The returned loss is measured **before** the update:

```text
loss_b = (1/2) * sum_j[m_j (y_j - w_j k)^2] / max(1, sum_j m_j).
```

It is zero when the row has no observations. The mandatory `origin` may have
shape `(168,70)` or `(B,168,70)`; there is no implicit zero-origin fallback.
For a row with at least one observed output, the whole matrix `W_trial` is
projected to the Frobenius ball around `origin` with the requested positive
radius (default `2`). Global projection can also move an individual output
whose mask is false, although that output's `P` remains exactly unchanged.
An entirely unsupported row keeps its input `W` exactly, even if a caller has
supplied it outside the ball. Inputs are not modified in place.

## What the inverse precision does and does not imply

Given the initial conditions `W=W0`, `P=I`, and no active projection, the
per-output state after observations with masks `m_s` matches the ridge
solution centered on `W0`:

```text
P_j = (I + sum_s m_s k_s k_s^T)^-1
w_j^T = P_j (w0_j^T + sum_s m_s k_s y_sj).
```

The closed-form equivalence was tested in double precision, including
different observation sets for different outputs. Repeated keys reduce the
corresponding gain in the synthetic tests. This fact is not a guarantee of
improved noisy-data forecasts or downstream risk decisions.

Once Frobenius projection is active, `P` continues to track inverse feature
precision, but projected `W` is generally **not** the unconstrained ridge or
Bayesian posterior mean. The tests explicitly demonstrate that distinction.
`P` is an algorithmic inverse precision here; it is not a calibrated
uncertainty estimate. The caller, not this primitive, must ensure the initial
state is the intended `W0, P0` pair and that each update's target has arrived.

Symmetrization addresses ordinary antisymmetric roundoff. The implementation
does not run an eigenvalue decomposition, Cholesky validation, or spectral
clamp on every update; a positive denominator alone does not certify an
arbitrary caller-supplied `P` is positive definite. Positive definiteness is
an invariant of exact RLS from a positive-definite prior, subject to numerical
roundoff. Synthetic tests check symmetry, finiteness, and Cholesky on selected
matrices, but do not establish stability for every dtype or real trajectory.

## Verification and integration limits

- TDD red: the new test file initially failed to collect because `rls.py`
  did not exist. After implementation, the RLS tests passed.
- Combined CPU-only verification, with CUDA hidden, bytecode/cache writes
  disabled, and one CPU thread: **91 passed** across the RLS and existing
  direct-memory core test files. The new tests include an integration of real
  synthetic `dynamic_key`/`direct_target` with RLS and existing `innovation`.
- Tests use CPU tensors and metadata-only `meta` device mismatch cases. No GPU,
  raw shot data, network, full stream, or outcome data were used. CUDA
  execution remains unverified and must be checked by the integration owner.
- A float32 inverse-precision state alone occupies about 3.14 MiB per batch row
  and feedback arm; intermediate tensors require additional memory. CUDA
  runtime, peak memory, and end-to-end latency were not measured here.
- The caller must preserve predict-before-reveal, shot-local reset, and the
  fixed first-20 cost. The supervised connection should be fitted from the
  **actual RLS** first-20 trajectories, not reused from the NLMS experiment.
  Compare the real arm with the prespecified controls and report diagnostic
  forecasting separately from timely-risk outcomes.
