# Direct fast-memory core review — 2026-11-04

## Scope and coordinates

The core is a functional tensor-only component. It does not read data, fit a
forecaster, inject into a model, maintain a stream, or manage shot identities.
`level` and `future` are supplied in the existing source-normalized asinh
coordinates. No inverse transform, unit conversion, or derived physical units
are used. History has shape `(B, 128, 23)` and future `(B, 8, 23)`; their
availability tensors must be boolean and shape-matched. Observed values must
be finite, including observed channels excluded from the target. Unobserved
values may be nonfinite and cannot affect the outputs.

The API accepts tensors on CPU or CUDA, with all tensors in a call on the same
device and floating inputs sharing a dtype. Meta tensors in the tests are used
only to exercise device-mismatch rejection before their values are inspected.
The implementation does not move inputs between devices.

## Explicit definitions

Let `z[b,t,c]` be an input level and `a[b,t,c]` its availability. Define
`v = where(a, z, 0)`. The summary uses only the last 16 history positions:

- `last[b,c] = v[b,127,c]`, with no fallback to an earlier observation.
- `n_old` and `n_new` are available counts over positions `112:120` and
  `120:128`, respectively. `mean_old` and `mean_new` divide the corresponding
  masked sums by their respective counts. `delta = mean_new - mean_old` only
  where both counts are positive; otherwise `delta = 0`.
- `fraction = (n_old + n_new) / 16`.
- `key = concat(last, delta, fraction, 1)`, with its first 69 entries divided
  by `sqrt(69)` and its last entry equal to `1`.

Let `C = [0, 1, 3, ..., 9, 11, ..., 22]` be RAW23 indices in their original
order, excluding MNM (2) and mode_number_n (10). For each future time `h` and
channel `c` in `C`,

```text
valid[b,h,c]  = future_available[b,h,c] AND a[b,127,c]
target[b,h,c] = where(valid[b,h,c], future[b,h,c] - z[b,127,c], 0)
```

The `(8,21)` time/channel axes are flattened in time-major order to 168
outputs. The function only accepts the explicit eight-step future tensor; the
caller is responsible for ensuring every `future_available=True` sample has
actually arrived before using it for an update.

For batch row `b`, output `j`, and a key `k`, define `p_j = W_j @ k` and
`e_j = where(mask_j, p_j - target_j, 0)`. The returned pre-update row loss is

```text
loss_b = (1/2) * sum_j(e_j^2) / max(1, sum_j(mask_j)).
```

It is zero for an unsupported row. The normalized LMS update uses a
per-output rate, not the derivative of the displayed mean with respect to a
whole row:

```text
W_trial[b,j,:] = W[b,j,:] - lr * e_j * k / (1 + sum_i(k_i^2)).
D_b = W_trial[b] - origin[b]
W_new[b] = origin[b] + D_b * min(1, radius / ||D_b||_F)
```

The last line applies only when that row has at least one observed target;
otherwise `W_new[b]` is exactly `W[b]`, even if the caller supplied an
out-of-ball state. Consequently the bound is invariant for inactive rows only
when their input state was already in the ball. Projection of an active row
acts on its entire matrix, including outputs with a zero mask. Floating-point
comparisons of the radius should allow normal numerical tolerance. Finite
inputs whose derived results overflow are rejected rather than returned.

`origin=None` in `fast_step` denotes a zero origin. Otherwise the origin can
be shared `(168,70)` or batched `(B,168,70)`. `innovation` requires an
explicit origin and computes

```text
innovation[b] = (W[b] - origin[b]) @ key[b].
```

It takes no target or current label, and is exactly zero for equal initial
weight and origin. Each shot must start from its own caller-owned copy/clone of
the common `W0`, and every update must be used only after the relevant target
is revealed. This module cannot verify chronology, shot membership, locked
splits, or the source of labels; integration must enforce those boundaries.

## Verification and remaining limits

- TDD red: the new test module initially failed collection because
  `direct_fast_memory_20261104.memory` did not exist. A separate red check
  caught CPU-only device validation before it was generalized.
- CPU verification, with CUDA hidden, bytecode/cache writes disabled, and one
  CPU thread: `49 passed` in the requested test file. Coverage includes exact
  update/loss formulas, masks and nonfinite values, temporal and channel
  indexing, zero innovation/reset, actual spatial target differences,
  per-row independence, repeated projection, float32 and float64, and
  differentiation including the projected step.
- CUDA execution and end-to-end forecaster/injection/stream integration have
  not been run here. The caller must verify those separately, including
  predict-before-reveal and shot-local reset behavior.
