# Source update utility: bounded implementation review

[utility.py](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/utility.py) adds the offline actual-update objective. It does not change `meta.py`, the source classifier, data preparation, a training runner, or deployment risk scores. No model was fitted.

## API and behavior

`objective(base, head, origin_x, origin_mask, query_x, query_mask, truth25, target_mask, labels, *, initial_head_state, threshold)` returns `(loss, info)`. `threshold` is a required finite probability strictly between 0 and 1; labels and the independent head-initialization snapshot follow the frozen `meta.objective` contract.

The function reuses `meta.objective` for validation, the proxy gradient and detached signed-logit teacher; applies the native `.05` normalized step with radius `1` through `inner_state(..., create_graph=True)`; and compares query margin BCE after that step against a no-grad zero-delta baseline. BCE differences are calculated in float64. For each eligible row, the denominator is `.05 × sqrt(sum(detached_desired_gradient²) + 1e−12)`. Eligibility requires observed diagnostic targets and a **nonzero raw** teacher gradient, before epsilon. Fully unobserved and all-zero-teacher batches raise the existing explicit `no_target` / `no_teacher_direction` errors.

The optimized loss is `risk_utility + 0.20 × absolute_proxy + 1e−4 × anchor`. The `info` dictionary retains the original cosine `alignment` and other meta diagnostics, and adds `risk_utility`, `perrowrisk_delta`, `sensitivity` (the smoothed gradient norm), `normalization_scale` (step × sensitivity), `updated_state`, detached `baseline_logit`, and `risk_utility_rows`. The baseline, teacher and scale are detached; the updated state and risk utility retain the head's outer-gradient path, including when called inside `no_grad`.

## CPU verification

- Tests were written first and initially failed because `utility.py` did not exist. The final synthetic/checkpoint-only regression ran 18 [utility tests](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_source_update_utility_20261104.py), 22 meta tests and 2 existing scoring tests: **42 passed**. Ruff lint and format checks passed for both new Python files.
- Tests independently reconstruct the native inner state, float64 per-row BCE differences, normalization and full loss; verify nonzero finite head gradients, detached baseline/scale, preserved cosine diagnostics, mixed/all-missing and zero-sensitivity cases, bad thresholds/labels, no-grad callers and hook cleanup. The hash-locked source state and gradients are checked unchanged after each test.
- For identical `+0.2` logit moves and equal sensitivity, correctly confident versus incorrectly confident synthetic examples had absolute BCE-utility ratios `5.568783627782071e−6` and `6.801727659069387e−6`; this checks the intended confidence weighting, not a learned benefit.
- Fixed-direction finite difference of **risk utility** through the actual source path: float64 step `2e−5`, analytic `4.038222032170793`, numerical `4.038221847769973`, relative error `4.5663863571128565e−8`. On the same synthetic setup, float32 step `1e−3` gave `4.038251876831055` versus `3.991945200558747`, relative error `0.011467010400709878`. Double BCE reduces loss-arithmetic cancellation but cannot recover precision already lost in float32 model logits; this single check does not establish the cause of the discrepancy.
- Two synthetic zero-offset logits matched exactly between no-grad and grad-enabled source forwards with CPU MHA fastpath enabled and disabled. This is not a GPU equivalence guarantee. Frozen `meta.py` SHA-256 remained `c7804fe5765de3c6fbfedff63be406df8e7a77e55dabe94ce60112d81bb7f721`.

No raw shots, GPU, optimizer loop, saved head, or empirical source/EAST gain was evaluated. The main process remains responsible for data/label validity, matched schedules, training, and independent acceptance checks.
