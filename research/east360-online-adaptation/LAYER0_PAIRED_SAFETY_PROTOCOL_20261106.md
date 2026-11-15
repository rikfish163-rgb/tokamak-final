# Matched safety in a learned within-shot direction (TRAIN-only)

Prior positive-only peak shaping recovered little and exposed more false alarms.
Test a new **first pre-attention LayerNorm affine** direction, not an alarm gate.
Use the locked J-TEXT parent/threshold and existing sealed TRAIN1353 native,
no-MNM, and no-MNM/CIII views. No VAL, TEST or EAST access in this experiment.

- At the same 32-point clocks `m=63,95,...`, use the fixed decoder's prediction
  made at `m-25` and the real observations that have since reached `m`. Compute
  group gradients of the half-difference between Huber lag-8 transition losses
  on the true and reversed observed samples (MIR, SXR, XUV). A reversal swaps
  the gradient's sign, not the mask. Normalize each nonzero group direction;
  require at least two groups. A small learned mixer uses only current prefix
  static risk/slope, elapsed time, and group gradient magnitudes. Its bounded
  coefficients can change signs. The action starts strictly after `m+8`, expires
  at `m+32`, and resets every shot. No labels, current shot length, or outcome
  are mixer features. All tested directions have the same per-release norm:
  cap `.35`, downweight as `.35/(1+abs(prefix_logit-tau))`.
- Whole physical shots (all masks) are split by the existing deterministic
  five folds. For each training fold complement, select one positive release
  per disruptive shot/view with at least one future target satisfying the
  original `max(25 ms, AdvancedTime)` window and strict lead `>10 ms`. Among
  eligible releases take the one closest to threshold at `m+8`. Pair it with a
  non-disruptive training shot with the same observed family pattern, prefix
  logit difference ≤1.0, and elapsed-time difference ≤0.5 s; distance uses these
  two values and prefix logit slope. A safe shot is used at most once per view
  per fit. Outcome is used for offline teaching strata, never during prediction.
- At most four highest native-logit frames in each eligible future block
  supply offline first-layer Jacobians. Fit the mixer on the linearized
  maximum subsequent margin per release, with the same logistic losses on
  danger and safe plus a squared hinge penalizing `shift_safe + .05 > shift_danger`.
  Both fits use the same positives, safety windows, weights, 160 fixed optimizer
  steps, and coefficients; the placebo cyclically deranges safe partners
  within the view. Exact model forwards are used for holdout replay, not the
  training approximation. Compare no-write, correctly paired, shuffled pairs,
  sign-reversed paired and the same paired logit shift added as a scalar.
- At each release, choose the largest shared old scale in `[1,.5,.25,.125,.0625]`
  for all directions including their negatives, with actual absolute logit
  shifts at `m` and `m+8` ≤`.10-.00002`. Unsupported releases write zero.
  The safeguard never changes the alarm threshold. The complete true model
  curve is re-run on the already fixed prior difficulty cohort: all non-timely
  disruptive, 48 near-threshold static-safe/no-alarm, 24 static-safe/alarming,
  and 24 near-threshold timely per view. Report score flips once per shot/view
  and deduplicate physical cases across views. The held cohort is a diagnostic,
  not a prospective estimate of FPR; the parent knew the source TRAIN set.
- Advance to a full source replay only with ≥2 independently distinct matched
  true timely rescues absent from **all** matched controls, no lost timely case,
  and no increased safe false-alarm count per view compared to frozen. Failing
  the gate does not reopen EAST or justify another parameter scan. Independently
  recount archived first alarms and witness prefixes from truncated raw signals.
