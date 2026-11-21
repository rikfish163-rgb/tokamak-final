# Matured within-shot readout, then disjoint Q/K feedback

2026-10-29. Development experiment, not independent validation. Source classifier
SHA c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02 unchanged.
Keep the audited qk_forecast_east360_v1 order, cold20 operator, density map and
threshold .7258946895599365. All360 including cold20 must be reported. EAST
strict lead>30ms. No J-TEXT online experiment or source classifier retraining.

Prior evidence: readout_identifiability_v2 cold20 conditional OOF last-token
increment readout is the least-bad new neural readout (.034219 vs hold .026810).
Using this development evidence to choose it is explicit, not confirmatory.
Existing lag_bias_feedback subtracts an older constant residual; forecast_profile
removes error means. We do not repeat them or claim local bias removal as new.
The new hypothesis is that adapting the full latent-to-increment linear mapping
from this shot's older observations gives more relevant gradients on newer
observations. Same-shot learning, not merely all20 in-sample fitting, is tested.

After20 shots have ended, load the exact last_token increment ridge prior from
readout_identifiability_v2. It is trained on those20 signals only; recompute that
prior from cold20 before auditing the full experiment. Readout inputs are the
64 final last-signal-token features (not CLS), with per-target-channel feature
mean/scale fixed by cold20. Coefficients include an intercept:65x25 perchannel.
No effect on risk scores unless a Q/K update changes the internal forward.

Maintain at most16 chronological forecast-origin rows within each new shot.
Each row uses the frozen cold encoder coordinates, its last observed normalized
value, and25 future values only once all25 have arrived. Whole-channel targets
need last observation plus25 valid points, and an estimable cold prior.
MNM and mode_number_n remain ineligible auxiliary targets as before.
At clocks256,384,... form the four disjoint mature H25 targets. Insert the
older two into the readout fit, then use the newer two only for neural-gradient
feedback. Insert those newer rows into history only after that update. Thus
calibration targets end strictly before gradient targets begin.

Solve in float64, independently per channel, with a fixed unit prior precision:
Delta = X^T (X X^T + I)^(-1) (Y - X W0), W = W0 + Delta.
X includes standardized64 features plus1; Y is future-minus-last signal in
source-asinh units. This is sum-of-squares plus ||W-W0||^2, not a hyperparameter
scan. If no eligible matured rows exist, retain the explicitly pre-existing
prior and record zero fit rows; do not invent observations or zero labels.
No differentiation through the fitted decoder. This split procedure is NOT
exact variable projection or a reproduction of a recursive RLS bank.

Arms: cold (no intra); frozen_readout (prior fixed, true recent feedback);
local_readout (past readout fit, true recent feedback); local_reverse (reverse
each already observed25-target sequence in both fit/gradient); local_hold
(zero future increments in both fit/gradient). All update arms use the same
two recent examples, .05 normalized full Q/K gradient in first Transformer
block, reset Q/K proposal to cold parent at each feedback, apply only to future
predictions. Readout history resets per shot; no inter-learning after cold20,
no alarm changes, no risk-head/score correction. Auxiliary head-only cannot
alter risk; regression test this invariant.

Record per-shot decision changes and actual updates. At at most4 evenly spaced
eligible feedback clocks per shot, seal H25 auxiliary predictions before
examining those future targets; score them retrospectively after sealing the
shot. Compare before/after neural update and hold on identical observed masks.
Sampling this diagnostic never affects adaptation. Missing targets are excluded
from loss, not imputed. Auxiliary evaluation also excludes post-disruption
targets using outcome information only AFTER prediction seal; that validity
mask never enters the predictor. Report early alarms>3000ms separately as before.

Necessary candidate screen: local_readout TPR no lower than cold, FPR<=.20,
and J strictly above cold, frozen_readout, local_reverse and local_hold.
Passing is necessary, not sufficient for reproducible scientific success.
Failing does not change this rule. All data are development-exposed.
Outputs on mounted disk; no raw-data copies/deletions; original files preserved.
