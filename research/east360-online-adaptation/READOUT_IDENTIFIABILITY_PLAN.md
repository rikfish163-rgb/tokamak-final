# Frozen-representation readout versus dynamics diagnostic

2026-10-29. Previous full360 source forecast improves one timely decision but
reverse/hold controls do the same. Cold20-head future error remains9.95x hold.
This diagnostic asks whether a better-conditioned auxiliary readout can recover
current levels and predict changes without moving the frozen risk representation.
It is not a new online360 result or a demonstration of physical mechanisms.

Preserve source SHA c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02,
original normalization, risk head, threshold, cold20 mapping and slow operator.
J-TEXT TRAIN1353 fits auxiliary probes, VAL160 evaluates; TEST220 is never read.
EAST only the already-ended first20 of qk_forecast_east360_v1 are used. No
suffix340 data or outcomes/events are read. No J-TEXT online experiment.

For each record choose at most16 evenly spaced clocks256,384,... with a complete
next25-point target. No outcome-based selection. Three frozen input views:
all17x64 final tokens; last signal token64; last8 normalized raw vectors plus
their masks (368). The latter is a non-neural positive/control baseline.
For every view fit current23 values, absolute next25x23 and increments relative
to last observed value, using a single independent multi-output linear ridge.
Per-channel require last8 samples and all25 future samples finite/available;
exclude mode_number_n. Missing targets are not zero labels. Channels unobserved
in a fitting fold remain explicitly unestimable, not assigned default predictions.

Ridge objective: equal-shot weighted mean squared residual plus .01||W||^2.
Each input column is centered/scaled using fitting rows only; a constant column
is exactly centered to zero with unit divisor. Intercept is not penalized.
Use deterministic float64 exact primal/dual solves by matrix dimension, not
an optimizer or held-out hyperparameter search. Verify normal-equation residuals.
Report masked SmoothL1 in original source-asinh coordinates and MSE, first
average clocks inside each shot then average shots. Keep counts and exclusions.
Also evaluate existing original/source/cold auxiliary heads, hold, and the
causal slope (last value minus8-points-earlier value)/8. For this slope require
that earlier observation too. No clipping by future target values.

Source: fit1353, report training fit and validation160. EAST: five deterministic
auxiliary-readout folds (index modulo5), fit16 and evaluate4; then fit all20 only
to characterize in-sample fit. These are conditional readout folds: the existing
fixed cold operator and mapping have already used all20. Thus this is NOT a
clean held-out transfer test, NOT independent validation, and NOT an operational
first-shot online score. Labels never enter the new readout fitting.

Compare current-level reconstructability versus future residual skill, and raw
versus internal representation. Failure of a linear probe does NOT prove the
representation contains no nonlinear information. A residual diagnostic readout
is not itself a new proposed risk adapter: increments and residual tasks have
already been tried. No candidate promotion or new risk trajectory this stage.
All output stays on mounted disk; source files/data remain unchanged.

Numerical correction after v1: independent recount exposed float32 accumulation
in baseline scores versus float64 in ridge scores (channel-sum relative error
about 1.7e-6 in the observed failure). Score all methods in float64, rerun v2
unchanged data/fits, and preserve v1. This does not change forecasts or risk.
For the training-versus-OOF comparison, additionally recount training predictions
on the exact OOF channel mask; one fold cannot estimate CIII from its16 shots.
