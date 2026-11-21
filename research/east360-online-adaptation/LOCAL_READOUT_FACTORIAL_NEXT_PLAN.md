# Next experiment: separate optimization validity from feedback information

Declared 2026-10-29 AFTER local_readout_east360_v1 failed. Not run yet; no
performance claims. This is a new development iteration, not a repair of v1's
scores. Keep all v1 files, coefficients, sealed predictions and reports intact.

## Evidence motivating the change

Full360 local125TP/33FP versus cold125/32; no rescue. Same observed loss only
decreased6325/10774 times. Three deliberately selected real-prefix CPU probes
show quarter steps decrease that loss while full steps increase it. This does
not prove a smaller step improves later signals or risks. Existing reverse and
hold controls alter both fitted history and recent gradient targets.

## Fixed new comparison

Use unchanged source classifier SHA, cold20 map/operator/prior, order, threshold,
same mature H25 windows, complete-channel masks, history16 and Q/K reset scheme.
No J-TEXT online, no risk-head correction, no alarm or postcold20 inter updates.

Compare cold plus ten adaptive arms: fixed prior/true recent target, and the
3x3 combinations of history fit target {true, reverse, hold} and recent neural
target {true, reverse, hold}. Reverse means reversing each25-point target, not
future access. All ten arms receive the same optimization acceptance rule.

At each mature feedback, form the current normalized negative gradient proposal
as in v1 with maximum norm0.05. Test at most eight factors
{1,1/2,1/4,1/8,1/16,1/32,1/64,1/128} in that order. Accept the first finite trial
satisfying observed-loss Armijo decrease with constant0.0001, using the actual
gradient dot trial displacement. Require a strictly negative directional dot
product and strict measured loss decrease. If no observed target, zero gradient,
or no trial passes, emit exact zero Q/K state, explicitly recorded as abstention.
Nonfinite model/gradient errors remain errors, not silent abstentions. This is a
bounded optimization rule on arrived observations, not a label-scored learning
rate sweep. Never select a factor using current outcomes or future signals.

Persist per-feedback raw gradient norm, proposed/accepted norm, trial count,
reason, loss before/after and directional dot product; these make optimization
quality and effective step sizes auditable. Different accepted norms change the
estimand; retain v1 equal-norm results as a separate reported reference. No claim
that acceptance on observed auxiliary loss is a safety or future-risk guarantee.

## Evaluation and stop rules

Run all arms over the same entire360, including cold20. Same first-alarm strict
lead>30ms; report TP/FP integer counts, FPR<=.20, J, lead and >3000ms alarms.
Predeclared candidate: true-history/true-recent. Compare against cold, fixed
prior/true, all same-history target controls, and all same-target history controls.
Necessary screen requires no lost TP versus cold and J strictly above those
matched comparators at FPR<=.20. Report all9 factorial combinations, including
their interactions; do not select a different winner after reading outcomes.

Also seal sampled next-window auxiliary predictions as in v1; separately report
before/after Q/K and plain hold on identical masks. Preserve the distinction
between matured training loss, future-signal loss and full-shot risk decisions.
Sampling by eventual shot length is retrospective diagnostic selection only.

If observed optimization becomes valid but risk benefit still fails, do not
continue scanning step radii in this family. Reassess the task/representation
interface using the factorial evidence and the literature memo; do not insert an
unvalidated reliability gate merely because auxiliary error fell.

## Implementation and verification ownership

Create a new bounded module/output rather than mutate frozen v1 modules.
Main integrates and schedules the single GPU. New terra/max children may own
disjoint acceptance/gradient tests and independent factorial audit. No raw-data
copies/deletions. About1.1GB free onD: check expected output before launch.
Tests first: accepted descent on convex/quartic overshoot example, empty/zero
feedback abstention, nonfinite rejection, causal prefix perturbation, independent
fit/target modes, state reset and frozen classifier. Independently recount all
sealed decisions, replay all changed shots plus fixed controls, preserve all
negative results. Goal remains independent intra gain, then intra+inter—not a
successful auxiliary loss or a passing test suite.
