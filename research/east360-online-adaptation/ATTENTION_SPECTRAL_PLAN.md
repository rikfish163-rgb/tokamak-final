# Historical joint-descent spectral diagnostic — 2026-10-29

Status: specified before running this candidate. Development-only; not independent validation.
Prior goal turn: progress (three full360 negative experiments and audits). The intervening
disk-space question verified both partitions have ~1.5GB free, not a research success.
This run uses the existing worktree/data links, no duplicate raw data, no deletions.
Configured named agent roles are not exposed by this runtime; work remains local.

## Question and distinction

The earlier learned Q/K basis selected eta=0, including 20 fixed-step training iterations.
Can a *joint* signal/risk descent subspace be extracted analytically from ended cold20,
and does it generalize to subsequent shots? This is not a new score-correction module.
Source, cold alignment and cold slow state remain unchanged. No J-TEXT online work.
Existing gradient_transport.py already implements a gradient map; this is a narrower
block-matrix PSD projection diagnostic, not a claim that gradient transport is novel.

## Construction

Reuse the exact frozen first-layer full Q/K MatrixAttention parameterization. For ended
shot i, take the cold maximum logit among operationally eligible endpoints (>30ms for
disruption; every valid point for safe shots). Sample later endpoints every8 points plus
the last eligible disruptive endpoint, as in the previous basis experiment. The fixed
first256 prefix is also included; if its maximum wins, its gradient is zero because it
cannot be retroactively changed. All supervision in this construction is historical.

G_i is the arrived H25 forecast gradient available at the preceding 128-point feedback
clock. H_i is the historical operational-loss gradient at the chosen cold peak. Use
equal class weight and equal shot weight within each class. No class label at inference.
For each Q/K matrix, S = weighted mean sym(H_i^T G_i / ||G_i||_F), with the norm shared
across Q and K. Choose the at-most-two positive largest eigenvectors of S, giving P=UU^T.
The new temporary update is Delta W = -eta (G/||G||_F) P. Normalize *before* projection,
unlike the older LoRA basis candidate. It is reset at each feedback and at each shot.

The trace identity <H,GP>=tr(P sym(H^T G)) gives the average linearized risk improvement.
PSD projection also ensures <G,Delta W> <=0 for the signal loss. Neither fact guarantees
finite-step improvement, every-shot benefit, future generalization, or a physical law.

## Historical finite-step check (predeclared)

Use all-channel forecast gradients (exclude noncontinuous mode_number_n as before).
No weight-policy fit or extra channel search in this candidate. Evaluate eta in
1,1/2,...,1/1024 in this order and choose the first with actual historical class-macro
operational loss change + positive loss change < -1e-6. Evaluate the entire historical
sampled trajectory for each step (not just the selected old peak). If none qualifies,
eta=0 and stop after the historical diagnostic: do not run a redundant zero-update360.
Also run five policy-only folds (shot index mod5); the cold parent itself uses all20,
so these folds are conditional diagnostics, NOT independent held-out validation.

## Full stream if nonzero

Full360 includes original first20 cold-start predictions, never retrospectively changed.
After cold20: cold / random_projector(seed131, matched eta) / spectral / reverse_target /
constant_feedback(historical mean G with spectral P, same clocks). First256 points cold;
feedback every128 observations using only fully arrived targets. Current labels blocked
until curves sealed. No additional inter-shot or alarm updates. Strict lead>30ms,
threshold=0.7258946895599365; report full360 and prefix20/suffix340, paired rescues/harms,
actual nonzero updates, first256 identity, constant and reversed-control differences.
Primary necessary screen: TPR not below cold, FPR<=.20, J above cold and random control.
Passing is only a screen; lack of distinction from constant/reversed controls precludes
claiming useful per-shot adaptation. Recompute decisions independently from stored curves.

## Literature read for this design

- Gauch et al., SubGD, CoLLAs2022, Methods 3.1–3.2:
  https://proceedings.mlr.press/v199/gauch22a/gauch22a.pdf
  Gradient-space spectral preconditioning; published matrix is an uncentered
  autocorrelation estimated from training trajectories, NOT our H/G cross moment.
- Dery et al., ATTITTUD, ICLR2021, sections3–4 and equations1–2:
  https://arxiv.org/pdf/2108.11346
  Auxiliary gradients decomposed using primary supervised gradients. Current target
  supervision is unavailable here; only ended-shot supervision may guide a frozen P.
  Its theorem is not a guarantee for unseen EAST shots or our construction.

Verification: unit trace/eigen optimum/descent/norm/zero tests, actual-network causality
and reset test, finite historical effects, immutable-source and causal full-stream audit.
