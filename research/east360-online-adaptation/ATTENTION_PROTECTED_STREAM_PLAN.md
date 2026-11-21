# Protected Q/K contraction — full-stream amendment, 2026-10-29

Prior goal turn: progress, full360 negative spectral experiment and a historical-only
LP feasibility diagnostic. The latter selected eta=.25; it was not an online result.
This amendment is specified before its crossfit/full-stream evaluation. No source or
previous experiment files may be edited. Use the same original J-TEXT source, cold20
map/slow state, fixed threshold, 360 IDs and order. EAST only; no J-TEXT online branch.

## Policy and capacity

Reuse attention_spectral_20261029.protection.contraction: ended-shot G/H pairs,
positive eigenatoms of mean disruptive-shot sym(H^T G/||G||), LP maximizing mean
disruptive gain while each fitted shot's linear gain>=0. Atom weights in[0,1];
trace<=2 per Q/K. The result is PSD contraction, NOT a rank2 orthogonal projector.
Expected full-fit ranks from prior diagnostic: Q2/K5. Do not conceal extra rank.
Update Delta W = -eta (G/||G||) P, reset at each feedback and each shot.

Five policy-only folds, shot index mod5: eigenspace, LP and finite-step selection use
only the16 fitted shots. The same cold parent uses all20: conditional diagnosis, not
independent cross-validation. Backtrack1,...,1/1024, first step satisfying max fitted
loss change<=1e-6 and mean fitted disruptive change< -1e-6. Held shots cannot affect
selection. Full fit must reproduce prior diagnostic matrix and step. If final eta0,
stop without redundant zero-update360. Crossfit is reported, not used to pick a model.

## Five full360 arms

cold; random_contraction; protected; reverse_target; constant_feedback.
Random control: same eigenvalues as fitted P separately for Q/K, orthogonally rotated
by seed131 CPUfloat64 QR, same eta. Thus rank/trace/operator norm are matched; unlike
the earlier random rank2 projector. Constant arm uses historical mean G, same clocks.
Reverse arm reverses only fully arrived targets/masks. First20 predictions preserved;
P activates only after20 reveals. First256 samples cold; every128 observations use
four matured direct H25 targets; no current outcome before seal. No post-cold20 inter
updates or alarm changes. Include all360 in primary metrics, strict lead>30ms.

Necessary screen: TPR>=cold, FPR<=.20, J>cold and matched random. Comparison with reverse
and constant controls is also required before claiming useful within-shot information.
No result here is independent confirmation or proof of new physical laws. No policy
promotion without full-stream evidence. All results and code snapshots retained on D;
no raw-data duplication or deletions. Named configured subagent role selection remains
unavailable in this runtime; this bounded GPU workflow runs in the main process.

Verification: matrix spectrum/PSD/norm tests; historical selection ignores held values;
actual-network causality, reset and zero-state tests; 360 sealed-curve recount and raw
hashes; independent NumPy reconstruction of LP matrices/constraints and finite losses.
