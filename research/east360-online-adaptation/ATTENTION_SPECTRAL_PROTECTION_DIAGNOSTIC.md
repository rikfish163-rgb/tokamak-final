# No-compensation historical diagnostic — 2026-10-29

Specified after spectral_v1 completed, before this diagnostic. Its full360 result
was125TP/33FP vs125/32 cold; reversed targets gave identical decisions. No promotion.
Historical macro loss concealed D mean loss increase +.03253 and safe decrease -.07823.

Question: within a PSD contraction of the full Q/K signal gradient, can ended cold20
provide a direction with no first-order harm to *any* of its14 mutable peaks?
This is not a universal impossibility test and not an independent future experiment.

Use only the20 saved G/H pairs, identical cold parent, no suffix outcomes. For each Q/K
block, diagonalize mean_D sym(H^T G/||G||) and retain all positive eigenvectors
(relative cutoff1e-12). These orthogonal atoms define P=sum_j w_j u_j u_j^T.
Solve a linear program: maximize mean_D <H,GP>/||G|| subject to each historical
shot's linear gain>=0, 0<=w<=1, and sum w<=2 separately for Q and K. Row-normalize
nonzero gain constraints; scipy HiGHS primal/dual tolerances1e-9. Zero is feasible;
solver failure is an error, not an invented zero result. P is PSD/operator-norm<=1
with trace<=2; unlike_v1 it is not necessarily idempotent or rank<=2. Report actual rank.

Verify actual finite updates on the historical sampled full trajectory, reusing exactly
the previous cache. Backtrack eta=1,...,1/1024; accept the first with every historical
shot loss change<=1e-6 AND mean disruptive-shot change< -1e-6. Otherwise report no
accepted nonzero historical policy. No full360 launch from this diagnostic alone.
No remaining-shot labels, source changes, threshold changes, or output corrections.
The original gradient_transport and ATTITTUD overlap in purpose; this LP is a bounded
diagnostic motivated by observed class compensation, not a claimed new physical law.

Artifacts: LP gains/weights, PSD spectra, all finite-step outcomes, source/code hashes,
comparison against_v1, and tests of constraints, zero solution and objective direction.
