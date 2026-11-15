# Matched auxiliary head for Q/K adaptation: source-only experiment

2026-10-29. Original qualified classifier SHA c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02 stays unchanged.
No J-TEXT online method is promoted: J-TEXT TRAIN1353 provides offline training
episodes for an auxiliary head intended for later EAST deployment. TEST220 and
all EAST labels/signals are inaccessible in this source experiment.

Question: the checkpoint trained fast token gain/bias, but recent candidates
updated first-layer Q/K. Does teaching the existing diagnostic forecast head
through the actual Q/K update improve its subsequent risk direction?

Two same-budget head-only arms: forecast-only and matched-meta; original head
is a reference. Same initialization,512 AdamW steps,12 episodes/step,lr1e-4,
weight_decay0, gradient clip1,seed131; final checkpoint only, no epoch selection.
All other weights and source scaler stay frozen. No new risk correction/head.

Each episode matches deployed clocks: observed_count=256+128*j; four complete
H25 target intervals end before observed_count. Query is in the following128
points and is not fed to the update. Per-episode normalized Q/K gradient step
has joint L2 budget0.05, reset every episode. No learned spectral projection.
The feedback mask requires observed target AND observed last input, and excludes
mode_number_n (index10), matching the current Q/K feedback contract.

Three offline strata: safe shot, disruptive shot early negative window, and
disruptive positive window with strict lead>10ms. Labels remain
max(25ms,AdvancedTime). Sample four episodes/stratum per step, shot-uniform then
endpoint-uniform. Other/too-short intervals are listed, not padded into an
available deployment-clock episode. Query labels train the head's outer
objective; the adaptation function accepts only signal feedback.

Forecast arm objective=0.20*signal SmoothL1.
Matched arm objective=0.20*signal +0.50*risk_after +0.10*(1-cos(G,H)).
G is the arrived signal gradient at Q/K; H is the detached later supervised
risk gradient at the SAME Q/K state. Smooth gradient norms sqrt(sum(g²)+1e-16)
avoid undefined second derivatives at zero. Coefficients reuse the qualified
source recipe; this is a controlled engineering hypothesis, not a new theorem.

VAL160 fixed endpoints: at most four evenly spaced queries per stratum/shot,
with those clocks and >10ms rule. Compare original token (location diagnostic),
original Q/K, forecast Q/K, matched Q/K, matched time-reversed and matched
hold-last targets. Pair all arms on the same endpoints. Equal L2 state budgets
do not equal functional capacities across token and Q/K, so their difference
alone cannot establish a location-caused performance gap.

Report shot-averaged risk-loss change separately by stratum, gradient alignment,
signal loss and actual parameter norm. Necessary source screen: matched safe
and timely-positive changes<=0, and macro change across all three strata better
than original Q/K, forecast Q/K, reverse and hold by>1e-6. This is not a
shot-level TPR/FPR result or EAST validation. If it fails, do not claim success
or launch the identical failed head into full360. If it passes, full360 from
the same cold20 parent with fixed threshold remains mandatory.

Related: MT3 (Bartler et al.,AISTATS2022), method§2–2.1, learns downstream
performance through self-supervised inner updates. Our frozen-classifier,
direct diagnostic prediction head and delayed plasma signal protocol differ;
no image result transfers as an EAST guarantee. Earlier token/head/meta
attempts remain negative controls; this is not a claim of inventing meta-TTT.
