# Mask-separated gradient and weight-capacity arithmetic audit

**Result:** the saved v2 trace and capacity arithmetic are internally consistent, but the offline LOSO estimates remain positive for disruptive and danger groups. Neither the fitted weights nor the first-order estimates establish deployable or neural efficacy.

Scope: read-only CPU arithmetic on the [v2 trace](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/mask_separated_channel_trace_20261104_v2), [capacity artifact](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/feedback_weight_capacity_20261104_v1), sealed pilot, and code; no GPU, raw files, training, or neural replay. The retained v1 `failure.json` reports a numerical decomposition failure after 50 episodes, with maximum absolute difference `5.430541932582855e−6`; it is not a completed trace.

## Integrity and trace

- Completion hashes match: pilot **9/9**, trace **3/3**, capacity **1/1**. Pilot prediction/state seals, trace→pilot and capacity→trace completion hashes, capacity code hash, and **115/115** trace-bound current code hashes match. These checks do not attest execution chronology.
- Exactly **200/200** unique trace keys match the pilot's real arms: 100 coordinates × two stems, no missing/extra rows. Outcomes/leads and each actual delta equal the pilot's scored row. Independent NumPy BCE recomputation from sealed logits and saved HDF outcomes differs by at most `3.7470027081099033e−16`.
- For all 200 rows, `sum(23 channel contributions) + numerical residual` matches first order within `1.3877787807814457e−17`; maximum absolute numerical residual is `1.9522597005684666e−7`. Actual and first-order deltas have the same nonzero sign **200/200**, although their maximum absolute difference is `0.0017144134876358047`.
- Independent shot-macro reconstruction matches every saved group count, actual/first-order/residual mean, sign count and channel mean exactly. Safe/disruptive/danger contain 12/8/8 shots and 60/40/24 points:

| Stem | Safe actual / first-order | Disruptive actual / first-order | Danger actual / first-order |
|---|---:|---:|---:|
| original | −0.000625274 / −0.000614802 | +0.008090056 / +0.008016919 | +0.004779658 / +0.004617626 |
| reference_bt | −0.000917993 / −0.000901028 | +0.008761465 / +0.008777846 | +0.006665705 / +0.006668698 |

The explicit residual accounts for the trace's whole-gradient versus summed-channel reduction discrepancy; it is not a physical channel. The code only checks the recomputed whole gradient against the pilot receipt with tolerances (`atol=2e−7`, `rtol=2e−4`); the saved trace does not permit independent vector-level proof that it equals the pilot step. No such replay was attempted.

## Capacity and limitations

For all 21 saved fits, weights range **0–2**, the 21 active-channel mean is 1 within `2.220446049250313e−16`, and the two inactive channels are zero. Every LOSO training-ID set excludes its held shot and contains the other 19. Independent shot-macro matrix multiplication reproduces all saved training group estimates and 100 held rows exactly. Numerical KKT certificates for the all-20 and 20 fold minimax solutions have maximum primal–dual gap `7.595920420433444e−16`; no new weights were fitted.

| Original stem, first-order only | All-20 uniform / fitted | LOSO held uniform / weighted |
|---|---:|---:|
| Safe | −0.000614802 / −0.001021194 | −0.000614802 / −0.000684753 |
| Disruptive | +0.008016909 / −0.001021194 | +0.008016909 / +0.004088245 |
| Danger | +0.004617618 / −0.003670126 | +0.004617618 / +0.002525709 |

Capacity uses channel sums **without** the separate numerical residual; its uniform values therefore differ slightly from the trace's full first-order values. The 21-channel mean-one constraint is global to each fit, not per episode: using saved observed counts, the effective observed-scalar weight mean ranges **0.95–1.00**. Reweighting would change local normalization, the metric step's normalization/backtracking and nonlinear response; these matrix estimates do not measure that new update.

Training groups use outcome-signed gradients; **19/20** LOSO folds include shots later than the held shot in the pilot order, and held evaluation itself uses the revealed outcome. All-20 results are in-sample. The capacity `held_rows` omit query IDs; safe-shot leads are null, so this audit matched each shot's five rows by producer order. This is a traceability defect, not a detected arithmetic mismatch. No oracle weight is claimed deployable.
