# Final seat A cross-review

2026-11-26. **Research-candidate approval, not deployment approval.** Independently read final D1–D6, both READMEs, directory protocol, historical proposal/state, seat C, scientific evidence rows/abstracts, replay and mapping/scoring code. No other seat notes were present. No new experiment, training or inference was run. `apply_patch` was unavailable; only this report was written with the dedicated writer.

## Independent votes on the final propositions

| Decision | Vote | Scope and remaining objection |
|---|---|---|
| D1 | **YES** | Checkpoint bytes independently hash to `45b17a4d56b710c8ba349af53778c175a46112776ed1525dcd508953bb1d51e7`. Simple23 keeps its source normalization, masks, network and 64-dimensional risk head; no EAST neural fitting occurs. The older 146/180 result uses a different prepared system and is not reproduced here. |
| D2 | **YES** | The first five available IDs are 53825, 54003, 54006, 54014, 54017, disjoint from 360 but included in 2,593. Raw HDF5 audit finds three normal and two disruptive donors; labels do not select or crop mapping inputs. Equal-shot statistics use up to 128 available observations per channel; the density gain is the source/target weighted-median ratio `0.0005490980893439213`. This is a zero-preserving **numerical transport**, not certification that `ne_nG` shares physical units or is a comparable Greenwald fraction. Five target statistical shots are not zero target information. Code retains identity predictions for all five donors, activating the map only afterward. |
| D3 | **YES** | Approve the absolute/relative policy comparison, **not relative centering as the preferred method**. The first 100 samples use `z − source_anchor`; subsequent samples use `z − median(z[:100])`. This is causal endpoint history, with startup alarms counted, but not a common absolute startup boundary across policies. Consequently the comparison includes startup-policy changes, not just post-prefix centering. A contaminated/high-risk prefix remains an unresolved failure mode; 100 samples are nominally, not exactly, 100 ms and need not represent a safe plasma phase. |
| D4 | **YES** | Source initial memory has 80 normal peaks (tolerance rank 71); the five-donor seed replaces it only after donor completion. Those five peaks include disruptions and are explicitly unlabeled seeds, not five certified normal examples. Subsequently only completed normal-shot peaks update the next threshold, with an 80-shot FIFO capacity. This consumes continuing target outcome feedback, separate from the five-shot scale budget. The small-sample maximum and binomial tolerance rule are empirical operating-point choices; mixture, drift and selection prevent a prospective 95%/20% guarantee. |
| D5 | **YES** | Directly opened all 2,953 HDF5 outcome records: 360 = 180 disruptive/180 normal; complement 2,593 = 1,018 disruptive/1,575 normal. Strict caps are FP ≤35 and ≤314. Sorted-ID replay is a **separate exploratory protocol**, not replacement of the original archived order/first-20 preparation contract; IDs are only a chronology proxy. Both sets are exposed retrospective research data. The first-180-labeled diagnostic also uses a separate budget: 1,130 extension shots are at/before fit boundary 74990, only 1,463 after it. |
| D6 | **YES** | Both candidate streams meet their empirical caps and improve timely detection versus identity under the same donor-seed construction, feedback eligibility and sorted order. Approve only that paired **mapping-plus-threshold-policy system** claim; realized FP counts and threshold trajectories differ, so this is not an equal-FP or isolated-physical-scale claim. It does not establish superiority over the old system, independent-device validation, or readiness to replace production. My vote does not assert that the other seats have voted YES. |

## Directly verified final evidence

Replay SHA256: `368c82ab0e2385b6530b5a619c81adf27c482516af7aa732ae97368854a3de42`. All rows below use `donor_seeded_tolerance`. Counted unique stored timely/FP IDs and checked summary denominators and paired set differences; did not regenerate model predictions.

| Input / score | 360 timely TP / FP | 2,593 timely TP / FP |
|---|---:|---:|
| Identity / absolute | 59 / 31 | 204 / 211 |
| Density gain / absolute | **126 / 35** | **436 / 237** |
| Density gain / relative | 63 / 32 | 301 / 217 |
| Multichannel moments / absolute | 82 / 20 | 274 / 211 |
| Multichannel moments / relative | 47 / 20 | 276 / 204 |

- Candidate density/absolute has **24 late, 30 missed, 145 TN** on 360 and **118 late, 464 missed, 1,338 TN** on 2,593: timely TPR **70.00% / 42.83%**, FPR **19.44% / 15.05%**. Startup FP **17 / 76** remain included. One additional development FP would fail the strict cap.
- Identity→density paired timely gains/losses are **69/2** and **275/43**; FP additions/removals are **15/11** and **142/116**. These exactly reconcile the reported totals.
- On the same density curves, relative scoring loses **63 / 135** timely detections for only **3 / 20** fewer total FPs. Its startup FPs actually rise to **22 / 166**. Keep this negative ablation, not a blanket recommendation against all relative policies.
- At the identical source checkpoint boundary, density transport changes 360 **140/121→139/61** and extension **776/1,125→656/622** (TP/FP): a genuine numerical-input effect, but neither stream meets the cap. Source-seeded rather than donor-seeded density tolerance gives development **132/38**, also failing. Initial construction is therefore material to the successful candidate.

## Scientific interpretation and disposition

The stored Shen-2026 evidence supports testing feature-distribution normalization, but its ten real **disruptive** EAST statistical donors, PGFE-U/backbone and warning interval are not this five-unlabeled-donor experiment. Ai-2026 supports examining startup and sequential references, not equating reconstruction-error multipliers with probability/logit multipliers or importing its retraining results. The corpus is largely abstract-level evidence; none certifies local diagnostic semantics or future false-alarm control.

**Agree with seat C's cautious scope, not deployment.** The final documents now explicitly account for numerical-only gain, mixed donors, startup, ongoing labels and retrospective exposure. Scientific semantics, campaign representativeness and prospective performance remain unresolved. Historical benchmarks already reported higher extension timely recall (58.74% for the older zero-cold-start policy; 594/1,018 for the stricter five-stage system), under different budgets/protocols. This candidate is worth retaining as a low-cost frozen-head archival result, **not a demonstrated upgrade of those systems**.

Evidence: [final propositions](panel.md), [final replay](paired_replay.json), [replay code](paired_replay.py), [scientific evidence](literature_evidence.tsv), [abstract corpus](corpus.tsv), [seat C objections](seat_c.md), [statistical mapping](../experiments/risk_aligned_online/warmstart_alignment_20261022/alignment.py), [raw-data and outcome contract](../../../zero_cold_start/data.py). Any future performance commitment requires rules frozen before a new continuous shot stream; this is a limitation, not an added gate for retaining the present research candidate.
