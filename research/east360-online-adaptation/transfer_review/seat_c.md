# Seat C: transfer evaluation, cohorts and fair baselines

**2026-11-26. Scope:** research review of frozen J-TEXT→EAST at **strict FPR < 20%**. Read directory instructions, README, proposal, current-state excerpts, primary-paper passages and actual local HDF5 metadata. No training, inference, GPU experiment, checkpoint deserialization or change outside this report was performed. Historical performance below is reported evidence, not a new replay. `apply_patch` was unavailable; this report was created with the dedicated file writer.

## Verdict and votes

**Proceed with the small paired frozen-source experiment; do not assume alignment or relative alarms will win.** All 2,953 local EAST files exist. Five earlier, outcome-blind statistical donors are feasible before the first member of the fixed 360. The archive and extension have already been used for research, so another replay is not independent prospective validation.

| Candidate decision | Vote | Evidence / objection |
|---|---|---|
| Freeze the available source checkpoint | **YES** | Local flat23 checkpoint exists and SHA256 matches documented artifact. Preserve normalization, masks, architecture, head and source threshold; do not select a seed on EAST. §Checkpoint audit. |
| ≤5 earlier unlabeled shots for fixed device map versus identity | **YES, with donor accounting** | Five donors below shot54157 exist, disjoint from 360/support110 and not selected by outcome. Zero target classifier training with five target statistical shots is not zero target information. Donors belong to extension2593: retain their unadapted startup results or separately name a donor-excluded set. |
| Fixed and relative logit alarms on same curves | **YES** | Isolates alarm policy from model changes. Fixed logit and fixed probability thresholds are equivalent. Relative centering can suppress an already-abnormal prefix. Shared startup rules and total alarm accounting matter. |
| No new input features/head | **YES for this experiment** | Good attribution and inexpensive reuse. Limits claim to this backbone/diagnostics; does not test complete PGFE-U/FLS, E-CAAD or multi-machine methods. |
| Max timely TPR at strict FPR<.2 | **YES on declared calibration/development; NO as test-set selection** | At most **35**, not36, FPs on180 safe shots. Lock operating point before downstream scoring; same-set optimized envelope is posthoc capacity. |
| Startup and post-shot feedback separated | **YES** | Count every startup alarm; reveal outcomes only after sealing. Feedback cannot retract a startup FP. A relative-only detector with an unprotected prefix is not continuous protection. |
| Call360 independent, or silently change order | **NO** | Repeatedly used development roster. Original archived order has178 adjacent shot-ID decreases. Sorted-ID replay is a new exploratory ordering. |
| Call2593 untouched or wholly later than development | **NO** | Prior extension results and supervised capacity studies exist; extension/development IDs interleave. Use an explicitly later subset for temporal transfer. |

## Actual cohort and chronology audit

I opened **all2,953 EAST HDF5 files** read-only. Every file had readable binary `meta/IsDisrupt`; disruptive `DownTime` values were finite. All23 expected channels existed with equal per-shot lengths, `SampleRate` and signal `StartTime`. This verifies storage/schema, not diagnostic calibration or live availability. Recursive source-file counting found **1,733 J-TEXT HDF5 files**. [L1–L4]

| Set | Actual available | Disruptive | Safe | Shot-ID range |
|---|---:|---:|---:|---|
| All local EAST | 2,953/2,953 | 1,198 | 1,755 | 53825–96905 |
| Locked development roster | 360/360 | 180 | 180 | 54157–96375 |
| Complement of360 | 2,593/2,593 | 1,018 | 1,575 | 53825–96905 |
| Historical support | 110/110 | **15** | **95** | 54314–96845 |
| Original archived first20 | 20/20 | 8 | 12 | 54157–94428 |
| First20 after sorting same360 by ID | 20/20 | **2** | **18** | 54157–61475 |

**360/support110 overlap is zero. All support110 shots are in the2593 complement.** “Not in360” therefore does not mean never used for target training. A source-only experiment must not load support-derived scalers/adapters/thresholds. Legacy class names and shotlists do not supersede HDF5 labels. Export catalog `usable_shots=1607` reflects old TV-list/AdvancedTime filters, not storage count; do not filter to that catalog to improve scores. [L1–L5]

Original archived order is `probe_outputs/stratified_cold20_context360_v1/order.json`, beginning:

`71304, 87494, 69458, 87487, 69854, 70404, 70323, 86167, 54157, 63097, 91054, 91885, 77002, 94428, 61468, 85632, 85320, 85316, 91723, 75657`.

Order-file SHA256: `11bfe1ebc59eb19c97b7282c545c733746823e609cb192787ee836295e72a96a`.
Membership-manifest SHA256: `657cab542bd987f9c97c3cb88867a9cc828373b32e6ee58d450451461c6ec825`.
Copied `zero_cold_start/artifacts/manifests.json` matches canonical source roles and legacy360 membership/order. [L1,L2,L6]

**Chronology:** no EAST root or metadata-group attributes contain acquisition dates. Union of metadata keys: `DownTime, IsDisrupt, StartTime, length, time_interval`. These describe within-shot clocks/termination, not cross-shot dates. Sorted IDs are a chronology **proxy**, not independently authenticated campaign chronology. Preserve legacy order for legacy comparisons; name/version sorted-ID arms separately. Directory instructions retain original membership/order/first20 for the established protocol. [L2,L3,L7]

**138 files** depart from1000Hz beyond strict numerical tolerance; all are within the declared EAST nominal ±0.5Hz contract. No channel timebase mismatches were found. Use actual timestamps/nominal contract rather than dropping files or rounding clocks. [L3]

## Donor feasibility and information budget

There are **11** non360/nonsupport files with IDs strictly below **54157**, smallest development ID:4 disruptive and7 safe by audit. Outcome-blind deterministic five-shot choice:

| Donor | Outcome, audit only | Samples | Signal start | Rate Hz |
|---|---|---:|---:|---:|
|53825|Safe|5,997|3.000s|1000|
|54003|Safe|5,699|3.000s|1000|
|54006|Disruptive|7,337|1.301s|999.8637230853094|
|54014|Safe|5,699|3.000s|1000|
|54017|Disruptive|6,744|1.301s|1000|

These are first five available IDs, not selected normal/disruptive examples. Outcomes must not enter map estimation. Using `DownTime` to crop a safe donor prefix is label use despite calling the estimator unlabeled. A causal validity/phase rule fixed beforehand is permissible. Full completed donor signals are permissible only **after donors ended**, with documented sampling; unlabeled does not imply normal-only or representative. [L3,L6;P1§3.4]

Accounting options:

- **Startup-inclusive archive:** score donors using identity/no new map, seal their alarms, freeze map only for later shots. Keep all2953 outcomes, distinguish5 statistical/commissioning shots from2948 later shots. This is online-from-first-available-shot evaluation, not disjoint all-archive holdout.
- **Disjoint post-donor evaluation:** keep360 intact; separately name **2588-shot extension** excluding donors, with1016 disruptive/1572 safe. Do not call it2593.
- **Zero target statistics:** identity plus source-selected policies is the baseline. Five-donor arm cannot imply protection before donor data existed.

**Zero local files predate minimum archive ID53825.** A five-donor map cannot be applied retrospectively to every archive shot and called pre-existing. Using first legacy ID71304 as the sole boundary is too weak: that order later includes older54157. All five donors below the **minimum** evaluation ID avoid this conflict for360.

Single density gain is a reasonable narrow falsification probe, **not** the published46-statistic EFD map or proof that stored `ne_nG` has shared physical calibration. State gain, source TRAIN reference, donors, sampling/phase coverage and commit boundary. Compare identity on identical shots. General diagonal alignment needs source-coordinate compatibility, finite floors and equal-shot/phase-aware statistics; preserve units/missingness and derived physical ratios.

## Checkpoint audit

Directly usable **flat23** artifact:

`code/zero_cold_start/artifacts/jtext_source.pt`

**408,581 bytes**, fresh SHA256:
`45b17a4d56b710c8ba349af53778c175a46112776ed1525dcd508953bb1d51e7`.

Matches README and prior result provenance. `Simple23.forward` returns a **raw logit**; masked23 input, stored center/scale then `asinh`, window128/patch8/dim64/four heads/two layers/existing risk head. Windows contain only endpoint history with unavailable left padding. Training code uses TRAIN-only statistics, source positive windows `max(25ms,AdvancedTime)`, VAL160 selection and strict>10ms source timing. I verified bytes/hash and code contract; **did not deserialize or rerun qualification**. Default Python lacked PyTorch. [L8]

Original360 research parent is **different**: `gradient_aligned512_20260929/source.pt`, enforced hash `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`; referenced file is absent here. Its loader enforces a trained forecast-head/token configuration and cannot accept available Simple23 merely because both use23 channels. **Old136/32 and146/34 are not this reused checkpoint's results.** [L9]

Available **rebuilt17** PADFIX checkpoints for seeds131/133/137 have freshly verified recorded hashes but different17-feature/median-IQR/clipping contracts. Alternatives through their own loader only, not flat23 replacements. No justification to add EAST-trained heads or borrow support110 normalization. [L10]

## Strongest fair baselines and scoring

**2×2 core**, one frozen source and unchanged score generation:

1. Identity + absolute logit threshold.
2. Five-donor fixed map + absolute threshold.
3. Identity + frozen current-shot relative logit statistic.
4. Same map + identical relative statistic.

Retain original source threshold as no-target-calibration operating point. Also compare **source-VAL-selected absolute and relative policies** at equal declared FPR budget. Tuned relative versus arbitrary/source5%-threshold absolute is weaker than both families' matched low-FPR envelopes. EAST-development-selected threshold is allowable but uses a separate **labeled tuning budget**, not five-shot-only adaptation. [P1,P3]

```text
absolute: z(t) >= a
relative after prefix arrives: z(t) - median(z(prefix)) >= h
odds(p(t))/odds(p_reference) >= exp(h)
```

Absolute is exactly `p(t)>=sigmoid(a)`; logits alone cannot improve ordering/AUC. Shared additive logit bias cancels in relative statistic. Reference only uses received points and is then frozen; no lowest-risk interval search over whole shot. Its full-prefix estimate first affects predictions **after** prefix arrival.

**Startup:** keep declared common source absolute protection while reference incomplete; first alarm latched. Relative-only post-prefix arm can diagnose behavior but cannot claim to fix earlier FPs or count an unprotected prefix as safe. In absolute-guard OR relative alarm, guard imposes FP floor: a guard already over cap cannot be repaired by relative branch. High/contaminated prefix is a real failure mode. Initial file samples can start at3s; first100 samples is not necessarily ramp-up/certified stable plasma. [P2,L3,L8]

**Scoring:** HDF5 `DownTime` only for disruptions; first valid alarm; strict lead **>30ms**, exactly30ms late. Full timely/late/missed/FP/TN, all safe shots in FP denominator, startup included. EAST125ms positive training window is not scoring deadline. Existing primary scoring has no upper-warning cutoff: separately report>3000ms and optional(40,500]ms Shen sensitivity. Do not penalize early alarms in only one arm. [L7,L11;P1§3.5]

**Strict empirical cap:** `max_fp=ceil(.2*n_safe)-1`.

|Safe denominator|Largest qualifying FP|
|---:|---:|
|180|**35**|
|1575|**314**|
|1572, donor-excluded extension|**314**|
|1755, startup-inclusive all EAST|**350**|

Old `commissioning_transfer.capacity` uses `<=cap*n_safe`, admitting36/180; cannot reuse unchanged for this strict task. With first-alarm/minimum-deadline-only scoring, lower thresholds monotonically increase timely detection, so smallest normal-cap-feasible threshold maximizes timely count. Ties, startup costs and confirmation need exact full-curve replay. Calibration cap is not downstream/population guarantee; report uncertainty and paired flips. [L11]

Same curves for fixed/relative, same map for each policy contrast, and boundaries selected on same allowed data. Common numeric boundary provides an additional trajectory ablation, distinct from matched-FP-budget performance.

## Current `paired_replay.py` code audit

Code visible during review; **not executed by this seat**. [L12]

- **Good:** frozen flat23 source; source TRAIN/first5 EAST donor density estimation without outcome selection; map only after donors; shared logits; strict FP-count cutoff; startup FP counts.
- `dev=sorted(dev360)` differs from archived order. Label new sorted-ID probe; do not overwrite/call original first20 protocol.
- `EAST_first180` boundary uses **180 labeled development shots**, beyond five unlabeled map donors. `dev360` includes calibration shots; `dev_last180` is the within-split downstream subset.
- Development midpoint is ID **74990**. Extension includes **1130** shots at/before boundary and **1463** after. `extension_all` under `EAST_first180` threshold is not all chronologically out-of-sample; `extension_after_dev_fit` is appropriate later subset, still exposed archive.
- Donors scored in `extension_all` before map activation are legitimate startup-inclusive accounting, not disjoint fixed-map holdout. Supply post-donor subset if claiming disjointness.
- Relative startup uses `z-source_anchor>=h`; effective absolute boundary is `source_anchor+h`, potentially different from absolute arm. Relative gains can partly be changed startup threshold. Report startup-only counts and optionally shared source guard to isolate post-prefix effect.
- Reference constant100 is **samples**, not exactly100ms at varying actual rates; document convention/use elapsed time. Preserve>30ms.
- `AdvancedTime` is separate annotated coverage, not all-shot onset truth; cannot select references or replace HDF5 labels.

## Primary transfer-literature comparison

Contextual comparators, **not ranked operating points**: deadlines, diagnostics, budgets and architectures differ.

|Study|Target information / transfer|Verified evaluation lesson|
|---|---|---|
|**Shen2026**, PPCF68:085055 [P1]|J-TEXT→EAST; zero target classifier shots but **10 real disruptive EAST statistical donors**; few-shot ten labeled disruptions|Same360, disjoint discharge roles, source validation selection. Source-stats AUC**.592** versus EFD**.892**, TPR@5%FPR**8.4%→67.2%**. Few-shot S-CORAL AUC**.957**, TPR@5%**85.0%**. First alarm40–500ms before disruption, not>30ms here. Read passages supply no donor IDs/chronology. Five mixed-outcome unlabeled donors are new budget, not published EFD reproduction.|
|**Ai2026**, NF66:086034 [P2]|J-TEXT E-CAAD→EAST; **350** random shots IDs54000–67000, sorted chronologically; retraining after each shot|First-slice error may rescale threshold; high first-slice error still triggers alarm, correction after configuration feedback for later shot. SAR**85.88%**, FAR**6.15%**, **20ms** response. Includes model learning plus thresholds, not frozen logit calibration.|
|**Zheng2023**, CommunicationsPhysics6:181 [P4]|J-TEXT→EAST; **20 labeled** target shots, freeze/fine-tune FFE, raw multi-rate diagnostics|Source-only and transfer controls essential; architecture/confirmation differ.20-shot budget is not physical requirement to be blind for20 shots.|
|**Shen2024**, NF64:066036 [P5]|PGFE plus S-CORAL, limited labeled target data|Compatibility/alignment deserve controls; supervised classwise CORAL unavailable under five-unlabeled/no-head restriction. Cannot import published performance to this backbone.|
|**FRNN2019**, Nature568:526–531 [P6]|DIII-D/JET, scalar/profile spatiotemporal features, cross-device predictions|Supports learned cross-machine representation. Accessible primary abstract does not establish zero-target-statistics/frozen relative-prefix calibration; data restricted. No matched quantitative claim here.|
|**DisruptionBench2025**, JFusionEnergy44:26 [P3]|Framework spans C-Mod/DIII-D/EAST; zero/few/many-shot, RF/HDL/Transformer/CCNN|**Training-only task-specific scaling** avoids whole-target statistics leakage. Causal forward-fill and configurable high/low thresholds/confirmation/useful time. Flags20-disruption-only target tasks as potentially unrealistic. **Published analysis focuses on C-Mod as new machine; EAST/DIII-D-new-machine analyses are future work.** Not published validation of our EAST setting.|

Shen directly supports identity/map contrast, but zero shot means **zero classifier fitting**, not zero target measurements. Ai supports reference/post-shot adaptation, but **RCE** is not sigmoid probability; logit amplification is engineering synthesis, not E-CAAD's equation. DisruptionBench supplies clean normalization-budget warning and multi-machine protocol vocabulary. [P1–P3]

HDL's often quoted “20 EAST shots” is not twenty total target examples: primary Zheng article describes **20 disruptive plus thousands of non-disruptive EAST shots**. Original HDL publisher fetch errored, so no independent HDL numerical AUC endorsed. Exa subsequently reached free rate limit; no further primary search completed. Direct Shen/Ai and DisruptionBench evidence remains available. [P4]

## Historical baselines to retain

Original360 **146/180 timely,34/180 FP** includes labeled cold20-fitted input/readout/threshold. Static**136/32** is already prepared, not raw source. Original2593 extension: **706/1018 timely,434/1575 FP=27.56%**, failing cap; stricter archived threshold**594/1018,283/1575=17.97%**, lower timely recall. These motivate matched-budget baselines, not evidence new relative/map wins. [L7]

Same available flat23 checkpoint already has separate `zero_cold_start/results` extension outcomes. Reported chronological rows: unaligned fixed **76.23%TPR/71.43%FPR**, align-only source threshold **45.58%/8.95%**, blind updates **53.44%/12.76%**, reflective **58.74%/17.52%**. Another one-shot policy has **1487/1575 FPs**, predominantly startup. Different alignment/feedback budgets, not matched five-donor controls; evidence extension/checkpoint **already examined**, and feedback not best at every FP budget. [L13]

## Recommended action and claim

Run narrow paired frozen-source comparison with explicit donors/calibration budgets. Preserve old360 order/results, separately label temporal probe. Distinguish original source threshold, source-selected policy and EAST-development-selected policy. Use chronologically later non-fitting shots for temporal estimate, explicit startup/donor subsets; retain negative arms.

Strongest defensible archival claim: **“On previously used EAST archive, frozen source plus fixed five-shot statistical map and/or causal per-shot logit reference improved the specified timely/FPR tradeoff in a locked downstream replay.”** Requires actual paired results. Not information-free zero shot, protection guarantee or independent future validation. No new training/expensive literature-wide benchmark needed to decide this narrow experiment.

## Evidence references

Paths relative to `/home/dianrobot/Fuison-final`. Line/section references identify inspected code, not instructions to execute.

- **L1:** `code/config/splits/{east_hdf5_authoritative_balanced360_v1,east_paper131_split_all_in}.json`; splits README lines3–11. Manifest membership intersected with HDF5 inventory.
- **L2:** `code/research/east360-online-adaptation/probe_outputs/stratified_cold20_context360_v1/order.json`; `code/zero_cold_start/artifacts/manifests.json`. Direct list/hash/order checks.
- **L3:** `data/EAST/unified_hdf5/*/*.hdf5`, all2953: read-only h5py audit of outcomes/channels/lengths/timebase/metadata keys/donor details. `code/zero_cold_start/data.py:35–84`; legacy `io23.py:24–32,98–115`.
- **L4:** `data/J-TEXT/unified_hdf5`, recursive1733 count; `code/splits/{train,val,test}.txt` and copied lists.
- **L5:** `data/EAST/exports/east_shot_catalog_summary.json:2–22`; `code/DATA_CONTRACT.md:14–48`.
- **L6:** `code/zero_cold_start/data.py:96–118,160–175`, cohort/order/complement/path contracts.
- **L7:** `code/research/east360-online-adaptation/{AGENTS.md,README.md,EAST360_TECHNICAL_PROPOSAL_20261114.md}`; proposal§§4–6; `CURRENT_SOURCE_UPDATE_STATE_20261104.md` top entries including§76 capacity study/HDF5 reconciliation excerpts.
- **L8:** `code/zero_cold_start/artifacts/jtext_source.pt`, hash verified only; `backbone.py:17–97,101–136,186–253`; README18–24.
- **L9:** `code/research/east360-online-adaptation/experiments/risk_aligned_online/redesign_20261004/prepare_prior.py:21–22,53–72`, missing legacy parent/hash/config.
- **L10:** `code/experiments/transfer_methods_2026/paper_grade_package/source_padfix_confirm_e32/seed{131,133,137}/{source_checkpoint.pt,training_summary.json,test220_metrics.json}`; `rebuilt_harness.py:42–44,66–121`; `grid_search.py:138–172`. Alternative hashes match recorded JSONs.
- **L11:** legacy `hdf5_score_recheck_20261103.py:18–33`; `commissioning_transfer_20261108.py:245–264`, first-alarm/inclusive-cap behavior.
- **L12:** `code/research/east360-online-adaptation/transfer_review/paired_replay.py:15–26,37–62,65–111`, visible during audit, not executed.
- **L13:** `code/zero_cold_start/results/{final_holdout_indep2593,dev360_locked_comparison,dev360_one_shot,extension2593_one_shot}.json`; README38–62. Historical summaries/provenance only.
- **P1:** Shen et al., *The role of data quality and alignment in cross-tokamak disruption prediction*, PPCF68:085055(2026), [DOI](https://doi.org/10.1088/1361-6587/ae9985). Local `literature/Shen_2026_Plasma_Phys._Control._Fusion_68_085055.pdf`; directly read extracted printed pp8–10,§§3.4–4,Tables2–3 and relevant EFD/enlargement passages.
- **P2:** Ai et al., *From cold start to sustainable operation: a strategy for disruption prediction across the tokamak lifecycle*, NF66:086034(2026), [DOI](https://doi.org/10.1088/1741-4326/ae7e03). Local `literature/Ai_2026_Nucl._Fusion_66_086034.pdf`; directly read printed pp6–8,Eq2/cross-device experiment.
- **P3:** Spangher et al., *DisruptionBench and Complimentary New Models*, JFusionEnergy44:26(2025), [primary full text](https://link.springer.com/article/10.1007/s10894-025-00495-2), Benchmarking Tasks/Implementation Details/Machine-Specific Versus Test-Case-Specific Data Standardization/Results and Discussion; [original repository](https://github.com/MIT-PSFC/DisruptionBench). Full publisher text/README fetched, relevant passages inspected.
- **P4:** Zheng et al., *Disruption prediction for future tokamaks using parameter-based transfer learning*, CommunicationsPhysics6:181(2023), [primary article](https://www.nature.com/articles/s42005-023-01296-9). Abstract/introduction/source evaluation fetched. HDL budget is this article's account, not independently reread HDL full text.
- **P5:** Shen et al., *Cross-tokamak disruption prediction based on domain adaptation*, NF64:066036(2024), [primary article](https://iopscience.iop.org/article/10.1088/1741-4326/ad3e12), abstract/introduction fetched; no unverified numbers asserted.
- **P6:** Kates-Harbeck et al., *Predicting disruptive instabilities in controlled fusion plasmas through deep learning*, Nature568:526–531(2019), [primary article](https://www.nature.com/articles/s41586-019-1116-4), accessible abstract/data availability; no inaccessible operating-point table claimed.
