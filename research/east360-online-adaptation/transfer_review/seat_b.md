# Seat B: frozen J-TEXT → EAST transfer

**Independent research review · 2026-11-26.** Goal: improve timely first alarms with **FPR strictly below 20%**, without retraining the available source model, adding a risk head/features, or borrowing future target information. These are independent votes, not panel consensus. Proposed designs are hypotheses; this review ran no training or model replay.

## Decision and votes

**Test a fixed, causal per-shot logit reference on frozen trajectories. Keep identity input mapping as the primary full-cohort control. Treat a ≤5-shot device prior as a separate, chronology-qualified experiment. Reject raw probability multiplication as the default detector and reject any claim that source calibration or five unlabeled donors guarantees target FPR.**

Input-coordinate mismatch, bounded-score saturation, startup errors and operating-point selection are different problems. A relative threshold cannot be assumed to solve all four.

| Requested decision | Vote | Objection / qualification |
|---|---|---|
| Available source checkpoint, frozen | **YES** | Current file/hash and metadata verified below. Do not substitute the historical unavailable parent or reselect a checkpoint using EAST outcomes. |
| ≤5 genuinely earlier unlabeled donors | **CONDITIONAL** | Allowed as a frozen input-statistics prior, not a class prior/probability calibrator. No local donor precedes the first shot of the complete extension2593 stream. |
| Identity input mapping | **YES; primary full-cohort branch** | Identity retains existing units, masks and source normalization, including known J-TEXT kA→A conversion. It is an honest fallback, not a promise of good target transfer. |
| New head/features, target gradients, S-CORAL | **NO for this task** | Changes the question and information budget. Paper results do not authorize these under the frozen restriction. |
| Fixed source vs per-shot logit reference on identical trajectories | **YES; preferred mechanism test** | Compute each trajectory once per fixed map; change only reference location. Separate shared-margin ablations from individually dev-calibrated operating points. |
| Raw probability multiplier as default | **NO; retain negative comparator** | Ai's RCE is not a bounded classifier probability. A raw threshold >1 is unreachable. Caps/floors are additional interventions, not proof of better transfer. |
| Preserve per-shot relative reference | **YES as a hypothesis** | Ai supplies direct precedent for relative normal/precursor contrast. Absolute classifier risk may still contain useful signal that centering removes. |
| Startup policy | **Explicit scored fallback; no silent skip** | Before reference formation use the same source fallback in paired arms. Startup FP alone can make every later operating point infeasible. |
| Dev-only operating point | **YES** | Select margin/persistence on the declared dev cohort only; score actual first persistent alarms and retain every first shot. |
| Shot-number order | **YES for the new chronology experiment** | Preserve archived-order historical results separately. Shot ID is an available chronological proxy, not certified wall-clock acquisition time. Never choose order by score. |
| FPR budget | **STRICT <20%** | With 180 negatives: at most **35 FP**. With 1,575 negatives: at most **314 FP**. 36/180 and 315/1,575 fail this task, although older protocols allowed ≤20%. |
| Historical vs new scores | **SEPARATE** | Saved JSONs and historical 146/34 describe different protocols. Reading counts is not a new replay; already inspected archival shots are not fresh prospective validation. |

## 1. What the actual papers establish

### Ai 2026: relative anomaly scale has precedent; frozen probability transfer does not

The supplied paper is **From cold start to sustainable operation: a strategy for disruption prediction across the tokamak lifecycle**, NF 66 086034, DOI 10.1088/1741-4326/ae7e03 [P1]. It covers E-CAAD, adaptive learning and **disruption budget consumption (DBC)**, not EFD/CORAL.

- Its score is **reconstruction error (RCE)**. Eq. (1) minimizes normal-sample RCE and uses an inverse-RCE precursor loss. This is not a classifier probability or a calibrated probability of shot disruption [P1, printed p. 4].
- Fig. 3 contrasts no target fine-tuning with one and 200 target shots. Absolute RCE changes with the model, while precursor-versus-normal contrast remains visible and can grow after fine-tuning. This motivates a relative-reference experiment; it does not prove frozen Simple23 scores behave similarly [P1, pp. 4–5].
- Eq. (2) uses `ThR(i+1)=max(m*mean_i, mean_i+n*std_i)` from the previous shot's normal period. Fig. 5 estimates the current normal scale from the first flat-top slice. Outside the similarity zone but below `ThR`, it scales the threshold by `e_first/mean_i`. If the initial score already exceeds `ThR`, it **alarms**; subsequent scenario correction depends on device stability confirmation. A high persistent initial score is not automatically declared safe [P1, pp. 6–7].
- **85.88% successful alarms / 6.15% false alarms**, with **20 ms** reserved, comes from a **350-shot** chronological archival experiment with adaptive retraining. These numbers do not estimate frozen-weight, zero-post-shot-label performance on our 360 shots or our >30 ms criterion [P1, pp. 6–8].
- DBC is an operational impact/budget proxy, not disruption probability or a learned boundary. Eq. (3) includes more than current squared. Reordering archives by `Ip²` is not reproduction of prospective DBC-guided learning [P1, pp. 8–10; C3].

The primary 2025 predecessor [P3, §§4.1–4.2] makes startup risk explicit: **m=2, n=8, k=.5**, **100 ms initial running period**, and J-TEXT threshold **0.855**. Target normal RCE exceeds the source threshold; the first EAST discharge **alarms at its start**, followed by scenario-change correction. Its IRP policy is richer than the simplified 2026 diagram. “Protection from first discharge” does not establish first-shot false-alarm-free operation. Copying these constants into a bounded classifier score is unjustified.

### Shen 2026: alignment can dominate; the information budget differs

The supplied paper is **The role of data quality and alignment in cross-tokamak disruption prediction**, PPCF 68 085055, DOI 10.1088/1361-6587/ae9985 [P2].

- PGFE-U, floating labeling and EFD are separate interventions. The predictor is a **DART/LightGBM tree model**, not our Transformer. New feature extraction or floating-label source retraining would violate this task [P2, §§3.2–3.5].
- **EFD supplies target per-feature mean/std for z-score normalization.** Experimental statistics come from **ten known-disruptive EAST shots**, even though zero EAST shots train the zero-shot classifier. “Zero classifier-training shots” is accurate; “zero target information,” “five unlabeled donors” and “simulation-only prior already validated” are not [P2, §3.4, Table 3, §6].
- EFD improves zero-shot **AUC .592→.892**, low-FPR **pAUC .084→.619**. Nevertheless **TPR@FPR5%=67.2%**, versus 93.6% for the abundant-target model. AUC alone does not establish our operational budget [P2, Table 3 / printed p. 10].
- Its timely window is **40–500 ms before disruption**: <40 ms is tardy/FN; >500 ms is premature/FP. Our >30 ms first-alarm criterion and separate >3 s audit are different. Do not directly rank paper rates against project rates [P2, §3.5].
- Fig. 10 identifies sensitive global features, but wording about a vertical-position proxy conflicts with Table 1's **`Z_proxy=Vloop/Ip`**, a resistance proxy. Preserve stored field meaning; do not infer a geometry/sign repair from the inconsistent prose [P2, Table 1, §5.1].
- The roughly **.3 standard-deviation mean / 15% scale** sensitivity statement is a Monte Carlo acceptance criterion relative to an unperturbed scan, not a deterministic bound for arbitrary five-shot priors. No end-to-end simulation-prior test is supplied [P2, §5.1, §6].
- SHAP sign disagreements are 6/23 and 7/23; importance-rank correlations with the abundant model are **.07 and −.09**. Broad sign agreement is not preserved importance or causal validation [P2, §5.2].

**Interpretation:** audit input scales before blaming thresholds. A short prefix is not necessarily a device-wide distribution estimate, and affine remapping every physical quantity can erase real scenario/risk differences.

### MIT primary works: machine-specific normality and regime coverage

- Zhu/Rea et al., **Hybrid deep-learning architecture…**, NF 61 026007 [P4]: publisher abstract reports EAST AUC .959 with **20 disruptive EAST discharges plus thousands of non-disruptive EAST discharges**, combined with source-machine training. “20 shots total” is incorrect. It distinguishes machine-specific non-disruption data from transferable disruption knowledge. **Reading limit:** accessible primary abstract, not full paywalled article; no finer claim inferred.
- Zhu/Rea et al., **Scenario adaptive disruption prediction study…**, NF 61 114005 [P5]: full primary author manuscript reports poor LP→HP same-device transfer, improved transfer through operational-parameter matching, and benefit from combining target LP with other-machine HP data. This does not establish LP priors guarantee HP safety. Historical scenario ranges are not a causal first-shot oracle.
- Montes/Rea et al., **Machine learning for disruption warnings on Alcator C-Mod, DIII-D, and EAST**, NF 59 096015 [P6, §§2–4]: thousands of device-specific shots, required causal filtering, machine-dependent precursor behavior and flat-top validity restrictions. Large same-device results do not remove startup/regime-domain gaps.

## 2. Calibration, AD thresholds and saturation

### Score interpretation

Source training samples uniform class → shot → endpoint, with precursor labels `max(25 ms, AdvancedTime)` [C2]. Its sigmoid is a precursor-class score under that training sampling distribution, not automatically EAST shot-disruption probability. Neither unlabeled input moments nor a per-shot logit median provides labeled probability calibration.

Guo et al. [P7, §4, Eq. 9] fit positive temperature `T` by **labeled validation NLL**; binary form is `p_T=sigmoid(z/T)`. Positive common temperature preserves score order; transforming the threshold correspondingly preserves crossing times. It creates no new discriminative information. Ovadia et al. [P8, §§4–5] show IID validation calibration **does not guarantee calibration under shift**. No target temperature without permitted labels; do not call relative scores calibrated probabilities.

### AD threshold risk is not shot FPR

Siffer et al. [P9, §4] initialize POT/SPOT from a substantial batch, typically **~1,000 observations**, fit a tail and exclude flagged anomalies from updates. SPOT's stationary case requires a time-invariant distribution; DSPOT models a drift component. Risk `q` concerns observation exceedance, not **any persistent first alarm in a whole shot**. Correlated 1 kHz points and varying shot durations prevent equating these units; five donors do not automatically supply thousands of independent tail observations. Moving thresholds can absorb growing precursors. **Vote: no GPD/SPOT machinery in the minimal design.**

Tong/Feng/Li [P10, Proposition 1] distinguish empirical FPR selection from high-probability population type-I control. Even with independent **labeled normal calibration shots** and a fixed scoring function, their largest-order-statistic approach needs `n≥log(delta)/log(1-alpha)`; `alpha=.2, delta=.05` requires **14 independent calibration negatives**. With five continuous IID normal scores, worst-rank violation probability is `.8^5≈.328`, not .05. Our donors estimate inputs, not those held-out normal labels; drift adds further limitations. This refutes a **guarantee claim**, not empirical development with five donors; it adds no sample-size gate.

### Exact raw/odds algebra

For native `p(t)=sigmoid(z(t))`:

```text
Raw multiplier:   p(t) ≥ M*p0.
Odds multiplier:  p(t)/(1-p(t)) ≥ M*p0/(1-p0)
                  ⇔ z(t)-logit(p0) ≥ log(M).
Odds-derived probability threshold = M*p0/(1-p0+M*p0).
```

- If `M*p0>1`, raw-ratio post-reference alarms are impossible. `p0=.2, M=10` gives raw threshold 2, versus odds threshold .7142857. Raw/odds approximations agree only in the low-score regime. This is algebra, not a performance claim.
- **Use actual native pre-sigmoid logits**, already emitted by `Simple23.forward` [C2, lines 44–59]. Exposing them is not a new risk head. Current float32 inference discards logits after sigmoid [C2, lines 79–97]. Recovering `logit(clip(p,eps,1-eps))` only approximates them; current `eps=1e-6` introduces ±13.8155 plateaus [C1, lines 236–238]. True logits retain information before rounding, not information absent from the learned representation.
- Fixed `p≥tau` and fixed `z≥logit(tau)` have identical crossings absent numerical clipping. Renaming probabilities logits cannot improve AUC. Subtracting **different shot-specific** references changes cross-shot rankings and is a genuine intervention.
- Current `logit_margin` is **not pure odds centering**: anchor cap `[1e-4,.45]`, MAD-dependent margin and threshold cap `.995` all affect alarms [C1, lines 245–266]. Its historical score cannot be attributed solely to logits.

## 3. Local evidence constraining the protocol

### Available source checkpoint

Verified with `sha256sum` and metadata-only loading in the existing `lerobot` environment; no inference/training:

```text
File: code/zero_cold_start/artifacts/jtext_source.pt
SHA256: 45b17a4d56b710c8ba349af53778c175a46112776ed1525dcd508953bb1d51e7
Seed133 / selected epoch15 / 91,905 parameters excluding normalization buffers
Architecture: window128 / patch8 / dim64 / heads4 / layers2 / FFN128
VAL5% source threshold: 0.8429462313652039
VAL2.5% alternate source threshold: 0.981286346912384
Splits: TRAIN1353 / VAL160 / TEST220
Stored VAL: timely72/80, FP4/80; TEST: timely93/110, FP14/110
```

Metadata explicitly records **`test220_previously_inspected=True`** and prior source runs. Use the available fixed checkpoint, but do not describe TEST220 as newly untouched. Source-seed choice is documented as VAL-based; this review does not reconstruct all past decisions [C2, C4].

The historical parent's loader points to a different file/hash, `c7270c1e…`; its default checkpoint is absent locally. Historical **136/32 and 146/34 are not this available checkpoint's zero-shot scores** [C5–C6]. This pre-existing asset gap is not permission to retrain or mix lineages.

### Donor chronology, without donor outcome labels

Enumerated HDF5 **filenames only**, compared against saved manifests [C3–C4]:

```text
All local EAST:   2,953 shots; earliest ID 53825
Development360:     360 shots; earliest ID 54157
Extension2593:    2,593 shots; earliest ID 53825
Earlier than dev360: 53825,54003,54006,54014,54017,54068,
                     54078,54094,54098,54099,54105
Earlier than complete extension2593: NONE
```

First five `53825,54003,54006,54014,54017` are earlier donor candidates **for dev360**, but belong to extension2593. They are not globally pre-existing for that complete extension stream.

Three honest scopes, no outcome-dependent donor selection:

1. **Unchanged full extension2593:** begin at 53825 with identity/no local target prior; score all shots.
2. **Explicit donor-commissioned suffix:** reserve first ≤5 chronological archive shots, use signals only, freeze prior after completion and evaluate later shots. If donors are unscored, declare changed cohort/denominators; this is not “same2593.”
3. **Counted commissioning full stream:** score first donor shots under identity, reveal no labels to the prior estimator, freeze the prior afterward for later shots. Membership is preserved, but it is a two-stage protocol, not a prior available from shot one.

The new brief requests shot-number order; old directory conventions freeze archived first20 preparation order. Preserve old evidence and explicitly name the new sorted-order protocol [C3, C5].

### Historical scores and the startup lower bound

These are **saved evidence, not new runs** [C7]:

| Saved protocol | Timely | Tardy / miss | FP | Meaning |
|---|---:|---:|---:|---|
| `dev360_one_shot`, raw-ratio M909/hold10 | 49/180 | 3 / 128 | 31/180 | Selected from 23 dev trials; 16 FP before reference formation. |
| Same selected config, `extension2593_one_shot` | 885/1018 | 4 / 129 | 1487/1575 (94.41%) | 1,466 FP have unreachable post-reference thresholds; their first alarms necessarily precede reference formation. |
| Older reflective COH6 M100 extension | 598/1018 | See JSON | 276/1575 (17.52%) | Different input/feedback protocol using ended target labels; not frozen zero-label performance. |
| Older prepared + intra-shot EAST360 | 146/180 | 22 / 12 | 34/180 (18.89%) | Different parent, supervised first20 preparation, scoring formula and order. |

For a fixed map/common startup fallback, **startup FP is a lower bound on full-shot FP**. Later references cannot retract an issued alarm. Startup >35 FP on dev360 makes every post-reference margin infeasible under strict <20%. The 1,466 historical extension cases show that increasing M is not a startup remedy. Earlier frozen input calibration may change startup trajectories; suppressing startup is a separate detector policy with early-disruption costs.

Current one-shot alignment calls a prefix “safe” after reading the ended shot's `IsDisrupt/DownTime` and excluding 250 ms before disruption or 50 ms before normal end [C1, lines 391–414]. Valid causal post-shot label use in that saved protocol is **not an unlabeled donor estimator**.

## 4. Parsimonious concrete next design

### Frozen model; only reference/confirmation state per shot

1. Lock the verified checkpoint, existing units/masks/source normalization and declared membership. Use native 23 inputs, native 128-point causal windows and existing scalar logits. No gradients, head fitting, extra features or target temperature.
2. **Identity primary.** Optional donor map: ≤5 explicitly earlier shots, signals only, fixed observed prefix such as first200 ms; no outcome/time-to-disruption/AdvancedTime/score-based selection or terminal exclusion. Record donor IDs, signal cutoff, counts and map hash. No full covariance.
3. Optional diagonal map: `x_to_source=cS+sS*(x-cD)/sD`, in the checkpoint's actual source coordinates. Fixed source-informed scale floors; missing/near-constant/unsupported channels retain identity. Lock channel policy before evaluation; do not presume current/Bt/scenario changes are sensor-gain errors. No per-shot physical-input centering or moving target moments. Five prefixes provide a **hypothesized nuisance prior**, not Shen's full ten-disruptive-shot distribution.
4. Generate **native logits once per fixed map**. The reference arms consume identical logits, masks and timestamps. Identity-vs-prior is a separate matched input intervention.
5. First100 ms: same source fallback `z_src=logit(.8429462313652039)` and same fixed persistence in all reference arms. Gather arrived/valid initial scores. Once available, freeze `mu_i=median(initial logits)` for the rest of that shot. Short/inadequate records stay on source fallback and remain scored; do not rewrite startup predictions or use later points to revise the reference.
6. Minimal shared-margin mechanism comparison:

```text
Before reference completes, both arms: z_i(t) ≥ z_src.
After reference completes:
  Fixed-source reference:             z_i(t)-mu_S ≥ delta.
  Current-shot relative reference:    z_i(t)-mu_i ≥ delta.
mu_S: fixed source-TRAIN normal-prefix median; shot-balanced;
      same prefix/validity definition as current-shot reference.
mu_i: current-shot initial logit median, causally formed then frozen.
```

   This source score-reference statistic is not a new risk head. If unavailable, retain checkpoint constant-threshold control rather than fabricate a source median. Also retain native constant-source-threshold baseline.
7. First version has **no anchor cap, MAD inflation, threshold ceiling, rolling baseline, inter-shot multiplier update or new OR danger guard**. Historical safeguarded-logit mode remains a separate comparator; these extras otherwise obscure location-only attribution.
8. Score **first completed persistent alarm**, not its first confirming sample. Gaps/missing scores reset persistence. Log hot/unstable prefix contamination without declaring normality from future outcome. A constant high-risk trajectory or precursor present from startup can defeat relative centering. An absolute guard, if tried later, is a separate matched arm with total union-alarm FPR counted.
9. No post-shot learning in the minimal test. Model/map/source-reference/margin/hold remain fixed; ephemeral reference/confirmation reset per shot. This isolates zero-label inference effects from historical reflection.

### Development, locking and reporting

- Declare **sorted dev360** as the new development order; preserve old archived-order scores. Constants above are reversible initial engineering choices, not established paper optima.
- **Mechanism table:** common delta/hold, identical trajectories, reference-only change. **Deployment table:** optionally dev-select each margin to maximize timely TPR under exact strict budget; tie-break lower FP, then more conservative margin. Separately calibrated thresholds are operating-point comparisons, not pure ablations. Keep persistence common when claiming reference-only gains.
- If startup alone exhausts the budget, report **no feasible operating point**, with counts. No discarded initial shots, extension-driven tuning or order search to rescue the claim.
- Lock checkpoint/prior/reference/margin/hold before extension replay. Use the actual unchanged-full, counted-commissioning-full or donor-suffix scope. Already inspected archive remains historical engineering replay, not newly unseen data.
- Ground truth only HDF5 `meta/IsDisrupt`, disruptive `meta/DownTime`. Primary timely **lead>30 ms**, exactly30 ms tardy, misses, FP/TN, every first shot retained. Source >10 ms and training label windows remain separate [C3, C5].
- Also report (30,3000] ms warnings, >3 s counts, startup/post-reference FP, saturation and contaminated-reference counts, class counts, paired shot flips and actual confirmation completion time. Main budget is exactly **`5*FP<N_normal`**.
- With 180 negatives, strict <15% permits 26 FP; strict <10% permits17 FP. If supplemental conventions instead use ≤15%/≤10%, label them. Do not ROC-interpolate an unattainable persistent-first-alarm operating point.
- Native-score AUC is not relative-detector AUC. Label any relative statistic and account for startup/persistence consistently. Primary evidence remains timely/FP counts and paired first-alarm flips [C8].

## 5. Hypotheses retained; no assumed consensus

| Hypothesis | Supporting evidence needed | Objection |
|---|---|---|
| H-input: nuisance input mismatch drives startup FP | Earlier frozen donor map beats identity on matched sorted membership, without losing early TP | Five donors may encode operating regimes, not calibration; full extension has no earlier local donor. |
| H-bounded: raw ratio causes unreachable alarms | Already supported by algebra and saved JSON | Logits remove this particular impossibility, not startup FP or poor representation. |
| H-reference: shot-specific location helps | Relative beats fixed source on identical logits/common hold under strict FP budget, with paired flips | Hazard already in prefix; constant high risk; subtraction removes genuine absolute signal. |
| H-operating-point: apparent gains are retuning only | Individual dev-calibration improves while shared-rule ablation does not | Cannot call this new physics or representation gain. |
| H-generalization: rule transfers to later operation | Locked later chronological replay, eventually genuinely new shots | Existing 2,953 archive shots are already inspected; a new score is not prospective evidence. |

I favor the small reference-location experiment over a new adaptation mechanism. I oppose globally initializing complete extension2593 with its own first five shots, importing Shen's EFD result as a five-shot guarantee, or attributing cap-heavy detector gains to logit centering alone. Keep negative identity/startup results visible.

## References and inspection record

### Primary literature

- **[P1]** Ai et al. (2026), *From cold start to sustainable operation…*, NF 66 086034. [DOI](https://doi.org/10.1088/1741-4326/ae7e03); [supplied PDF](/home/dianrobot/Fuison-final/literature/Ai_2026_Nucl._Fusion_66_086034.pdf). Read manuscript and references, directly inspected equations/figures. PDF page=printed page+1 (cover).
- **[P2]** Shen et al. (2026), *The role of data quality and alignment…*, PPCF 68 085055. [DOI](https://doi.org/10.1088/1361-6587/ae9985); [supplied PDF](/home/dianrobot/Fuison-final/literature/Shen_2026_Plasma_Phys._Control._Fusion_68_085055.pdf). Read manuscript §§1–6; directly inspected Tables1–5/Figs4–11. PDF page=printed page+1.
- **[P3]** Ai et al. (2025), *Adaptive anomaly detection disruption prediction starting from first discharge on tokamak*, NF 65 036011. [Primary publisher text](https://iopscience.iop.org/article/10.1088/1741-4326/ada9a9/pdf); [author preprint](https://arxiv.org/abs/2404.08241). Startup/threshold sections read; distinct from P1.
- **[P4]** Zhu et al. (2021; online December2020), *Hybrid deep-learning architecture for general disruption prediction across multiple tokamaks*, NF61 026007. [Primary publisher abstract](https://google.iopscience.iop.org/article/10.1088/1741-4326/abc664). **Abstract-level support only.**
- **[P5]** Zhu et al. (2021), *Scenario adaptive disruption prediction study for next generation burning-plasma tokamaks*, NF61 114005. [DOI](https://doi.org/10.1088/1741-4326/ac28ae); [full primary author manuscript at OSTI](https://www.osti.gov/pages/servlets/purl/1832010).
- **[P6]** Montes et al. (2019), *Machine learning for disruption warnings on Alcator C-Mod, DIII-D, and EAST*, NF59 096015. [Primary text](https://google.iopscience.iop.org/article/10.1088/1741-4326/ab1df4). Database/causality/flat-top sections read.
- **[P7]** Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML/PMLR70:1321–1330. [Primary paper](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), §4/Eq9.
- **[P8]** Ovadia et al. (2019), *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*, NeurIPS. [Primary paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf), §§4–5.
- **[P9]** Siffer et al. (2017), *Anomaly Detection in Streams with Extreme Value Theory*, KDD. [Primary paper](https://dl.acm.org/doi/pdf/10.1145/3097983.3098144), §§4.1–4.3.
- **[P10]** Tong, Feng, Li (2018), *Neyman-Pearson classification algorithms and NP receiver operating characteristics*, Science Advances4 eaao1659. [Primary article](https://www.science.org/doi/full/10.1126/sciadv.aao1659), Algorithm1/Proposition1/sample-size discussion.

### Local source and evidence

- **[C1]** [Current engine](/home/dianrobot/Fuison-final/code/zero_cold_start/engine.py:236); [post-shot estimator](/home/dianrobot/Fuison-final/code/zero_cold_start/engine.py:391).
- **[C2]** [Native backbone/source training](/home/dianrobot/Fuison-final/code/zero_cold_start/backbone.py:20). Checkpoint metadata directly inspected, no inference/retraining.
- **[C3]** [Input/outcome/cohort contracts](/home/dianrobot/Fuison-final/code/zero_cold_start/data.py:35); [cohort/order handling](/home/dianrobot/Fuison-final/code/zero_cold_start/data.py:160).
- **[C4]** [Manifests](/home/dianrobot/Fuison-final/code/zero_cold_start/artifacts/manifests.json); [checkpoint/hash documentation](/home/dianrobot/Fuison-final/code/zero_cold_start/README.md:18). Donor chronology used filenames only, no donor labels.
- **[C5]** [Scoped protocol](/home/dianrobot/Fuison-final/code/research/east360-online-adaptation/AGENTS.md); [historical README](/home/dianrobot/Fuison-final/code/research/east360-online-adaptation/README.md); [historical proposal](/home/dianrobot/Fuison-final/code/research/east360-online-adaptation/EAST360_TECHNICAL_PROPOSAL_20261114.md); [historical status](/home/dianrobot/Fuison-final/code/research/east360-online-adaptation/CURRENT_SOURCE_UPDATE_STATE_20261104.md).
- **[C6]** [Historical parent/hash loader](/home/dianrobot/Fuison-final/code/research/east360-online-adaptation/experiments/risk_aligned_online/redesign_20261004/prepare_prior.py:22); [historical supervised readout](/home/dianrobot/Fuison-final/code/research/east360-online-adaptation/experiments/risk_aligned_online/commissioning_transfer_20261108.py:158).
- **[C7]** [Dev one-shot JSON](/home/dianrobot/Fuison-final/code/zero_cold_start/results/dev360_one_shot.json); [extension one-shot failure](/home/dianrobot/Fuison-final/code/zero_cold_start/results/extension2593_one_shot.json); [older labeled-feedback extension](/home/dianrobot/Fuison-final/code/zero_cold_start/results/final_holdout_indep2593.json); [dev selection code](/home/dianrobot/Fuison-final/code/zero_cold_start/experiments.py:165).
- **[C8]** [First-alarm metrics/native AUC](/home/dianrobot/Fuison-final/code/zero_cold_start/metrics.py:7).

**Verification boundary:** primary-paper reading, existing-code/saved-score inspection, filename chronology, checkpoint hash/metadata only. No new training, inference, source tests, PR or claimed performance gain. Only this review file created. The scoped `apply_patch` command was unavailable (`apply_patch: 未找到命令`); the dedicated file-write tool was used instead.
