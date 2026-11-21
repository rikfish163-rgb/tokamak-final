# Primary evidence: frozen J-TEXT → EAST input alignment

**Date:** 2026-10-31. **Scope:** bounded, non-systematic review of four primary papers and the requested local dissertation; no novelty claim. This is not an implementation or an experimental result. The project details and shot counts below are supplied by the requester, not verified by inspecting the implementation or data in this review. No private data were uploaded.

## Bottom line

For a classifier trained in a **fixed source input coordinate system**, the relevant test-time operation is to take an EAST input **toward the J-TEXT source coordinate system**, while leaving the source preprocessing, classifier parameters, and alarm threshold fixed. Original CORAL instead transforms source training examples toward the target and trains a classifier on those transformed examples. The CORAL authors explicitly discuss the reverse direction, but their reported comparison favoured source-to-target **with retraining**; that does not settle the frozen-classifier case [P1].

A per-channel map fitted using **source TRAIN safe-only** and **previously ended EAST safe-only** calibration data is a plausible, deliberately limited candidate: it preserves channel axes and avoids a direct mixture of safe and disruptive shots in the moment estimates. This is an inference for this task, **not a method or outcome established by the papers**. It does not establish alignment of disruptive precursors, equal posterior probabilities, improved performance, or learned physics. Its utility must be evaluated out of sample with chronological availability and the same already-fixed threshold.

## Source register: exactly four primary papers

“Full text inspected” means the indicated sections of the actual paper or author manuscript were inspected, not that every section was read or independently reviewed. Short extracts below are source words, not claims of human reading.

1. **[P1] Sun, Feng & Saenko, 2016.** “Return of Frustratingly Easy Domain Adaptation,” AAAI 30(1), DOI [10.1609/aaai.v30i1.10306](https://doi.org/10.1609/aaai.v30i1.10306). **Full text inspected:** [publisher PDF, §§3.1–3.3, especially equations (1)–(2), Algorithm 1, and the reverse-direction paragraph](https://ojs.aaai.org/index.php/AAAI/article/download/10306/10165#page=3); see also [§1](https://ojs.aaai.org/index.php/AAAI/article/download/10306/10165#page=1). Evidence anchor: “training a classifier on the transformed source features.” The paper concerns object recognition and sentiment, not tokamak data or a frozen classifier.

2. **[P2] Zhao, Tachet des Combes, Zhang & Gordon, 2019.** “On Learning Invariant Representations for Domain Adaptation,” ICML, PMLR 97:7523–7532. **Full text inspected:** [proceedings PDF, §§4.1 and 4.3, Theorem 4.3 and Corollary 4.1](https://proceedings.mlr.press/v97/zhao19a/zhao19a.pdf#page=7); [official record](https://proceedings.mlr.press/v97/zhao19a.html). Evidence anchor: “not sufficient to guarantee target generalization.” Its lower bound has stated assumptions, concerns a joint classification error and distributional invariance, and is not a theorem that every individual covariance map will hurt.

3. **[P3] Tachet des Combes, Zhao, Wang & Gordon, 2020.** “Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift,” NeurIPS 33. **Full text inspected:** [proceedings PDF, §3.1 Definition 3.1 and §§3.2–3.3](https://proceedings.neurips.cc/paper_files/paper/2020/file/dfbfa7ddcfffeb581f50edcf9a0204bb-Paper.pdf#page=3); [official record](https://proceedings.neurips.cc/paper/2020/hash/dfbfa7ddcfffeb581f50edcf9a0204bb-Abstract.html). Evidence anchor: “conditioned on the label, there exists a representation of the input that is invariant between the source and target domains.” Generalized label shift is an **assumption about every class**; matching two moments of SAFE data does not establish it.

4. **[P4] Shen et al.** “Cross-tokamak Disruption Prediction based on Physics-Guided Feature Extraction and domain adaptation,” **author manuscript, arXiv:2309.05361v2, 2023**. **Full text inspected:** [author-deposited PDF, §§2.1–2.2, Fig. 2, §§3.1–3.2, §4.2/Table 3, and §5.1/Figs. 7–8](https://arxiv.org/pdf/2309.05361v2#page=6); [versioned record](https://arxiv.org/abs/2309.05361v2). The corresponding published article is “Cross-tokamak disruption prediction based on domain adaptation,” *Nuclear Fusion* 64, 066036 (2024), [DOI 10.1088/1741-4326/ad3e12](https://doi.org/10.1088/1741-4326/ad3e12); the publisher version was inaccessible, so section and number claims here are anchored to the inspected **2023 preprint**, not assumed identical in the final version. Evidence anchor: “aligning the disruptive samples and non-disruptive samples separately between the target and source domains.” The S-CORAL experiment aligns **training** data and trains a new classifier; it is not the present frozen-model procedure.

### Additional local primary context, not a fifth paper

Shen Chengshuo, *Cross-Tokamak Disruption Prediction Using Machine Learning Based on Physics Features*, 2024 doctoral dissertation. The supplied local text is [paper_131.txt, §4.5.1](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/literature/paper_131.txt:2832), with [§4.5.2 and Table 4-9](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/literature/paper_131.txt:2859) and [§5.3.3](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/literature/paper_131.txt:3560). **Full-text extraction inspected at these passages only**; this is a text extraction, not an independently checked original PDF. The local document reports that means and standard deviations were calculated using **ten EAST disruptive discharges**, not used for classifier training. Its “zero-shot” means zero EAST shots for *classifier training*, not zero EAST data for normalization. §4.5.2/Table 4-9 reports AUC 0.892 for that particular PGFE-U/FLS/EFD study; it is not a result or expected target for the current raw23 model. Separate target-statistics provenance and train/evaluation disjointness should be checked in the original study before comparing protocols.

## 1. Exact direction and covariance equations

Let \(z\in\mathbb R^d\) be a **column vector in the exact coordinate system at the input of the frozen source classifier**, \(f_S(z)\). Let source and target calibration populations in that coordinate system have means \(\mu_S,\mu_T\) and symmetric positive-definite covariances \(C_S,C_T\). For unregularized full-rank matrices, a covariance-only target-to-source map with mean correction is

\[
\widetilde z_T=\mu_S+C_S^{1/2}C_T^{-1/2}(z_T-\mu_T),\qquad
\operatorname{Cov}(\widetilde z_T)=C_S.
\]

The order matters: first whiten the **target**, then colour it with the **source**; the displayed identity follows by direct multiplication. This is the reverse-direction counterpart of [P1], not P1's default training procedure. With row vectors the equivalent is \((z_T-\mu_T)C_T^{-1/2}C_S^{1/2}+\mu_S\). A mean correction is shown explicitly here; CORAL's derivation assumes normalized means. Matching first and second moments does not match the joint distribution or guarantee label preservation. Other covariance-matching maps exist, so the matrix order and coordinate convention must be specified.

In contrast, original CORAL's row-vector transformation is \(z_S C_S^{-1/2}C_T^{1/2}\), followed by **training** on the transformed source. Feeding untransformed EAST into the existing source classifier after computing this source-to-target map would not implement the frozen target-to-source candidate. Transforming the classifier itself or retraining it would be a different, out-of-scope intervention.

Rank deficiency, strong cross-channel correlations, few independent shots, or measurement-scale differences can make \(C_T^{-1/2}\) unstable. A regularized illustration is

\[
A_\lambda=(C_S+\lambda_S I)^{1/2}(C_T+\lambda_T I)^{-1/2}.
\]

It maps the **regularized target covariance** to the **regularized source covariance**, but in general \(A_\lambda C_T A_\lambda^\top\ne C_S\) for the actual unregularized data. Regularization changes the map and does not supply a validated choice of \(\lambda\). Full covariance also mixes physical channels and can amplify estimation error; axes and channel meanings must not be assumed preserved.

## 2. Label-mixture confounding and a limited SAFE-only alternative

For illustration, let \(Y\in\{0,1\}\) denote a **window-level** safe/disruptive label, \(\pi_D=P_D(Y=1)\), and \(\mu_{Dy},C_{Dy}\) be class-conditional moments in domain \(D\in\{S,T\}\). The law of total covariance gives

\[
\mu_D=(1-\pi_D)\mu_{D0}+\pi_D\mu_{D1},
\]
\[
C_D=(1-\pi_D)C_{D0}+\pi_D C_{D1}
 +\pi_D(1-\pi_D)(\mu_{D1}-\mu_{D0})(\mu_{D1}-\mu_{D0})^\top.
\]

Thus a change in label mixture alone can change the marginal mean and covariance even if both class-conditional distributions are unchanged. Conversely, matched marginal covariance does not prove either class is matched. This is a direct mathematical illustration of the conditional/mixture issue discussed in [P2–P3], not a measurement of the present data.

The requester reports cold20 as **8 disruptive / 12 safe shots**, and source TRAIN as **188 disruptive / 1,165 safe shots**. Those are shot-level fractions 8/20 = 40% versus 188/1353 ≈ 13.9%. They are **not** estimates of \(\pi_D\) above: window counts, shot length, time cuts, sampling, and weighting can all change the relevant mixture. Nor is the cold20 necessarily the eligible calibration pool. Do not use these fractions to infer the size or direction of window-level shift. Note also that [P4]'s own training split is 1,354 total/188 disruptive, not the requester's 1,353; do not silently identify the two datasets.

If “SAFE” is operationally defined and labels become available only after a discharge ends, estimate \(\mu_{S0},\sigma_{S0}\) from source TRAIN SAFE shots and \(\mu_{T0},\sigma_{T0}\) from **eligible previously ended** EAST SAFE shots. The same diagonal affine map is then applied to *all* future EAST windows, not just windows believed to be safe:

\[
\widetilde z_{T,j}=\mu_{S0,j}
 +\frac{\sigma_{S0,j}}{\max(\sigma_{T0,j},\epsilon_j)}
 (z_{T,j}-\mu_{T0,j}).
\]

Here \(j\) indexes a physical channel in its established order. The formula is a **task-specific proposal** inspired by moment alignment, not P4's supervised CORAL and not the dissertation's exact EFD recipe. With nonzero standard deviations and no active floor it matches the selected safe subset's empirical marginal means and variances; it does not match cross-channel covariance. An active floor, clipping, shrinkage, excluded channels, or missing-value handling breaks exact variance matching. The map preserves the identity of each axis, **not necessarily physical values, units, meaningful thresholds, or relationships**.

This SAFE-only choice removes one direct source of mixed safe/disruptive shot composition from the fit, but does **not** remove all selection bias. Stable phases of eventual disruptive shots can differ from completed safe shots; operating regimes, pulse lengths, missing diagnostics, and within-shot sampling may differ. In particular, using only completed SAFE shots says nothing by itself about \(P_T(z\mid Y=1)\). [P3]'s all-class conditional-invariance assumption and [P4]'s class-specific training alignment are therefore not established by this one SAFE-only map. Selecting a different transform using a live sample's true disruption outcome would be oracle leakage.

For a concrete estimation convention to pre-specify rather than infer from these papers: equal weight per eligible shot and equal weight per valid window within that shot gives

\[
\widehat\mu_{D0,j}=\frac1{N_D}\sum_{i=1}^{N_D}\frac1{n_i}\sum_{t=1}^{n_i}z_{itj},\quad
\widehat\sigma_{D0,j}^{2}=\frac1{N_D}\sum_i\frac1{n_i}\sum_t(z_{itj}-\widehat\mu_{D0,j})^2.
\]

These are descriptive weighted moments, not a claim of independent windows or an unbiased variance estimator. Apply a declared, compatible validity mask and phase definition in both domains. Equal-shot versus equal-window weighting changes the estimand; do not silently treat long EAST discharges as thousands of independent shots, and do not assert either weighting is inherently optimal.

## 3. Coordinate-system compatibility with raw23/asinh

The requester describes raw23 source training as an asinh center/scale pipeline. **Conditional on the actual implementation** being

\[
z_j=\phi_{S,j}(x_j)=\operatorname{asinh}\!\left(\frac{x_j-c_{S,j}}{s_{S,j}}\right),
\qquad s_{S,j}>0,
\]

the mathematically unambiguous candidate above fits source and target moments **after this same frozen \(\phi_S\)** and before the frozen classifier. The source scaler parameters \(c_S,s_S\) are not the SAFE-only moments \(\mu_{S0},\sigma_{S0}\). Do not refit or replace the source transform, apply asinh twice, change feature ordering, or align values in one space while estimating moments in another. The exact asinh expression, padding/missing-channel semantics, clipping, masks, and insertion point remain for the main code investigation to verify; this review has not inspected them.

The dissertation's EFD is different: §4.5.1 uses per-device **z-score** parameters, with J-TEXT statistics for source and estimated EAST statistics for target, after PGFE-U. In that setting separate per-domain standardization is algebraically related to per-channel target-to-source moment mapping before the source z-score. It is **not** a license to substitute target z-scoring for an already-trained source asinh transform or to claim equivalence after a nonlinear asinh. [P4, §§2.1 and 3–4](https://arxiv.org/pdf/2309.05361v2#page=5) concerns engineered PGFE features and separate per-device normalization, not these 23 raw channels.

## 4. What the tokamak evidence actually says

[P4, §2.1/Table 1](https://arxiv.org/pdf/2309.05361v2#page=5) designs related physical features for the two machines despite differing diagnostics; this is feature engineering for comparable physical phenomena, not proof that same-index raw channels are calibrated equivalently. [P4, §2.2/Fig. 2](https://arxiv.org/pdf/2309.05361v2#page=6) contrasts marginal U-CORAL with labeled, separate-class S-CORAL. [P4, §3 and Table 3](https://arxiv.org/pdf/2309.05361v2#page=10) uses 110 EAST training discharges, 10 disruptive, and reports AUC 0.797 for U-CORAL and 0.890 for S-CORAL in that study. It trains its classifier after alignment; it does not establish the efficacy of a frozen target-to-source SAFE-only map. Its warning-time comparison uses different model thresholds, including 0.31 for U-CORAL and 0.58 for S-CORAL, so it is not a same-fixed-threshold comparison.

Further, [P4, §5.1/Figs. 7–8](https://arxiv.org/pdf/2309.05361v2#page=12) describes the study's non-disruptive and disruptive feature distributions separately and reports that PGFE plus normalization aligned some but not all features. That directly cautions against assuming that alignment of safe features carries over to disruptive features. Its results are on PGFE-derived features and study-specific train/test settings, not on the requester's raw23 pipeline.

The dissertation's [§5.3.3](/media/hetaisheng/044A81D94A81C83E/Fuison/docs/literature/paper_131.txt:3560) reports an EFD sensitivity analysis for its own model and chosen perturbations. It is not a general robustness bound for asinh-transformed raw signals, a fixed alarm threshold, or this calibration sample.

## 5. Evaluation and leakage boundaries for the current task

These are methodological checks, not completed experiments:

- **Chronology:** for each scored shot, use only source TRAIN and target shots whose full discharge and SAFE status were available before that shot. Freeze each map for the entire scored shot. If no eligible target SAFE calibration exists yet, use the unchanged source pipeline and record that state. Earlier scored shots may inform later ones only after they end and their SAFE status is established under the allowed protocol; never use a shot's later outcome to map its own earlier windows. This is a prequential ended-feedback protocol, not an evaluation using a completely disjoint calibration set; declare which protocol is used. Do not compute statistics over all 360 shots or use future target shots.
- **Compare fairly:** unchanged frozen-source baseline versus the single pre-specified SAFE-only candidate, on the same chronological scored shots and with the **same fixed threshold**, same window/alarm logic, and no target-result-based threshold selection. If exploring multiple transforms, bounds or weightings, pre-specify a selection protocol and keep final evaluation separate; otherwise treat the comparisons as exploratory.
- **Measure both ranking and decisions:** report shot-level TPR/FPR and counts, missed/late alarms and warning-time distribution under the fixed threshold, along with the project's correctly defined AUC if applicable. AUROC alone does not determine fixed-threshold alarm quality. On 8 disruptive and 12 safe scored shots, one changed shot changes the corresponding empirical rate by 12.5 and 8.3 percentage points respectively; this is arithmetic, not a confidence interval. Use shot-level uncertainty or paired shot-level comparisons where feasible and disclose the small sample.
- **Inspect the actual fit:** record eligible shot IDs/timestamps locally, valid-window counts, per-shot weighting, per-channel sample moments and gains, any variance floors/bounds, missing-channel handling, and calibration dates. Check score traces, baseline alarm rate, and whether physical precursor excursions remain distinguishable. Any parameter floor, extreme gain, unstable estimate, regime drift, or new diagnostic configuration needs an explicit fallback or warning; the papers do not prescribe a safe numeric bound for this pipeline.

## 6. What the reviewed literature does **not** support

1. It does not show that target-to-source full CORAL or the diagonal SAFE-only candidate improves this frozen raw23 classifier on EAST, at any threshold. P1's preference for its own training direction is not a comparison under the frozen constraint.
2. It does not show that SAFE-only moment matching aligns the disruptive class, proves generalized label shift, calibrates probabilities, or preserves physically meaningful warning signals. P3's assumption concerns full class-conditional distributions, not just the first two SAFE moments.
3. It does not establish that PGFE-U/EFD is equivalent to the current raw-channel/asinh preprocessing, that the dissertation used no target data, or that its AUC transfers to cold20 or fixed-threshold performance.
4. It does not justify assuming away temporal dependence, converting shot-level mixture fractions into window-level prevalence, using current/future labels for live alignment, changing the source checkpoint, or fitting all 360 target shots.
5. This bounded search is not a comprehensive literature survey, a priority search, or evidence of novelty.

## Source-check and limitations log

| Pairing checked | Outcome in this review | Boundary |
|---|---|---|
| P1: source-to-target whitening/recolouring followed by source training | Supported by inspected text | Reverse direction for a frozen model is the mathematical adaptation described above, not an experimentally established advantage. |
| P2/P3: marginal invariance is insufficient in general; conditional distributions matter | Supported by inspected text | Do not turn their assumptions/bounds into a universal claim that any finite-sample CORAL map must hurt. |
| P4: class-specific alignment for J-TEXT → EAST | Supported by inspected author manuscript | It transforms training examples and trains a model; not live true-label gating or this frozen model. |
| Local thesis: EFD uses ten EAST disruptive shots for statistics | Supported by inspected §4.5.1 text extraction | “Zero-shot” refers to classifier training, not zero target statistics; original PDF/publisher copy not inspected. |
| SAFE-only diagonal map improves this project at the fixed threshold | **Unverified** | No relevant experiment was performed or supplied, and no reviewed paper directly tests it. |

**Rejected source–claim pairings: 3**, all `does_not_support_claim`: P1 as evidence that its original training-direction recipe works unchanged with a frozen source classifier; P4 as evidence that true live target class can select a map without leakage; and the local thesis as evidence that no EAST data at all entered its normalization. These are rejected pairings, not rejected papers.

**Candidate accounting:** four primary papers included; one additional potentially relevant newer title, “The role of data quality and alignment in cross-tokamak disruption prediction,” was found in indexing with the reported [DOI 10.1088/1361-6587/ae9985](https://doi.org/10.1088/1361-6587/ae9985), but its publisher record/full text could not be accessed. Its metadata and contents are **unverified here**; it is blocked/excluded, not counted among the four or used to support a claim. One user-specified local dissertation was inspected as contextual primary material, not counted as a separate paper. No abstract-only paper is included as verified evidence.

**Method and review limitation:** Searches used public web tools and official publisher/proceedings or author-deposited primary texts, with queries for CORAL direction, label/conditional shift, and J-TEXT/EAST PGFE/EFD. Selection required direct relevance and a readable primary text, within the requested 3–5-paper bound; it was not a PRISMA search and hit counts are not claimed. The source-check skill's independent fresh-context skeptic could not be run because no in-conversation child-agent interface was exposed; checks above are single-reviewer, source-anchored assessments, not independent confirmations. Fast/priority service was likewise unavailable. AI-assisted literature search and synthesis; no GPU job or project code change was performed.
