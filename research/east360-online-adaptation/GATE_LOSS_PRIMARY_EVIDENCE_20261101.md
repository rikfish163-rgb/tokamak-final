# Primary evidence for the within-shot gate loss

**Date:** 2026-11-01. **Status:** bounded, AI-assisted primary-source research and algebraic analysis; no experiment or code change was made. Six sources are included, not an exhaustive literature review or a novelty claim. This note is for the main agent to inspect independently.

## Bottom line

1. **The closest forecasting evidence does not establish the project's desired implication.** SAF adapts a jointly trained forecaster using an observed-history backcasting loss; TAFAS updates adapters on already arrived forecast targets while the forecaster is frozen. Both evaluate forecasting, not a frozen disruption classifier driven by a separate forecast head. Neither reports the project's true-versus-reversed 32-gate gradient test. [SAF](https://arxiv.org/html/2202.02403v3), Implementation and Algorithm 1; [TAFAS](https://arxiv.org/html/2501.04970v1), “Gated Calibration Module” and Eqs. 3–7.
2. **The gradient is the interface that must be tested.** Sun et al. establish task/auxiliary gradient alignment under specific assumptions and examine its association with adaptation benefit on corrupted images, not EAST. Agrawal et al. construct an explicitly order-sensitive proxy, but their experiment is offline representation pretraining on clinical notes, not forecasting TTT. Neither implies that reversing a 25-point target will change these particular gate gradients, or that an order-sensitive update improves an alarm. [Sun et al.](https://arxiv.org/html/1909.13231), §4 and Fig. 4; [Agrawal et al.](https://proceedings.mlr.press/v151/agrawal22a/agrawal22a.pdf), §§2–4 and 6.
3. **The current DC/AC and saturation checks are well motivated as diagnosis, not fixes established by the papers.** Under complete, equally weighted horizon data, squared forecast error splits exactly into time-mean and time-centered terms. Reversing only the target preserves the mean term and its gradient at the same gate state. For SmoothL1, exact additivity fails and saturation can make gradients identical even when values differ; see the derivation below. Near-identical directions can have several causes, and the reported cosine alone does not identify one.
4. **Keep persistence and the frozen parent in view.** A recent preprint reports that causal delay, a persistence reference, and backbone training materially affect forecasting-TTA conclusions; some methods help on one benchmark while failing to improve or harming others. This is a challenge to generalizing a reported improvement, not proof that the original TAFAS study leaked future observations. [Fentazi et al.](https://arxiv.org/html/2610.12232v1), §§3, 5.2, 6–7.

## Project boundary and current observations

The main agent supplied the context that the qualified J-TEXT classifier and its source forecast head are frozen; temporary within-shot adaptation acts through **32 internal gates**, using the frozen head's complete 25-by-23 future in source asinh coordinates. The reported current run had true-versus-reversed feedback gradient cosine about **0.99966**, approximately **10,910 gradients**, and no observed alarm benefit. These figures are **main-agent-supplied context, not results re-run or independently audited by this research sidecar**. They are not a human attestation. In particular, gradient count is not a count of independent shots or evidence of effective learning. Cosine near one does not establish equal gradient norms or equal classifier effects.

The main agent's in-scope work is the DC (per-channel mean-error) / AC (time-centered error) and Huber/SmoothL1 saturation investigation on **ended20**; no suffix training is authorized for this sidecar. The first 20 are part of full360, not a disjoint calibration cohort: their source-only predictions must be sealed before their outcomes are used for preparation. Any within-shot update uses only causally arrived signal feedback and resets at the shot boundary. The fixed source classifier, risk head, input contract, alarm threshold and lead-time policy stay unchanged. No proposed item below adds a risk-score correction, a confidence veto, a new alarm rule, or an online label oracle.

The two prior local notes were searched by title for all six papers below: none appears in either `META_ADAPT_PRIMARY_EVIDENCE_20261101.md` or `TIMESCALE_PRIMARY_EVIDENCE_20261101.md`. That is a bounded duplicate check, not a claim that these papers are new to the wider project. The earlier notes' cautions remain: the first-20 prepared parent may already depend on all 20, and neither the source/target label windows nor record length defines a fusion physical time scale.

## Six primary sources: method, observation, and transfer boundary

### 1. Forecasting TTT with masked-history backcasting

Sercan Ö. Arık, Nathanael C. Yoder and Tomas Pfister, **“Self-Adaptive Forecasting for Improved Deep Learning on Non-Stationary Time-Series,”** [author manuscript, arXiv:2202.02403v3](https://arxiv.org/html/2202.02403v3). Read “Principles behind self-adaptive forecasting,” Algorithm 1, “Implementation,” “Forecasting performance,” Table 4 and Appendix Algorithm 2. Cited version is the author preprint; no later publication status is inferred.

- **What they do.** Given a past input window, SAF masks its earlier part and fills it with the first remaining observed value, predicts the masked history from the later observed portion, and takes a gradient step on the encoder **and backcast decoder**. The adapted copy then forecasts using the complete observed window; optional backcast errors are fed as extra input features. The authors explicitly write that the original pretrained model is “not altered between different inference runs.” Their training procedure also optimizes the forecast task. This is per-inference adaptation, not the project's accumulated per-shot gate state.
- **What they show.** On their COVID/LSTM ablation in Table 4, the full method's test MAE is 793.5 versus 1157.7 without the encoder update, 1421.1 without the decoder update, and 941.2 without the error signal. The combined design matters in that experiment; it does not isolate a clean pure update-versus-zero-update for the fixed classifier here. Their synthetic Table 1 contains both improvements and small degradations, depending on process/model.
- **What transfers / does not.** The observation that the proxy is actually used in an update and the comparison of adapted versus unadapted components are relevant. Their backcast learns from a later observed segment to an earlier segment; it is **not** an experiment that keeps the same forecast and merely reverses its 25 target values. Its jointly trained forecast decoder, optionally added error features, full encoder update and per-window reset are not an unchanged qualified J-TEXT risk classifier. We do not suggest adding the error features to the classifier.

### 2. Frozen forecast backbone with matured and partial targets

HyunGi Kim, Siwon Kim, Jisoo Mok and Sungroh Yoon, **“Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation,”** AAAI 2025, 39(17), 17868–17876. [Official record and DOI](https://ojs.aaai.org/index.php/AAAI/article/view/33965); [author full text, arXiv v1](https://arxiv.org/html/2501.04970v1). Read “Challenges and Opportunities in TSF-TTA,” “Periodicity-Aware Adaptation Scheduling,” “Gated Calibration Module,” “Prediction Adjustment,” Eqs. 1–7 and Table 1; Appendix “Component-wise Ablation Studies.”

- **What they do.** For an `H`-step forecast, the first `p` values can be used only after those `p` observations arrive. Their PAAS chooses `p` from an FFT of a mean-centered *observed lookback*; two gated modules calibrate inputs and outputs around the **frozen forecasting backbone**. The input form is `X + Tile(tanh(α)) ⊙ (Concat_c(W^c X^c) + b)`; the paper initializes `W^c` and `b` to zero. Its adaptation loss is `L_partial + L_full`, consisting of MSE on an arrived prefix and on older fully matured targets. It recalculates the still-unobserved portion of past-origin **forecasts** after adaptation, not already issued historical ground-truth values.
- **What they show.** Table 1 compares forecasting MSE with and without TAFAS; the component ablation changes the adaptation mechanism as well as removing components. The intended adaptation target is the *same forecasting task* measured in the paper, not a separately frozen event classifier. It is inaccurate to equate its input/output calibration modules with 32 internal gates or to call the frozen source model's outputs invariant: adapting modules can change the composite function while the backbone weights are fixed.
- **Boundary.** Use of partial matured feedback is an available causal concept, not a recommendation to change this project's specified complete-25 feedback contract. Its revised unobserved forecast is not permission to revise past EAST alarms. The periodicity schedule is based on the paper's data and is not an EAST period, a target label window, or a physically established clock. This paper does not report a target-reversal gate-gradient or downstream alarm ablation.

### 3. A causal-delay audit with an actual persistence reference

Mohamed Readh Fentazi, Mazene Ameur and Adlen Ksentini, **“Training on the Future: A Delay-Aware Audit of Test-Time Adaptation for Time-Series Forecasting,”** [arXiv:2610.12232v1, 8 October 2026](https://arxiv.org/html/2610.12232v1). Read §3 “The Causal Delayed Evaluation Protocol,” §5.1, §5.2 and Table 2, §6, §7, and Appendix A/F. This is a recent **preprint**, and its runs or patches were not independently reproduced here.

- **What they do.** The audit sets `H=24` and requires a complete target window from origin `s` to be released no earlier than `s+H`, optionally later; its partial-target procedures use only the observed prefix. It runs four published gradient-adapter methods with modified maturity handling and compares them with frozen forecasts, persistence, seasonal-naive, and its own closed-form alternatives. The deliberately leaky `d=1` setting is an audit reference, **not evidence that a particular cited method necessarily used that setting in its original paper**.
- **What they report, with counterexample.** In their Table 2, on ETTh2 at minimum delay, persistence MSE is 1.1196, frozen PatchTST 1.3728, and TAFAS 1.3864; the authors say TAFAS is statistically indistinguishable from frozen and that no published method is significantly better than persistence there. On ETTm1, in contrast, all four published methods improve over frozen in their reported audit, with differing significance across seeds. Weather/Electricity/Traffic use one checkpoint; the ETT results have three seeds at minimum delay. The authors use a nonstandard 25/5/70 split, so absolute values should not be compared with the original papers' tables.
- **Boundary.** This establishes a forecast evaluation counterexample under the audit's settings, not a universal failure of TTA, proof of a bug in TAFAS, or a result for EAST. Its output correctors are **not** proposed as risk-score add-ons. The useful transplant is a same-origins, same-mask persistence comparison, plus explicit maturity and paired-shot accounting.

### 4. Task-gradient alignment is a condition, not a synonym for proxy improvement

Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros and Moritz Hardt, **“Test-Time Training with Self-Supervision for Generalization under Distribution Shifts,”** ICML 2020, PMLR 119:9229–9248. [Official record](https://proceedings.mlr.press/v119/sun20b.html); [author full text](https://arxiv.org/html/1909.13231), §§2–4, especially Eqs. 4–11, Theorem 1 and Fig. 4.

- **What they establish.** They jointly train an image classifier and a self-supervised rotation head on a shared feature extractor, then adapt the shared extractor using the self-supervised loss. In a smooth convex setting with bounded gradients and a specified step size, a sufficiently positive task/auxiliary gradient inner product ensures a one-step main-task loss reduction. Their Fig. 4 plots the task/proxy inner product against performance gains across corruption datasets and reports positive empirical correlations. The paper explicitly frames the condition as “positive gradient correlation between the loss functions.”
- **Boundary.** The proof's convexity, smoothness, boundedness and step-size conditions are not established for the nonlinear fixed Transformer and gate coordinates. Their image experiment is not a time-series experiment. The observation is about gradients of the **main task and proxy**, not about the cosine of two versions of the proxy. Thus `cos(g_true,g_reverse)≈1` is concerning for this specific order control but says nothing by itself about `g_alarm·g_true`, and decreasing forecast loss is not proof of alarm gain.

### 5. Mean/variance normalization can erase relevant nonstationarity

Yong Liu, Haixu Wu, Jianmin Wang and Mingsheng Long, **“Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting,”** NeurIPS 2022. [Official conference record](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html); [conference paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/4054556fcaa934b0bf76da52cf4f92cb-Paper-Conference.pdf), §§3.1–3.2, Eqs. 1–6, §4.2 Table 5 and §4.3 Table 6.

- **What they do.** For a lookback of length `S`, they form per-variable `μ=(1/S)Σ_i x_i`, `σ²=(1/S)Σ_i(x_i−μ)²`, feed `(x_i−μ)/σ` to a forecaster, and restore statistics on its output. Their de-stationary attention learns factors from the unnormalized series/statistics and uses `softmax((τ Q'K'^T + 1 Δ^T)/sqrt(d_k)) V'`. Their analysis first assumes linear pointwise embedding/feed-forward maps and a shared scalar variance; the learned network is an approximation beyond those simplifying assumptions.
- **What they report.** The paper notes that original nonstationarity “cannot be fully recovered only by De-normalization,” since distinct shifted/scaled series can produce the same normalized input and attention. In Table 5, for example, the Transformer Exchange average MSE is 0.569 with Series Stationarization and 0.461 with their complete design. The model and normalizer are trained together for forecasting; this is not an online gate-loss ablation.
- **Boundary.** A warning against discarding level information is relevant to EAST, but a time-centered **forecast error** is not the same operation as normalizing the classifier input. DC is not automatically nuisance, AC is not automatically physics, and subtracting a 25-point mean does not prove stationarity or separate trend from dynamics. Do not normalize the already-qualified classifier input on the authority of this study.

### 6. An order-sensitive proxy can learn the wrong information

Monica N. Agrawal, Hunter Lang, Michael Offin, Lior Gazit and David Sontag, **“Leveraging Time Irreversibility with Order-Contrastive Pre-training,”** AISTATS 2022, PMLR 151:2330–2353. [Official record](https://proceedings.mlr.press/v151/agrawal22a.html); [full paper](https://proceedings.mlr.press/v151/agrawal22a/agrawal22a.pdf), §§2–4 and §6.

- **What they do.** For each series they sample ordered segment pairs `(X_W, X_{W+1})` and reversed pairs `(X_{W+1}, X_W)`; a representation and classifier predict which order is correct. Their method is explicitly designed to “predict whether the tuple Z is in the correct order.” In synthetic experiments and small-labeled-data clinical-note progression extraction, the learned representation can outperform alternatives; representations are then frozen and a downstream classifier is fitted.
- **Crucial qualification.** Their theory assumes a restricted feature-selector setting with time-irreversible features, reversible background under stated conditions, nonredundancy, **and** that the irreversible features suffice for the downstream task. The authors warn in §6 that dates and task-irrelevant global nonstationarities can be shortcuts and that not all tasks are irreversible. Their downstream task and input data are not forecasting-TTT or plasma signals, and they do not test the gradient of a reversed H25 forecast target. Ordering may be detectable while reflecting acquisition drift, shot age or benign evolution rather than disruption.

## Algebra for the frozen forecast head and the 32 gate coordinates

The following is **our derivation**, not a reported result from the six papers. Let `a∈R^32` denote the actual temporary gate coordinates. At origin `s`, hold the past input `x_s`, gate state `a`, head, valid-channel mask and all weights fixed. Let `H=25`, let `C` be the number of channels with a complete valid target under the existing contract, and define, in the existing source-asinh coordinates,

```text
ŷ_hc = frozen_forecast_head(frozen_source_with_gates(x_s, a))_hc
e_hc = ŷ_hc − y_(s+h),c                  J_hc = ∂ŷ_hc / ∂a ∈ R^32
ē_c  = (1/H) Σ_h e_hc                   e^AC_hc = e_hc − ē_c
J̄_c  = (1/H) Σ_h J_hc
```

The model's nominal output is 25×23; use the same eligible channels for all terms and all controls, not missing values filled as observed zeros. Any other preprocessing, weighting, non-complete mask, reduction or gate parameterization must be taken from the actual implementation before applying these identities.

### What a square-loss decomposition proves

For equal time weights within each eligible channel,

```text
L_MSE = (1/(H C)) Σ_c Σ_h e_hc²
      = L_DC + L_AC
L_DC  = (1/C) Σ_c ē_c²
L_AC  = (1/(H C)) Σ_c Σ_h (e_hc − ē_c)²

g_DC  = (2/C) Σ_c ē_c J̄_c
g_AC  = (2/(H C)) Σ_c Σ_h (e_hc−ē_c)(J_hc−J̄_c).
```

Let `y^R_hc = y_(H+1−h),c` reverse **only** the already completed target, while holding `x_s`, `a`, `ŷ`, `J` and the channel mask fixed. Because `mean_h(y^R)=mean_h(y)`, `L_DC` and `g_DC` are exactly unchanged. Moreover,

```text
g_MSE,true − g_MSE,reverse
  = −(2/(H C)) Σ_c Σ_h J_hc [y_hc − y_(H+1−h),c]
  = −(2/(H C)) Σ_c Σ_(h=1..12)
        (J_hc − J_(H+1−h),c) (y_hc − y_(H+1−h),c).
```

The center position of odd `H=25` contributes zero. Therefore a real-vs-reversed difference requires both target asymmetry **and** a horizon-varying gate-to-forecast Jacobian that does not cancel across channels. For example, if every `J_hc` is horizon-constant within a channel, reversal changes no square-loss gradient, even with a strongly varying target. This is a sufficient failure mechanism, **not an empirical diagnosis of the current run**. A strong common DC component can also make full-gradient cosines high while a smaller AC component differs; conversely, equal directions need not imply equal magnitudes. For any MSE weighting, robust loss, partial mask, or rollout that violates the assumptions, recompute the correct expression rather than assuming the identity remains exact.

### Why SmoothL1 can suppress order differences

If the implementation uses PyTorch-style SmoothL1 with transition `β>0`, then `ρ'_β(e)=clip(e/β,−1,1)` and

```text
g_SmoothL1 = (1/(H C)) Σ_c Σ_h J_hc ρ'_β(e_hc).
```

For a channel, if **both** true and reversed target assignments produce residuals all beyond `β` on the same side of zero, their derivatives have the same constant sign and that channel contributes exactly the same gradient under both orders, even if its targets are not constant. A large saturated fraction alone is not sufficient: signs, the pairing with `J_hc`, masks and per-channel weights matter. Confirm the actual loss definition, `β`, reduction and precision; the above is not a claim that the run uses any particular `β`.

Importantly, `Σρ(e_hc)` generally **does not equal** `ρ(ē_c)+Σρ(e_hc−ē_c)`. A loss that separately applies robust penalties to DC and AC is a *different objective*, not an exact decomposition of the existing SmoothL1. Do not interpret a larger AC gradient as a proven improvement: reweighting or scaling may magnify noise, and an apparently different direction can remain invisible to the risk head.

### Why a useful forecast gradient still may not move the alarm correctly

For a later causally available classifier input `x_u`, denote its fixed-head logit by `z_a(x_u)`. For an actual gate displacement `Δa` after all clipping/projection, the local change is `Δz ≈ ∇_a z_a(x_u)·Δa`. Thus an auxiliary gradient in a locally risk-insensitive direction may reduce forecast loss and barely move the logit.

Retrospectively on ended20 only, one may define a differentiable task loss with the *existing valid retrospective labels*, `g_task=∇_a ℓ_task(z_a(x_u), y_u)`, evaluated at the matching gate state. With an unprojected small step `Δa=−ηg_aux`, `Δℓ_task≈−η g_task·g_aux`. With projection, clipping, optimizer state, or a nonlinear finite step, use the **actual** `Δa` and check the realized task-loss/logit change; the Taylor expression is not a safety guarantee. Labels and outcome-derived times are for this after-shot diagnostic only, never a running-shot update. Even favorable BCE/logit movement does not necessarily improve the existing fixed-threshold, lead-time-aware shot-level alarm.

## Minimal diagnostic readout for the main agent

This section is an **unexecuted proposal**, aligned with the already stated ended20 check; no extra training on suffix340 is proposed.

1. **Fork at the same state.** At matched origin/maturity and identical `a`, compare the true and reversed completed feedback with the same complete-channel masks. Record both gradient norms, cosine where norms are nonzero, `||g_true−g_reverse||/(||g_true||+||g_reverse||+ε)`, and the actual clipped/projection-adjusted displacement. For MSE, separately record DC/AC gradient norms, cosine and their sum; validate the exact decomposition numerically. A separately rolled-out reversed arm is useful operationally but is *not* the same-state attribution because its gate trajectory diverges.
2. **Test the proposed mechanisms.** Per shot and per eligible channel, summarize `|ē|`, time-centered residual energy, target reversal asymmetry, horizon differences in `J`, and SmoothL1 fractions inside/outside `β` and same-sign saturation. Compare the observed gradient difference with the square-loss paired-Jacobian identity. Small target asymmetry, small `J` differences, cross-channel cancellation, DC dominance and saturation are competing explanations; none should be asserted without measurements. If gradients are numerically zero or feedback is unavailable, mark the comparison uninformative.
3. **Compare against persistence without changing the risk path.** At the same origin and eligible channels, forecast each future as the last observed source-asinh value, `ŷ^hold_hc=z_sc`. Report per-shot masked MSE and, separately, DC/AC errors for the frozen forecast and persistence; compare on identical origins and masks. The first difference is an observed **forecast skill** diagnostic, not a new classifier input or alarm score. Persistence can be optimal under a zero-conditional-mean random-walk innovation model for squared loss, but that assumption is not established for these plasma features.
4. **Inspect task coupling only after outcomes are revealed.** On ended20, compare proxy update directions with fixed-head task gradients and with realized later logit/loss changes at the same states; report distributions by shot rather than treating thousands of windows as independent. The ended20 parent may already have been fit using all 20; any folds made afterward are conditional on that parent, not an independent end-to-end holdout. No outcome-dependent action may be applied within a replayed live shot.
5. **If the main agent later chooses an update test**, predeclare any DC/AC weight and any fixed per-channel scale using allowed source/ended20 data only; compare original loss, the chosen alternative, no update, and matched true/reverse feedback on the same shots. Preserve source/forecast/risk-head weights, preprocessing, fixed threshold, lead-time rule and per-shot reset. Report gradient/step magnitudes, forecast skill, fixed-policy shot-level TPR/FPR and lead times separately; a forecast-only improvement or order sensitivity without alarm benefit does not pass the task goal. Any later suffix evaluation is evaluation only, with no suffix training and no retrospective revision of prior alarms.

## Source coverage and verification limits

- **Discovery:** public Web and Exa searches used combinations of `test time training time series forecasting`, `forecast adaptation delayed labels`, `time reversal order contrastive`, `nonstationary transformer normalization`, and `persistence baseline`. URLs above point to author manuscripts, official proceedings or the primary audit preprint. Six included sources had their relevant full method/protocol sections inspected; this does **not** mean every appendix, experiment, code repository or later version was audited.
- **Breadth:** 3/6 directly study forecasting TTT/TTA or its causal evaluation (SAF, TAFAS, Fentazi); 1/6 is forecasting normalization (Liu); 1/6 is task-gradient theory and image evidence (Sun); 1/6 is order-sensitive clinical pretraining (Agrawal). Neither of the last two is represented as forecasting adaptation. This is a relevance-selected shortlist, not a systematic search, comprehensive evidence inventory or proof of novelty.
- **Claim-to-source challenges (inline, not independent):** rejected four initially tempting pairings: (i) SAF is a true-versus-reversed H25 gradient study; (ii) TAFAS freezes the entire composite prediction function; (iii) high auxiliary order sensitivity itself guarantees task alignment; and (iv) demeaning proves plasma stationarity or identifies safe nuisance. All six papers were retained for narrower claims; the four rejections are **claim-pairing rejections**, not rejected papers. The additional spatial-temporal bidirectional-cycle TTT paper was considered but not included in the six: its jointly trained, multi-location architecture is less directly diagnostic of the present frozen gate interface; no claim is made that it is scientifically invalid.
- **Independence:** no fresh-context skeptic or separate agent was available in this sidecar. The above checks were performed in the same context as discovery and must not be described as an independent adversarial confirmation; the main agent's requested independent inspection remains outstanding. Publication/retraction status beyond the linked records was not exhaustively audited.
- **Algebra sanity check:** a deterministic, synthetic in-memory calculation with `H=25`, three channels and 32 coordinates checked the MSE decomposition, its gradient split, reversal-invariant DC gradient, and the paired-horizon reversal formula; the largest absolute residual was below `4.6e-16`. A constructed all-same-sign saturated SmoothL1 example gave zero gradient difference after target reversal. This checks the stated identities in one numerical construction, not their relevance to EAST, actual masks, or the current model.
- **Safety and scope:** no private data or workspace text was uploaded in a search; queries used public generic terms. No GPU, model/data experiment, code edit, threshold change, suffix training, or external deployment was performed. The `25/125 ms` supervision windows supply no inferred fusion time-scale ratio.
