# Independent verification: forecasting adaptation and label timing

**Checked:** 2026-10-31. **Role:** fresh-context skeptic. I checked the five supplied claim–paper pairings against the papers and their primary bibliographic records, without reading the excluded proposer report. A CONFIRMED verdict means that the paper supports the stated, scoped description; it is not an independent replication of its experiments or an audit of its implementation. Each listed excerpt is verbatim; the surrounding analysis is my own paraphrase or inference.

## Resolution log

| # | Pairing | Verdict | What the primary source establishes |
|---|---|---|---|
| 1 | Kim et al. 2025, TAFAS | CONFIRMED | Frozen forecaster, trainable gated input/output calibration, and adaptation using an observed prefix and, when available, matured full labels. |
| 2 | Arık, Yoder and Pfister 2022, SAF | CONFIRMED | Backcasting masks part of the observed lookback, updates a per-instance encoder/backcast decoder copy, and uses a matching adaptation stage in training. |
| 3 | Liu et al. 2022, Non-stationary Transformers | CONFIRMED | Output de-normalization is paired with learned de-stationary attention; the paper argues that output restoration cannot by itself restore distinctions lost inside attention. |
| 4 | Sun et al. 2020, TTT | CONFIRMED | Theorem 1 provides a conditional single-step loss result; the rotation auxiliary task does not require the test class label. |
| 5 | Fentazi, Ameur and Ksentini 2026, delay-aware audit | CONFIRMED | The preprint exists with the stated date and authors; its full-window delay, available-prefix and pretraining-statistics descriptions are supported. The audit alone is not proof that original TAFAS leaks. |

## 1. TAFAS: frozen backbone, calibration and observed labels

**Metadata.** HyunGi Kim, Siwon Kim, Jisoo Mok and Sungroh Yoon, “Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation,” *Proceedings of the AAAI Conference on Artificial Intelligence* 39(17), 17868–17876 (2025), DOI [10.1609/aaai.v39i17.33965](https://doi.org/10.1609/aaai.v39i17.33965); [arXiv:2501.04970v1](https://arxiv.org/abs/2501.04970), submitted 9 January 2025. The AAAI publisher landing page denied the direct fetch; the publisher's indexed record supplied its bibliographic fields, and its [conference PDF](https://ojs.aaai.org/index.php/AAAI/article/download/33965/36120) and the author-deposited full text were accessible in the retrieval system.

**Primary excerpts.** Figure 2 says: “The source forecaster is frozen”. The PAAS discussion states: “we must wait for p time steps.” [arXiv v1, Figure 2 and PAAS](https://arxiv.org/html/2501.04970v1); conference PDF pp. 17870–17871.

**What was actually read and checked.** In the [paper's challenges, PAAS, GCM and Prediction Adjustment sections](https://arxiv.org/html/2501.04970v1), including Figure 2 and equations (4)–(7), a forecast made at time `t` can use its first `p < H` target observations only at `t+p`. Input and output gated calibration modules are adapted while the source forecaster remains frozen. The loss can combine that observed-prefix error with a *past minibatch whose entire ground truth has already matured*; initially the partial loss is used alone. These statements concern the method as described, not a line-by-line certification of its released code.

**Important scope restriction.** Prediction Adjustment recomputes and substitutes only the still-unobserved suffix after the later adaptation time. Such a revised forecast has a later information vintage and shorter effective lead time than the original issuance. It must not be scored as though the revised whole horizon was available at the original `t`. This is a comparison-design warning, not a finding of leakage. No code or prediction artifacts were inspected.

## 2. SAF: reconstruct an observed lookback and adapt before forecasting

**Metadata.** Sercan Ö. Arık, Nathanael C. Yoder and Tomas Pfister, “Self-Adaptive Forecasting for Improved Deep Learning on Non-Stationary Time-Series,” [arXiv:2202.02403v3](https://arxiv.org/abs/2202.02403), first submitted 4 February 2022, v3 dated 26 September 2022; arXiv DOI [10.48550/arXiv.2202.02403](https://doi.org/10.48550/arXiv.2202.02403). This verification uses the preprint record, without asserting an unverified conference publication.

**Primary excerpt.** “SAF integrates a self-adaptation stage prior to forecasting based on ‘backcasting’ the masked inputs, a form of test-time training.” [arXiv v3, Conclusions](https://arxiv.org/html/2202.02403v3).

**What was actually read and checked.** [Algorithm 1 and Implementation](https://arxiv.org/html/2202.02403v3) define the complete observed input window through time `t` and mask its older part; the newer, unmasked part provides the input for backcasting. The masked values are replaced with repeated values from the unmasked part rather than assumed to be missing from the real data. The backcast loss updates the encoder and backcast decoder before a forecast is generated. The described experiments use one gradient step; different inference runs use separate copies of the pretrained weights. The [Appendix, Algorithm 2](https://arxiv.org/html/2202.02403v3) performs backcast adaptation during training before forecasting and supervised forecast-loss updates.

**Scope restriction.** The self-supervision uses values already present in the lookback, not future target labels. The paper's concrete construction is backcasting from the later observed part toward the earlier observed part; it is not evidence that arbitrary masking directions, continual accumulation across instances, or this particular task will improve a disruption alarm. Matching the adaptation stage in training is more specific than merely adding an unrelated auxiliary training loss.

## 3. Non-stationary Transformers: restore statistics and attention information separately

**Metadata.** Yong Liu, Haixu Wu, Jianmin Wang and Mingsheng Long, “Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting,” *Advances in Neural Information Processing Systems* 35, main conference track (2022), [NeurIPS proceedings record](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html), DOI [10.52202/068431-0718](https://doi.org/10.52202/068431-0718). The [arXiv record](https://arxiv.org/abs/2205.14415) was first submitted in 2022; the separately examined HTML is v4, revised in 2023.

**Primary excerpt.** “De-stationary Attention to re-incorporate non-stationary information of raw series.” [arXiv v4, §3](https://arxiv.org/html/2205.14415v4); compare [2022 proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/4054556fcaa934b0bf76da52cf4f92cb-Paper-Conference.pdf#page=4), §§3.1–3.2.

**What was actually read and checked.** In the [full text, §§3.1–3.2, Figure 2 and equations (1)–(6)](https://arxiv.org/html/2205.14415v4), window statistics normalize the input and restore the scale/location of the output. The authors explain that different raw series can share the same normalized input and thus produce identical internal attention even when output statistics are restored. Their projector learns `log τ = MLP(σ_x, x)` and `Δ = MLP(μ_x, x)` from the raw input and its statistics; these factors modify attention logits.

**Scope restriction.** This is the paper's analysis and approximate architectural remedy, not a theorem that output de-normalization is useless, nor that the learned factors recover every original attention exactly. The analytic derivation makes simplifying linearity and shared-variance assumptions, whereas the implemented projectors approximate the factors for deep models. It does not establish any guaranteed disruption-alarm improvement.

## 4. TTT: theorem conditions and rotation labels

**Metadata.** Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros and Moritz Hardt, “Test-Time Training with Self-Supervision for Generalization under Distribution Shifts,” *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119:9229–9248 (2020), [PMLR publisher record](https://proceedings.mlr.press/v119/sun20b.html).

**Primary excerpt.** “Theorem 1 reveals gradient correlation as a determining factor of the success of Test-Time Training in the smooth and convex case.” [PMLR PDF, §4, Theorem 1, PDF page 8](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf#page=8).

**What was actually read and checked.** The [published theorem](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf#page=8) assumes the main loss is differentiable, convex and `β`-smooth in the parameters, and bounds the norms of both main and auxiliary gradients by `G`. For a positive margin `ε`, it uses the fixed step `η = ε/(βG²)` and requires the per-instance gradient inner product to exceed `ε`; then one auxiliary gradient step strictly decreases that instance's main loss. The [authors' arXiv v3 HTML, §4 and Appendix A3.2](https://arxiv.org/html/1909.13231) was also read to resolve symbols that were not cleanly extracted from the PDF. The [PMLR PDF, §2](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf#page=2) describes rotations by known multiples of 90° and predicts the applied rotation as the auxiliary label; test-time adaptation does not use the image's ground-truth class label.

**Scope restriction.** Gradient positivity alone is shorthand, not the theorem's complete premises or step-size rule. The main-task gradient in the theorem depends on the true label, so its alignment condition is not automatically observable at deployment. The result is for one step and the stated loss assumptions, not a guarantee for a nonconvex Transformer, future multi-step adaptation, classification AUC/TPR, or a plasma disruption alarm. Test class labels may still be used separately to evaluate accuracy; they are not used to form the rotation update.

## 5. 2026 delay-aware audit: existence, label release and normalization

**Metadata and existence.** Mohamed Readh Fentazi, Mazene Ameur and Adlen Ksentini, “Training on the Future: A Delay-Aware Audit of Test-Time Adaptation for Time-Series Forecasting,” [arXiv:2610.12232v1](https://arxiv.org/abs/2610.12232), submitted **8 October 2026 at 16:16:58 UTC**, as shown in the arXiv submission history. The arXiv page displays DOI `10.48550/arXiv.2610.12232` with registration marked pending at access. This is a real, accessible preprint; no peer-review status is asserted here.

**Primary excerpts.** The paper says “standardization uses pretraining statistics only” and, of the lag patch, “At ℓ=0 the patched behavior is bit-identical to the authors’ code.” [arXiv v1, §3 Setup and Appendix A.2](https://arxiv.org/html/2610.12232v1).

**What was actually read and checked.** I read the [arXiv record](https://arxiv.org/abs/2610.12232) and [v1 full text](https://arxiv.org/html/2610.12232v1): Introduction; §3 Setup, Delay semantics and Audited methods; §5 scope; §7 limitations; Appendix A.1–A.4. The audit releases the *complete* target window for origin `s` at `s+d`, with `d ≥ H`; `d=H` is minimum full-window maturity. For `d>H`, extra label lag is `ℓ=d−H`. The usable partial target prefix is shortened accordingly, so already arrived individual observations need not wait for the entire window. The paper's deliberately nonstandard 25% pretraining / 5% validation / 70% online split uses only pretraining statistics for standardization. Its `d=1` full-window reference is leaky for `H>1`; that does **not** mean an update using just the first arrived label is automatically leaky.

**TAFAS-specific inference and restriction.** The paper's Introduction cites Kim et al. among work correcting the earlier leaky protocol, and Appendix A says that at zero additional lag the patched behavior is bit-identical to the reference code it audited. The paper's controlled leaky full-window references and its delayed-label evaluation do not by themselves establish that the original TAFAS method or original TAFAS repository leaked. Appendix A identifies the audited TAFAS implementation as bundled in the COSA reference repository; do not silently equate that with an independent inspection of Kim et al.'s original repository. Conversely, neither the paper's description nor the present document certifies that every implementation, benchmark, release or score is leak-free. I did not inspect or execute the repositories or cached experiments.

## Warnings carried forward

- Do not cite TAFAS as having immediate access to an entire future target window, or silently treat its later revised suffix as an original fixed-vintage forecast.
- Do not describe SAF's per-instance weight-copy updates as necessarily cumulative online training, or its masked observed history as a future label.
- Do not turn Non-stationary Transformers' approximation into exact attention recovery, or Sun et al.'s conditional convex one-step theorem into a general nonconvex/alarm-performance guarantee.
- Do not cite the 2026 audit as proving that TAFAS leaked. Also do not extrapolate its pretraining-only standardization and 25/5/70 split to all forecasting studies. Its experiments have stated architecture, dataset, seed and compute limitations and were not reproduced in this verification.

## Counts, limitations and paths

- Proposed claim–paper pairings checked: **5**. Verdicts: **5 CONFIRMED, 0 PARTIAL, 0 REJECTED, 0 BLOCKED**. Multiple versions and publisher records for the same paper are counted as one pairing, not as extra proposals. There is no invented rejection quota.
- Rejected pairings: **0**; rejection classes `does_not_support_claim`, `metadata_wrong`, `wrong_site_or_population`, `not_primary_but_required_to_be`, and `fabricated`: **0 each**. The warnings above reject overbroad extrapolations, not the five scoped claims. The inaccessible AAAI landing page was not treated as an inaccessible paper because the primary PDF and author full text were available; its metadata was obtained from the publisher's indexed record.
- No experiment was rerun; no code, data, checkpoints, or predictions were audited. The original papers' performance findings are author-reported, not independently validated here. Browser extraction can omit typographic details; exact sections/equation numbers and versioned source links are supplied for inspection.
- Output: `/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/LEVEL_CHANGE_LITERATURE_VERIFICATION_20261031.md`. The explicitly excluded proposer report was not read. No other artifact was intentionally created or changed.
