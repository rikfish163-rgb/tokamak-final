# Internal fast memory: bounded literature and design note

Date: 2026-11-04. Scope: six primary papers, their exact relevant locations,
and one **unimplemented hypothesis** for the existing frozen Simple23 model.
This is not a novelty, exhaustive-search, performance, physical-mechanism, or
independent-validation claim. I read the linked primary passages and checked
authors, titles, publication status, and versions against publisher/arXiv
records. The source check was performed by the same assistant; no independent
reviewer or child-agent check was available. No raw data, model inference,
training, or GPU was used for this note.

## 1. What the primary sources actually say

### 1.1 Learn the self-supervised task and a fast weight state

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang,
Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto,
and Carlos Guestrin, **Learning to (Learn at Test Time): RNNs with Expressive
Hidden States**. First arXiv version 2024; ICML 2025, PMLR 267, 57503–57522.
Read [arXiv v1, §2.1–2.4 and §2.7](https://arxiv.org/html/2407.04620v1)
and checked the [official conference record](https://proceedings.mlr.press/v267/sun25h.html).
These are two versions of **one** paper, not two independent studies.

- §2.1, Eqs. (1)–(3) makes the recurrent state the weights `W` of a small
  model and updates it with an unlabeled loss:
  `W_t = W_(t-1) - η ∇_W ℓ(W_(t-1); x_t)`. The section's first illustration
  reconstructs `x_t` from a corrupted view.
- §2.3, Eqs. (4)–(5) learns the training, target and query projections:
  `ℓ(W;x_t) = ||f(θ_K x_t;W) - θ_V x_t||²` and
  `z_t = f(θ_Q x_t;W_t)`. The projection/task parameters belong to the
  outer loop; sequence-specific `W_t` is the inner state. §2.7 also learns
  initialization and step-size-related parameters.
- **Transfer and limit:** the useful idea is learning what to write/read into
  a small recurrent weight state, not assuming any reconstruction objective is
  helpful. The published experiments are language modeling, not tokamak
  disruption alarms. The paper's `z_t` uses the current updated `W_t`; the
  proposed schedule below deliberately predicts first and writes later using
  matured real observations. It is inspired by TTT, not a reproduction or
  evidence that this particular state improves FPR.

### 1.2 Self-supervised adaptation inside an observed time-series window

Sercan O. Arik, Nathanael C. Yoder, and Tomas Pfister, **Self-Adaptive
Forecasting for Improved Deep Learning on Non-Stationary Time-Series**,
arXiv:2202.02403v3, 2022, preprint. Read [§“Principles behind self-adaptive
forecasting”, Algorithm 1 and §“Implementation”](https://arxiv.org/html/2202.02403v3).

- Algorithm 1 masks an earlier segment of the **already observed** lookback,
  backcasts it from a later observed segment, and applies a backcast-loss
  gradient to both encoder and backcast decoder. Its encoder update is
  `Ω_e^(m) = Ω_e - α ∇_(Ω_e) L(b, x[t-m+1:t])`; a separate forecast
  decoder then predicts the future. The later segment is still no later than
  the forecast origin `t`.
- **Transfer and limit:** the paper motivates an internal-representation
  self-supervision task and training for the adaptation procedure. It does
  **not** keep its encoder frozen like the present requirement; its backcast
  loss is not our matured multi-horizon future-observation loss. Its temporal
  forecasting experiments do not establish disruption-warning benefit or
  disentangled operating context.

### 1.3 Patched representation and direct multi-horizon forecast

Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam,
**A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**,
ICLR 2023, arXiv:2211.14730v2. Read [§3.1 Model Structure and §3.2
Representation Learning](https://arxiv.org/html/2211.14730v2); the
[arXiv metadata](https://arxiv.org/abs/2211.14730) records acceptance.

- §3.1 forecasts the vector `x_(L+1:L+T)` in one forward pass with a
  flatten-plus-linear prediction head; its stated supervised objective is
  `E_x [(1/M) Σ_i ||x̂^(i)_(L+1:L+T) - x^(i)_(L+1:L+T)||²]`.
  §3.2 pretrains using randomly masked patches and reconstruction of the
  masked values.
- **Transfer and limit:** vector output is a precedent for direct horizons,
  not recursive generation of future pseudo-observations. PatchTST processes
  each channel independently with shared weights; Simple23 instead embeds a
  multichannel value-and-mask patch. PatchTST's masked pretraining is not a
  claim about online fast weights, causal alarm safety, or forecast-loss/risk
  alignment.

### 1.4 Context statistics can matter, but are not identified hazard factors

Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long, **Non-stationary
Transformers: Exploring the Stationarity in Time Series Forecasting**,
NeurIPS 2022; read [arXiv v4, §3.1 and §3.2, Eqs. (1), (6)](https://arxiv.org/html/2205.14415v4)
and checked the [NeurIPS record](https://papers.nips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html).

- §3.1 uses an input-window mean and variance for stationarization, then
  restores output scale. In §3.2, Eq. (6) learns factors from the original
  series and its statistics: `log τ = MLP(σ_x, x)` and `Δ = MLP(μ_x, x)`;
  attention logits are adjusted as `(τ Q'K'^T + 1 Δᵀ)/√d_k`.
- **Transfer and limit:** removing context statistics can discard information
  relevant to forecasting. A past-only slow summary plus a fast residual is a
  testable design hypothesis, **not** proof that the summary is operating
  regime, that the residual is hazard, or that either is physically causal.
  The paper modifies attention and normalization; here those source modules
  remain frozen and unchanged.

### 1.5 Auxiliary loss decrease does not imply main-task improvement

Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz
Hardt, **Test-Time Training with Self-Supervision for Generalization under
Distribution Shifts**, ICML 2020, PMLR 119, 9229–9248. Read the
[official paper, §4, Theorem 1 and Eqs. (10)–(11), pp. 7–8](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf)
and checked its [publication record](https://proceedings.mlr.press/v119/sun20b.html).

- Under differentiable, convex, β-smooth main loss, bounded gradient norms
  `G`, and `⟨∇ℓ_main, ∇ℓ_ssl⟩ > ε`, Theorem 1 gives a main-loss improvement
  after one SSL step with `η = ε/(βG²)`. Its image-shift experiments are not
  tokamak experiments.
- **Transfer and limit:** this is a conditional alignment result, not a
  general guarantee for a nonconvex frozen network, alarm threshold, FPR, or
  the proposed memory. Even with an SPD preconditioner `M`,
  `g_sslᵀ M g_ssl > 0` for nonzero `g_ssl` does not fix the sign of
  `g_riskᵀ M g_ssl`. Main-task gradient diagnostics need labels and therefore
  belong offline or after an eligible shot is sealed, never in its live
  prediction/update. The prior local probe did **not** save gradient vectors;
  it did not measure this cosine.

### 1.6 Matured labels and delayed feedback

Mohamed Readh Fentazi, Mazene Ameur, and Adlen Ksentini, **Training on the
Future: A Delay-Aware Audit of Test-Time Adaptation for Time-Series
Forecasting**, arXiv:2610.12232v1, 8 October 2026, preprint, not verified as
peer reviewed. Read [§3 “Delay semantics”, §5.2, §7 and Appendix F](https://arxiv.org/html/2610.12232v1)
and checked [version metadata](https://arxiv.org/abs/2610.12232).

- §3 distinguishes full-horizon targets, released no earlier than delay
  `d ≥ H`, from a genuinely observed partial prefix, which can be used when
  actually available. A full `H`-step target used after one step would reveal
  unavailable future values. §5.2 reports method/dataset-dependent outcomes,
  not universal benefit or universal harm from adaptation.
- **Transfer and limit:** use only measurements that have actually arrived,
  never autoregressive forecasts as truth. This is a recent preprint covering
  forecasting benchmarks, not a safety or tokamak validation; Appendix F
  describes limitations in backbone/checkpoint coverage. Its forecast labels
  are future signal measurements, distinct from HDF disruption outcomes.

## 2. Local evidence and what it does not prove

- The frozen model contract in [simple23.py](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple23.py:44)
  is `W=128`, `P=8`, 16 multichannel patches, `d=64`, two Transformer encoder
  layers and four heads. For the source input view, each patch embedding uses
  8 × (23 values + 23 availability indicators). The CLS token, positional
  embedding, existing source fast-origin behavior, post-encoder norm,
  risk head, and direct 25 × 23 forecast head are part of the frozen parent.
- The existing [source local review](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/SOURCE_MASK_REPAIR_LOCAL_REVIEW_20261104.md)
  records reconstruction decrease in 777/777 fitting and 759/777 held-out
  patch cases, while the real update's safe risk proxy worsened by
  `+0.000232686` and the timely-positive proxy improved by `-0.001764898`;
  reverse was `+0.000162169` and `-0.002023743`, respectively. Both
  prespecified main gates failed. These are local source-VAL finite-step
  results, not EAST online outcomes or measured gradient alignments.
- The existing [forecast-fidelity audit](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/SOURCE_FORECAST_FIDELITY_AUDIT_20261104.md)
  reports that, on 764 complete windows from 133 source-VAL shots, the native
  zero-fast forecast head's shot-averaged SmoothL1 was `0.091421096` versus
  `0.053994268` for hold-last, using the documented native-view contract.
  This selected subset is not an independent evaluation and does not prove
  an adapted forecaster must fail; it **does** make native-head fidelity an
  explicit stop/go check rather than an assumed strength.

## 3. Small architecture to consider, not an implementation claim

### 3.1 Frozen path, separate shot-local fast state

Keep every source parameter and source normalization buffer frozen, including
the patch embedding, positional/CLS tokens, both encoder layers, original
source fast origin, final norm, risk head, and native forecast head. Let
`E_(s,j) ∈ R^64` be the existing value-plus-availability patch embedding for
window ending at sample index `s`, before position/CLS. A patch is valid only
if it contains at least one actually available value; padded/all-unobserved
patches never receive a correction. “23” describes input channels, not a
claim that every channel, including EAST CIII/MNM, is physically available.

Maintain two separate, shot-reset states:

- `c ∈ R^64`: a slow EMA of past valid patch-embedding means, initialized to
  zero with the first received timestamp as its reference. Use `c_(s-)` for the
  current prediction; update the EMA only after that prediction is sealed.
  For elapsed time `Δt` since its last supported update, a proposed
  weight is `exp(-Δt/τ)` with `τ` chosen on source TRAIN before evaluation;
  use actual timestamps rather than assume a nominal sampling rate. If no
  patch is valid, keep the prior state. The EMA uses neither risk scores nor
  labels nor future-shot information. It is a *candidate operating-context
  feature*, not a proven operating/hazard separation: slow hazard information
  and fast changes in operating conditions can occur in either component.
- `U ∈ R^(8×8)`: a 64-number fast memory, initialized to exactly zero at
  each shot. With source-TRAIN-learned and then frozen `P_q ∈ R^(8×64)` and
  `P_o ∈ R^(64×8)`, form, independently for each valid patch,
  `q_(s,j) = P_q LN_nonparam(E_(s,j) - c_(s-))` and
  `δE_(s,j) = clip_RMS(P_o U q_(s,j), ε_token)`.
  Set `δE=0` for invalid patches and pass `E+δE` through the *unchanged*
  parent, retaining its source fast-origin behavior. Use a predeclared
  per-token bound and a per-shot state-radius bound; these bound intervention
  size, not risk. `U=0` must reproduce parent logits/forecasts exactly.

This is a low-rank internal-memory proposal, not an extra risk head. Yet the
adapter changes the inputs to the frozen risk head, so risk can change: a
frozen head is **not** a risk-invariance guarantee. Source asinh
normalization remains unchanged. Neither `P_q` nor `P_o` should be initialized
as the zero matrix, which would kill the initial `U` gradient; even nonzero
initialization does not guarantee useful gradients. Verify them through a
source-TRAIN unroll. At `U=0`, a zero-step forward alone gives no ordinary
adapter-gradient signal for these projections.

### 3.2 Direct real-future self-supervision, only after it matures

First measure whether the *existing* frozen 25-step head is useful and
state-sensitive under the precise target/mask contract, against the same
hold-last baseline; the local finding above is a serious warning. If it fails,
do not present its adaptation loss as validated. One optional contingency,
requiring a separately authorized source-TRAIN fit and source-VAL gate, is a
small fixed auxiliary forecast-only head `ψ`:

- Flatten the parent masked, normalized 17 × 64 output to `z ∈ R^1088`;
  use `A ∈ R^(8×1088)` and `B ∈ R^(575×8)` to output all 25 × 23
  corrections simultaneously. Add each to that channel's last **real**
  available observed value within the lookback, transformed with the frozen
  source asinh coordinates. If no valid past anchor exists for a channel,
  it has no target support for this loss; report it rather than invent an
  anchor. Keep the existing target-contract exclusions, true future masks and
  per-channel coverage. This is direct multihorizon prediction, never an
  autoregressive rollout or treating a prediction as a measurement.
- `A,B` contain 13,304 weights without biases, `P_q,P_o` contain 1,024;
  total **14,328** separately learned persistent weights, plus `U` and `c`
  as nonparameter per-shot states. Source model weights are unchanged.
  Source-TRAIN optimization can train `ψ` and differentiate an outer
  *real-future* forecast objective through one or more causal `U` updates;
  an optional HDF-outcome outer loss would make that branch explicitly
  supervised, not fully unsupervised. Freeze all learned components before
  VAL/EAST; neither VAL, TEST nor EAST chooses the architecture or step.
  Do not initialize both low-rank factors `A` and `B` to zero. Nonzero useful
  gradients and non-degenerate output must be tested, not assumed from
  parameter counts.

For an origin `s`, queue its already-observed embedding, masks, `c_(s-)`, the
sealed direct forecast, and, for `ψ`, its past-only anchor values/masks. After
all indices `s+1 … s+25` have actually arrived, form a masked SmoothL1 loss
against their **real** available values
in the frozen source coordinates. At sample `t=s+25`, first produce and seal
the current risk output using `U_(t-)`; only then recompute the queued origin's
forecast loss with current `U_(t-)`, update
`U_t = project_radius(U_(t-) - η ∇_U L_s)`, and use `U_t` no earlier than
the next prediction. This recomputation is a present update objective, not a
replacement for the forecast that was sealed at `s`. Use each eligible origin
at most once on a fixed, causal cadence; no eligible targets means no update.
If the shot ends before maturity, discard the unfinished update, not the shot
or its sealed predictions. A horizon of 25 **indices** must not be silently
called 25 ms without checking actual timestamps.

### 3.3 Source/EAST authority and an optional end-of-shot branch

- Before a current shot's predictions are sealed, neither `meta/IsDisrupt`
  nor `meta/DownTime`, event CSV labels, shot-list classes, complete-shot
  length, or any future samples may enter the current predictor, state update,
  queue scheduling, map, or selection. `meta/IsDisrupt` and `meta/DownTime`
  are the HDF hard-outcome/time authority for post-seal scoring; `DownTime`
  is meaningful for disrupted shots under the [local outcome contract](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/HDF5_OUTCOME_CONTRACT_20261103.md).
  Preserve the existing locked cohort/order, fixed parent threshold, and true
  masks.
- The user permits **first 20 EAST shots' outcome labels only after each
  respective shot ends and its predictions are sealed**. One optional,
  explicitly *supervised inter-shot* experiment is a small persistent
  adapter `S ∈ R^(8×8)`, fit once after shot 20 using only those 20 ended
  shots, then frozen for shots 21–360. It could enter through `U+S` under a
  common predeclared total correction bound; it is not a free, risk-neutral,
  or self-supervised update. This branch is not required for the basic
  within-shot proposal. During shots 1–20 take `S=0`; use the already locked
  `map_after20` rule, not a new map or threshold. No current/later shot label
  may affect this fit.
- Compare native, `U` only, `S` only, and `S+U` with matched bounds and
  fixed cadence; report prefix 20, suffix 340 and full 360. The paired
  `S+U` versus `S` comparison tests incremental within-shot memory in that
  particular branch, not an identified causal decomposition or independent
  validation. First-20 adaptation changes the protocol and must be labeled
  separately from static source-only evaluation.

### 3.4 Preregistered checks and failure conditions

1. **Contract:** parent equality at `U=S=0`; source/head/buffer hashes before
   and after; batch/shot independence and reset; strict no change to padding;
   no fake CIII/MNM support; no eligible target or zero `U` gradient leaves
   `U` exactly unchanged, while no valid current patch leaves `c` unchanged.
   Record actual available-target counts, unmatured origins, and queue cost.
   Perturb later real samples or the current outcome and verify already sealed
   predictions and earlier updates are unchanged.
2. **Controls:** native/no update, matched-size fixed or shuffled-past update,
   frozen versus live `c`, and hold-last forecast on identical masks; if
   reverse-context is included, reverse within the same window and disclose
   potential padding confounding.
   Report auxiliary loss, state gradients and forecast-vs-hold fidelity by
   shot/channel/support, not only a pooled decrease.
3. **Alignment:** offline source-TRAIN/VAL only, compare true SSL and risk
   gradient directions or finite-step paired risk changes by safe and timely
   strata; never use current-shot labels for live update selection. A lower
   auxiliary loss or a positive-definite state geometry is insufficient.
4. **Gates and exposure:** select gate criteria before fitting, retain the
   parent threshold and all locked shots, use paired per-shot outcomes and
   relevant groups; source timely means strictly `>10 ms`, EAST strictly
   `>30 ms`. Preserve full-record primary scoring; prefix windows are only
   separately labeled sensitivity analyses. Do not treat correlated frames
   as independent trials. VAL has participated in development and is not an
   independent verification set.

No actual model has been built or evaluated here. The literature supports
testing these choices, not claiming novelty, identified operating/hazard
factors, or improvement on EAST.
