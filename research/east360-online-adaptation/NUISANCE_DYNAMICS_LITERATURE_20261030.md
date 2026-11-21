# Nuisance operating conditions versus hazard dynamics: bounded methods sidecar

2026-10-30. **Status: literature-based proposal only; no implementation, run, or
EAST performance claim.** This is a deliberately bounded reading of six primary
papers, not a systematic or 100-paper review. The main agent is separately
reading LSTD, arXiv:2502.12603, so it is not duplicated here. The first-hand
inputs are the method sections linked below and the existing project notes and
code interfaces. AI-assisted research was used.

## Bottom line

The literature does **not** supply a generic unlabeled rule that can tell a
benign operating-condition change from an incipient disruption. Its useful
ideas come with assumptions: some latent relationships remain stable, observed
channels distinguish competing explanations, and the necessary feedback is
actually available. A slow change can be hazardous, and a fast one can be
benign. In a minimal observation model \(x_t=n_t+h_t\), every alternative
\((n_t+d_t,h_t-d_t)\) has the same observed \(x_t\). Additional sensors,
controlled variation, or testable structural restrictions are needed to break
that ambiguity; the restrictions themselves are not verified by reducing a
training loss.

The concrete proposed intervention below is **conditional temporal-relation
attention adaptation**: construct a stop-gradient relationship target from
already-arrived diagnostics after conditioning on measured operating
covariates, then take one bounded gradient step on a temporary, first-layer
Q/K attention adapter. It is not a scalar score adjustment, a forecast-loss
gradient, a learned representation target, or an entropy update. It is also
not a reproduction of any paper. Its key unproven step is that an internal
temporal attention pattern should match those measured signal relationships.
An ablation can falsify that step before a full-stream experiment.

The user supplied the newer observation that a source-trained vector target
improves safety BCE while worsening danger on 59/64 validation shots, and that
real/reversed updates have cosine about 0.99998. I treat these as parent
context, not independently re-evaluated results. Existing project reviews
already document a failed context-vector screen, the danger of origin-CLS
Jacobian bottlenecks, and failed task-aligned Q/K forecasting; see
[context-vector review](</media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/CONTEXT_VECTOR_REVIEW_20261030.md>),
[source-teaching review](</media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/CONTEXT_VECTOR_SOURCE_TEACHING_REVIEW_20261030.md>),
and [history audit](</media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/NEXT_DIRECTION_HISTORY_AUDIT_20261030.md>).
The new source diagnostic is running separately and was not inspected or awaited.

## Primary methods read

### 1. State augmentation makes the identifiability problem explicit

**Joshi, González and Upchurch (2014), “Identifiability of Additive Actuator
and Sensor Faults by State Augmentation.”** Journal of Guidance, Control, and
Dynamics. [NASA copy of the complete paper, §§II–IV and Theorems 2–3](https://ntrs.nasa.gov/api/citations/20140006008/downloads/20140006008.pdf);
[publisher DOI](https://doi.org/10.2514/1.62523).

- **Method/equations.** The starting model is
  \(\dot x=Ax+Bu+w_p,\ y=Cx+w_s\). For a candidate sensor-bias pattern, the
  bias is appended to the state and modeled as a random walk, yielding an
  augmented observation \([y_1;y_2]=[C_1x;C_2x+b]+w_s\). Banks of
  Kalman–Bucy filters and hypothesis tests estimate competing fault patterns.
  The paper's Theorem 2 requires detectability/observability of \((C,A)\)
  **and** observability of zero-frequency modes through the bias-free outputs.
  For simultaneous actuator and sensor faults, Theorem 3 additionally requires,
  among other conditions, \(l\ge m_k+q\) and no relevant invariant zero at the
  origin. In the discrete-time remark, zero-frequency modes correspond to
  eigenvalue 1.
- **What changes and feedback.** A filter estimates augmented state and
  constant/slow bias from measurements; this is state estimation and fault
  discrimination in the stated linear model, not online neural-weight
  adaptation or plasma identification.
- **Boundary.** In their model, simultaneous actuator and sensor faults cannot
  be identified by this method when all sensors have unknown biases and at
  least one actuator fault is included. That is a counterexample to assuming
  every observed shift can be separated. It is **not** a theorem that the
  nonlinear EAST problem is unidentifiable in every possible formulation.
  Source anchor, §III: “All zero-frequency modes of A are observable with
  respect to the bias-free sensor outputs.”

### 2. Stable *structure* does not mean stable full dynamics

**Cai et al., “Time Series Domain Adaptation via Latent Invariant Causal
Mechanism,” arXiv v1 (2025); subsequently published in IEEE TPAMI.**
[Inspected manuscript, §§III–V, especially Assumption 1, Lemma 1, and
Eqs. 12, 16–19](https://arxiv.org/html/2502.16637);
[publication record](https://doi.org/10.1109/TPAMI.2025.3642245).

- **Method/equations.** The paper posits \(x_t=g(z_t)\) and
  \(z_{t,i}=f_i(\mathrm{Pa}(z_{t,i}),u,\epsilon_{t,i})\), with independent
  innovations. Its generalized causal conditional shift explicitly permits
  \(P(z_{t+1}\mid z_{\le t},S)\ne P(z_{t+1}\mid z_{\le t},T)\) while assuming
  the latent graph \(A_S=A_T\). An encoder/decoder and learned inverse
  transition functions estimate latent innovations; a sparsity penalty on the
  inverse-transition Jacobian and a masked cross-domain Jacobian-alignment
  loss implement the structural inductive bias. The total objective combines
  task, reconstruction, KL, sparsity and alignment losses.
- **Identifiability, not magic.** Its Lemma 1 assumes an invertible mixing,
  conditional independence of latent components given their past, and
  sufficient linearly independent derivatives of conditional log densities as
  past latent states vary. The stated result is component-wise identification
  **up to permutation and invertible component-wise transformations**. Graph
  recovery also relies on accurately modeled transitions and sparsity.
  Source anchor, §III-B: “the latent causal structures are stable across
  different domains.”
- **What changes and feedback.** This is joint unsupervised domain adaptation
  using source and target datasets, not a demonstrated single-shot prequential
  update. Its target-domain samples must be available during training. The
  target cannot simply be presumed to preserve the source graph during an
  instability. Forecast/classification benchmark results in §VI do not
  establish an EAST first-alarm benefit.

### 3. Variable-interaction masks are assumed causal, not observed causal edges

**Lu and Sun (2024), “CauDiTS: Causal Disentangled Domain Adaptation of
Multivariate Time Series,” ICML.** [Full methods, §§3–4.3 and Algorithm 1](https://raw.githubusercontent.com/mlresearch/v235/main/assets/lu24i/lu24i.pdf);
[proceedings record](https://proceedings.mlr.press/v235/lu24i.html).

- **Method/equations.** Per-variable LSTMs give \(h_{i,t}\). Their attention
  for variable pair and adjacent times is
  \(e_{ij,t:t+1}=v^\top\sigma(W_1h_{i,t}+W_2h_{j,t+1}+b_1)+b_2\);
  normalized coefficients are aggregated into an inter-variable matrix
  \(I\). A learned binary mask partitions this into
  \(A=I\odot M\) and \(B=I\odot(1-M)\). Separate graph networks produce a
  class-related representation and a domain-related representation. Their
  training uses source class labels, target unlabeled data, source class
  prototypes, nearest-prototype target consistency, augmentations, a
  representation recombination/backdoor-motivated term, and target conditional
  entropy. The entropy term is part of this multivariate graph-domain method;
  it is not evidence that image entropy adaptation solves this task.
- **Assumptions.** The method assumes the class-related rationale and
  \(P(Y\mid Z_c,G_c)\) are invariant, with each class having an invariant
  rationale. Its backdoor derivation assumes access to the relevant
  domain-specific confounder and a uniform prior over the considered domain
  correlations. An attention mask and low correlation do not by themselves
  prove a physical causal graph or independence.
- **What changes and feedback.** Network and mask parameters are optimized
  during offline source/target UDA. At inference the paper uses the target
  query branch, not an online within-shot update. The experiments include
  human-activity recognition and boiler faults, not tokamak disruption alarms.
  Source anchor, §4.3: “In the inference phase, we use only the query branch.”

### 4. Delayed feedback can guide updates, but it is not a hazard label

**Li, Teng, Zhou and Duan (2026), “Beyond Uniform Updates: Drift Pattern Aware
Online Time Series Forecasting Under Delayed Feedback,” IJCAI, PADRE.**
[Full paper, §§3.1–3.6 and Table 1](https://ijcai.org/proceedings/2026/0503.pdf);
[proceedings text](https://ijcai.org/proceedings/2026/503).

- **Method/equations.** At origin \(t\), the paper stores an \(H\)-step
  forecast and releases its actual signal target at \(t+H\). Offline, it
  clusters regime embeddings and pairs each with its typical forecast
  residual. Online it represents the original input, a chronological buffer of
  matured residuals, and a retrieved memory view. A triangle-area statistic
  measures their geometric consistency. An MLP produces a scalar
  \(\gamma_t\in(0,1)\); after recomputing the current forecast on the old
  input, it takes
  \(\theta\leftarrow\theta-\eta\gamma_t\nabla_\theta
  \mathrm{SmoothL1}(f_\theta(x_t),y_t)\). Its policy also learns using a
  one-step surrogate, and a threshold controls insertion into episodic memory.
- **Assumption.** Coherent context, residual evolution and retrieved memory
  are assumed to be evidence of systematic drift, not a proof that the drift
  is safe. The memory can miss a new mechanism; the method does not certify a
  hazard/nuisance split. Source anchor, §3.6: “recomputed under the current
  parameters.”
- **What changes and feedback.** The paper actually changes the forecaster
  weights and policy parameters, and accumulates memory. Its feedback is
  delayed **diagnostic values**, not future class or shot outcome. This
  proposal borrows the arrival discipline, not its scalar gating or
  cross-shot memory. Results are ordinary forecasting benchmarks; for example,
  Table 1's TCN/Traffic/H=96 cell is 0.515 MSE for PADRE versus 0.511 for
  PROCEED, so “best in every cell” would overstate that table.

### 5. TCV state mapping combines operating regimes and disruption information

**Poels et al. (2025), “Plasma State Monitoring and Disruption Characterization
using Multimodal VAEs,” Nuclear Fusion 65, 096012.** The IOP page was
inaccessible here; the authors' [full arXiv v1, §§3.2, 4.1–4.4 and
5.1–5.3](https://arxiv.org/html/2504.17710) contains the methods inspected;
[journal DOI](https://doi.org/10.1088/1741-4326/adf121).

- **Method/equations.** The recurrent encoder uses a past window and previous
  latent mean, with
  \(\mu_t=\mu_{t-s}+f_\phi(\mu_{t-s},x_{t-w:t})\), while a static decoder
  maps the two-dimensional latent state back to the diagnostic space. A
  Gaussian-mixture prior encourages regime structure. Training combines
  reconstruction, KL, a mixture-coverage term and a disruption-risk BCE
  formed from time-to-disruption proxy labels.
- **What changes and feedback.** Encoder, decoder and risk-map weights are
  trained offline; at inference the **latent state** changes as observed input
  arrives, but the paper does not demonstrate within-shot weight adaptation.
  The authors explicitly distinguish their state map from a complete future
  disruption prediction, and their risk-related labels/calibration use
  outcomes during training/post-hoc analysis. Their signals use last-available
  samples, not future interpolation. Source anchor, §3.2: “we refrain from
  non-causal interpolation methods.”
- **Limit.** This is TCV flat-top characterization with a risk-shaped latent
  representation, not proof of an independently identified nuisance axis, an
  EAST alarm improvement, or a reason to add the paper's risk map/calibration
  to the qualified J-TEXT classifier.

### 6. A tokamak dynamics model can restrict *which* component is refit

**Wang et al. (2025), “Learning plasma dynamics and robust rampdown
trajectories with predict-first experiments at TCV,” Nature Communications.**
[Full author manuscript, Methods: “The Neural State-Space Model,” “The
Dynamics Function,” and “Training Data Distribution”](https://arxiv.org/html/2502.12327);
[journal DOI](https://doi.org/10.1038/s41467-025-63917-x).

- **Method/equations.** Their neural state-space model is
  \(\dot x=f_\theta(x,a)\), \(o=O_\theta(x,a)\). It combines learned
  confinement scalings, bounded neural corrections, and 0-D energy/particle
  balances; for example the thermal-energy equation contains
  \(-W_{\rm tot}/\tau_E\), heating and radiated-power terms. The H-mode
  correction has an explicit smooth regime transition, and the confinement
  neural correction is bounded to [0.75, 1.25]. The model is integrated along
  measured action trajectories and fitted against observations.
- **What changes and feedback.** Following initial fitting, the paper
  fine-tunes only parameters in its energy and particle confinement laws on
  44 shots, freezing other weights; later experimental retraining is between
  completed shots/run days. Source anchor, Results: “all other model weights
  frozen.” It designs rampdowns rather than modifying a deployed J-TEXT
  disruption classifier; this is not a demonstrated intra-shot EAST learner.
- **Limit.** The authors note small high-performance sample sizes and
  unmodeled off-normal events. Conditioning on actual control actions and
  physically structured equations is stronger than merely calling measured
  \(I_p\) or \(B_t\) an independent nuisance variable. Their particular
  balance equations cannot be assumed identifiable from the project's
  available 23 features.

## Interpretation across the six methods

The most relevant tension is between the attractive structure in LCA/CauDiTS
and the observability requirement in Joshi et al.: a stable or sparse graph is
useful **if** the assumed decomposition is identifiable, but a learned
partition is not itself proof that the physical hazard survived. Poels et al.
show that real TCV regimes and disruption-relevant structure can coexist in
one state map; their risk labels also show that this was not an entirely
unsupervised nuisance/hazard separation. Wang et al. show a concrete restricted
model refit, but with different data and between-shot timing. PADRE shows how
to respect signal-target delay, while remaining a forecast-error method whose
success criterion is not an alarm. These are complementary, not direct
head-to-head experiments.

Do not equate low-frequency variation with nuisance, high-frequency variation
with precursors, a high attention weight with causality, or lower
signal-prediction error with greater alarm utility. The diagnostic
\(I_p\)/\(B_t\) are observed channels, not established exogenous instruments;
both can change with plasma evolution. Conditioning on them may remove hazard
information, and the proposed cross-channel similarities may simply reflect
noise or sampling artifacts. No inspected paper establishes otherwise for
EAST.

## One sharply testable intervention for the main agent

**Proposal, not a reported result: operating-conditioned temporal-relation
attention (OCTRA).** The causal question is whether the *arrived trajectory's*
conditional relationship structure can select useful internal attention
directions, not merely rescale a common update. The published papers motivate
conditioning and relational structure; the equations and experiment below are
new project-specific choices and have not been validated by those papers.

### Fixed interface and offline reference

1. Freeze the qualified J-TEXT checkpoint, preprocessing, risk head, cold20
   mapping/slow state, threshold and alarm policy. No J-TEXT online branch, risk
   correction, pseudo-outcome, score blending, or learned vector target.
   Only EAST has an online temporary neural update. Reuse actual timestamps,
   masks and permitted prior-shot data; preserve the first 20 archived cold
   predictions in a later stream evaluation.
2. Initial, deliberately limited covariates are the already available
   normalized \(u=(B_t,I_p)\); initial diagnostic set
   \(C=\{\mathrm{Mir\_avg\_amp},\mathrm{sxr\_mean},
   \mathrm{xuv\_ratio},\mathrm{ne\_nG}\}\). These are candidate measured
   channels, **not** certified redundant sensors, independent actuators, or
   a complete set of hazard channels. Their existence in the flat-23 order
   follows the project's
   [input contract](</media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/io23.py>).
   Use the model's exact frozen asinh/availability transformation, not new
   normalization fitted on a test shot.
3. On previously completed and permitted EAST development shots only, fit
   per-channel ridge maps
   \(m_c(u)=\beta_{c0}+\beta_{c1}u_1+\beta_{c2}u_2\), shot-weighted,
   ridge penalty 1 on the two slopes, from each shot's first 256 samples.
   This early reference is not asserted to be safe. Fit a diagonal scale
   \(D=\operatorname{diag}(\max(\operatorname{mean}(d_c^2),10^{-4}))\)
   from the same development material's consecutive valid 8-sample
   conditional-residual means. Freeze \(m,D\) before evaluating a held shot.
   In a retrospective cold20 component screen, refit them without the held
   shot. The other completed cold20 shots may include shots that occurred
   later, and the pre-existing cold parent has seen all 20, so this is only a
   **conditional-parent component** screen, not a chronological deployment
   estimate. A later suffix stream may use a reference frozen after the first
   20 shots have completed; it may not incorporate later suffix shots.
   If the valid design is rank-deficient,
   observed scales are degenerate, or prior-shot support is inadequate, stop:
   do not improvise a substitute channel or pool outcome groups after seeing
   the results.

### Actual mature-feedback update

Use the existing zero-based scheduler \(o=127+25k\). Score with the previously
active state through \(t=o+25\), wait for all available samples through \(t\)
and their real acquisition latency, then compute the new state for \(t+1\)
onward. Never use a current shot's eventual outcome for the target, support
test, state, selection, or update. At the next update replace rather than
accumulate the temporary state; reset all state and queues at shot boundaries.

At \(t\), divide the latest 128 arrived points into the same 16 chronological
8-point patches as the frozen transformer. For each valid patch \(i\), form
\(\bar r_i=\operatorname{mean}_{a\in i}[z_C(a)-m(u(a))]\) and
\(d_i=\bar r_i-\bar r_{i-1}\). Require both covariates and all four diagnostic
channels on at least six samples of each used patch; never turn missing values
into observed zeros. For query rows \(i\in\{13,14,15,16\}\) and valid keys
\(j\le i-2\), require at least four keys and define the **observed,
stop-gradient** target

\[
 T_{ij}=\operatorname{softmax}_{j}
 \!\left(\operatorname{clip}(d_i^\top D^{-1}d_j,-4,4)\right).
\]

This is a conditional **temporal similarity**, not a causal edge estimate or
a claim that \(\bar r\) contains only hazard. Restrict the first transformer's
first preselected attention head to the same query/key indices and renormalize
its Q/K logits over those keys to obtain \(A_{ij}(\Delta;X_{\le t})\). The
actual forward pass continues to use the original input and original
attention mask; the restricted distribution is only the auxiliary loss tap.
Define

\[
 L_{\rm rel}(\Delta) =
 \frac{1}{|I|}\sum_{i\in I}\mathrm{KL}
       (T_i\Vert A_i(\Delta;X_{\le t})),\quad
 g=\left.\nabla_{\Delta}L_{\rm rel}\right|_{\Delta=0},\quad
 \Delta_{\rm new}=-0.05\,g/\lVert g\rVert_2 .
\]

Here \(\Delta=(\Delta Q^{(0)},\Delta K^{(0)})\) changes only the
\(16\times64\) Q and K projection rows of that one head; all original network
weights remain frozen. Concatenate the two tensors for the norm. A zero
gradient or no eligible row means **zero temporary state**, not a fabricated
direction. The first head, 0.05 radius, patch rules, diagonal scale floor and
temperature 1 are fixed candidates, not optimized post hoc. Compare actual
finite-step network outputs rather than just a local Taylor approximation.

This differs from the earlier
[temporal-relation experiment](</media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/TEMPORAL_RELATION_PLAN.md>):
that method trained a source-domain real-versus-shuffled sequence discriminator
and updated embedding gain/bias. Here no discriminator or permutation
classifier is trained; an explicit conditional similarity matrix drives one
head's temporary Q/K weights. It also differs from the previous Q/K
forecast-gradient/gradient-transport and origin-CLS target-vector approaches.
These differences do **not** establish novelty, identifiability, or success.

### Discriminating controls and stop/go

Use the same scored prefixes, timestamps, support intersection, temporary
tensor budget and eligible update schedule for every arm:

- **Cold:** exact zero temporary state.
- **Real conditional relation:** the intervention above.
- **Reverse:** keep the model's actual input unchanged, but reverse only the
  25 newly arrived residual/mask rows when constructing \(T\); this is a
  negative control, not a physically realistic counterfactual.
- **Unconditioned:** use the same formula with \(m=0\), common masks and
  the same prefit scaling. This asks whether conditioning helps at all.
- **Fixed teacher/direction:** compare a relation target averaged solely over
  permitted training-shot windows and, separately, an independently
  precomputed mean Q/K gradient direction from those windows. Do not choose
  either sign or norm using the current real update. Norm-match nonzero
  controls to 0.05 and report genuine zeros.
- **Existing forecast-gradient and temporal-order controls**, if already
  available under the same parent and clock, provide context; they are not
  relabeled as this method.

Before any full stream, use a held-shot, conditional-parent cold20 diagnostic
whose fitting folds exclude the held shot. All endpoints must be generated
chronologically without using that shot's outcome; only after the shot is
sealed may outcomes label its safe windows or its strictly timely
\(30 < \text{lead} \le 125\) ms danger windows for analysis. Do not treat
windows as independent shots or change the established label contract.
Require all three known safe development shots and at least ten distinct
disruptive shots to have eligible later-query evidence; otherwise the screen is
**inconclusive**, not passed. Report support and all exclusions.

Predeclare this screen's necessary conditions: real feedback must reduce
mean per-shot later-query BCE relative to cold in both safety and timely-danger
groups; it must beat reverse, unconditioned, and both fixed controls in both
group means; and at least two of three safe and at least half the eligible
disruptive shots must improve relative to cold. Report the **entire**
per-shot distribution, including the early-disruptive region and later
windows outside the supervised danger interval. These thresholds are
exploratory engineering filters, not statistical significance claims.

Additionally audit whether the supposed direction actually changes: on matched
nonzero opportunities report real/reverse Q/K cosine, the fraction of exact
zeros, and the component orthogonal to the current frozen-risk gradient.
Require median real/reverse cosine below 0.99 **and** a nonzero finite-step
real/reverse logit difference on at least half the matched later queries
(absolute difference above \(10^{-4}\)); otherwise stop for directional or
functional collapse. This is intentionally stricter than the reported
0.99998 direction similarity, but even passing is not proof of useful causal
information. Report actual norms, not just cosines, and do not give zero
vectors invented cosine values.

If the screen passes, freeze the entire protocol before any separately
authorized paired EAST stream. Seal each prediction and first alarm before
reading that shot's outcome; report paired rescues **and harms**, strict
first-alarm lead \(>30\) ms, false alarms, \(\mathrm{FPR}\le0.20\), and
the first20/suffix split using the fixed original policy. Require an actual
paired alarm advantage over cold **and** the feedback controls without
violating the predeclared FPR bound. BCE, auxiliary fit, cosine, or a single
boundary flip is not a substitute. The repeatedly developed 360 stream is
exploratory, not an independent confirmation.

### Falsifiers and prerequisite checks

- The reference regression must have out-of-shot support; \(B_t\) and
  \(I_p\) cannot be called nuisance instruments merely because they were
  selected as covariates. Compare conditioned versus unconditioned relations
  on held completed shots stratified by operating range and by later known
  safe/danger outcome. Large residual dependence on operating range or loss of
  danger contrasts refutes the intended separation for this specification.
- The proposed head-level relation match is an inductive bias, not a theorem:
  a correct classifier may need attention unrelated to the hand-built target.
  If the unconditioned, reverse, constant, or zero controls match it, or if
  Q/K differences collapse back to a single risk direction, reject this fixed
  intervention rather than adding a score patch.
- Test actual-clock causality, per-shot reset, channel masks, exact zero state,
  head indexing, no effect on already sealed logits, and original-weight
  hashes. A perturbation of samples after \(t\) must not change the state or
  score through \(t\). A synthetic nuisance or injected precursor test has
  known construction labels but cannot prove that real EAST disturbances
  obey the construction.

## Source-check record and scope

Searches used public web/primary repositories for time-series causal domain
adaptation, delayed-feedback forecasting, system-identification observability,
and tokamak latent/state-space dynamics. Inclusion required an accessible
primary methods text; abstracts alone were insufficient. Six papers above were
retained. The full methods sections identified in each entry were read, not
merely the abstracts. Two tempting claims were explicitly rejected: CauDiTS
does not demonstrate causal disentanglement in EAST merely by learning a
mask, and PADRE's Table 1 does not establish superiority in every forecast
cell. Exa returned a credit-limit error; public web retrieval supplied the
primary texts. The journal page for Poels was blocked, so the inspected
version is identified as arXiv v1. LSTD was excluded because the main agent is
already reading it; Ai2026 was not used.

The claim-to-source check was performed in the same context as discovery;
no independent-context skeptic/spawn interface was available, so it is not
represented as an independent adversarial review. No GPU job, model or data
change, data upload, code edit, commit, deletion, or experiment was performed.
Only this note was written.
