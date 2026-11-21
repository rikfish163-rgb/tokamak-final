# Gate-loss claim–source check — 2026-11-01

Scope: independent, default-REJECT check of the three supplied pairings, using public primary texts only. No proposer reasoning was supplied. No raw data, GPU, model execution, or proposer-file edits were involved. A confirmation applies to the stated, appropriately bounded claim, not to a new method or a downstream result.

## Resolution log and counts

| ID | Verdict | Reason |
| --- | --- | --- |
| C1 | **CONFIRMED** | TAFAS adapts input/output calibration modules around a frozen source forecaster using observed partial targets and, when available, older complete targets; its reported evidence is forecasting, not a disruption-classifier guarantee. |
| C2 | **CONFIRMED** | Theorem 1 gives a sufficient alignment condition for a one-step decrease in the main loss under specified assumptions; it does not establish the result for the specified tokamak Transformer. The appendix has a sign inconsistency in a proof line, discussed below. |
| C3 | **REJECTED** | The paper normalizes forecast inputs and restores output statistics; it does not prove that demeaning a plasma forecast loss separates safe drift from dangerous evolution. |

**Counts:** 3 pairings proposed; 2 CONFIRMED, 0 PARTIAL, 1 REJECTED, 0 BLOCKED. The one rejection is `does_not_support_claim`; its mismatch of task and population is additional context, not a second rejection. No source was found fabricated or non-primary. No PARTIAL items remain.

## C1 — TAFAS

**Claim checked:** TAFAS adapts auxiliary calibration modules around a frozen forecasting backbone using arrived partial targets and older complete targets; this is forecasting evidence, not a demonstrated guarantee of downstream disruption-classifier improvement.

**Primary metadata:** HyunGi Kim, Siwon Kim, Jisoo Mok, and Sungroh Yoon, “Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation,” *Proceedings of the AAAI Conference on Artificial Intelligence* 39(17), 17868–17876 (2025), [DOI: 10.1609/aaai.v39i17.33965](https://doi.org/10.1609/aaai.v39i17.33965). Checked the [author-posted full text, arXiv:2501.04970v1](https://arxiv.org/html/2501.04970v1) and [AAAI proceedings PDF](https://ojs.aaai.org/index.php/AAAI/article/download/33965/36120). The publisher landing page returned an access error on live opening; its indexed publisher record supplied the issue/pages/DOI, and the full primary texts were accessible.

**Verbatim support**, Figure 2 caption: “The source forecaster is frozen to preserve its core semantic information learned.” The *Gated Calibration Module* section places adapted GCMs before and after that forecaster. Equations 4–6 define the adaptation loss from the newly available partial ground truth and an older mini-batch whose full ground truth has arrived. The section also explicitly states that, at the beginning or whenever no older full batch is available, only the partial term is used. Thus the claim describes the method overall; it would be wrong if read to require both kinds of targets at every update.

**Scope check:** *Experimental Setup* lists seven multivariate time-series forecasting benchmarks, and Table 1 reports forecast MSE. The inspected paper does not present a downstream tokamak disruption-classifier evaluation or prove an improvement guarantee for one. Searches of the full text for “plasma,” “tokamak,” and “disruption” found no matches. This is a conclusion about this paper, not an exhaustive claim about all later work. **Verdict: CONFIRMED**, with the availability qualification above. [Primary full text](https://arxiv.org/html/2501.04970v1)

## C2 — Sun et al. test-time training

**Claim checked:** Sun et al. 2020 establish a sufficient task/auxiliary gradient-alignment condition under smooth convex assumptions; this does not prove the corresponding guarantee for a fixed nonlinear tokamak Transformer.

**Primary metadata:** Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt, “Test-Time Training with Self-Supervision for Generalization under Distribution Shifts,” *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119:9229–9248 (2020). Checked the [PMLR record](https://proceedings.mlr.press/v119/sun20b.html), [PMLR full PDF](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf), and [author-posted full text, arXiv:1909.13231v3](https://arxiv.org/html/1909.13231).

**Verbatim support**, §4 immediately before Theorem 1: “Our main theoretical result extends this to general smooth and convex loss functions.” The theorem itself states the alignment condition and one-step conclusion; the following is a mathematical paraphrase, not a quotation. Let \(g_m\) be the gradient of the labeled main-task loss and \(g_s\) that of the self-supervised loss, both with respect to the parameters being updated. The main loss is differentiable, convex and \(\beta\)-smooth in those parameters, both gradient norms are bounded by \(G\), and with \(\eta=\epsilon/(\beta G^2)\), the condition \(\langle g_m,g_s\rangle>\epsilon\) yields a strict decrease in that example's main-task loss after one auxiliary gradient step. This is not a theorem about repeated updates, aggregate classification metrics, or disruption alarms. [§4 and Theorem 1](https://arxiv.org/html/1909.13231)

**Proof-line caveat and independent check:** In [Appendix A3.2 of the published PDF](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf), the first displayed smoothness bound prints a **plus** before \(\eta\langle g_m,g_s\rangle\), although the step is \(\theta-\eta g_s\); the subsequent negative-term bound does not follow from that printed line as written. Independently, the standard smoothness bound for this step has a minus sign. For positive \(\epsilon,\beta,G\), substituting the stated step and gradient bound gives a loss change strictly below \(-\epsilon^2/(2\beta G^2)\), so the theorem's stated one-step conclusion is consistent with that corrected algebra. This check does not establish its premises for another model.

**Scope check:** The theorem's assumptions concern the actual trainable parameterization and example; a nonlinear architecture alone does not establish or refute them. The paper's non-convex evidence is empirical, not a proof for the requested Transformer, and its experiments concern image classification. The main-loss gradient in the condition depends on the true label. No claim is made here that the condition is satisfied or verifiable for the specified fixed tokamak model. **Verdict: CONFIRMED**, including the explicit limitation. [§§3–4](https://arxiv.org/html/1909.13231)

## C3 — Non-stationary Transformers

**Claim checked:** “Non-stationary Transformers proves that demeaning a plasma forecast loss separates safe operating drift from dangerous physical evolution.”

**Primary metadata:** Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long, “Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting,” *Advances in Neural Information Processing Systems* 35, NeurIPS 2022 main conference track, [DOI: 10.52202/068431-0718](https://doi.org/10.52202/068431-0718). Checked the [publisher record](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html) and the [primary conference PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/4054556fcaa934b0bf76da52cf4f92cb-Paper-Conference.pdf).

**Verbatim relevant text**, §1: “And De-stationary Attention approximates the attention of unstationarized data and compensates the intrinsic non-stationarity of raw series.” This describes what the authors' method is intended to recover, not the suspect safety claim. In §3.1, the paper subtracts an input-series mean and scales by its standard deviation, then restores statistics at the output. That is not demeaning a forecast loss. Section 3.2 warns that stationarization can erase temporal distinctions and introduces de-stationary attention; it does not establish a safe/dangerous physical decomposition.

**Scope check:** Section 4.1 lists electricity, transformer electricity, exchange rates, influenza-like illness, traffic, and weather forecasting benchmarks. Text searches of the conference PDF found no “plasma,” “tokamak,” or “disruption.” There is no supporting sentence or relevant validation for the asserted plasma-safety separation in this source. This is not a proof that such a separation is impossible; it is a rejection of this source–claim pairing. **Verdict: REJECTED** — `does_not_support_claim` (also wrong task/population). [§§3–4, conference PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/4054556fcaa934b0bf76da52cf4f92cb-Paper-Conference.pdf)

## Rejection carried forward

Do not cite Liu et al. 2022 as proof that demeaning a plasma forecast loss separates benign drift from dangerous physical evolution. Its normalization and forecasting findings support only their own stated scope.
