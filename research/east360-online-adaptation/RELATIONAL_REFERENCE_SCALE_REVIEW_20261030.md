# Standardized relational reference — independent CPU review

**The saved summary and six failed gate conditions reproduce exactly. The new compensation changes the outputs substantially, but it is not an accepted method.** No concrete implementation or summary defect was found in the reviewed new code. This review covers only `relational_reference_standardized_v1` (the v2 reference experiment), conditional on the shared cold20 parent. **No full360 claim.**

## Verified results

I independently computed stable BCE from saved NPZ logits, averaged within each shot/group and then equally across shots. All **270 per-shot/group/arm rows, 20 group summaries and gate fields match exactly**, with maximum numerical discrepancy zero. No original summary or acceptance helper was imported. [Saved results](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/relational_reference_standardized_v1/complete.json:1); [gate specification](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/relational_attention_20261030/run_reference_scale.py:23).

| Arm | Safe mean ΔBCE | Safe improved | Timely-danger mean ΔBCE | Danger improved |
|---|---:|---:|---:|---:|
| real | +0.0001283748 | 0/3 | -0.0003448592 | 8/17 |
| reverse | +0.0001092507 | 0/3 | +0.0000425825 | 6/17 |
| unconditioned | +0.0001663056 | 0/3 | -0.0009332209 | 7/17 |
| mean_target | +0.0008409372 | 0/3 | -0.0002235220 | 4/17 |
| mean_direction | -0.0037196811 | 3/3 | +0.0020897221 | 11/17 |

Real worsens all three safe shots; timely danger improves on 8/17 and worsens on 9/17. All six failure reasons match: safe mean does not improve cold; safe improvement count is insufficient; safe does not beat reverse or mean_direction; danger improvement count is insufficient; danger does not beat unconditioned. Support and specificity pass. Relative to v1, safe harm is smaller (+0.000197669→+0.000128375), while danger mean benefit is smaller (−0.000727162→−0.000344859). The number of improved primary-group shots does not change.

**Pairing is exact across all 2,707 opportunities:** cold logits, edge masks and support flags match v1 bit for bit for all 20 shots; records and folds are also identical. All opportunities remain supported, with 124,522 edges and 10,828 supported rows. Thus the observed difference is not explained by a changed baseline, endpoint population or common-support selection. [Runner's paired checks](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/relational_attention_20261030/run_reference_scale.py:121).

## Conditioning is now functionally distinguishable

| Saved real versus unconditioned query logits | v1 | Standardized reference |
|---|---:|---:|
| Median absolute difference | 0.0000038147 | 0.0012664795 |
| Mean absolute difference | 0.0000182482 | 0.0030442475 |
| Differences >1e-4 | 88/2,707 (3.25%) | 2,472/2,707 (91.32%) |
| Safe-window differences >1e-4 | 41/527 | 507/527 |
| Danger-window differences >1e-4 | 3/63 | 58/63 |

The near-identity seen in v1 has therefore been removed **at the saved query-output level**. Within v2, real improves safe mean ΔBCE relative to unconditioned by 0.000037931, but worsens danger mean ΔBCE by 0.000588362. Both still fail the joint utility requirements.

Real targets also change relative to v1: median total variation across supported rows is **0.206576**. Fixed mean targets change in all three folds; mean-gradient cosines relative to v1 are 0.990707, 0.771147 and 0.928775. These are saved-reference comparisons, not refits. Unconditioned targets and per-window Q/K vectors are not saved, so exact conditioned/unconditioned target and direction differences are not claimed here.

Real/reverse specificity still passes: median saved direction cosine **0.811150**, with **92.6117%** of future logits differing by >1e-4. Real auxiliary KL decreases by >1e-6 at **all 2,707 opportunities**, with window mean 0.475385358→0.469845316. This coexists with safe-shot harm. Reverse also decreases at all opportunities; unconditioned increases once and mean_target increases eight times. Each auxiliary arm uses its own target, so their absolute KL values are not interchangeable risk scores.

## TRAIN-only standardization and saved-reference checks

The new API accepts only supplied exact 256×23 CPU prefixes. It has no held-shot, label, outcome or file-reader argument. The runner supplies only `first[s]` for each fold's `fit` IDs; held shots are used afterward for evaluation. The center is the mean of within-shot covariate means, and covariance is the mean of within-shot second moments around that shared center. This preserves equal-shot weighting and includes between-shot mean variation. Ridge1 penalizes standardized slopes, leaves the intercept unpenalized, and converts beta back before use by the unchanged target code. [API/moments](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/relational_attention_20261030/standardized_reference.py:22); [conversion](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/relational_attention_20261030/standardized_reference.py:63); [fit call](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/relational_attention_20261030/run_reference_scale.py:103).

From the saved fold artifacts, independently checked without fitting:

- Fit/held IDs are disjoint, match v1 exactly, and match the saved fold specifications. Reference tensors are finite, detached float64 CPU arrays; covariate standard deviations pass the declared >1e-12 threshold.
- Reconstructing original-coordinate beta from saved standardized beta, center and scale agrees within **5.69e-14**.
- Saved covariance/design eigenvalues satisfy the standardization and ridge algebra within **1.34e-15**. Standardized covariance eigenvalues span **0.585792–1.414208**, rather than the original tiny coordinate variances.
- All folds record rank3, 256 valid prefix points per fit shot, 32 valid patches and 31 differences. These support records were inspected, not regenerated from raw signals.
- D changes in every fold, and fixed controls are rebuilt using fit shots only. Therefore v2 changes the complete reference **m and D plus its fixed controls**; it cannot be described as a pure subtraction comparison holding D fixed across versions. Within each version, real and unconditioned share that version's D. [D recomputation](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/relational_attention_20261030/standardized_reference.py:83); [control rebuilding](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/relational_attention_20261030/run_reference_scale.py:107).

No numerical refit was performed, so these are API/dataflow and saved-array consistency checks rather than independent recovery of the TRAIN moments from raw prefixes. The code deliberately uses prior saved outcome groups and does not reread HDF5 outcomes. The shared parent and input mapping were already exposed to all20; component-fold isolation does not make the parent independently held out.

## Coverage and limits

**174/174 hash comparisons pass**, covering v2 artifacts/code, frozen references, the source checkpoint, prediction seals, pinned v1 completion and the paired v1 comparison artifacts. All **73 archived code members** match their recorded hashes. The qualified source checkpoint matches its pinned SHA256; cold weights and input-map files also match. This establishes current file identity, not an independent replay of in-memory parameter immutability.

All 540 v2 saved arrays are finite and aligned. Step norms remain within 3e-9 of 0.05. Both new test files were read completely (14 test functions, plus parameterization); tests involving fitting were not executed. [Reference tests](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_relational_standardized_reference.py:79); [gate tests](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_relational_reference_run.py:16).

The results-analysis workflow separates the observed change in compensation from failed risk utility. The reference specification was chosen after v1, as disclosed in the [plan](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/RELATIONAL_REFERENCE_SCALE_PLAN_20261030.md:11); neither stronger conditioning, KL decrease nor a passing specificity check makes it an accepted method. Main owns all neural replay; none is claimed as this sidecar's work.

Only this report and [independent_cpu_review.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/relational_reference_standardized_v1/independent_cpu_review.json) were written. The JSON includes independent calculations and exact verifier code. **No GPU, raw-file/signal or outcome access, fitting, model replay, source edits, deletion or commit. No full360 claim.**
