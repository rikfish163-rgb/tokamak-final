# Input-coordinate diagnostic: qualified source versus ended EAST20

2026-11-04. This is a **CPU-only, retrospective input-coordinate description**.
It neither diagnoses a physical unit error nor establishes a cause of false
alarms. There was no fitting, new alignment, GPU use, outcome read, model
inference, or change to a shared producer.

## Outcome in brief

The existing cold20 map is a `density_gain` map: it changes **only** `ne_nG`
by the already saved gain `0.00024050718742271607`; the other 22 channels are
identity. On this sampled EAST20, the source-coordinate `ne_nG` median was
`9.6804` before the map and `-0.0008511` after it. Its sampled fraction with
`|asinh| > 5` went from `2560/2560` to `0/2560`. The comparison on these
same first 20 shots is **in-sample and retrospective**: their original
operational predictions used unmapped inputs, and this fixed map first
applies operationally from shot 21. It cannot establish suffix behavior.

After the map, `n=1 amplitude` remains a large coordinate outlier under the
declared descriptive flag: `2558/2560` whole-record samples and `2539/2560`
first-128 samples have `|asinh| > 5`; the map leaves this channel untouched.
`qa_proxy`, `ip`, `Bt`, and the sampled SXR/XUV channels do not meet that
flag, although their distributions can differ substantially. In particular,
`qa_proxy` has a whole-record numerical IQR ratio of `37.13` and `Bt` `9.66`
(EAST/source); neither ratio proves matched physical units or a unit bug.

The source itself is not uniformly concentrated: its `mode_number_n` has
`13.31%` of samples above `|asinh| > 5` and `13.17%` above 10, with frozen
source scale `1e-6`. The source64 sample is not the entire TRAIN population
and was not selected to represent it. These observations are development
diagnostics, not independent validation or evidence that a distribution
change explains EAST FPR.

## Input contract and selection

- Source IDs are the first 64 sorted canonical TRAIN IDs; all 64 exact IDs,
  sample indices, and per-shot receipts are in the [protocol](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/input_coordinate_diagnostic_20261104_v1/protocol.json).
  There was no source VAL/TEST signal read. The source samples total 7,996:
  61 records contribute 128 points each and three shorter records contribute
  48, 52, and 88 points.
- The exact first 20 in the completed RLS protocol and the fixed map's
  calibration IDs, in order, are: `71304, 87494, 69458, 87487, 69854, 70404,
  70323, 86167, 54157, 63097, 91054, 91885, 77002, 94428, 61468, 85632,
  85320, 85316, 91723, 75657`. No other EAST signal was read.
- The code calls the existing `simple_data`/`io23.read_signals`, the EAST-only
  `apply_input_contract`, and the saved map's existing `transform`. The
  source uses its original finite mask, without the EAST CIII sentinel rule.
  On EAST, model-input MNM is unavailable under `simple_data` regardless of
  the HDF value; CIII uses the existing pointwise `-10 ± 0.01` sentinel rule.
  Other finite zero values are **not** blanket-masked. The EAST model masks
  are identical before and after the map.
- For CIII, `1920/2560` sampled points from 15 of 20 records are model
  available in both the spread and prefix samples. The HDF reader's finite
  mask is `2560/2560`; finite sentinel values explain why finiteness is not
  itself model availability. EAST MNM likewise has `2560/2560` finite HDF
  sample values but `0/2560` available to this model. This does not establish
  whether its finite HDF entries are physical observations. These are sample
  counts, not claims about every unsampled point.
- Existing `simple_data` multiplies **source J-TEXT `ip` by 1000** before the
  model; the diagnostic uses that exact existing contract. All other ratios
  and raw numerical summaries refer to post-contract, pre-map or mapped
  numbers, not inferred physical-unit conversions. HDF `Unit` attributes were
  not inspected; equality of units across devices is not assumed.

## Sampling and 23-channel results

For each ended record, choose at most 128 deterministic, evenly spread
indices over its length. EAST also has a separate sample of its actual first
128 points; these two views can overlap. The existing reader loads the whole
23-channel record, but each coordinate-distribution summary uses no more than
128 points per shot **per view**. There is no label-based selection or
preterminal cropping. Quantiles pool the available sampled points, so a shot
with missing observations contributes fewer points; the samples are not
independent and the result is not an equal-shot inferential estimate.

The [full all-23 tables](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/input_coordinate_diagnostic_20261104_v1/tables.md)
give, for every channel, the frozen source center/scale, sample availability,
source and EAST whole-record q05/median/q95 in source asinh coordinates,
before/after `|asinh| > 5` and `>10` fractions, numerical median/IQR ratios,
and the separate EAST prefix q05/median/q95 and fractions. Full precision,
the prefix ratios, HDF-finite counts, and counts of represented shots are in
[summary.json](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/input_coordinate_diagnostic_20261104_v1/summary.json);
the 115 group-by-channel rows are also in
[summary.csv](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/input_coordinate_diagnostic_20261104_v1/summary.csv).

The following compact all-channel table makes the outlier flag visible.
`S`, `E`, and `E-map` are source, EAST raw and EAST mapped **asinh medians**;
fractions are sampled `|asinh| > 5` before → after map. The flag means that
at least half the available pooled samples exceed 5 in **both** EAST
whole-record spread and first-128 views. It was set in the probe before the
pass. It is not a physical criterion, does not imply every shot or every time
point is extreme, and is unavailable when there is no model support.

| RAW23 channel | S median | E median | E-map median | Full >5 raw → map | First128 >5 raw → map | Flag raw → map |
|---|---:|---:|---:|---:|---:|---|
| Bt | 0.6277 | 1.4815 | 1.4815 | 0 → 0 | 0 → 0 | no → no |
| CIII | 0.7523 | -0.9536 | -0.9536 | 0 → 0 | 0 → 0 | no → no |
| MNM | -0.6202 | — | — | — | — | no support |
| Mir_VV | -0.1262 | -0.5403 | -0.5403 | 0 → 0 | 0 → 0 | no → no |
| Mir_avg_amp | -0.2542 | -0.2542 | -0.2542 | 0 → 0 | 0 → 0 | no → no |
| Mir_avg_fre | -0.0553 | -0.0553 | -0.0553 | 0 → 0 | 0 → 0 | no → no |
| Z_proxy | 0.0121 | -1.4972 | -1.4972 | 0.000781 → 0.000781 | 0 → 0 | no → no |
| dx_a | 0.3314 | 2.7861 | 2.7861 | 0 → 0 | 0 → 0 | no → no |
| dy_a | -0.4755 | 3.0465 | 3.0465 | 0.000391 → 0.000391 | 0 → 0 | no → no |
| ip | 0.5008 | 2.1745 | 2.1745 | 0 → 0 | 0 → 0 | no → no |
| mode_number_n | 0 | 0 | 0 | 0.004688 → 0.004688 | 0 → 0 | no → no |
| n=1 amplitude | 1.3927 | 5.9558 | 5.9558 | 0.999219 → 0.999219 | 0.991797 → 0.991797 | yes → yes |
| ne_nG | 0.4561 | 9.6804 | -0.000851 | 1 → 0 | 1 → 0 | yes → no |
| qa_proxy | -0.000064 | 0.1275 | 0.1275 | 0.000781 → 0.000781 | 0 → 0 | no → no |
| sxr_kurt | 0.5441 | 1.1439 | 1.1439 | 0 → 0 | 0 → 0 | no → no |
| sxr_mean | 0.7008 | -0.9510 | -0.9510 | 0 → 0 | 0 → 0 | no → no |
| sxr_skew | 0.1279 | -0.1813 | -0.1813 | 0 → 0 | 0 → 0 | no → no |
| sxr_var | 0.8709 | -0.5269 | -0.5269 | 0 → 0 | 0 → 0 | no → no |
| v_loop | 0.1796 | -1.4341 | -1.4341 | 0 → 0 | 0 → 0 | no → no |
| xuv_kurt | 0.1288 | 0.4278 | 0.4278 | 0 → 0 | 0 → 0 | no → no |
| xuv_ratio | -0.0552 | 2.1140 | 2.1140 | 0.000391 → 0.000391 | 0 → 0 | no → no |
| xuv_skew | 0.0590 | 0.4742 | 0.4742 | 0 → 0 | 0 → 0 | no → no |
| xuv_var | 1.1802 | -0.5616 | -0.5616 | 0.00625 → 0.00625 | 0 → 0 | no → no |

For `ne_nG`, `|asinh| > 10` is `934/2560` for the whole-record spread and
`1024/2560` in the first-128 view before mapping, and zero in both after
mapping. Its raw numerical whole-record EAST/source median and IQR ratios
are `3337.70` and `10210.46`; after mapping they are `0.8027` and `2.4557`.
Ratios are reported only where the source denominator is nonzero; `—` is
undefined, not zero. Ratios compare stored model-input numbers and **do not
assert common units or a physical conversion**.

## Reproducibility and checks

- Actual run: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.coordinate_probe_20261104`.
  It exited 0, reported `source_shots=64`, `east_shots=20`, and `0.64173 s`
  internal elapsed time; CPU only. The reader accesses `/data` signals and
  their existing rate/start attributes, not HDF outcomes. Outcome access is
  explicitly guarded in the execution path.
- Qualified checkpoint SHA-256:
  `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`;
  loaded source state SHA-256:
  `99affd49f71f41129b87b03624201ef94ef6dc44e652eaef411b2924f8186dc8`.
  Both match the completed qualified baseline and RLS protocol; no old
  125/35 checkpoint was substituted. Map SHA-256:
  `43c37ad981261e059ac8bf8d11d89324bad375dd7405decc194c51e53614aff2`.
- The [completed receipt](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/input_coordinate_diagnostic_20261104_v1/completed.json)
  binds protocol, summary JSON/CSV and tables; all four hashes were
  independently recomputed and matched. The protocol binds the checkpoint,
  index, old-RLS/baseline protocols, map, source code, and test code; the six
  input/model dependencies match the archived RLS code hashes. Each shot has
  a hash of typed **sampled signal arrays and masks**, plus file size/mtime.
  This is not a whole-HDF hash; none was computed, to avoid reading outcome
  bytes. Size/mtime plus sampled-content hashes do not prove whole-file
  identity.
- TDD: the initial test invocation failed at collection with
  `ModuleNotFoundError` for the not-yet-created probe; after implementation,
  `6 passed in 1.43s` on CPU. Tests cover fixed sampling, map-mask invariance,
  source/EAST CIII distinction, MNM model mask, observed zeros, prefix versus
  whole sampling, ratios and exact cohort validation. Saved-output checks
  confirmed 23 channels, 115 CSV rows, unchanged masks, and identical
  pre/post-map summaries for all channels except `ne_nG`.

This sidecar used the results-analysis workflow to separate observed
coordinate differences from claims about physical calibration or model
performance; it made no selection of a new map, threshold or model.
