# RAW23 input-pipeline audit, 2026-10-31

## Scope and bottom line

This is a bounded, CPU-only audit of the recorded `anchored_full360_diagnostic_v1` run and its lifecycle reference, not proof of what an independently running production service currently executes. No model, code, data, or experiment artifact was changed. The material findings are:

1. The source-training and recorded deployment readers use the **same RAW23 order** and the same J-TEXT `ip *= 1000` rule. EAST `MNM` is explicitly unavailable, whereas J-TEXT `MNM` is not categorically masked. EAST additionally masks CIII values near -10; the hash-matched J-TEXT source reader treats any finite CIII value, including -10, as available. The follow-up below found no such CIII sentinel in the current canonical-hash-matched J-TEXT TRAIN1353 files, so the source-side sentinel-rule difference is not observed on those files.
2. `mode_number_n` is retained as a risk input with center 0 and scale approximately 1e-6; a value of 1 maps to `asinh(1e6) = 14.5087`. Among suffix340 samples, 69,868/1,438,358 (4.86%) are nonzero and 66,015 exceed 10 in absolute asinh coordinates. This is a verified sharp numerical transformation, not evidence by itself that the model is wrong or that a nonzero mode is out of source support.
3. Cold20 `qa_proxy` is on a markedly different scale from the separate J-TEXT source-statistics cache: equal-shot weighted medians 7.51065e-8 versus 4.12120e-6, about 54.87-fold. Its median asinh coordinate is about -2.106 using the checkpoint center/scale. This is **not universal to EAST**: 93/340 suffix shots have a per-shot median below 2e-7, and 247/340 above 2e-6. The calculation is descriptive; no causal or physical explanation is established here.
4. The recorded affine map changes **only `ne_nG`**, by 0.0003150853015380819 with zero offset. It is an empirical source/cold20 median ratio, explicitly not a certified physical calibration. Cold20 predictions themselves were made before this map existed; the mapped inputs apply to the suffix and were also used later in fitting from ended cold20 shots.

## Evidence and reproducibility

Let **WT** be `/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028`; **MAIN** is `/media/hetaisheng/044A81D94A81C83E/Fuison`; **R** is `WT/probe_outputs/anchored_full360_diagnostic_v1`; **P** is `MAIN/experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028`. The WT `data`, `artifacts`, and `experiments/risk_aligned_online/runs` paths are symlinks into MAIN. Line citations below are relative to these definitions; ZIP citations use `archive!entry:line`. A colon on a checkpoint or JSON identifies an array/key rather than a text line.

* The locked `MAIN/experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt` SHA-256 was recomputed as `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`, matching `R/protocol.json:392` and `WT/experiments/risk_aligned_online/redesign_20261004/prepare_prior.py:22-23,54-70`. CPU `torch.load` was performed only after the hash check. Its config is source view, W128/P8, dim 64, horizon 25, token fast site and normalized fast rule. Checkpoint metadata center/scale are bitwise equal to its actual `model.center`/`model.scale` buffers.
* The initial source checkpoint `MAIN/experiments/risk_aligned_online/runs/simple23_source_20260929/source.pt` hashes to `2291666ac48203d4e50468ca5fc242648620e8599c28d436309b6e3aaef4f0ae`, the parent recorded in `gradient_aligned512_20260929/protocol.json:11`. Initial and final center/scale arrays are bitwise identical. The current joint-training source `WT/experiments/risk_aligned_online/simple_train.py` hashes to its recorded joint-run hash `5716a805...d6ab9`; its `load_shots` uses `data.read_shot` at lines 67-68. The current `simple23.py` hashes to the joint-run recorded `2234ab9c...e7b22` (`gradient_aligned512_20260929/protocol.json:1530-1537`).
* Initial source `config.json:19-23` records historical `simple_data.py` SHA `751ee1f3...f70fa` and `io23.py` SHA `99ad1ff5...9d26a`. Exact-byte matches were recovered from `MAIN/experiments/risk_aligned_online/runs/redesign_ordered_error_full360_20261004/code_snapshot.zip!simple_data.py` and `!io23.py`. The former is also the joint-run recorded `simple_data.py` hash. These are historical implementations, not assumptions based on the live filenames. The original initial-training `simple_train.py` and `simple23.py` hashes were **not** recovered in the 283 run snapshots checked; we did not reconstruct all initial training data or independently reproduce scaler fitting.
* The map hash was independently recomputed as `a1516e9ce835dc8803797712003e80604e49aa69b4f726ea7045db49bca21e4b`, matching `R/protocol.json:381-388`. Reading the 20 actual masked records with the current hash-matched run code, rebuilding `fit_prefix_map` against the saved source-statistics cache returned a Python-object **exact match** to `P/input_map.json`; the source-statistics cache and embedded source statistics also matched exactly. Independently, all 20 recorded `cold` arrays and timestamps in R were bitwise equal to P's respective arrays.
* Recomputed SHA-256 of all 360 current EAST HDF5 files against `P/input_hashes.json`: **360/360 matched**. A separate CPU, read-only pass over only CIII, mode_number_n, ne_nG, and qa_proxy read every sample, cast to float32 as the reader does, and used finite masks plus the CIII sentinel rule. Quantiles below are exact sample-weighted quantiles over those arrays, not endpoint-weighted model-window distributions, shot-weighted estimates, or preterminal-only summaries. No outcome labels were read. The source/cold20 cached median statistics have a different estimator: at most 128 available points per shot and equal shot weights (`WT/experiments/risk_aligned_online/warmstart_alignment_20261022/alignment.py:37-83`).

## Input construction and masks

`R/code_snapshot.zip` contains byte-identical copies of the live WT `io23.py`, `simple_data.py`, `simple23.py`, `causal_input.py`, `warmstart_alignment_20261022/alignment.py`, `attention_lowrank_20261029/model.py`, `anchored_stream_20261031/run.py`, and `redesign_20261004/prepare_prior.py`. The corresponding common input files are also byte-identical in P's and `WT/probe_outputs/arrival_reference_east360_v1`'s snapshots. This supports using the cited live WT line numbers for the recorded input code; it does not prove every file in those snapshots is unchanged.

The source hash-matched `redesign_ordered_error_full360_20261004/code_snapshot.zip!io23.py:25-29,43-76` and deployed `WT/experiments/risk_aligned_online/io23.py:35-39,62-96` have identical channel ordering and both load HDF5 arrays as float32, stack in that order, mark finite values available, and zero nonfinite stored values. The original source required exactly 1000 Hz; the recorded EAST run explicitly opts into nominal 1000 ± 0.5 Hz without resampling or rounding the actual clock (`io23.py:25-33,47-53,74-93`; `anchored_stream_20261031/run.py:74-77`). Thus current timing acceptance is not byte-identical to historical source acceptance.

The source-matched snapshot `!simple_data.py:15-26` and live `simple_data.py:15-29` both multiply only J-TEXT `ip` by 1000 and leave EAST `ip` unchanged; the archived reader comments describe this as kA-to-A (`!io23.py:43-48`). This report verifies code behavior, **not** upstream instrument/unit provenance. Examined HDF5 attributes for EAST 71304 and J-TEXT 1051500 on ip/CIII/mode/ne_nG/qa_proxy contained only SampleRate and StartTime; those examples do not certify physical units for the whole dataset. No other physical unit conversion is established by these readers.

For EAST, `simple_data.py:20-26` forcibly masks `MNM` for every sample. `causal_input.py:8-17,87-107,126-133` then masks CIII within absolute tolerance 0.01 of -10, leaving its stored value unchanged; zero remains available. J-TEXT historical `!simple_data.py:15-23` has only finite masking and no CIII sentinel treatment. At model entry, `attention_lowrank_20261029/model.py:31-39` recomputes finiteness, substitutes the source center for unavailable values (so normalized value is zero), concatenates the availability flag, and marks a patch valid if any feature is available. Padding is unavailable (`simple_data.py:41-48`). Therefore absent and observed zero are distinguishable by the flag, but source versus EAST missingness patterns differ. In this recorded run mode_number_n remains in the risk input; it is excluded from the auxiliary *arrived-change* target at `context_vector_20261030/observation.py:14-26`, not automatically removed from the classifier.

Measured CIII: cold20 has 63,156 sentinel points out of 70,996 and only 1/20 shots has any nonsentinel point; suffix340 has 214,476/1,438,358 sentinel points and 290/340 shots with any nonsentinel point. All CIII values in these 360 records were finite before the sentinel rule; combined, 277,632/1,509,354 (18.39%) are masked by that rule. The convention is explicitly a local candidate, not a universal physical fact. A change in source CIII prevalence or mask semantics cannot be inferred from this EAST-only count.

## Checkpoint coordinates and scale floor

The hash-matched historical `!simple_data.py:48-63` defines center as sampled median and scale as `max(q75-q25, 0.01*abs(median), 1e-6)` using valid/available J-TEXT training samples, with per-shot stride `max(1, len(v)//128)`. The separately saved alignment statistics are **not** those same statistics or estimator. The final checkpoint stores `mode_number_n` and `qa_proxy` scales at float32 1e-6; this is consistent with the hard floor, and the hash-matched scaler function specifies it, but its historical call and exact source-training quantiles were not independently replayed. The separate source cache has mode median/IQR 0/0 and qa median/IQR 4.12120e-6/6.99334e-7, consistent with this interpretation but not proof of the exact original quantiles.

The deployed source-view embedding is `asinh((x-center)/scale)` plus availability, not a plain z score (`attention_lowrank_20261029/model.py:31-36`; `simple23.py:74-105`). Asinh compresses large standardized values, but does not eliminate a large jump produced by a tiny denominator. In particular 0 -> 1 on mode changes the coordinate 0 -> 14.5087; this transformation was also in the hash-matched *joint-training* `simple23.py`, so a large value is not automatically novel to EAST. The separate source cache has mode mean 0.4292, std 2.2222, median/IQR zero, indicating sparse nonzero source activity; exact nonzero training distribution was not reconstructed.

Full stored checkpoint centers/scales (rounded for presentation; zero-based RAW23 order; units are checkpoint input units, not independently verified physical units):

| # | Channel | center | scale |
|---:|---|---:|---:|
| 0 | Bt | 1.79189909 | 0.297771335 |
| 1 | CIII | 1.02935922 | 0.914565027 |
| 2 | MNM | 1.81992531 | 2.75427389 |
| 3 | Mir_VV | 0.160240591 | 0.271385074 |
| 4 | Mir_avg_amp | 3.85892153 | 15.0171614 |
| 5 | Mir_avg_fre | 200.161911 | 3616.10596 |
| 6 | Z_proxy | 1.03947832e-5 | 4.10272014e-6 |
| 7 | dx_a | 0.373035371 | 1.68570709 |
| 8 | dy_a | 0.450730920 | 0.504392982 |
| 9 | ip | 163239.5 | 32276.6953 |
| 10 | mode_number_n | 0 | 9.99999997e-7 |
| 11 | n=1 amplitude | 3.88152623 | 3.51119494 |
| 12 | ne_nG | 0.00229155039 | 0.00119009311 |
| 13 | qa_proxy | 4.12109330e-6 | 9.99999997e-7 |
| 14 | sxr_kurt | -1.50039411 | 0.586274564 |
| 15 | sxr_mean | 0.197468817 | 0.164281279 |
| 16 | sxr_skew | 0.449276865 | 0.348565429 |
| 17 | sxr_var | 0.0322138742 | 0.0574340038 |
| 18 | v_loop | 1.68996024 | 0.568500102 |
| 19 | xuv_kurt | -0.111745745 | 1.34991932 |
| 20 | xuv_ratio | 0.315701723 | 0.148420706 |
| 21 | xuv_skew | -0.121434525 | 0.501304805 |
| 22 | xuv_var | 0.0224563740 | 0.0329195671 |

For qa in cold20, the sample-weighted raw p25/median/p75 are 7.14594e-8 / 7.46106e-8 / 7.75682e-8, and the asinh p25/median/p75 are -2.10668 / -2.10593 / -2.10522. Thus within-cold20 variation is small in these coordinates, not necessarily amplified by asinh; the large effect is the offset from the source center. In suffix340, raw median is 3.48806e-6 and asinh median is -0.59694, with raw min/max 1.78593e-8 / 1.91962e-4. Do not extrapolate the cold20 mismatch to every suffix shot. This audit does not determine whether low/high groups reflect real physics, upstream preprocessing, units, time periods, or a data defect.

For mode, cold20 has 140/70,996 nonzero samples and max asinh 16.4904; suffix340 has 69,868/1,438,358 nonzero, raw max 7.99236 and max asinh 16.5871. Within suffix nonzero samples, asinh median is 14.8510; 66,015 total suffix samples have |asinh| > 10. Very small but nonzero mode values exist, so nonzero and threshold counts are not identical.

## Density map and cold-start timing

`P/input_map.json:2-51,114-136,193-214,1902-1905` is policy `density_gain`, gain 1 and offset 0 everywhere except ne_nG (index 12) gain 0.0003150853015380819, offset 0. `alignment.py:86-101,118-125,128-156` shows the gain is the positive source median divided by target median and only available values are mapped; availability is unchanged. The cached equal-shot source median 0.0022905373952627536 divided by cold20 median 7.269578695297897 yields that gain. It is a fitted statistical alignment, **not** a certified conversion of units. Other channels including mode, qa, and Bt stay identity; no conclusion about their physical comparability follows.

The recorded full360 runner `anchored_stream_20261031/run.py:58-78,120-127` loads archived predictions for indices 0-19 and applies this map only to raw suffix indices 20-359. P's **archived** `code_snapshot.zip!experiments/risk_aligned_online/lifecycle_stream_20261025/run.py:135-144,159-170` shows original cold20 predictions were computed before mapping, the map was fitted after shot 20, and the completed cold20 samples were subsequently mapped for cold fitting. Direct bitwise curve/time comparison confirms the R and P cold20 artifacts agree; no GPU inference replay was performed.

For perspective, in reconstructed cold20 input the unscaled ne_nG sample median is 9.53467 and its median asinh coordinate is 9.68157. In mapped suffix340, input median is 0.00288953 and median asinh is 0.483416; min/max are -0.000349533 / 0.202254 and corresponding asinh range -1.53759 / 5.81726. These are different shot groups, not a paired causal effect estimate.

## Historical/live difference and limits

P's archived `lifecycle_stream_20261025/run.py` SHA is `8b7da79886a7275264eeb5352ab4ab7d0c8c799f375796fbb5ed910c88f7cf2b`; the current WT file hashes to `01adc4eaf8afd11b8209e8727a7e0fa1a4eeb53cca949e723f5cee9e261c283c`. A read-only diff shows that the current file loads context memory into all three arms instead of only the cold arm before state copying. The cited archived file, not the later live version, establishes historical execution. The specific input reader, CIII contract, scaler model, and affine-map implementation compared above were byte-identical in the checked P/R snapshots.

This is a diagnostic run marked `component_screen=FAILED`, `promoted=false`, `independent_confirmation=false` (`R/protocol.json:397-402`); it is not a deployment qualification or an estimate of the effect of fixing any input. No GPU, training, model inference replay, code edits, raw-data edits, commits, or deletes were performed. Full 23-channel raw census and Bt/qa/Z physical algebra were intentionally left to the main audit owner; only the four assigned channels above were numerically scanned. Exact historical source fit data provenance and physical unit/extraction equivalence remain unresolved. Recommended next verification is to trace acquisition/feature-generation provenance and source sampling/masks before proposing or running retraining, with any corrective candidate evaluated independently rather than treating a statistical median match as physical calibration.

## 补充：J-TEXT TRAIN1353 的 CIII 和 mode 全样本核验

只读取 canonical TRAIN 的 1,353 炮 HDF5 中 `data/CIII` 和 `data/mode_number_n`，不读取 HDF5 outcome、不按结局或有效区间筛选。名单与 `gradient_aligned512_20260929/protocol.json` 的 `train_ids` 顺序完全一致；1,353 个当前 HDF5 文件的 SHA-256 均匹配 canonical index 对应 `hdf5_sha256`。该 index 的 SHA-256 为 `30b7ddbd6ab9f1fc1386c250c01663f80af23f1e31a9c987101ca32c4dc6c1b5`。这验证当前文件与该 canonical index 一致，但不单独证明历史训练时所有输入文件的哈希已被锁定。

两通道各有 **476,194 个原始采样点**，覆盖 1,353 炮；每炮长度 32–516 点，其中 36 炮短于 128 点，**全部纳入**以下统计。原始 HDF5 两通道在全部炮中均为 float64；同时检查了源读取器使用的 float32 转换口径。

| 检查项 | 精确计数 |
|---|---:|
| CIII 有限值 / NaN / ±Inf 点 | 476,194 / 0 / 0 |
| CIII 满足 `abs(x + 10) <= 0.01` 的点 | 0 / 476,194 |
| 含至少一个 CIII 哨兵的炮 | 0 / 1,353 |
| mode 有限值 / NaN / ±Inf 点 | 476,194 / 0 / 0 |
| mode 非零点 | 71,966 / 476,194（15.112748%） |
| mode 非零至少一点的炮 | 430 / 1,353 |
| mode `abs(asinh((x-center)/scale)) > 10` 的点 | 71,298 / 476,194（14.972469%） |
| mode 上述坐标超过 10 至少一点的炮 | 429 / 1,353 |

CIII 的原始 float64 与转换后 float32 哨兵判定差异为 0；mode 的原始与 float32 非零判定差异也为 0。mode 坐标使用哈希锁定检查点的 center=0、float32 scale=9.999999974752427e-7，转换后最大 mode=15.9975080，最大 asinh 坐标=17.28109。按 `abs(float32(x)) > sinh(10) * scale` 的独立第二次扫描复核了 71,298 点；没有坐标落在距阈值 10 的 1e-4 范围内。

**解释：** 对这些经过 canonical 哈希核验的当前源 TRAIN 文件，历史源读取器与 EAST 规则的 CIII 哨兵区别在源侧没有产生可观察的哨兵处理差异，因为源侧根本没有该哨兵；EAST 侧上述统计中的哨兵和屏蔽则确实存在。mode 的大 asinh 值也并非源 TRAIN 原始信号中未出现：源侧全样本 `|asinh| > 10` 为 14.972469%，而此前 EAST suffix340 对应全样本为 66,015/1,438,358（4.589608%）。这些是按时间点加权、未按结局裁剪的原始信号比例，不能直接当作优化器训练采样分布，也不能证明各点实际进入训练窗口或解释跨设备物理等价性。
