# Fixed-source threshold headroom at the first possible adaptation point

2026-11-04. This is a **posthoc capacity diagnostic of the sealed fixed source**, not a measurement of adaptation performance. The fixed stream was designed using existing data and is not an independent future sample. The primary curve is `map_after20`: unmodified source on the first 20 shots and the already-prepared numerical map from shot 21. Its map is not a certified physical calibration. The source is fixed throughout; this report does not score a trained JointLoRA, warm-static, inter-shot, or neutral-runner parent.

At the unchanged exact threshold **0.7258946895599365**, the fixed-source curve has 29 safe-shot false alarms and 16 disrupted-shot first alarms with lead >3 s. If the parent and every risk score at zero-based indices 0–134 are identical to this source, **8 false alarms and 5 >3 s first alarms have already occurred before the earliest possible update**. In the original frozen-startup protocol, the first 20 shots are source-only, so the source-specific floor is **10 false alarms and 5 >3 s alarms**, and at most 19 of the baseline 29 false alarms and 11 of the baseline 16 >3 s alarms could be removed through later within-shot changes. These are necessary constraints, not predictions or achievable gains.

## First-alarm constraints by stream segment

The feedback schedule gives the earliest possible update at index 135 (`127 + 8`): arrival/update precedes the risk at that index. A supported, nonzero update is not guaranteed there. Columns marked `<135` include indices 0 through 134; `≥135` includes 135 itself. The last two subcolumns in each group partition the baseline alarm shots, not all shots.

| Segment | Disrupted / safe | Baseline safe FP | FP first `<135` | FP first `≥135` | Baseline >3 s | >3 s first `<135` | >3 s first `≥135` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Startup 20 | 8 / 12 | 6 | 4 | 2 | 2 | 2 | 0 |
| Suffix 340 | 172 / 168 | 23 | 4 | 19 | 14 | 3 | 11 |
| Full 360 | 180 / 180 | 29 | 8 | 21 | 16 | 5 | 11 |

The `<135` floor assumes the same source predictions, same threshold, same clock and first-alarm rule; a different initial model, calibration, prior state, or earlier intervention can change the prefix, so these floors do **not** automatically apply to a different warm/static or inter-shot parent. Conversely, an update later than 135 can only increase the set of already-committed baseline first alarms, if the source prefix remains unchanged until that later update.

The first-20 source-only condition is stronger than the 135-point thought experiment: all six startup FPs remain, including two occurring after index 134. Adding the four suffix FPs before 135 gives the 10-shot floor. The startup two and suffix three >3 s first alarms similarly give the five-shot floor. A future candidate may also create new FPs or new >3 s alarms; suppressing one original crossing is insufficient unless all later disqualifying crossings are also suppressed.

## Disrupted shots with a possible timely point

The scoring contract separates an outright **miss** (no pre-disruption alarm) from a **tardy** alarm (an alarm exists, but first lead is not strictly >30 ms). The latter is also not a timely TP. A point is *eligible* here if its zero-based index is at least 135 and its HDF5-based, nine-decimal-ms lead is strictly >30 ms. The table counts shots with at least one such point and a strictly positive margin, not the number of points.

| Segment | No-alarm misses | Misses with eligible point and positive margin | Tardy shots | Tardy shots with eligible point and positive margin | All not-timely shots with eligible point |
|---|---:|---:|---:|---:|---:|
| Startup 20 | 0 | 0 | 1 | 1 | 1 / 1 |
| Suffix 340 | 26 | 26 | 37 | 37 | 63 / 63 |
| Full 360 | 26 | 26 | 38 | 38 | 64 / 64 |

All 64 also have an eligible point at index **>135**, and all 64 have one in the more restrictive lead band **30 ms < lead ≤ 3,000 ms**. The latter band matters when a new >3 s first alarm would be undesirable. The single startup tardy shot is not a fast-adaptation opportunity if the first 20 shots are frozen; the corresponding maximum number of suffix shots with a post-update timely candidate is 63, comprising 26 no-alarm misses and 37 tardy shots. Existence of a candidate does not establish that a causal update can use it, that its threshold crossing would be first, or that other shots would remain unchanged.

For each shot and eligibility set, define its baseline probability margin as

`threshold − max(source_probability at eligible points)`.

For the 26 no-alarm misses, all in the suffix, the minimum, median and maximum margins are **0.0014019012, 0.4037835598, and 0.7206151839**, respectively. One shot has margin ≤0.01 and four have margin ≤0.1. Restricting to the 30–3,000 ms band leaves all 26 eligible but changes the median margin to **0.5835598037**; only two have band-restricted margin ≤0.1. For example, shot 65944 has margin 0.0014019012 and shot 69859 has margin 0.7206151839. These are probability-space distances to the existing threshold, **not** observed or feasible LoRA score changes, per-shot recovery probabilities, or a reason to retune the threshold. Outcome-based point selection and these shot IDs are available only to the offline evaluator, not the predictor.

## Verification and interpretation boundaries

- The locked 360-shot order matched the hash-bound original order exactly. The fresh v5 HDF metadata inventory passed its completion hashes; all 360 source HDF file size/mtime pairs still matched it. The inventory supplies `meta/IsDisrupt` and, for disrupted shots, `meta/DownTime`; neither old shot-list classes nor event labels override it. This analysis did **not** reopen HDF contents, re-read metadata itself, or read any raw signal arrays.
- All four static completion artifacts matched their recorded hashes. For all 360 shots, the curve file hash matched both its seal and the seal record in the completion artifact; the two seal records also agreed. Each of the 1,080 arm–shot curves was independently re-scored from its sealed time and probability arrays against v5 metadata and matched its saved per-shot record. All nine saved arm/segment summaries matched, including TP, FP, tardy, miss and >3 s counts. All saved clocks matched inventory start/rate/sample count; all curves were finite and in [0,1], and none of their 4,528,062 scores equaled the threshold exactly. Prefix map identities were checked pointwise. Seal records and hashes establish artifact consistency, not independent proof of historical execution order.
- The score is the first `score ≥ threshold`; for disrupted shots the valid preterminal condition is `(DownTime − time) × 1000 ≥ −1e−6 ms`. The first lead is rounded to nine decimal places in milliseconds before testing `>30` or `>3000`. Synthetic checks exercised 134/135 and exact 30/3,000 ms boundaries. A separate direct Python recount over the primary 360 curves independently reproduced the reported segment counts, candidate eligibility, selected margin counts, and minimum/median/maximum no-alarm margins.
- [Analysis script](probe_outputs/joint_lora_neutral_headroom_20261104_v1/headroom.py) and [receipt with per-shot margins](probe_outputs/joint_lora_neutral_headroom_20261104_v1/receipt.json) preserve the calculations and hashes. The source completion SHA-256 is `ef41bc2db11c19d12f7fd2f6b2fea24a5208d0daa4cf5b818c996bf80d66d78f`; v5 metadata SHA-256 is `286f838542f5186be61696485bd3952fda66bcacc72a32a27f72a96971ea79b1`. The receipt also records the unmapped and retrospective-map arms for verification; the latter is not an operational startup baseline.
- No GPU, model inference, training, threshold tuning, existing-code edits, or change to historical artifacts was performed. This analysis does not infer population performance, online benefit, a physical mechanism, or an achievable intervention from the designed stream.
