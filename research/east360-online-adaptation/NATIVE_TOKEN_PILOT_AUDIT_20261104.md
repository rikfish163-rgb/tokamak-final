# Native-token EAST20 pilot: independent bounded audit

**Result: the prespecified `native_absolute` gate failed.** Its safe and overall disruptive shot-macro risk losses improved, but danger-window loss worsened and real feedback was worse than hold. None of the three secondary configurations passed. These retrospective local losses are not operational alarm performance.

Scope: CPU-only arithmetic on the [native pilot artifacts](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/native_token_ended20_20261104_v1), [plan](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/NATIVE_TOKEN_PROBE_PLAN_20261104.md), code, previous mask pilot, and 20 HDF metadata records. No raw signal arrays, GPU, training or neural replay were used.

## Integrity, support and scoring

- Completion hashes match **8/8** native, **9/9** previous pilot and **2/2** fresh-metadata artifacts. Prediction and state seals match: `86bef1720e9b2cde7d506b234ae8437965371a5da602e9c72448d5655967fcc8` and `f52df8aae8f4991fb402868adb38055f49f7a50d0009509d93b547e1d3c1f470`. All **7/7** bound inputs and **118/118** current code hashes match; their ZIP entries and archived plan match too.
- The **1,600 unique** rows are exactly 100 prior coordinates × four head/loss configurations × static/real/reverse/hold, five coordinates per shot, all `q−origin=26`. For each coordinate, all 16 rows have matching mask digests/counts and timestamps; target digests match across configurations within mode. All 1,600 before logits and 400 static after logits match the previous original/static source exactly: maximum difference **0**. Repeated static rows are not independent experiments.
- Metadata-only HDF rereads agree with the saved outcomes and fresh metadata for all 20 shots: **8 disruptive, 12 safe**, zero outcome/clock mismatches, and query-time error **0 s**. Independently, `logaddexp(0,(1−2y)×(logit−logit(threshold)))` with threshold `0.7258946895599365` reproduces 1,600 scored rows. Maximum errors: before/after BCE `2.220446049250313e−16`, delta BCE `4.163336342344337e−16`, and unrounded lead `1.2505552149377763e−12 ms`. Labels match; disruptive margin targets include all leads >30 ms, including >125 ms, while the 125 ms point label is separate.
- Reconstructed per-shot-then-group counts, means, point-sign counts and balanced deltas agree with all saved summaries within `4.440892098500626e−16`; independently reconstructed primary and secondary gates all match. Groups comprise 12 safe shots/60 points, 8 disruptive/40, and 8 danger/24.

## Risk-loss deltas and controls

Each entry is `real − same-source static`, averaged within shot and then across shots; lower is better. Balanced equally weights safe and disruptive means.

| Configuration | Safe | Disruptive | Danger 30–125 ms | Balanced | Gate |
|---|---:|---:|---:|---:|---|
| **native absolute, primary** | −0.000002442 | −0.000054903 | +0.000080027 | −0.000028673 | fail |
| native centered | +0.000141601 | +0.000631511 | +0.000425871 | +0.000386556 | fail |
| external source-decoder absolute | +0.000019756 | −0.000001465 | +0.000254015 | +0.000009146 | fail |
| external source-decoder centered | +0.000053189 | +0.000529654 | +0.000539740 | +0.000291421 | fail |

For the primary, balanced real-minus-reverse is `−1.0062603888582484e−6`, just `6.260388858e−9` beyond the required `1e−6` advantage; real-minus-hold is **+4.665515936838476e−6**, so hold is better. Only 3/12 safe and 3/8 disruptive shots improve under primary real; the group means reflect heterogeneous magnitudes. Native centered beats both controls but worsens all three groups; the external-head variants also fail their gates. Across the **1,200** dynamic sampled points, the counted fixed-threshold crossings are **0**; this is not a full-shot alarm metric.

## Updates and process limits

All **1,200/1,200** dynamic receipts report a computed gradient and strictly lower auxiliary loss; there are no equal/increased losses. Updates are unconditional, not accepted through an auxiliary-loss gate. All 1,600 CPU-loaded states match their keys and receipts; 400 static states are zero, dynamic norms range `0.049999930902199785–0.05000000566943628`, and maximum norm/receipt discrepancy is `1.3877787807814457e−17`.

The inspected path loads the qualified source and source-only external head, checks frozen source/native-head/external-head digests and gradients, and records no old cold-state restore. Each zero-start episode uses origin `q−26`, matured targets `q−25…q−1`, then scores at `q`. The runner blocks `read_outcome` during neural calculation, then seals predictions/states before the HDF outcome read. These are code and recorded-process evidence, not independent runtime attestation. Whole ended-shot arrays are loaded before prefix slicing; coordinate selection and the density map use the same ended 20 shots. The slices therefore do not demonstrate literal streaming I/O or target-blind preparation.

The native rule uses an L2-normalized **0.05 parameter-space step and radius 1**, unlike the prior FFN's **0.05 hidden-RMS step bound**. Smaller risk changes cannot isolate update site: geometry, head/loss and update rule differ. The main process reports 33 native-core plus 2 scoring CPU tests passed; they were not independently rerun here. No efficacy or physical-law claim follows.
