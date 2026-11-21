# Source-only anticipation annotations: data review

**Result:** the locked source archive yields 10,044 TRAIN records from 1,303 shots and 946 VAL records from 145 shots. Fresh HDF5 outcome/clock metadata agrees with every archived outcome, source point label, group and lead, and every archived field is preserved. The 125 ms field is an auxiliary source-training hypothesis, **not** the original J-TEXT label or an online EAST result.

## API and boundary

[annotate](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/data.py:42) copies a record, retains its archived `label`, `group`, `disruptive` and `lead_ms`, and adds authoritative `hdf5_disruptive`, `hdf5_down_time`, `hdf5_query_time`, `hdf5_lead_ms`, `original_source_point_label`, `future_disruption_125ms`, `timely_eligible`, `anticipation_stratum`, and explicit mismatch fields. The recomputed source label uses `max(25 ms, AdvancedTime)` and the existing `1e-6 ms` source-label tolerance. Safe-shot DownTime is not treated as a disruption time.

The auxiliary label is 1 for an HDF5-disruptive shot with lead in the closed interval 0–125 ms; timely eligibility is true for safe shots and disruptive shots with lead strictly greater than 30 ms. `auxiliary_lead_ms` rounds the raw HDF5-derived lead to nine decimal places in milliseconds for these boundary comparisons, avoiding binary subtraction errors at exactly 30 or 125 ms; the unrounded lead is retained separately. Post-disruption rows are explicitly ineligible. A disruptive shot earlier than 125 ms has auxiliary label 0 but **is not a safe shot**.

[build_records](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/data.py:179) returns `(rows, receipt)`, accepts only `train` or `val`, reuses the hash-locked [source archive loader](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_event_tasks_20261102/data.py:66), and always reads the current AdvancedTime mapping. It checks every role/identity and `(shot, query)` uniqueness before raw access, enforces `maturity=origin+25` and `origin+26≤query≤origin+50`, and reads metadata once per used shot. A test-injected metadata reader is explicitly marked `metadata_source=injected`; production reads are marked `hdf5`. No event families are invented or attached.

## Fresh metadata census

| HDF5 stratum | TRAIN records / shots | VAL records / shots | Auxiliary / timely |
|---|---:|---:|---|
| Safe | 9,293 / 1,163 | 638 / 80 | 0 / yes |
| Disruptive, 0–30 ms | 194 / 135 | 91 / 65 | 1 / no |
| Disruptive, >30–125 ms | 378 / 123 | 155 / 53 | 1 / yes |
| Disruptive, >125 ms | 179 / 65 | 62 / 20 | 0 / yes; still disruptive |

Shot counts can overlap across disruptive strata; do not sum them as distinct shots. There were no post-disruption records. In the >30–125 ms group, TRAIN consists of **350 archived early/0 + 28 danger/1**; VAL consists of **125 early/0 + 30 danger/1**. Source-point positives overall are 196 TRAIN and 117 VAL; auxiliary positives including the ineligible late group are 572 and 246. In VAL, two >125 ms records are original source positives because their source horizon exceeds 125 ms: the two label definitions can differ in either direction.

An in-memory HDF5 guard allowed only the current role's J-TEXT shot paths and dataset-value reads of `/meta/IsDisrupt` and `/meta/DownTime`. It recorded exactly 1,303 TRAIN and 145 VAL file opens; `/data/ip` supplied only attributes and shape. No signal arrays, TEST/EAST raw files, model checkpoints, or event CSVs were read. Mismatch counts for `disruptive`, `label`, `group`, and `lead_ms` were all zero; a separate comparison confirmed every original archive field was preserved.

The receipt's `source_metadata_fields_sha256` hashes canonical JSON over sorted shot IDs and the normalized `disruptive`, `down`, `start_time`, `sample_rate_hz`, and `samples` fields; it is **not a whole-HDF5 or signal hash**:

- TRAIN: `fade918c5d2e1e3867e70cb47bd2bfedde851264cd1763998e0ae60a8265e767`
- VAL: `0c690b904ccf72df11e97f823a113ee5780156d1b0fb8c4380a23f542f573542`

## Verification and reproducibility

TDD red: initial collection failed because the module was absent; additional tests exposed and then verified fixes for the locked-ID guard and floating-point 30/125 ms boundaries. Final CPU checks: **49 new tests passed**, and **104 combined new/parent source-data tests passed in 2.57 s**; Ruff E9/F passed. The tests use injected synthetic metadata and mocked HDF5 objects, not raw shot files. No training, fitting, new model, GPU job, output manifest, or modification of the frozen native-token implementation was performed.

From the worktree root, the final census can be reproduced without writing artifacts:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -c 'from collections import Counter; from experiments.risk_aligned_online.source_anticipation_20261104.data import build_records
for role in ("train","val"):
 r,p=build_records(role); c=Counter((x["group"],int(x["label"])) for x in r if x["anticipation_stratum"]=="disruptive_30_to_125ms"); print(role,"records",p["records"],"shots",p["shots"],"D30-125",dict(c),"mismatches",p["mismatch_counts"],"metadata_source",p["metadata_source"],"metadata_fields_sha256",p["source_metadata_fields_sha256"])'
```

Final SHA-256: [data.py](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/data.py) `644b44ed4e5a5c771ed60e48ffd6d8e8b22588461a726cd21744d29ada9abeed`; [tests](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_source_anticipation_data_20261104.py) `8a7d51625a4fab9ffff4b3cb7892e6fe39c8626d328f886af9f9eca927e4a607`; `__init__.py` `4789f56462092debb013fced7a57cf87ede3f0dbf5e45733edd7c8b7fdb8f0e8`. Archive TRAIN/VAL record hashes are respectively `75954fa4af6d3800a80d204ad8df7ac8013f0cc70ad3d065f392eb8b6e389008` and `132f96ef81719cec7dbd91c4d24f0ba99363c0980b3cb734a75a0a2174d6452c`; AdvancedTime JSON hash is `559da4f719373b0c13b96d34e27a2b46b80eba7f0ef185c3b6ac58b3761e0199`.

These are existing, outcome-selected source episodes, not an independent future test set. Metadata labels are offline supervision and must not be revealed to a predictor before its query prediction. The 1,303/145 used shots are subsets of the locked 1,353/160 roles, not new split membership; metadata-only fingerprints do not certify raw signal contents or the efficacy of any later auxiliary task.
