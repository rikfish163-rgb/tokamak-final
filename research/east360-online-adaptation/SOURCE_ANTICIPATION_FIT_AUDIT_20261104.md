# Source anticipation fit: bounded independent audit

**Result:** the audited fit is bound to V2 and remains `TRAINED_NOT_VALIDATED`. Saved labels, schedule, per-shot exposure and history reconcile; the shuffle control is globally and per-shot exposure matched, but **not matched by masking view or lead stratum**. This is not evidence of efficacy or physical validity.

## Integrity and historical producer

- All **20 completion-listed artifact hashes** matched across [V1 design](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_anticipation_design_20261104_v1/completed.json), [V2 design](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_anticipation_design_20261104_v2/completed.json), [heads fit](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_anticipation_heads_20261104_v1/completed.json), and the [2026-11-04 HDF5 audit](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/hdf5_outcome_authority_20261104_v1/completed.json). All ZIP members matched: V1/V2 each 73 pinned code files plus their separately pinned plan, and fit 121 code files plus its pinned plan. The fit's three `input_sha256` bindings also matched, including V2 completion `cb60329d102be5c7deb6c901a07acee50dbf7f599882b855b3f74b0848327562` and source `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`.
- The [V1 marker](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_anticipation_design_20261104_v1/superseded.json:2) says superseded before training; V1 completion also records no model training. The marker itself was added after, and is **not** covered by, the V1 completion manifest. The audited fit is independently bound to V2, not V1; this does not prove no unrecorded V1 run existed elsewhere.
- Historical `train.py` inside the [fit code ZIP](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_anticipation_heads_20261104_v1/code.zip) hashes to `f9f3e1b72b626ec2a4e5586c00a2f84f21cd71fb90e0f9ee7f92155c0b020f22`, matching its historical protocol. The current file hashes to `f503b94563406ae50981a8cdb80e7509c91a04a67cdf585b42545c7a282a2ffd`; the sole diff extends `load_windows` with an explicit TRAIN/VAL role. The archive, not current equality, identifies the producer.

## Data and target reconstruction

Recomputed all packet rows against the hash-verified [saved fresh HDF5 metadata](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/hdf5_outcome_authority_20261104_v1/hdf5_metadata.json), current AdvancedTime JSON, and original archive fields: **TRAIN 10,044/1,303 shots; VAL 946/145**, zero outcome, clock, source-label, auxiliary-label, role or archived-field differences. Metadata-fields fingerprints match both design receipts. The separately recomputed `original_source_point_label` remains distinct from the 125 ms auxiliary field; archived `label`/`group` are unchanged. In >30–125 ms, TRAIN is **350 original 0 + 28 original 1**; VAL is **125 + 30**. No event-CSV outcome was used.

Independently regenerated the seed-131 schedule and all three arrays in [V2 targets](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_anticipation_design_20261104_v2/training_targets.json): exact match. V1's corpus-only permutation reproduces **256 vs 301** positive draws and 45 shots with unequal weighted exposure. V2 permutes only within each disruptive shot and equal scheduled-frequency group: per-shot corpus and weighted positive counts all match; safe and unsampled coordinates do not change. Of 512 draws, only **56 change** (28 each direction), representing **48 distinct coordinates across 22 shots**; 456 draws retain the same label. This is a limited single shuffle, not a comprehensive temporal-randomization test. See [inventory](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_anticipation_design_20261104_v2/inventory.json:66).

## Matched-arm limits

Each arm has the same 512 scheduled draws, but the positive counts are **123 point / 256 anticipation / 256 shuffle**. Per lead stratum, the counts are:

| Scheduled HDF stratum | Point | Anticipation | Shuffle |
|---|---:|---:|---:|
| Safe | 0 | 0 | 0 |
| Disruptive >30–125 ms | 12 | 128 | 118 |
| Disruptive >125 ms | 0 | 0 | 28 |
| Disruptive 0–30 ms | 111 | 128 | 110 |

Anticipation/shuffle positive draws also differ by view: `native` **86/88**, `no_MNM` **86/88**, and `no_MNM_CIII` **84/80**. Thus same rows and global/per-shot prevalence do not imply matched view-specific or lead-specific labels; this is a qualification on the temporal control, not a basis for a success claim.

All **128 × 3** saved [history](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_anticipation_heads_20261104_v1/training.json:1) arm labels match the indexed arrays and scheduled rows/tasks; all three arms log four observed rows at every step. The archived producer's `train.py:153–160, 239–255` constructs one shared view/mask batch before the arm loop. Full masks were not saved, so identical actual mask bytes are **not independently replay-proven**. The 411 selected coordinates cover 244 recorded TRAIN shots; their recorded raw-file hashes match the source archive, but raw files were not opened in this audit.

## Neural evidence and limits

Archived `train.py:184–198, 284–320` records three independent copies of the source native head, source-state/gradient checks, and unchanged static-query logits. CPU inspection, without restoring a model, found three finite `weight[575,1088]`/`bias[575]` heads in hash-verified [heads.pt](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_anticipation_heads_20261104_v1/heads.pt); final heads differ from the source and each other. First-step logged proxy values agree across arms, but **final weights and logs cannot independently prove their initial equality or in-memory base immutability**; those remain producer-code/receipt claims.

This results-analysis audit used CPU and saved artifacts only: no fitting, GPU, raw signal arrays, fresh HDF5 opens, or in-progress VAL outputs. The producer records zero VAL/TEST/EAST signal reads during fitting; this audit did not independently monitor its I/O. Main owns VAL evaluation and final verification.
