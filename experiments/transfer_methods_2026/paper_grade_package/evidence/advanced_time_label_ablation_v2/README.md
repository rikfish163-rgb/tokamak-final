# AdvancedTime label ablation v2 evidence

This directory contains the corrected three-seed evidence for the label-only
2×2 ablation.  The v1 bare-encoder result is superseded and must not be used.

For each seed, `source_dynamic/test220_metrics.json` records the J-TEXT source
result.  Each target-arm directory contains the portable arm summary and the
360-row prediction seal produced before balanced360 outcomes were read.  Raw
HDF5 data and retrainable model states remain external to Git.

The four target arms are:

```text
source_fixed__east_fixed
source_dynamic__east_fixed
source_fixed__east_dynamic
source_dynamic__east_dynamic
```

Re-run one seed from the repository root with `rl_env` and the external data:

```bash
FUSION_DATA_ROOT=/absolute/path/to/data \
python experiments/transfer_methods_2026/run_advanced_time_label_ablation.py \
  --seed 131 \
  --data-root "$FUSION_DATA_ROOT" \
  --jtext-advanced-time /absolute/path/to/Fuison/shot_list/J-TEXT/AdvancedTime_J-TEXT.json \
  --east-advanced-time /absolute/path/to/Fuison/shot_list/EAST/AdvancedTime_EAST.json \
  --output results/generated/advanced-time-label-v2-seed131
```

Repeat for seeds 133 and 137.  The fixed/fixed arm fails closed unless its
target-parent hashes, threshold, operational metrics, and all 360 archived
alarm decisions match the current mainline.
