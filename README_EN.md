# J-TEXT → EAST: causal transfer and chronological online adaptation

**Language:** [简体中文（默认）](README.md) · **English**

[Chinese technical report](experiments/transfer_methods_2026/paper_grade_package/DYNAMIC_MAINLINE_TECHNICAL_REPORT_zh.md) · [XeLaTeX source](experiments/transfer_methods_2026/paper_grade_package/technical_report_dynamic/main.tex) · [Data contract](DATA_CONTRACT.md)

![Python](https://img.shields.io/badge/Python-3.11-3776AB) ![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C) ![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900) ![HDF5](https://img.shields.io/badge/HDF5-authoritative%20labels-00599C)
![model](https://img.shields.io/badge/model-causal%20Transformer-0072B2) ![online](https://img.shields.io/badge/online-chronological%20active%20learning-009E73) ![reproducibility](https://img.shields.io/badge/reproducibility-hash--bound-2F80ED)

![J-TEXT source, EAST few-shot transfer, and online-active architecture](docs/figures/jtext_east_mainline_architecture_v2.png)

This repository packages one auditable development chain:

```text
J-TEXT TRAIN1353 source model
-> EAST support110 few-shot transfer
-> sealed EAST balanced360 static replay
-> reset to the identical transferred parent
-> chronological online-active replay on balanced360
```

J-TEXT TEST220 also has an independent online diagnostic branch. Its online state is never used to initialize EAST transfer. Shot outcome and disruption time are authoritative only in HDF5 `/meta/IsDisrupt` and `/meta/DownTime`.

> Evidence boundary: all results are `TEST_DRIVEN_EXPLORATION`. TEST220 and balanced360 were inspected during method development; they are not independent future confirmation cohorts or deployment evidence.

## Current results

Equal-weight means across seeds 131, 133, and 137 are shown below. Operational TPR requires lead `>10 ms` on J-TEXT and `>=30 ms` on EAST.

| Stage | TPR | FPR | AUC | J = TPR - FPR |
|---|---:|---:|---:|---:|
| J-TEXT static TEST220 | **0.866667** | **0.130303** | **0.967906** | **0.736364** |
| EAST static balanced360 | **0.725926** | **0.220370** | not reported | **0.505556** |
| EAST online balanced360 | **0.733333** | **0.220370** | not reported | **0.512963** |

EAST online-minus-static delta-J is `+0.005556`, `+0.016667`, and `0` by seed, mean `+0.007407`. FPR is unchanged for every seed; the limited gain comes from converting 1, 3, and 0 disruptive shots into timely alarms. This is a small, seed-unstable timing gain, not reliable false-alarm reduction. Seed 133 fails the TPR gate and seed 137 fails the FPR gate, so `T0_ADMITTED=false`.

The retired paper-v3c operating point used a different source checkpoint, label contract, and alarm lineage. It is archived and must not be combined with this table.

## Label and method contract

For disruptive J-TEXT shot `s`, the positive horizon is `max(25 ms, AdvancedTime_s)`, with a 25 ms fallback. AdvancedTime changes only the positive-window boundary; it cannot override HDF5 outcome or DownTime. EAST support110 retains its fixed 125 ms target label. The selected arm is `source_dynamic__east_fixed`.

- The source is a causal `128 x 17` Transformer trained on TRAIN1353; VAL160 locks checkpoint and threshold before TEST220 access.
- Cross-device shared17 uses the source median/IQR scaler. Eight EAST-private signals enter a bounded target-only residual branch instead of forced semantic alignment.
- EAST support110 adapts high target layers and an independent EAST risk head. The J-TEXT online state is forbidden as a transfer parent.
- Intra-shot updates use only arrived endpoints and modify a disposable fast state. Persistent updates happen only after prediction sealing, query selection, and label reveal.
- Active and causal-random J-TEXT controls use identical query counts, optimizer attempts, windows, replay volume, and reported FLOPs.

## Reproduction

Use the CUDA-enabled `rl_env` environment:

```bash
export FUSION_DATA_ROOT=/absolute/path/to/data
PY=${CONDA_PREFIX}/bin/python
$PY -c 'import torch,h5py; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

CPU CI installs only audit dependencies and does not replace GPU replay:

```bash
uv sync --frozen --only-group ci --no-install-project
```

Keep the GPU development environment inside the active Conda environment:

```bash
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --frozen --extra dev
export FUSION_PROJECT_ROOT="$PWD"
```

Run one formal seed with:

```bash
$PY experiments/transfer_methods_2026/run_selected_full_mainline.py --seed 131 --stage all --output-root /ext4/runs/selected_full_mainline/prefix_causal_v2c
```

Repeat for seeds 133 and 137. Then run evidence assembly, report generation, and the test suite using the scripts under `experiments/transfer_methods_2026/paper_grade_package/scripts/`.

`fusion-reproduce` is an editable-checkout CLI, not a standalone wheel; it still needs repository configs, rosters, and scripts under `FUSION_PROJECT_ROOT`.

## Evidence and limitations

The portable evidence object is [dynamic_mainline_report.json](experiments/transfer_methods_2026/paper_grade_package/evidence_dynamic/dynamic_mainline_report.json), SHA-256 `04e7319843304d28ec58e6b2b483bf85c5b4dea8dedd521825a6625d319dae2e`. It binds 80 input artifacts and records `summary_metrics_used=false`.

The engineering chain is executable and hash-auditable. Scientific admission is not achieved: EAST transfer remains below the registered three-seed gate, J-TEXT active selection does not outperform equal-compute random querying, and an independent future chronological stream is still required.

License: [MIT](LICENSE). Citation metadata: [CITATION.cff](CITATION.cff).
