# Source-window compatibility audit, 2026-11-04

**Result:** on the four requested real J-TEXT windows, `original` and `reference_bt` were exactly equal (`torch.equal`, no tolerance) in first-block tokens, token validity, risk logits, native-head forecasts, external `forecast_only` input and forecasts. The frozen original source model's logits and native forecasts also matched the `original` wrapper exactly. This is a bounded CPU compatibility result, not evidence of EAST or online efficacy.

## Evidence and scope

- The reproducible [test](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_mask_separated_compatibility_20261104.py:48) reads only the first 256 `/data` samples per channel of [JT1051510](/media/hetaisheng/044A81D94A81C83E/Fuison/data/J-TEXT/unified_hdf5/1051500/1051510.hdf5) and [JT1051650](/media/hetaisheng/044A81D94A81C83E/Fuison/data/J-TEXT/unified_hdf5/1051600/1051650.hdf5); no HDF5 outcome fields were opened. Endpoints 127 and 255 both existed for both shots. All 128 points in each window had Bt, CIII and MNM coavailable; no CIII sentinel adjustment was needed. Masks and values were not modified by the wrappers.
- The prefix digests below are SHA-256 of the 256×23 raw signal values stacked in `RAW23` order, converted to float32 and serialized C-order **before** J-TEXT `ip` unit conversion; they are not hashes of whole HDF5 files:
  - 1051510: `e2603b334ab3c17fae95385d06da7e726fa0ab4c03a42d6a46be0328eee21b2f`.
  - 1051650: `5e254f5737664475ddde6e699ea25cb965dfb66a79ea61e83c0561437fee02a3`.
- The CPU fixture loads the hash-locked source, makes two fresh zero-slow wrappers with identical initial state, and strictly loads only the source `forecast_only` tensors into an independent external head. Source SHA-256: `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`; decoder SHA-256: `cdde96086517cbc468a911bbb5926dde1c783b3711b0a8abc09caf0a6d2e6730`. See [fixture](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_mask_separated_compatibility_20261104.py:81) and [source loading](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/redesign_20261004/prepare_prior.py:54).
- The head weight/bias shapes are `575×1088` / `575`; the actual hooked input is `1×1088`, reshaped output `1×25×23`. The test verifies the complete flatten/reshape and selected horizon/channel indices against `h*23+c`, plus selected source `asinh` target/loss coordinates: [checks](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_mask_separated_compatibility_20261104.py:141). This does not assert forecast accuracy.
- A per-test guard verifies unchanged source, both wrapper and external-head states, plus forward/pre-hook registries. Deliberate exceptions separately verify embedding-hook and decoder-hook cleanup while preserving pre-existing hooks: [state guard](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_mask_separated_compatibility_20261104.py:130), [exception tests](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_mask_separated_compatibility_20261104.py:263).

## Synthetic missingness: static contrast only

Using values equal to source centers, one all-missing eight-sample patch and otherwise available samples, the test separately hides CIII, MNM, or both while Bt remains available. For each, the mask and token validity stayed unchanged, the all-missing patch's added shift was zero, and the added shift matched explicit source-weight redistribution (`rtol=2e-5`, `atol=2e-7`). The resulting static max absolute logit differences were respectively `0.0234918594`, `0.00765800476`, and `0.00179862976`; external forecast max differences were `0.883975923`, `0.463647366`, and `1.15578055`. These are implementation contrasts on synthetic inputs, **not gains or physical claims**. See [synthetic test](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/tests/test_mask_separated_compatibility_20261104.py:219) and [stem](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/mask_separated_intra_20261103/model.py:13).

## Exact checks run

From the worktree root, Python 3.11.14, PyTorch 2.11.0+cu128, pytest 9.0.2; CUDA was hidden, all tensors and computations were CPU-only, and pytest caching/plugin autoload were disabled:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -m pytest -p no:cacheprovider -v -s tests/test_mask_separated_compatibility_20261104.py
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -m pytest -p no:cacheprovider -v tests/test_mask_separated_model_20261103.py
```

Actual results: **9 passed** (2.78 s) and **6 passed** (1.91 s), respectively. No GPU job, training, EAST raw file, `cold.pt`, target head, or other J-TEXT shot was accessed. The two-shots/four-windows check does not replace all-source evaluation, nonzero-state/gradient parity, the main agent's EAST direct-FFN parity checks, or an end-to-end pilot.

File hashes at verification: test `ddeb25c31a6ea5acb7f9691b06cfe8aa49ee0a9024bafb354b9c7573365ada45`; stem `4154126f01401c296dfbfa55257f7a11c393f4ac7f347ea2b5fb322b3a1e8e0f`; direct-decoder core `0f32cd722151bc0f2cc4a31e1049e094a175779e5c66a94964e93080d9aaf443`; plan `f532bdb04dde1de294cf2abfc7c0858c4faabade9fc9fb8424cdf090a3fa258d`. The stem and plan hashes matched before and after testing.
