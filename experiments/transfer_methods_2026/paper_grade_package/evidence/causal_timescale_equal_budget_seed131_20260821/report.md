# Causal online ablation (seed 131)

Primary utility is `TPR@30ms - FPR`; raw TPR is reported separately.

| Arm | Raw TPR | TPR@30ms | FPR | Median lead (ms) | Utility |
|---|---:|---:|---:|---:|---:|
| static | 0.9500 | 0.8222 | 0.1556 | 208.0 | 0.6667 |
| intra_only | 0.9556 | 0.8222 | 0.1556 | 185.5 | 0.6667 |
| inter_only | 0.9500 | 0.8333 | 0.1556 | 202.0 | 0.6778 |
| intra_inter | 0.9556 | 0.8278 | 0.1500 | 181.8 | 0.6778 |

## Effects

- Intra main effect: `+0.000000`
- Inter main effect: `+0.011111`
- Joint total effect: `+0.011111`
- Interaction: `-0.000000`

## Equal-label-budget verdict

Online minus best direct-frontload utility: `+0.034315`.
This is label-budget matched, not exact compute matched. The balanced360 stream was used for method development and is not an untouched future test.

The Q30 identities were first selected causally by the label-free shot-level
query-value policy and were then frozen as a common manifest for the inter-only,
intra+inter and direct-frontload controls. “Fixed” therefore means controlled
reuse after selection, not an arbitrary predeclared 30-shot cohort. This
shot-level label-query policy is distinct from the intra-shot window-value
selector used for unsupervised adaptation.

The recovered immutable Q30 manifest is packaged at
`../fixed_query_low_budget30_seed131.json`; it records all 30 shot identities,
their stream positions, `label_oracle_used=false`, and the SHA256 of the lost
360-row decision ledger.  The original per-shot decision ledger was not
recovered after the tmpfs reboot, so the historical cutoff, score and route of
each decision cannot be independently recomputed from the compact package.
