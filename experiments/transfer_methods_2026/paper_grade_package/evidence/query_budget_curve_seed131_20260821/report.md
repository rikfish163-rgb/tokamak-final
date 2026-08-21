# Query-budget curve (seed 131)

Primary utility: `TPR@30ms - FPR`.

## Full balanced360 online curve

| Queries | Rate | TPR@30ms | FPR | Raw TPR | Lead (ms) | Utility | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.8222 | 0.1556 | 0.9500 | 208.0 | 0.6667 | — |
| 16 | 4.44% | 0.8278 | 0.1556 | 0.9500 | 208.0 | 0.6722 | +0.0056 |
| 35 | 9.72% | 0.8278 | 0.1556 | 0.9500 | 185.0 | 0.6722 | +0.0000 |
| 66 | 18.33% | 0.8444 | 0.1500 | 0.9611 | 208.0 | 0.6944 | +0.0222 |
| 97 | 26.94% | 0.8389 | 0.1556 | 0.9611 | 208.0 | 0.6833 | -0.0111 |

## Paired equal-label-budget comparisons

Complements differ between rows and must not be compared across budgets.

| Queries | Complement | Online J | Best direct J | Online-direct | Online wins |
|---:|---:|---:|---:|---:|:---:|
| 16 | 344 | 0.6800 | 0.6523 | +0.0277 | yes |
| 35 | 325 | 0.7223 | 0.7250 | -0.0027 | no |
| 66 | 294 | 0.7832 | 0.6969 | +0.0862 | yes |
| 97 | 263 | 0.8367 | 0.7439 | +0.0929 | yes |

## Claim boundary

- Full360 online rows are comparable across budgets.
- Each direct-frontload comparison excludes that budget's queried training shots; complement composition changes with budget.
- Label access is matched within a budget, but optimizer steps, replay and FLOPs are not exactly matched.
- The balanced360 stream was used for method development and is not an untouched future confirmation set.
