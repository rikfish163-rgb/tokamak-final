# Causal active versus random-query online baseline

Primary utility is `TPR@30ms - FPR`.

| Seed | Active TPR@30 | Active FPR | Active J | Random mean J | Delta J |
|---:|---:|---:|---:|---:|---:|
| 131 | 0.8444 | 0.1333 | 0.7111 | 0.6907 | +0.0204 |
| 133 | 0.8389 | 0.1333 | 0.7056 | 0.6963 | +0.0093 |
| 137 | 0.8611 | 0.1889 | 0.6722 | 0.6537 | +0.0185 |

Mean active-minus-random J: `+0.016049`.

Realized shared inter-shot update attempts (equal-seed mean): active `36.67`, random `16.56`. The selector comparison is query-budget matched but not equal-FLOP.

This remains development-stream evidence, not prospective confirmation.
