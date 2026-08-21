# EAST-only versus J-TEXT-initialized transfer-static baseline

All values are balanced360 development-stream means over seeds 131/133/137.

| Arm | TPR@30 | FPR | TPR@30-FPR | Median lead (ms) |
|---|---:|---:|---:|---:|
| J-TEXT initialized transfer-static | 0.8407 | 0.1667 | 0.6741 | 242.0 |
| target_only_local17 | 0.7963 | 0.1444 | 0.6519 | 165.6 |
| target_only_matched17 | 0.7315 | 0.1056 | 0.6259 | 150.4 |

This is a target-data/initialization control on the reused development stream, not an independent generalization estimate.
