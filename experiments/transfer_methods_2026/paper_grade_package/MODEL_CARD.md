# Model card: causal J-TEXT → EAST transfer and online-active development model

## Model identity

- Status: `DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION`
- Architecture: causal 17-feature patch Transformer, executed source contract
  `128x8-d64-l2-h4`, target temporal/risk adaptation, dynamic-hazard adapter,
  eight-channel EAST-private sidecar, and a causal dual-route alarm state machine
- Seeds: 131, 133 and 137
- Intended hardware: one CUDA-capable GPU; recorded release environment is
  `rl_env` on an NVIDIA GeForce RTX 5070 Laptop GPU

## Intended use

The model is intended for retrospective research on cross-tokamak feature
semantics, low-shot target adaptation, causal online learning and false-alarm
analysis. It may be used to reproduce the archived development experiments and
to pre-register a future confirmation protocol.

## Out-of-scope use

The model must not directly trigger plasma-control or mitigation actions. It is
not validated for unseen campaigns, other devices, missing-channel patterns,
real-time latency, actuator integration, safety certification or autonomous
operation.

## Input and output contract

- Shared input: 17 reconstructed causal features with availability masks
- Target-private input: eight EAST-only diagnostics summarized causally
- Outcome authority: HDF5 `meta/IsDisrupt`
- Disruption time authority: HDF5 `meta/DownTime`
- J-TEXT operational TP: first accepted causal alarm with lead strictly >10 ms
- EAST operational TP: first accepted causal alarm with lead at least 30 ms
- EAST alarm below 30 ms: operational FN/TARDY
- Alarm persistence: one accepted supported crossing; no 10-point requirement

## Data and evaluation limitations

- J-TEXT TRAIN1353/VAL160/TEST220 are zero-overlap but count-aligned and not
  strict chronological role blocks; VAL160 and TEST220 are outcome balanced.
- EAST support110 has HDF5-authoritative composition 15 disruptive/95 safe and
  is used both for target fitting and support-derived threshold selection.
- EAST balanced360 has 180/180 and is chronological by shot identifier, but it
  was used for architecture and 972-candidate parameter development.
- There is no independent future EAST stream. The primary causal
  block-matched random-query baseline has unequal realized error-triggered
  update counts. A separate fixed-shape selector-roster screen exactly matches
  realized windows and optimizer attempts, but the archived active roster does
  not outperform the random mean in any seed.

## Recorded development metrics

Three-seed arithmetic means:

| Stage | Operational TPR | FPR | AUC | Operational median lead |
|---|---:|---:|---:|---:|
| J-TEXT count-aligned TEST220 | 0.8515 | 0.2121 | 0.9219 | 116.6 ms |
| EAST static balanced360 development | 0.8407 | 0.1667 | not archived consistently | 242.0 ms |
| EAST online balanced360 development | 0.8481 | 0.1519 | not archived consistently | 229.5 ms |

These are descriptive development results, not prospective generalization
estimates.

The strict EAST-only support110 baseline obtains equal-seed mean operational
TPR `0.7963`, FPR `0.1444`, and TPR@30-FPR `0.6519`. Transfer-static improves
the combined objective by `0.0222` and median lead by `76.5 ms`, while raising
FPR by `0.0222`. A random-initialized model using only the source scaler is a
normalization diagnostic (`0.7315/0.1056` TPR/FPR), not a pure EAST-only arm.
The target-only and transferred models do not have equal trainable capacity or
FLOPs.

## Known failure modes

- Primary source representations expose padding/context bucket information
  (probe macro-AUC 0.955/0.929/0.937).
- PADFIX reduces the shortcut but loses source sensitivity and transfers
  negatively under the frozen target policy.
- Worst chronological blocks exceed the aggregate FPR gate.
- Safe-shot scores retain shot/campaign correlation in some seeds.
- Twenty to twenty-three disruptive alarms per EAST seed occur more than 3 s
  before disruption and may be operationally ambiguous.
- Active-query error enrichment is post-hoc on the development stream. The
  full primary mechanism exceeds the block-matched random-query mean in all
  three seeds, but it also realizes more error-triggered updates. When every
  queried shot is forced through the same 16+16-window, 2+2-optimizer-attempt
  operator (with potentially different guarded commits), the
  archived active roster has utility differences of `-0.00185/0/-0.00556`
  versus the random mean. Archived-roster superiority under that new operator
  is therefore unsupported; the adaptive selector-policy effect is not estimated;
  equivalence is not established.

## Reproducibility and provenance

Per-shot source/static/online predictions, query/reveal/update ledgers, model
hashes, roster hashes, figure/table hashes and validation checks are bundled in
`paper_grade_package`. Source alarms include endpoint/time fields, allowing
lead to be recomputed from HDF5 rather than trusted from the metric summary.
Prediction-time A1 loading excludes HDF5 outcome metadata and uses a lazy
post-seal oracle. The three source parents and the three fidelity-verified
post-support EAST target parents are bundled. Historical optimizer states and
exact sampled-window RNG states are not, so the sealed target parent is the
authoritative replay start and cross-hardware bytewise retraining is not
promised. Raw HDF5 data are external. A public repository DOI and data access
route are not yet available.

## Ethical and safety considerations

False negatives can miss disruptions; false positives can trigger unnecessary
interventions; very early alarms may reflect campaign state instead of an
actionable precursor. Device operators and safety engineers must define the
operating policy and validate real-time behavior before any control use.
