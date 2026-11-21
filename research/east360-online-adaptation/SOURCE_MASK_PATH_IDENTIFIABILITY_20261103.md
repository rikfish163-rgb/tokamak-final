# Source mask-path identifiability: CPU audit

**Conclusion.** For source inputs on which Bt, CIII and MNM availability coincide, the individual mask-column weights are not identifiable from the model's data-only input/output behavior. Their redistribution can leave every source embedding unchanged in exact arithmetic yet change an embedding when CIII or MNM is unavailable and Bt is available. This is an architectural/data-support statement, not a demonstration of the cause of EAST false positives.

## Evidence and exact indices

The existing [coavailability audit](probe_outputs/source_missing_diagnostics_audit_v1/audit.json) reports zero CIII/Bt and MNM/Bt mask disagreements in TRAIN 1,353 shots / 476,194 samples, VAL 160 / 48,606 and TEST 220 / 67,375. It reports `CIII_sentinel_points = 0` for each split. This audit relied on that saved result; it did not reread raw signals or repeat the coavailability scan.

The trusted parent loader in [prepare_prior.py](experiments/risk_aligned_online/redesign_20261004/prepare_prior.py) hash-checks the qualified checkpoint before loading it on CPU. The actual configuration is source view, window 128, patch 8, hidden width 64. In [simple23.py](experiments/risk_aligned_online/simple23.py), each timestep is `[asinh23, availability23]`, flattened **time first** into an 8 × 46 = 368 vector, then passed to `embedding.weight` with shape `[64, 368]`. Channel order starts `Bt=0, CIII=1, MNM=2` ([io23.py](experiments/risk_aligned_online/io23.py)). Thus, with zero-based patch position `p=0,…,7` and channel `c`, the mask-column index is `46p + 23 + c`:

| Mask | Eight columns in `embedding.weight` |
|---|---|
| Bt | 23, 69, 115, 161, 207, 253, 299, 345 |
| CIII | 24, 70, 116, 162, 208, 254, 300, 346 |
| MNM | 25, 71, 117, 163, 209, 255, 301, 347 |

Rows are hidden coordinates 0–63; the corresponding **value** columns are `46p+c`, not the columns listed above. For an unavailable input, the source-view preprocessing replaces its value by the source center, so its transformed value is zero; its mask remains false. Left padding in [simple_data.py](experiments/risk_aligned_online/simple_data.py) sets all channel masks false.

## Algebra and its boundary

Write the three 64-vector mask columns at position `p` as `B_p, C_p, M_p`. For arbitrary 64-vectors `U_p, V_p`, change only a **copy** of the weights:

`C'_p = C_p + U_p`, `M'_p = M_p + V_p`, `B'_p = B_p - U_p - V_p`.

For the same input and bias, the embedding difference is

`Δh = Σ_p [ U_p (m_C,p − m_B,p) + V_p (m_M,p − m_B,p) ]`.

It is exactly zero whenever the three masks agree at every position, including complete observations and shared zero padding. Thus the data can at most constrain the mask-column sum for each position; it need not even uniquely identify that sum. The construction supplies at least `2 × 8 × 64 = 1,024` independent flat directions in the data-only parameterization. With `m_C=0, m_B=1`, a nonzero selected `U_p` can change the target embedding; different positions can also cancel. Identical embeddings and otherwise unchanged inputs/weights imply identical downstream behavior in exact arithmetic, including under a fixed fast state. The target embedding difference alone does **not** guarantee a changed risk score.

The actual historical training objective has this invariance only if its actual inputs, including any augmentations, obeyed the same equality. The saved current coavailability audit alone cannot establish that: the archived training protocol records different hashes for `simple_data.py` and `joint_meta.py` than their current files. A parameter penalty need not be invariant. For example, equal L2 penalties prefer equal allocation among columns when their sum is fixed; optimizer dynamics, initialization, weight decay and early stopping can select a particular allocation without the data identifying it. AdamW's decoupled decay should not be conflated with proof of a uniquely determined physical effect.

## One bounded CPU example

On CPU, load the hash-locked source via `load_parent(DEFAULT_PARENT)`; copy its weight and bias tensors, never modifying the model. On the copies use `U_p = −C_p`, `V_p = 0`: transfer each CIII mask column to Bt and set the copied CIII mask column to zero. The synthetic 8-timestep patch uses transformed values `z_k=0.8 sin(0.17k)` for flattened value index `k=0,…,183` and initially all masks true. No raw signals are involved.

For the coavailable patch, the original and redistributed linear embeddings agree to maximum absolute error **4.44 × 10⁻¹⁶** in float64; repeating with float32 gives **1.79 × 10⁻⁷**. For the same synthetic patch with CIII unavailable at all eight positions, Bt available, and CIII transformed values set to zero, the float64 embedding difference has maximum absolute component **0.2257422**, L2 norm **0.7248166**, and matches the formula above within **2.22 × 10⁻¹⁶**. This is a finite linear-embedding illustration, not a full-risk-model or real-shot run; floating-point equality is not claimed to be bitwise.

## Interpretation and integrity

The user reports source TEST220 native **100 TP / 13 FP**, observed-center CIII **99 / 21** versus unavailable CIII **100 / 27**, and observed-center both **96 / 21** versus unavailable both **97 / 29**. Under the reported matched observed-center design, these contrasts are consistent with both loss of channel information and a mask/availability-path effect. They do not establish additive contributions or explain all EAST FPR. In particular, a normal missing-channel ablation changes both the value path and the mask; this algebra does not isolate either effect in real data.

If pursued later, keep the qualified base frozen and masks truthful; distinguish time padding from diagnostic absence, teach missingness using source TRAIN only, and compare a matched static control before any online adaptation. A frozen sidecar can expose the distinction, but the existing base's `available.any()` attention validity still conflates an all-missing real patch with padding; changing that validity rule would require separate parity and qualification checks. Never mark an actually absent diagnostic as observed as a deployment fix.

Checkpoint SHA-256 before and after the CPU checks: `c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02`. A digest over every loaded model state tensor, including name, dtype, shape and bytes, was also unchanged: `56d68c4338fddc1fc7b6fbaf6c03f835b6dff226a5470fc76bd4de7afa18e116`. The numerical work used CPU only, with no training, raw-signal access or full risk-model evaluation.

**Process disclosure:** a read-only `git status --short --untracked-files=no` was inadvertently run despite the no-git instruction. No modifying Git command was issued; possible Git index metadata refresh was not checked.
