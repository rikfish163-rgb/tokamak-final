# Q/K feedback freshness × non-compensating objective

2026-10-29. Follow-up to failed `qk_matched_source_v1`; diagnostic source
preparation for EAST, NOT J-TEXT online adaptation or a new EAST result.
Qualified source classifier SHA c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02
and threshold remain fixed. Only the existing diagnostic forecast head changes.

The failed head lowered risk in 30/32 eligible VAL disruptive shots' timely
positive windows. Two mechanisms will be separated in a fixed 2×2 experiment:

- Clock: `blocked` count=256+128*floor((query-256)/128); `fresh` count=query.
  Both use four non-overlapping H25 arrived targets and W128 inputs. All
  feedback ends <=query-1; query-time risk input may include query itself.
  Same query population >=256, deliberately NOT changing eligibility/startup.
  Fresh feedback can require more frequent deployment computation; this source
  endpoint experiment does not prove a real-time throughput budget.
- Objective: mean versus worst of the SAME three group objectives. Groups
  are safe, early negative on disruptive shots, and timely positive (>10ms).
  For group g: Cg=.5 mean(BCE_after-BCE_before)+.1 alignment_loss.
  Training loss=.2 mean(signal_loss)+mean(Cg), or +max(Cg).
  The second does not let improvements in one group's C compensate another
  group's worse C. It is NOT a hard no-harm constraint or future guarantee;
  signal and alignment terms may still conflict with finite-step risk.
  Group identities/labels are used only during offline TRAIN supervision.

All four heads: same source initialization, same exact 6144 episode choices as
the preceding run,512 steps,batch12 (four per group),AdamW lr1e-4,wd0,clip1,
seed131, normalized full Q/K step .05, final checkpoint only. Original head
and previous forecast-only head evaluated at both clocks as references.
Each trained head is evaluated with true, reversed and hold-last feedback.
No validation-driven checkpoint/lr/step tuning. TRAIN1353 only; VAL160 fixed
521 endpoints; TEST220/EAST signals and labels not read.

Report each group, shot-weighted changes/counts and macro, all four arms and
all controls. Inspect probability shift and feedback lag; no endpoint gain
may be called TPR/FPR. Screen requires safe and danger changes<=0 AND true
macro improvement over original/forecast head at the SAME clock and own
reverse/hold controls by>1e-6. If all fail, do not launch them on full EAST360.
If a candidate passes, only same-parent full-stream evaluation can establish
EAST benefit; no automatic promotion based on this screen.

MT3: official AISTATS2022 paper, reread §2–2.1; learn consequences of unlabeled
inner steps. Our frozen classifier/head-only diagnostic task is different.
Sagawa et al.,ICLR2020, arXiv1911.08731v2, read §1–2 and Eq1–3: worst-group
training and worst-group generalization differ. Here maximum change-plus-
alignment is only an engineering hypothesis, not their algorithm/theorem.
All existing data development-exposed; no independent-confirmation claim.
