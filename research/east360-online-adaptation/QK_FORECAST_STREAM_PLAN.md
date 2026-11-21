# Source/cold-head forecasting feedback in the complete EAST360 stream

2026-10-29. Preserve qualified J-TEXT classifier, threshold, cold20 mapping and
internal64×64 cold operator. Independent source-only qk_matched_source_v1
forecast head improved all three eligible source groups; its matched-meta head
failed. Only the ordinary forecast head is investigated here; no rejected meta
head enters this stream. No J-TEXT online adaptation.

Hypotheses: (1) ordinary head retraining may produce better EAST intra feedback
than the original token-meta-trained head; (2) a forecaster additionally fitted
to ended cold20 signals may reduce target-task mismatch. Neither is established.

Eight arms, all identical classifiers/threshold/order/first20:
`cold`, `original_qk`, `source_forecast`, `source_reverse`, `source_hold`,
`cold_forecast`, `cold_reverse`, `cold_hold`.
All adapted arms use the same full first-layer Q/K8192 temporary coordinates,
one normalized gradient step .05, first feedback at256 points then every128.
Each step resets from the fixed cold parent, and each shot discards its state.
Four arrived nonoverlapping H25 targets, W128 inputs, original availability
rules, mode_number_n excluded. Reversal acts within each H25 target with masks;
hold uses the last observed input. No current-shot outcomes/events in feedback.

At the end of cold20 only, fit `cold_forecast` starting from source_forecast:
256 AdamW steps,lr1e-4,wd0,clip1,seed131,batch four uniformly sampled ended shots,
one eligible feedback clock per shot, giving16 complete input/target windows.
Only existing forecast-head parameters train. Input uses the verified prefix
map and fixed cold operator. Targets are actual arrived signal values; no
outcome, future shot or current-risk pseudo-label. Missing windows are excluded
by availability/length only and explicitly reported. No selection by suffix
performance. Final head only, no hyperparameter scan.

Every shot is predicted and durably sealed before outcome access. No persistent
updates after cold20, no expert-weight/alarm updates. First20 raw source curves
retained and included. Full360=180D+180S; first20 actually17D+3S, not20safe.
Ip/Bt order is a prior offline commissioning proxy, not measured damage. Use
exact reference order/mapping/cold.pt from lifecycle_context8360_service_20261028.
Every reproduced cold curve and crossing sequence must match the sealed
attention_protected_east360_v1 comparator within2e-6 (identical crossings).

Report full360,cold20,suffix340; strict EAST lead>30ms, FPR<=.20, separate>3000ms
alarms, rescue/harm shot IDs, update count/norm and auxiliary-loss improvement.
Source/cold forecast candidate requires TPR>=cold, J>cold and own reverse/hold,
and FPR<=.20. This necessary screen alone cannot prove stable benefit or physics.
If neither passes, reject both and retain all negative results. Full continuous
curves support later error analysis; no optimistic selected-shot result.

Source-head provenance: SHA94442cb888068289a598e57c2318c127cdad3f93fc6a5ae33bf76b91605a9867.
The earlier source training entry was not contemporaneously snapshotted; its
weights/data receipts and validation were audited, not claimed a fully archived
original training command. This stream snapshots its entry and all imported
project modules before starting. Original classifier file and cold state must
remain unchanged. All data are development-exposed, not independent confirmation.
