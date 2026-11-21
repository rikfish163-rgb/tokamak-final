# First30 signal calibration / frozen330 diagnostic

Date: 2026-09-29. Development-only, same qualified source, fixed VAL threshold
0.7035074234, same existing commissioning order. First30=26 disruptive+4 safe;
evaluation330=154 disruptive+176 safe. Statistics never read first30 labels.

| Arm | TP (>30ms)/154 | FP/176 | TPR | FPR | J |
|---|---:|---:|---:|---:|---:|
| Original source coordinates |127|99|0.824675|0.562500|0.262175|
| Same source coordinates, calibration-unusable mask |117|55|0.759740|0.312500|0.447240|
| Same mask + target median/IQR |120|57|0.779221|0.323864|0.455357|

Calibration-unusable channels are CIII, MNM, mode_number_n. This means no
identifiable calibration IQR/availability, NOT proof these measurements are
physically useless on all future shots. qa_proxy remains usable: a regression
test rejects the incorrect absolute-epsilon criterion for small-unit channels.

Most of the net J difference from original arises in the mask-only arm. Adding
target scale on top gives +3 net TP and +2 net FP, deltaJ=0.008117. Neither arm
meets the FPR<=.20 requirement. This does not demonstrate intra-shot neural
adaptation, nor imply that all scale alignments or all23 features are appropriate.
No neural parameter changed in either run; each run's receipt SHA was checked
for all330 predictions. Evaluation shot IDs match the original source tail330.
Fourteen targeted tests and Ruff F/E9 pass; these are implementation checks,
not scientific success criteria.

## Filesystem incident / storage

The earlier run on NTFS stopped after partial output. Kernel logs explicitly
associate PID192920 with BUG fs/iomap/buffered-io.c:1061 and
ntfs_file_write_iter. Its main thread was Zsl; TERM then returned exit143.
That directory has no complete result and uses an earlier incorrect scale
degeneracy test; do not combine it with these completed runs.

Completed runs are TEMPORARILY in /dev/shm/fuison-alignment-CqqYA4.
This is RAM storage and is lost on reboot. No original data moved; no filesystem
repair/remount attempted. Durable export to a healthy filesystem is pending.

## Next scientific step

Compare target-supervised learning against these input-only controls, using
the same30 ended shots and same330 evaluation. Inspect whether dense negative
window training suppresses timely detections. Refit signal forecast heads when
their input representation changes. The final full360 prequential experiment
still requires independent intra benefit; do not substitute this frozen330
diagnostic for the user goal.
