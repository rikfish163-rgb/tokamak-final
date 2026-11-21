# Already-observed fit versus next-window diagnostic

2026-10-29, after seeing all eight full-stream results. Retrospective diagnostic,
not a new online performance result and not an update-policy selector.

The source-head arm rescues one shot but reverse/hold rescue the same shot.
Cold-head fitting rescues two and harms two. Before trying another head, check
whether the arrived-signal update even improves the next *unobserved* H25 target.

Use the exact three auxiliary heads, classifier, fixed cold operator and mapping
from qk_forecast_east360_v1. No optimization except replaying its one .05 Q/K step.
For every shot after cold20, choose up to four evenly spaced eligible feedback
clocks from 256,384,... with a complete subsequent25-point target. Selection
depends on signal length alone, not outcome, error, or candidate performance.
At each selected clock c, the update sees only indices0..c-1. All heads forecast
c..c+24 once from input c-128..c-1, with and without that update. Compare against
holding the last observed value. No recursive use of model outputs as truths.

Use exactly the existing asinh source coordinate and masked SmoothL1 loss.
Exclude mode_number_n; intersect target availability with last-input availability.
Keep per-channel sums/counts, per-clock losses, and per-shot means. Report before
and after next-window losses, persistence baseline, and already-observed losses.
Outcomes are used only afterward to annotate safe/disruptive cases and discard
postterminal forecast targets from the primary descriptive summary. All selected
records, including such exclusions, remain in the diagnostic file. Single seed;
sampled feedback clocks, not exhaustive endpoint coverage or new360 TPR/FPR.
