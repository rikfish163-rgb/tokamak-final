# Generate shared internal directions under danger–safe constraints

In the prior TRAIN audit two canonical danger-only directions failed when
transferred to matched safe shots. Does including the safety partner during
direction generation recover any real neural-only first alarms, and does the
answer depend on write location? This is an ended-shot diagnostic. No policy
training, no new J-TEXT VAL/TEST or EAST signal/outcome reads.

- Start with all frozen nontimely disruptive J-TEXT TRAIN shots in the three
  original views and the two **already sealed** radiation stress views.
  Use the same 32-sample schedule and no prior alarm by the observed
  commitment at `m+8`. Require an actual remaining suffix sample within
  the original positive window with strict lead >10 ms and ≥2 observed
  diagnostic families. Match one previously static-no-alarm safe TRAIN
  shot per release using prior risk, elapsed-time/slope distance, and the
  same family pattern. Reserve the three archived exact witnesses before
  greedy matching; their original safe IDs/clocks do not move. No safety
  partner reuse within a view and no matching on future response.
- Compare input motion-patch and inter-layer motion-patch writes. Same
  frozen model, 64D temporary matrix, cap .35 and affected frames
  `m+9..m+32`. At each site differentiate all reached anchors and each
  future sample at the identity state. For every canonical timely danger
  frame, optimize its linearized score over the same L2 ball. Both arms
  impose four symmetric danger history limits, avoid new earlier crossings
  outside the canonical window, and constrain the same-anchor scalar
  below the threshold on every timely sample in this suffix. The paired
  arm additionally imposes four symmetric **safe** history limits and
  caps every safe suffix sample below the alarm line. Report the dual
  upper bound on this local affine problem only, never a global neural bound.
- The direction-generation arms share a finite norm grid: for each
  target/site use the smaller solved norm, then its scales
  `1,.5,.25,.125,.0625` for both orientations. Run the frozen model
  exactly at every tested scale. History protection requires both signs
  at `m-16,m-8,m,m+8` on both shots to change by no more than
  `.10-.00002` logit. Future writes use only the positive signed candidate.
  Count (a) danger canonical recovery under danger protection; (b) same
  actual direction also passes safe history and leaves the whole safe shot
  without alarm; (c) additionally the exact danger same-anchor scalar fails.
  Also flag whether **any** positive scalar respecting the .09998 anchor cap
  could rescue the same release. Do not replace exact first alarm with peak.
- Deduplicate by physical source shot, report stress separately and show
  the three locked witnesses. Require ≥3 original non-stress independent
  successes to justify a subsequent source learnability study at a site;
  keep the previous ≥20 independent teachers gate before proposing an
  augmentation-trained policy. No EAST promotion on an oracle result.
