# Frozen TRAIN diagnostic: peak versus sustained internal margin

Test whether changing the state-generation objective, rather than another
readout of existing candidate curves, produces useful timely threshold margin.
This is a hindsight feasibility probe; it neither trains nor evaluates an
online policy. No new VAL, TEST, or EAST reads. No change to the frozen source
checkpoint, existing alarm threshold, split IDs, or online contract.

- Use the three previously sealed J-TEXT TRAIN1353 views (native, MNM masked,
  MNM/CIII masked). In each include every disruptive non-timely shot, the 48
  closest non-alarming safe shots from below the fixed threshold, the 24
  closest already alarming safe shots from above, and the 24 timely
  disruptive shots with the smallest above-threshold timely peak. Tie-break by
  shot ID. Control strata use HDF5 outcomes and static traces only.
- Same motion-token site and eight-by-eight state as the old bank. Use only
  its existing releases `m=63,95,...`; the first eight future samples stay
  unchanged. A state may act from `m+9` through `m+32`, capped at the last
  actual sample, if the frozen trace has not crossed by `m+8`. Timely target
  frames for disruptive shots must individually have lead strictly >10 ms
  from HDF5 DownTime; safe shots have no lead cutoff. Compare on blocks with
  at least one adjacent valid target pair; the single arm uses all endpoints
  of those pairs. No padding, gaps in eligibility or late frames create pairs.
- Differentiate the frozen model's actual logit at zero state for all target
  frames. Both arms share projection out of the reached logit gradients at
  `m-16,m-8,m,m+8` and the existing 0.35 state-norm cap. Peak maximizes one
  linearized margin; sustained maximizes the smaller of two adjacent
  linearized margins. Solve the two-affine maximin on the Euclidean ball;
  choose one state per arm and release, deterministic earliest tie. No fitting
  or temperature grid. The negatives of both states are sign controls.
- Use the old descending radius schedule to find one common multiplier for
  both arms and signs, with all four reached logit shifts checked by exact
  forward against `0.10-0.00002`. Reject both if no shared scale or either
  direction has no support. Re-run actual frozen network forward on the
  whole allowed suffix for both signs. Evaluate earliest alarms using the
  original one-crossing rule; the pair is only a generation objective and a
  separate durability readout, not a new alarm rule.
- Report single-edit timely gains and losses and safe FP additions/removals
  separately for the primary positive orientation and the signed menu; also
  report exact two-adjacent-timely-frame rescues. Compare the old menu on the
  same release support and outcome masks (12 old states, not equal menu size),
  and report its full-clock availability separately. Deduplicate original
  shot IDs across views; views are stress checks, not independent replicates.
- Advance the sustained objective to a shot-disjoint causal learning trial
  only if it yields at least two distinct new two-frame timely rescues outside
  the matched peak/old single-edit timely sets, with neither additional
  positive-orientation false-alarm shots nor loss of originally timely shots
  relative to peak on these stress strata. New cases require review of whether
  their advantage persists long enough to be recognized from reached data.
  Failure closes this matched experiment, not all possible state spaces.
