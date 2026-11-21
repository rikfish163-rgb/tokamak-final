# Protected Q/K: post-hoc error opportunity diagnosis

Date: 2026-10-29. This is analysis of already sealed full360 curves, not a new
online method or an independent confirmation. No model, threshold, order or
input mapping is changed. Future outcomes are used only by this offline audit.

Question: did the actual change have enough amplitude, at the right time, to
alter a wrong decision? For each point let b be cold probability, a the adapted
probability and e=abs(a-b). Consider the deliberately optimistic curve envelope
[b-e,b+e], clipped to [0,1]. Before sample index256 (zero-based), and for the
cold-start20 shots, the envelope is fixed to b. All four candidate arms are
reported; none is selected after looking at labels.

For a missed/late disruptive shot the envelope can correct it only if some
valid point strictly more than30ms before disruption has b+e>=threshold.
For a false-positive safe shot it can correct it only if **every** valid point
has b-e<threshold. A later reduction cannot erase an earlier sealed alarm.
This is a mathematical bound under observed pointwise probability amplitudes,
NOT a feasible neural policy, a gradient-step recommendation, a causal recovery
upper bound for other architectures, or an achieved rescue count.

Also report the actual signed change of operational peaks, the separation
between real and time-reversed feedback curves, and their risk-change cosine.
Separate cold20/full360/suffix340; keep all wrong shots, including immutable
early alarms. Recount actual decisions from curves and compare stored records.
No p-value or independent generalization claim for this repeatedly developed,
single-stream, single-seed diagnostic. Output is small JSON; no HDF5/cache copy.
