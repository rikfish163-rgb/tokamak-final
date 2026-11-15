# Teaching support for real internal pre-alarm correction

Determine whether a causal direction family has enough independent ended-shot
examples to learn an actual early-warning improvement instead of a score shift.
No policy fitting, alarm retuning, new J-TEXT VAL/TEST or EAST scoring access.

1. Census every disruptive shot in the three original sealed TRAIN1353 views.
   A potential teacher must be nontimely under the frozen score, have an action
   generated from reached data with the archived symmetric norm/logit guard
   at all four reached clocks, and take effect after `m+8` but before any frozen
   alarm. Its first crossing must be strictly >10 ms before the authoritative
   HDF5 DownTime and in the original `max(25 ms,AdvancedTime)` supervision
   window. Use the existing actual neural action bank; compare a control
   that holds each same action's exact observed logit shift at `m+8`.
   Count ordinary timely support, canonical-window support, and the subset
   for which **no legal scalar action** can make this shot timely, even outside
   the canonical positive window. One earliest write per original shot/view;
   no selection of a safe match by outcome of the edit.
2. For a strict teacher, find the nearest static no-alarm safe shot in that
   view: match reached logit within 1.0, elapsed time within .5 s, and same
   observed diagnostic-family pattern; distance also uses reached risk
   slope. A safe shot is used at most once per view. Apply the **same
   actual temporary state tensor** at the safe release, check both signs
   against the same guard, and verify no new first alarm for the actual sign.
   No match, rejected guard and false alarm are distinct failure modes.
3. If original support is below 20 distinct paired teachers, add exactly
   two radiation gain stress views on the same source TRAIN, both with MNM
   masked. Read only waveform/masks of the frozen, completed EAST cold20, no
   hard outcomes. From each shot's first 256 samples use the median observed
   SXR and XUV variances (at least 64 observations each, positive medians).
   Relative to the cross-shot medians, form two amplitude gains by square
   root. Order these observed pairs by mean log gain; take the floor-index
   25th and 75th percentile donor pairs, not artificial combinations.
   Require at least 10 supported donors and distinct profiles. Apply the
   assumed positive detector gain to SXR mean and its square to SXR variance,
   and the XUV gain squared to XUV variance. Preserve shape statistics,
   ratios, valid observations, labels and times. Measured shot differences
   can be plasma variation, not certified detector errors; the stress
   views are only measurement perturbations, not new physical discharges.
4. Seal full stressed frozen scores for all 1,353 source TRAIN shots;
   replay the same actual internal action family on each stressed late
   shot, and apply the same matching and counting. Also count stress-induced
   late positives and new false alarms against the no-MNM comparator.
   Proceed to learning from this augmentation only if it adds ≥20 independent
   original-shot teachers that were timely in no-MNM, with ≥3 distinct
   originals in each of at least four deterministic existing folds. A win
   on this capacity screen is not a deployable result or EAST gain.
