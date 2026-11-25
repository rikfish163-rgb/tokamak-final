"""First causal alarm scoring; outcomes and AdvancedTime are evaluation-only."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def score_shot(times, scores, threshold_curve, disruptive, down_time, *,
               hold_ms=1, deadline_ms=30., adv_time_ms=None):
    times, scores = np.asarray(times, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if times.ndim != 1 or scores.shape != times.shape or not np.isfinite(times).all():
        raise ValueError("matching finite timestamps and one-dimensional scores required")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must strictly increase")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities")
    threshold = np.broadcast_to(np.asarray(threshold_curve, dtype=np.float64), scores.shape)
    if not np.isfinite(threshold).all() or not np.isfinite(hold_ms) or hold_ms < 1:
        raise ValueError("finite thresholds and hold_ms >= 1 required")
    if not np.isfinite(deadline_ms) or deadline_ms < 0:
        raise ValueError("nonnegative deadline required")
    if disruptive and (down_time is None or not np.isfinite(down_time)):
        raise ValueError("disruptive shots require raw DownTime")
    if adv_time_ms is not None and (not np.isfinite(adv_time_ms) or adv_time_ms < 0):
        raise ValueError("AdvancedTime must be nonnegative or None")
    eligible = times <= down_time + 1e-9 if disruptive else np.ones(len(times), bool)
    hit = (scores >= threshold) & eligible
    # A 1 ms hold alarms at the first 1 kHz sample; H ms at sample H.
    # At later endpoints the width uses the already-observed previous interval.
    starts = np.flatnonzero(hit & ~np.r_[False, hit[:-1]])
    ends = np.flatnonzero(hit & ~np.r_[hit[1:], False])
    alarm = None
    for start, end in zip(starts, ends):
        indices = np.arange(start, end + 1)
        widths_ms = np.ones(len(indices))
        noninitial = indices > 0
        widths_ms[noninitial] = 1000. * (times[indices[noninitial]] - times[indices[noninitial] - 1])
        duration = 1000. * (times[indices] - times[start]) + widths_ms
        satisfied = indices[duration >= hold_ms - 1e-9]
        if len(satisfied):
            alarm = float(times[satisfied[0]])
            break
    lead = max(0., round(1000. * (down_time - alarm), 6)) if disruptive and alarm is not None else None
    if disruptive:
        outcome = "miss" if lead is None else "TP_timely" if lead > deadline_ms else "tardy"
    else:
        outcome = "FP" if alarm is not None else "TN"
    return dict(disruptive=bool(disruptive), down_time=down_time if disruptive else None,
                outcome=outcome, alarm_time=alarm, lead_ms=lead,
                shot_max_score=float(scores[eligible].max()) if eligible.any() else 0.,
                adv_time_ms=adv_time_ms)


def _ratio(n, d):
    return float(n / d) if d else None


def _stats(values):
    return (float(np.mean(values)), float(np.median(values))) if values else (None, None)


def summarize_stream(rows, *, deadline_ms=30., checkpoints=(1, 10, 20, 50, 100, 180, 360)):
    """DBC cost=(Σ missed/tardy DBC_ip2 + FP)/(Σ disruptive DBC_ip2 + N).

    AdvancedTime denominator is exactly disruptive shots with available truth,
    never all disruptive shots with guessed/fallback annotation. AUC uses raw
    shot maximum probabilities (not alarm outcomes or threshold margins).
    """
    rows = list(rows)
    if not np.isfinite(deadline_ms) or deadline_ms < 0:
        raise ValueError("nonnegative deadline required")
    d = [r for r in rows if r["disruptive"]]
    n = [r for r in rows if not r["disruptive"]]
    timely = [r for r in d if r["lead_ms"] is not None and r["lead_ms"] > deadline_ms]
    tardy = [r for r in d if r["lead_ms"] is not None and r["lead_ms"] <= deadline_ms]
    misses = [r for r in d if r["lead_ms"] is None]
    fp = sum(r["alarm_time"] is not None for r in n)
    tp, late, miss, tn = len(timely), len(tardy), len(misses), len(n) - fp
    tpr, fpr = _ratio(tp, len(d)), _ratio(fp, len(n))
    bounded = [r for r in timely if r["lead_ms"] <= 3000.]
    leads = [r["lead_ms"] for r in d if r["lead_ms"] is not None]
    bounded_leads = [lead for lead in leads if 0 <= lead <= 3000.]
    covered = [r for r in d if r.get("adv_time_ms") is not None]
    beat = sum(r["lead_ms"] is not None and r["lead_ms"] >= r["adv_time_ms"] for r in covered)
    weights = [r.get("dbc_ip2") for r in d]
    have_dbc = all(w is not None and np.isfinite(w) and w >= 0 for w in weights)
    weighted_miss = sum(r["dbc_ip2"] for r in tardy + misses) if have_dbc else None
    lm, lmed = _stats(leads)
    bm, bmed = _stats(bounded_leads)
    result = dict(n_shots=len(rows), n_disruptive=len(d), n_normal=len(n),
                  tp_timely=tp, tardy=late, miss=miss, fp=fp, tn=tn,
                  timely_tpr=tpr, any_alarm_recall=_ratio(tp + late, len(d)), fpr=fpr,
                  timely_acc=_ratio(tp + tn, len(rows)), f1_timely=_ratio(2 * tp, 2 * tp + fp + late + miss),
                  youden_J=tpr - fpr if tpr is not None and fpr is not None else None,
                  roc_auc=float(roc_auc_score([r["disruptive"] for r in rows],
                                              [r["shot_max_score"] for r in rows])) if d and n else None,
                  bounded3000_tpr=_ratio(len(bounded), len(d)),
                  premature_gt3000_count=sum(r["lead_ms"] is not None and r["lead_ms"] > 3000. for r in d),
                  lead_ms_mean=lm, lead_ms_median=lmed,
                  bounded3000_lead_mean=bm, bounded3000_lead_median=bmed,
                  dbc_weighted_cost=_ratio(weighted_miss + fp, sum(weights) + len(n)) if have_dbc else None,
                  dbc_covered_disruptive=sum(w is not None for w in weights),
                  adv_time_subset=dict(n_covered=len(covered), n_disruptive=len(d),
                                       coverage=_ratio(len(covered), len(d)), beat_adv_time_count=beat,
                                       beat_adv_time_rate=_ratio(beat, len(covered)),
                                       timely_tpr=_ratio(sum(r["lead_ms"] is not None and r["lead_ms"] > deadline_ms
                                                             for r in covered), len(covered))))
    if checkpoints:
        result["checkpoints"] = {int(k): summarize_stream(rows[:k], deadline_ms=deadline_ms, checkpoints=())
                                 for k in checkpoints if 0 < k <= len(rows)}
    return result
