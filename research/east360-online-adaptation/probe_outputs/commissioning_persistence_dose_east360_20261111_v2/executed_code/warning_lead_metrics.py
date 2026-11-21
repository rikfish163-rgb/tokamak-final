"""Secondary EAST warning deadlines and paired first-alarm lead summaries."""

import argparse
from pathlib import Path
from statistics import mean, median

from .history_gradient_metric_20261102.run import read, sha, write
from .source_anticipation_20261104.static_stream import timing_decomposition


DEADLINES_MS = (15, 20, 25, 30)
EARLY_MS = 3000


def _describe(values):
    return dict(count=len(values), mean_ms=float(mean(values)) if values else None,
                median_ms=float(median(values)) if values else None)


def summarize(rows):
    primary = timing_decomposition(rows, deadline_ms=30)
    alarms = [r['lead_ms'] for r in rows if r['disruptive'] and r['alarm']]
    normal = [lead for lead in alarms if lead <= EARLY_MS]
    deadline = {}
    n = primary['n_disruptive']
    for t in DEADLINES_MS:
        timely = [lead for lead in alarms if lead > t]
        timely_bounded = [lead for lead in normal if lead > t]
        deadline[str(t)] = dict(
            tp=len(timely), tpr=len(timely)/n if n else None,
            tp_at_most_3000ms=len(timely_bounded),
            tpr_at_most_3000ms=len(timely_bounded)/n if n else None,
            timely_lead_ms=_describe(timely), timely_lead_at_most_3000ms=_describe(timely_bounded))
    return dict(n_disruptive=n, n_safe=primary['n_safe'], fp=primary['fp'], fpr=primary['fpr'],
                misses=primary['miss'], early_over3000=len(alarms)-len(normal),
                all_alarmed_disruptions=_describe(alarms),
                alarmed_disruptions_at_most_3000ms=_describe(normal), by_deadline=deadline)


def paired(reference, current):
    """Compute gains on the same shots; missing alarms are separate events."""
    keys = ('shot', 'disruptive', 'down_s')
    if [tuple(r[k] for k in keys) for r in reference] != [tuple(r[k] for k in keys) for r in current]:
        raise ValueError('both arms require the same ordered HDF5 shots and outcomes')
    dis = [(a['shot'], a['lead_ms'], b['lead_ms'])
           for a,b in zip(reference,current,strict=True) if a['disruptive']]
    common = [(a,b) for _,a,b in dis if a is not None and b is not None]
    strict = [(a,b) for a,b in common if max(a,b) <= EARLY_MS]
    flips = {}
    for t in DEADLINES_MS:
        win = [sid for sid,a,b in dis if b is not None and b>t and (a is None or a<=t)]
        lose = [sid for sid,a,b in dis if a is not None and a>t and (b is None or b<=t)]
        flips[str(t)] = dict(gained=win, lost=lose, net=len(win)-len(lose))
    return dict(lead_gain_common=_describe([b-a for a,b in common]),
                lead_gain_common_at_most_3000ms=_describe([b-a for a,b in strict]),
                newly_alarmed_disruptions=[sid for sid,a,b in dis if a is None and b is not None],
                no_longer_alarmed_disruptions=[sid for sid,a,b in dis if a is not None and b is None],
                deadline_flips=flips)


def report(files, against):
    per = {}
    for f in files:
        for c, group in read(f).items():
            saved = per.setdefault(c, {})
            for k, rows in group.items():
                if k in saved and rows != saved[k]:
                    raise ValueError(f'duplicate arm {k} has inconsistent first-alarm records')
                saved[k] = rows
    return dict(
        status='POSTSEAL_EAST360_SECONDARY_WARNING_DEADLINES_NOT_RECALIBRATED',
        input_sha256={str(f): sha(f) for f in files},
        scorer_sha256=sha(Path(__file__)),
        deadline_rule='lead in milliseconds strictly greater than the stated deadline',
        early_definition='first alarm strictly more than 3000 ms before disruption',
        thresholds_unchanged=True, missed_shots_excluded_from_mean_but_in_tpr_denominator=True,
        metrics={c: {k: summarize(v) for k,v in group.items()} for c,group in per.items()},
        paired={c: {ref: {k: paired(group[ref], v) for k,v in group.items() if k!=ref}
                    for ref in against} for c,group in per.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--records', type=Path, action='append', required=True)
    ap.add_argument('--against', action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    write(a.output, report([f.resolve() for f in a.records], a.against))
