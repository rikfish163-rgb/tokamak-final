"""Integrity-checked diagnostic report; never tunes the deployed alarm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..representation_analysis import analyze_run, paired_flips
from ..simple_online import _durable_json


def acceptance(metrics, expected_count=None):
    required = {'static','intra','inter','both'}
    if not required <= metrics.keys():
        return dict(all_required_gates_pass=False, reason='four arms incomplete')
    counts = {(m['n_disrupt'],m['n_safe']) for m in metrics.values()}
    if len(counts) != 1:
        raise ValueError('paired arm denominator mismatch')
    static, intra, inter, both = (metrics[a] for a in ('static','intra','inter','both'))
    if any(m['J'] is None for m in (static,intra,inter,both)):
        return dict(all_required_gates_pass=False,reason='both shot classes required')
    gates = dict(
        independent_intra_gain=intra['J'] > static['J'],
        intra_timely_recall_preserved=intra['tp'] >= static['tp'],
        intra_within_fpr_cap=intra['fpr'] <= .20,
        combined_better_than_intra=both['J'] > intra['J'],
        combined_better_than_inter=both['J'] > inter['J'],
        combined_timely_recall_preserved=both['tp'] >= max(intra['tp'],inter['tp']),
        combined_within_fpr_cap=both['fpr'] <= .20)
    if expected_count is None:
        gates['full360'] = counts == {(180,180)}
    else:
        if type(expected_count) is not int or expected_count <= 0:
            raise ValueError('expected stream count must be a positive integer')
        gates['full_locked_stream'] = all(sum(c) == expected_count for c in counts)
    return dict(gates,all_required_gates_pass=all(gates.values()),
                warning='single development run; these gates do not establish significance or independent validity')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(argv)
    report = analyze_run(args.run)
    summaries = json.loads((args.run/'summary.json').read_text())
    measured = {a:s['metrics'] for a,s in summaries.items()}
    if {'static','both'} <= measured.keys():
        report['paired_flips']['both_vs_static'] = paired_flips(
            measured['static']['predictions'],measured['both']['predictions'],
            before_arm='static',after_arm='both')
    if {'intra','both'} <= measured.keys():
        report['paired_flips']['both_vs_intra'] = paired_flips(
            measured['intra']['predictions'],measured['both']['predictions'],
            before_arm='intra',after_arm='both')
    protocol = json.loads((args.run/'protocol.json').read_text())
    order = json.loads((args.run/'order.json').read_text())
    limit = protocol['args']['limit']
    emitted_order = [row['shot'] for row in (order[:limit] if limit else order)]
    for m in measured.values():
        if [p['shot'] for p in m['predictions']] != emitted_order:
            raise ValueError('sealed predictions differ from the predeclared stream order')
    flexible = protocol['args'].get('roster') is not None
    report['acceptance'] = acceptance(measured,expected_count=len(order) if flexible else None)
    report['early_alarm_diagnostics'] = {arm:dict(
        timely_alarms_above_1000ms=sum(p['timely'] and p['lead_ms'] > 1000. for p in m['predictions']),
        timely_alarms_above_3000ms=sum(p['timely'] and p['lead_ms'] > 3000. for p in m['predictions']),
        warning='still counted under official >30ms rule; not proof of precursor localization')
        for arm,m in measured.items()}
    report['lineage'] = 'independent zero-target-label internal redesign; not V3C'
    if args.output.exists():
        raise FileExistsError(args.output)
    _durable_json(args.output,report)
    print(json.dumps(dict(acceptance=report['acceptance'],metrics={a:{k:v for k,v in m.items()
        if k != 'predictions'} for a,m in measured.items()}),indent=2),flush=True)
    return report


if __name__ == '__main__':
    main()
