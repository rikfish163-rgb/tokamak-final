"""Recount completed commissioning arms, then report matched operating stages.

No threshold search, model updates or cohort filtering. A rising cumulative
score alone is not evidence: methods must be compared on the same later shots.
"""
import argparse
import json
import math
from pathlib import Path

from .persistent_ffn_verify import verify
from .simple_online import _durable_json, _hash
from .stage_metrics import summarize_commissioning


def paired_changes(reference, candidate):
    if [r['shot'] for r in reference] != [r['shot'] for r in candidate]:
        raise ValueError('paired comparisons need identical shot identities/order')
    result = dict(tp_rescued=[], tp_lost=[], fp_removed=[], fp_added=[])
    for b, c in zip(reference, candidate):
        if b['is_disrupt'] != c['is_disrupt']:
            raise ValueError('paired comparisons need identical revealed outcomes')
        if b['is_disrupt']:
            if not b['timely'] and c['timely']: result['tp_rescued'].append(c['shot'])
            if b['timely'] and not c['timely']: result['tp_lost'].append(c['shot'])
        else:
            if b['alarm'] and not c['alarm']: result['fp_removed'].append(c['shot'])
            if not b['alarm'] and c['alarm']: result['fp_added'].append(c['shot'])
    return result


def build_report(directory):
    protocol = json.loads((directory / 'protocol.json').read_text())
    # The new commissioning protocol hashes the SAVED manifests. Historical
    # shot-number runs hashed original files before JSON reserialization;
    # their identity/seal recount remains handled by the historical verifier.
    if (_hash(directory / 'order.json') != protocol['order_sha256']
            or _hash(directory / 'roster.json') != protocol['roster_sha256']):
        raise ValueError('locked order or roster hash changed')
    verification = verify(directory)
    comparisons = json.loads((directory / 'comparison.json').read_text())
    order = json.loads((directory / 'order.json').read_text())
    roster = json.loads((directory / 'roster.json').read_text())
    plan = roster['selection_protocol']
    if (plan.get('order') != 'offline_initial100ms_current_energy'
            or plan.get('ordering_uses_outcomes') is not False
            or plan.get('ordering_uses_predictions') is not False):
        raise ValueError('explicit outcome/prediction-free commissioning order required')
    stages = {arm: summarize_commissioning(result['metrics']['predictions'],
                    order, startup_budget=30, block_size=30)
              for arm, result in comparisons.items()}
    rows = {arm: result['metrics']['predictions'] for arm, result in comparisons.items()}
    q = math.ceil(len(order) / 4)
    for arm, stage in stages.items():
        stage['proxy_definition'] = plan['proxy'] + '; relative operating proxy, not damage or DBC'
        quarters = []
        for start in range(0, len(order), q):
            cohort = order[start:start+q]
            ids = {r['shot'] for r in cohort}
            observed = [r for r in rows[arm] if r['shot'] in ids]
            quarters.append(dict(start_index=start+1, end_index=start+len(cohort),
                metrics=summarize_commissioning(observed, cohort, startup_budget=0,
                                               block_size=30)['full']))
        stage['operating_quarters'] = quarters
    paired = {arm: paired_changes(rows['static'], rows[arm])
              for arm in ('intra', 'inter', 'both')}
    paired['addition_of_intra_to_inter_policy'] = paired_changes(rows['inter'], rows['both'])
    gates = {}
    for arm, data in stages.items():
        full = data['full']
        gates[arm] = {str(cap): bool(full['FPR'] is not None and full['FPR'] <= cap)
                      for cap in (.10, .15, .20)}
    return dict(evidence='DEVELOPMENT_OFFLINE_COMMISSIONING; not independent future validation',
        comparison_sha256=_hash(directory / 'comparison.json'),
        order_sha256=_hash(directory / 'order.json'),
        source_sha256=verification['source_sha256'], stages=stages,
        paired_changes=paired, fpr_caps=gates, verification=verification,
        inter_vs_both_interpretation='same update policy but different learned histories; factorial total effect, not a fixed-persistent-state intervention',
        cost_meaning='relative initial-current squared proxy only; not measured damage or DBC',
        thresholds_changed=False, startup_shots_included=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve earlier report')
    report = build_report(args.directory)
    _durable_json(args.output, report)
    print(json.dumps(dict(arms={a: s['full'] for a, s in report['stages'].items()},
                          fpr_caps=report['fpr_caps'])), flush=True)


if __name__ == '__main__':
    main()
