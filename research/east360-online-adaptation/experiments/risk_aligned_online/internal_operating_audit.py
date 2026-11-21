"""Retrospective ranking/operating-point check of completed internal updates.

Uses ended-shot outcomes and searches diagnostic thresholds. None of these
thresholds may be deployed or reported as causal online performance.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import io23
from .operating_point_audit import at_threshold, best_at_count, peak_for_shot, timely_auc
from .persistent_ffn_verify import verify
from .score_shift_audit import read_seals, sha
from .simple_online import _durable_json


def summarize(rows, threshold):
    if 'static' not in rows:
        raise ValueError('same-input static reference required')
    identities = [(r['shot'], r['disruptive']) for r in rows['static']]
    if len({r[0] for r in identities}) != len(identities):
        raise ValueError('unique paired identity required')
    if any([(r['shot'], r['disruptive']) for r in arm] != identities for arm in rows.values()):
        raise ValueError('identical ordered outcome/shot identity required')
    original = {arm: at_threshold(r, threshold) for arm, r in rows.items()}
    n_safe = original['static']['n_safe']
    return dict(
        original=original,
        matched_static={arm: best_at_count(rows['static'], point['fp'])
                        for arm, point in original.items() if arm != 'static'},
        ranking_auc={arm: timely_auc(r) for arm, r in rows.items()},
        caps={str(cap): {arm: best_at_count(r, int(np.floor(cap*n_safe)))
                        for arm, r in rows.items()} for cap in (.10, .15, .20)},
        interpretation='ALL TARGET LABEL POSTHOC; searched thresholds are diagnostic only, '
            'not an online policy or independent validation. Ranking gain is not proof of physics.')


def audit(directory):
    checked = verify(directory)
    protocol = json.loads((directory/'protocol.json').read_text())
    identities = [r['shot'] for r in json.loads((directory/'order.json').read_text())
                  [:protocol['selected_shots']]]
    arms = ('static', 'intra', 'inter', 'both')
    traces = {arm: read_seals(directory/arm) for arm in arms}
    rows = {arm: [] for arm in arms}
    shifts = {arm: [] for arm in arms if arm != 'static'}
    for sid in identities:
        reference = traces['static'][sid]
        disruptive, down = io23.read_outcome('EAST', sid)
        for arm in arms:
            current = traces[arm][sid]
            if (set(traces[arm]) != set(identities)
                    or current['threshold'] != protocol['threshold']
                    or not np.array_equal(current['times'], reference['times'])):
                raise ValueError('sealed cohort, threshold or time grid changed')
            rows[arm].append(dict(shot=sid, disruptive=bool(disruptive),
                peak=peak_for_shot(current['times'], current['scores'], disruptive, down),
                seal_sha256=current['sha256']))
            if arm != 'static':
                delta = current['scores'].astype(np.float64)-reference['scores']
                shifts[arm].append(dict(shot=sid, disruptive=bool(disruptive),
                    mean_probability_delta=float(delta.mean()),
                    maximum_absolute_delta=float(np.abs(delta).max())))
    report = summarize(rows, protocol['threshold'])
    for arm, point in report['original'].items():
        for key in ('tp', 'fp', 'n_disrupt', 'n_safe', 'tpr', 'fpr'):
            if point[key] != checked['arms'][arm][key]:
                raise ValueError('timely peak differs from independent first-alarm recount')
    return dict(scope='POSTHOC DEVELOPMENT DIAGNOSTIC; NO ONLINE THRESHOLD CHANGE',
        directory=str(directory.resolve()), source_sha256=checked['source_sha256'],
        protocol_sha256=sha(directory/'protocol.json'), code_sha256=sha(__file__),
        selected_shots=len(identities), routing=protocol['source_router']['routing'],
        source_model_changed=False, threshold_deployed=False, analysis=report,
        per_shot_peaks=rows, per_shot_probability_changes=shifts,
        verified_pair_changes={arm: checked['arms'][arm]['paired_vs_static']
                               for arm in arms if arm != 'static'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous diagnostic')
    report = audit(args.directory)
    _durable_json(args.output, report)
    print(json.dumps(report['analysis'], allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
