"""Posthoc shot-order description; never selects training shots or edits models."""
import argparse
import json
from pathlib import Path

from ..simple_online import _durable_json
from ..simple_train import sha


def class_ranges(rows):
    ids = [r['shot'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('distinct preparation identities required')
    groups = {label: sorted(r['shot'] for r in rows if r['is_disrupt'] == label)
              for label in (False, True)}
    if not all(groups.values()):
        raise ValueError('both preparation classes required')
    result = {name: dict(count=len(groups[label]), minimum=min(groups[label]),
                        maximum=max(groups[label]))
              for label, name in ((False, 'safe'), (True, 'disruptive'))}
    result['all_safe_earlier_than_all_disruptive'] = max(groups[False]) < min(groups[True])
    return result


def consecutive_cohorts(rows, early_ids, *, size=60):
    ids = [r['shot'] for r in rows]
    if size < 1 or ids != sorted(set(ids)):
        raise ValueError('positive size and distinct ordered shot IDs required')
    early = set(early_ids)
    safe_alarms = {r['shot'] for r in rows if not r['is_disrupt'] and r['alarm']}
    if not early <= safe_alarms:
        raise ValueError('early IDs must each be a safe alarm')
    result = []
    for start in range(0, len(rows), size):
        part = rows[start:start+size]
        disruptive = [r for r in part if r['is_disrupt']]
        safe = [r for r in part if not r['is_disrupt']]
        result.append(dict(first=part[0]['shot'], last=part[-1]['shot'],
            n_disrupt=len(disruptive), n_safe=len(safe),
            tp=sum(r['timely'] for r in disruptive), fp=sum(r['alarm'] for r in safe),
            early_fp=sum(r['shot'] in early for r in safe)))
    return result


def main(root):
    read = lambda p: json.loads(p.read_text())
    protocol = read(root/'protocol.json')
    verified = read(root/'verification.json')
    diagnosis = read(root/'posthoc_diagnostic.json')
    done, full = read(root/'completed.json'), read(root/'per_shot.json')
    assert verified['status'] == 'VERIFIED'
    assert sha(root/'completed.json') == verified['completed_sha256']
    assert sha(root/'verification.json') == diagnosis['verification_sha256']
    selection_path = Path(protocol['reference'])/'selection.json'
    selected = read(selection_path)['selected']
    assert [r['shot'] for r in selected] == protocol['selected_ids']
    groups = {}
    for arm, result in full.items():
        rows = result['predictions']
        assert [r['shot'] for r in rows] == protocol['contract']['shot_ids']
        groups[arm] = consecutive_cohorts(rows,
            diagnosis['false_alarms_before_first_possible_update'][arm])
        for key in ('tp', 'fp', 'n_safe', 'n_disrupt'):
            assert sum(r[key] for r in groups[arm]) == done['metrics'][arm][key]
    result = dict(scope='POSTHOC DESCRIPTION, NOT A CAUSAL ATTRIBUTION OR A DEPLOYMENT POLICY',
        grouping='six consecutive groups of60 in already fixed360 shot-ID order; no outcome-based grouping',
        caveat='shot ID approximates period, not a measured operating condition; IDs are not model inputs',
        code_sha256=sha(Path(__file__)), verification_sha256=sha(root/'verification.json'),
        per_shot_sha256=sha(root/'per_shot.json'), selection_sha256=sha(selection_path),
        preparation=class_ranges(selected), cohorts=groups)
    _durable_json(root/'cohort_diagnostic.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
