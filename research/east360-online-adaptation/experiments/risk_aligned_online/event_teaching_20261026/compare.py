"""Rescore all three complete streams; never select an arm using EAST labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..simple_train import sha
from ..simple_online import _durable_json
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from .learning import KINDS


def read(root, name):
    return json.loads((root/name).read_text())


def validate_protocols(protocols):
    if len(protocols) != 3:
        raise ValueError('all three predeclared arms required')
    first = protocols[0]
    for kind, p in zip(KINDS, protocols, strict=True):
        for key in first.keys() | p.keys():
            if key != 'source_memory' and first.get(key) != p.get(key):
                raise ValueError('changed paired protocol: '+key)
        a, b = first['source_memory'], p['source_memory']
        if (b['kind'] != kind or b['profile'] != 'event_teaching_v1' or b['epoch'] != 2
                or any(a[k] != b[k] for k in ('profile', 'epoch', 'source_protocol_sha256', 'audit_sha256'))
                or p['feedback'] != 'progressive' or p['source_deployment_feedback_match'] is not True):
            raise ValueError('matched audited epoch2 source event artifacts required')


def compare(roots):
    protocols = [read(r, 'protocol.json') for r in roots]
    validate_protocols(protocols)
    audits = [read(r, 'independent_audit.json') for r in roots]
    if not all(a['complete360'] for a in audits):
        raise ValueError('independently rescored complete360 runs required')
    for p in protocols:
        receipt = p['source_memory']
        checkpoint = Path(receipt['path'])
        assert sha(checkpoint) == receipt['sha256']
        assert sha(checkpoint.parent/'audit.json') == receipt['audit_sha256']
        assert sha(checkpoint.parent/'protocol.json') == receipt['source_protocol_sha256']
    same = ('order.json', 'input_map.json', 'source_replay.json', 'input_hashes.json')
    for filename in same:
        for root in roots[1:]:
            if read(roots[0], filename) != read(root, filename):
                raise ValueError('changed paired inputs: '+filename)
    arms = protocols[0]['arms']
    rows = [{arm: [] for arm in arms} for _ in roots]
    reverse_differences = [dict(changed_shots=0, endpoints=0, total_abs=0., max_abs=0.) for _ in roots]
    for index, item in enumerate(read(roots[0], 'order.json'), 1):
        sid = item['shot']
        name = f'{index:04d}_{sid}.npz'
        flag, down = io23.read_outcome('EAST', sid)
        reference = None
        for run, root in enumerate(roots):
            with np.load(root/'shots'/name) as curves:
                if reference is None:
                    reference = {k: curves[k].copy() for k in ('times', 'source', 'aligned')}
                else:
                    for key, value in reference.items():
                        assert np.array_equal(value, curves[key]), (sid, key)
                for arm in arms:
                    rows[run][arm].append(dict(shot=sid, disruptive=flag,
                        **classify(curves['times'], curves[arm], flag, down, protocols[0]['threshold'])))
                if index > 20:
                    difference = np.abs(curves['intra'].astype(float)-curves['intra_reverse'])
                    r = reverse_differences[run]
                    r['changed_shots'] += int(difference.max() > 0)
                    r['endpoints'] += len(difference)
                    r['total_abs'] += float(difference.sum())
                    r['max_abs'] = max(r['max_abs'], float(difference.max()))
    for run in range(3):
        assert {k: counts(v) for k, v in rows[run].items()} == audits[run]['metrics']
        r = reverse_differences[run]
        r['mean_abs'] = r.pop('total_abs')/r['endpoints']
    result = dict(scope='ONE_SEED_DEVELOPMENT_COMPLETE360', identical_source_aligned_shots=360,
        unchanged_inputs=list(same), threshold=protocols[0]['threshold'],
        runs={k: str(r.resolve()) for k, r in zip(KINDS, roots, strict=True)},
        metrics={k: a['metrics'] for k, a in zip(KINDS, audits, strict=True)},
        within_run={k: a['flips'] for k, a in zip(KINDS, audits, strict=True)},
        true_versus_none={arm: flips(rows[0][arm], rows[1][arm]) for arm in arms},
        true_versus_shuffled={arm: flips(rows[2][arm], rows[1][arm]) for arm in arms},
        reverse_probability_differences=dict(zip(KINDS, reverse_differences, strict=True)),
        attribution='Within-run intra versus cold; cross-run cold weights differ after source teaching.',
        independent_validation=False, physical_law_discovery=False)
    path = roots[1]/'paired_event_audit.json'
    if path.exists():
        raise FileExistsError(path)
    _durable_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for kind in KINDS:
        p.add_argument(kind, type=Path)
    args = p.parse_args()
    compare([getattr(args, k) for k in KINDS])
