"""Full-stream paired comparison; source teaching objective is the changed factor."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..simple_train import sha
from ..simple_online import _durable_json
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from .learning import PROFILE_KINDS, reference_policy, prior_policy


def read(root, name):
    return json.loads((root/name).read_text())


def validate_pair(a, b):
    for key in a.keys() | b.keys():
        if key != 'source_memory' and a.get(key) != b.get(key):
            raise ValueError('changed paired protocol: '+key)
    if a['feedback'] != 'progressive' or a['source_deployment_feedback_match'] is not True:
        raise ValueError('matched feedback required')
    am, bm = a['source_memory'], b['source_memory']
    if ((am['kind'], bm['kind']) != PROFILE_KINDS.get(am['profile'])
            or any(am[k] != bm[k] for k in ('profile', 'epoch', 'source_protocol_sha256', 'audit_sha256'))
            or am['epoch'] != 2):
        raise ValueError('paired final source objective artifacts required')
    if (am['profile'] in ('reference_gradient_ablation_v2', 'zero_prior_ablation_v3')
            and any(m.get('reference_gradient') != reference_policy(m['kind']) for m in (am, bm))):
        raise ValueError('paired reference gradient artifacts required')
    if (am['profile'] == 'zero_prior_ablation_v3'
            and any(m.get('prior_policy') != prior_policy(m['kind']) for m in (am, bm))):
        raise ValueError('paired prior-policy artifacts required')


def compare(bag, direction):
    roots = [bag, direction]
    protocols = [read(r, 'protocol.json') for r in roots]
    validate_pair(*protocols)
    kinds = PROFILE_KINDS[protocols[0]['source_memory']['profile']]
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
        if read(bag, filename) != read(direction, filename):
            raise ValueError('changed paired inputs: '+filename)
    arms = protocols[0]['arms']
    rows = [{arm: [] for arm in arms} for _ in roots]
    for index, item in enumerate(read(bag, 'order.json'), 1):
        sid = item['shot']
        name = f'{index:04d}_{sid}.npz'
        flag, down = io23.read_outcome('EAST', sid)
        with np.load(bag/'shots'/name) as a, np.load(direction/'shots'/name) as b:
            for key in ('times', 'source', 'aligned'):
                assert np.array_equal(a[key], b[key]), (sid, key)
            for run, curves in enumerate((a, b)):
                for arm in arms:
                    rows[run][arm].append(dict(shot=sid, disruptive=flag,
                        **classify(curves['times'], curves[arm], flag, down, protocols[0]['threshold'])))
    for run in range(2):
        assert {k: counts(v) for k, v in rows[run].items()} == audits[run]['metrics']
    result = dict(scope='ONE_SEED_DEVELOPMENT_COMPLETE360', identical_source_aligned_shots=360,
        unchanged_inputs=list(same), threshold=protocols[0]['threshold'],
        runs={k: str(r.resolve()) for k, r in zip(kinds, roots, strict=True)},
        metrics={k: audit['metrics'] for k, audit in zip(kinds, audits, strict=True)},
        within_run={k: audit['flips'] for k, audit in zip(kinds, audits, strict=True)},
        attribution='Within-run intra versus cold isolates fast effect; cold weights can differ across source objectives.',
        labels='source local supervision uses JT max(25ms,AdvancedTime), strict>10ms; EAST scoring strict>30ms',
        independent_validation=False, physical_law_discovery=False)
    result[f'{kinds[1]}_minus_{kinds[0]}'] = {arm: flips(rows[0][arm], rows[1][arm]) for arm in arms}
    path = direction/'paired_direction_audit.json'
    if path.exists():
        raise FileExistsError(path)
    _durable_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag', type=Path)
    p.add_argument('direction', type=Path)
    args = p.parse_args()
    compare(args.bag, args.direction)
