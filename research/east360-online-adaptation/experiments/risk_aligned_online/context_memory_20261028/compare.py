"""Compare completed matrix-zero and context8 streams, not partial results."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..simple_online import _durable_json
from ..simple_train import sha
from .checkpoint import read


def validate_pair(a, b):
    # Architecture/checkpoint and their archived implementation must differ;
    # all data access, alarm, ordering, seed and post-shot contracts stay fixed.
    for key in a.keys() | b.keys():
        if key not in ('source_memory', 'code_sha256') and a.get(key) != b.get(key):
            raise ValueError('changed stream contract: ' + key)
    am, bm = a['source_memory'], b['source_memory']
    if (am.get('profile') != 'zero_prior_ablation_v3' or am.get('kind') != 'zero_prior'
            or am.get('prior_policy') != 'fixed_zero' or bm.get('profile') != 'context8_v1'
            or bm.get('kind') != 'context8' or bm.get('context_dim') != 8
            or b['feedback'] != 'progressive' or b['readout'] != 'full'
            or b['source_deployment_feedback_match'] is not True):
        raise ValueError('fixed zero-matrix versus context8 comparison required')


def compare(reference, candidate):
    roots = (reference, candidate)
    protocols = [read(r, 'protocol.json') for r in roots]
    validate_pair(*protocols)
    audits = [read(r, 'independent_audit.json') for r in roots]
    if not all(a['complete360'] for a in audits):
        raise ValueError('complete independently rescored streams required')
    for p in protocols:
        receipt = p['source_memory']; path = Path(receipt['path'])
        assert sha(path) == receipt['sha256']
        assert sha(path.parent / 'protocol.json') == receipt['source_protocol_sha256']
        assert sha(path.parent / 'audit.json') == receipt['audit_sha256']
    same = ('order.json', 'input_map.json', 'source_replay.json', 'input_hashes.json')
    for name in same:
        assert read(reference, name) == read(candidate, name), name
    arms = protocols[0]['arms']; rows = [{k: [] for k in arms} for _ in roots]
    for i, item in enumerate(read(reference, 'order.json'), 1):
        sid = item['shot']; flag, down = io23.read_outcome('EAST', sid)
        name = f'{i:04d}_{sid}.npz'
        with np.load(reference / 'shots' / name) as a, np.load(candidate / 'shots' / name) as b:
            for key in ('times', 'source', 'aligned'):
                assert np.array_equal(a[key], b[key]), (sid, key)
            for n, data in enumerate((a, b)):
                for arm in arms:
                    rows[n][arm].append(dict(shot=sid, disruptive=flag,
                        **classify(data['times'], data[arm], flag, down, protocols[n]['threshold'])))
    for i in range(2):
        assert {k: counts(v) for k, v in rows[i].items()} == audits[i]['metrics']
    result = dict(scope='ONE_SEED_DEVELOPMENT_ARCHITECTURE_COMPARISON_COMPLETE360',
                  identical_source_aligned_shots=360, unchanged_inputs=list(same),
                  metrics={k: a['metrics'] for k, a in zip(('matrix_zero', 'context8'), audits, strict=True)},
                  within_run={k: a['flips'] for k, a in zip(('matrix_zero', 'context8'), audits, strict=True)},
                  context_minus_matrix={arm: flips(rows[0][arm], rows[1][arm]) for arm in arms},
                  attribution='Each intra/cold pair isolates its fast effect; architecture changes source teaching and cold weights.',
                  independent_validation=False, physical_law_discovery=False)
    path = candidate / 'paired_context_audit.json'
    if path.exists():
        raise FileExistsError(path)
    _durable_json(path, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path); p.add_argument('candidate', type=Path)
    args = p.parse_args(); compare(args.reference, args.candidate)
