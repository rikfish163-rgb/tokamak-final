"""Independent sealed-array recount and optional exact Q/K trajectory replay."""

import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..arrival_stream_20261030.verify import check_timing, decide, summary
from ..conditional_likelihood_20261029.run import read
from ..simple_online import _durable_json
from ..simple_train import sha
from .stream import NAMES


def check_receipt(r):
    """Check norm/direction contracts without calling the update implementation."""
    assert isinstance(r['observed'], int) and 0 <= r['observed'] <= 550
    for field in ('signal_gradient_norm', 'reverse_gradient_norm', 'risk_gradient_norm'):
        assert np.isfinite(r[field]) and r[field] >= 0
    norms = r['state_norm']
    assert set(norms) == set(NAMES)
    assert all(np.isfinite(v) and 0 <= v <= .05000002 for v in norms.values())
    assert norms['cold'] == norms['identical'] == 0
    assert abs(r['max_bound'] - max(norms.values())) < 1e-10
    for field in ('cosine_signal', 'cosine_risk', 'cosine_operational'):
        assert set(r[field]) == set(NAMES)
        assert all(np.isfinite(v) and abs(v) <= 1.000002 for v in r[field].values())
    if not r['observed'] or not r['signal_gradient_norm']:
        assert all(v == 0 for v in norms.values())
    assert abs(norms['signal_axis'] - norms['operational']) < 2e-8
    if norms['operational']:
        assert abs(r['cosine_signal']['signal_axis'] + 1.) < 2e-6
        assert abs(r['cosine_operational']['operational'] - 1.) < 2e-6
    if norms['risk_axis']:
        assert abs(norms['risk_axis'] - norms['operational']) < 2e-8
        assert abs(abs(r['cosine_risk']['risk_axis']) - 1.) < 2e-6
        assert r['cosine_risk']['risk_axis'] * r['cosine_risk']['operational'] >= 0
    else:
        assert (not norms['operational'] or not r['risk_gradient_norm']
                or abs(r['cosine_risk']['operational']) < 1e-6)


def verify(output):
    done, protocol = read(output/'complete.json'), read(output/'protocol.json')
    assert done['complete'] and not done['independent_confirmation']
    assert done['source_parameter_updates'] == 0 and done['source_unchanged']
    assert tuple(protocol['arms']) == NAMES and protocol['candidate'] == 'operational'
    order, threshold = protocol['order'], protocol['threshold']
    barrier = read(output/'all_predictions_sealed.json')
    assert barrier['count'] == len(order) == len(set(order)) == 360
    accesses = read(output/'accesses.json')
    assert len(accesses) == 360 and [r['shot'] for r in accesses] == order
    assert all(r['sealed_count'] == 360 for r in accesses)
    for name, digest in protocol['code_sha256'].items():
        assert sha(io23.ROOT/name) == digest, name
    reference, previous, policy = (Path(protocol[k]) for k in ('reference', 'previous', 'policy'))
    for name, digest in protocol['reference_hashes'].items():
        assert sha(reference/name) == digest, name
    for name, digest in protocol['policy_hashes'].items():
        assert sha(policy/name) == digest, name
    assert barrier['policy_sha256'] == protocol['policy_hashes']['policy.pt']
    assert sha(previous/'complete.json') == protocol['previous_complete_sha256']
    assert sha(previous/'all_predictions_sealed.json') == protocol['previous_barrier_sha256']
    hashes = read(reference/'input_hashes.json')
    arms = (*NAMES, 'archive')
    records, count, applied, points = {a: [] for a in arms}, {}, {}, {}
    for collection in (count, applied, points):
        collection.update({a: 0 for a in NAMES})
    changed = {a: 0 for a in arms}
    max_bound, receipt_count = 0., 0
    for i, sid in enumerate(order):
        name = f'{i+1:04d}_{sid}'
        path = output/'shots'/(name+'.npz')
        seal = read(output/'shots'/(name+'_seal.json'))
        assert sha(path) == seal['sha256'] == barrier['seals'][i]['sha256']
        assert barrier['seals'][i]['shot'] == seal['shot'] == sid and seal['index'] == i+1
        assert not seal['outcomes_used_in_prediction']
        assert seal['startup_archived'] == (i < 20)
        assert sha(io23.path_for('EAST', sid)) == hashes[str(sid)]
        flag, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as saved, \
                np.load(previous/'shots'/(name+'.npz'), allow_pickle=False) as old:
            times = saved['times']
            np.testing.assert_array_equal(times, old['times'])
            for arm in ('archive', 'cold', 'identical'):
                np.testing.assert_array_equal(saved[arm], old['cold'])
            assert seal['archive_max_abs_error'] == seal['archive_crossing_disagreements'] == 0
            valid = (down-times)*1000 >= -1e-6 if flag else np.ones(len(times), bool)
            for arm in arms:
                probability = saved[arm]
                assert probability.shape == times.shape and np.isfinite(probability).all()
                assert ((probability >= 0) & (probability <= 1)).all()
                if i < 20:
                    np.testing.assert_array_equal(probability, old['cold'])
                else:
                    np.testing.assert_array_equal(probability[:153], old['cold'][:153])
                records[arm].append(decide(sid, times[valid], probability[valid], flag, down, threshold))
                changed[arm] += int(np.sum(probability != saved['cold']))
            if i < 20:
                assert seal['updates'] == []
            else:
                check_timing(seal['updates'], len(times))
                for r in seal['updates']:
                    check_receipt(r)
                    receipt_count += 1
                    max_bound = max(max_bound, r['max_bound'])
                    for arm in NAMES:
                        nonzero = r['state_norm'][arm] > 0
                        count[arm] += int(nonzero)
                        applied[arm] += int(nonzero and r['apply_count'] > 0)
                        points[arm] += r['apply_count'] if nonzero else 0
    assert count == done['update_counts'] and applied == done['applied_update_counts']
    assert points == done['applied_point_counts'] and max_bound == done['max_state_norm']
    assert records == read(output/'per_shot.json')
    for arm, r in records.items():
        for section, selected in (('full360', r), ('cold20', r[:20]), ('suffix340', r[20:]), ('high_proxy90', r[-90:])):
            assert summary(selected, threshold) == done['results'][arm][section]
        rescued, harmed = [], []
        for base, changed_row in zip(records['cold'], r, strict=True):
            old = base['timely'] if base['is_disrupt'] else not base['alarm']
            new = changed_row['timely'] if changed_row['is_disrupt'] else not changed_row['alarm']
            if new and not old:
                rescued.append(base['shot'])
            if old and not new:
                harmed.append(base['shot'])
        assert dict(rescued=rescued, harmed=harmed) == done['comparisons'][arm]
    result = dict(saved_array_audit=True, complete_sha256=sha(output/'complete.json'),
                  shots=360, arms=len(arms), exact_decision_records=360*len(arms),
                  exact_summary_groups=len(arms)*4, update_receipts=receipt_count,
                  update_counts=count, applied_update_counts=applied, max_state_norm=max_bound,
                  changed_timepoints=changed, all_inputs_and_parents_unchanged=True,
                  neural_replay=False)
    _durable_json(output/'independent_audit.json', result)
    print(result, flush=True)


def replay(output, shots):
    import torch
    from .. import simple_data
    from ..attention_basis_20261029.learning import MatrixAttention
    from ..causal_input import apply_input_contract
    from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
    from ..warmstart_alignment_20261022.alignment import transform
    from .stream import predict

    protocol = read(output/'protocol.json')
    if not shots or len(set(shots)) != len(shots) or not set(shots) <= set(protocol['order'][20:]):
        raise ValueError('distinct suffix shot IDs required for neural replay')
    audit = read(output/'independent_audit.json')
    assert audit['complete_sha256'] == sha(output/'complete.json')
    for name, digest in protocol['code_sha256'].items():
        assert sha(io23.ROOT/name) == digest, name
    policy = Path(protocol['policy'])
    assert sha(policy/'policy.pt') == protocol['policy_hashes']['policy.pt']
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    reference = Path(protocol['reference'])
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    payload = torch.load(policy/'policy.pt', map_location='cuda', weights_only=True)
    results = []
    for sid in shots:
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k: raw[k] for k in ('values', 'available')}, read(reference/'input_map.json'))
        curves, info = predict(net, signal, slow, payload['rules'])
        index = protocol['order'].index(sid)+1
        name = f'{index:04d}_{sid}'
        with np.load(output/'shots'/(name+'.npz'), allow_pickle=False) as archive:
            for arm, curve in curves.items():
                np.testing.assert_array_equal(curve, archive[arm])
        assert info == read(output/'shots'/(name+'_seal.json'))['updates']
        results.append(dict(shot=sid, exact_risk_arrays=True, exact_update_receipts=True))
    _durable_json(output/'neural_replay.json', results)
    print(results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--replay', type=int, nargs='+')
    args = parser.parse_args()
    if args.replay:
        replay(args.output.absolute(), args.replay)
    else:
        verify(args.output.absolute())
