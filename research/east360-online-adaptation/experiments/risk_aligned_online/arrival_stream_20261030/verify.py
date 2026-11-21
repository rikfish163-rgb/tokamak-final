"""Independently recount sealed alarms; optional exact neural trajectory replay."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit

from .. import io23
from ..conditional_likelihood_20261029.run import read
from ..simple_online import _durable_json
from ..simple_train import sha


def decide(sid, times, probability, disruptive, down, threshold):
    hit = np.flatnonzero(probability >= threshold)
    lead = float(round((down-times[hit[0]])*1000, 9)) if len(hit) and disruptive else None
    return dict(shot=sid, is_disrupt=bool(disruptive), alarm=bool(len(hit)),
                lead_ms=lead, timely=bool(lead is not None and lead > 30.))


def check_timing(rows, n):
    origins = list(range(127, n-25, 25))
    assert [r['origin'] for r in rows] == origins
    for r in rows:
        assert r['maturity'] == r['origin']+25
        assert r['first_affected'] == r['maturity']+1
        assert r['apply_count'] == min(25, n-r['first_affected'])


def summary(records, threshold):
    nd = sum(r['is_disrupt'] for r in records)
    ns = len(records)-nd
    tp = sum(r['timely'] for r in records)
    fp = sum(r['alarm'] and not r['is_disrupt'] for r in records)
    leads = [r['lead_ms'] for r in records if r['timely']]
    return dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns, tpr=tp/nd if nd else None,
                fpr=fp/ns if ns else None, J=tp/nd-fp/ns if nd and ns else None,
                threshold=threshold, median_lead_ms=float(np.median(leads)) if leads else None,
                early_over3000=sum(t > 3000 for t in leads))


def verify(output):
    done, protocol = read(output/'complete.json'), read(output/'protocol.json')
    assert done['complete'] and not done['independent_confirmation']
    order, threshold = protocol['order'], protocol['threshold']
    barrier = read(output/'all_predictions_sealed.json')
    assert barrier['count'] == len(order) == len(set(order)) == 360
    assert sha(output/'feedback.pt') == barrier['feedback_sha256']
    accesses = read(output/'accesses.json')
    assert len(accesses) == 380
    assert [r['shot'] for r in accesses[:20]] == order[:20]
    assert all(r['sealed_count'] == 20 for r in accesses[:20])
    assert [r['shot'] for r in accesses[20:]] == order
    assert all(r['sealed_count'] == 360 for r in accesses[20:])
    for name, digest in protocol['code_sha256'].items():
        assert sha(io23.ROOT/name) == digest, name
    reference, previous = Path(protocol['reference']), Path(protocol['previous'])
    for name, digest in protocol['reference_hashes'].items():
        assert sha(reference/name) == digest, name
    hashes = read(reference/'input_hashes.json')
    arms = [*protocol['arms'], 'archive']
    records = {a: [] for a in arms}
    count = {a: 0 for a in protocol['arms'] if a != 'cold'}
    applied = count.copy()
    maximum_norm_error = 0.
    changed_points = {a: 0 for a in arms}
    for i, sid in enumerate(order):
        name = f'{i+1:04d}_{sid}'
        path = output/'shots'/(name+'.npz')
        seal = read(output/'shots'/(name+'_seal.json'))
        assert sha(path) == seal['sha256'] == barrier['seals'][i]['sha256']
        assert barrier['seals'][i]['shot'] == sid and seal['index'] == i+1
        assert sha(io23.path_for('EAST', sid)) == hashes[str(sid)]
        flag, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as saved, \
                np.load(previous/'shots'/(name+'.npz'), allow_pickle=False) as old:
            times = saved['times']
            np.testing.assert_array_equal(times, old['times'])
            np.testing.assert_array_equal(saved['archive'], old['cold'])
            np.testing.assert_array_equal(saved['cold'], saved['identical'])
            valid = (down-times)*1000 >= -1e-6 if flag else np.ones(len(times), bool)
            for arm in arms:
                probability = saved[arm]
                assert probability.shape == times.shape and np.isfinite(probability).all()
                assert ((probability >= 0) & (probability <= 1)).all()
                if i < 20:
                    np.testing.assert_array_equal(probability, old['cold'])
                records[arm].append(decide(sid, times[valid], probability[valid], flag, down, threshold))
                changed_points[arm] += int(np.sum(probability != saved['cold']))
            if i < 20:
                assert seal['updates'] == []
            else:
                check_timing(seal['updates'], len(times))
                for r in seal['updates']:
                    assert 0 <= r['support_count'] <= 23 and r['gradnorm'] >= 0
                    assert r['state_norm']['cold'] == r['state_norm']['identical'] == 0
                    for arm in count:
                        expected = expit(r['origin_logit'])-expit(r['origin_logit']+r['e'][arm])
                        if not r['support_count']:
                            expected = 0.
                        assert abs(expected-r['feedback'][arm]) < 1e-12
                        norm = .05*abs(expected) if r['gradnorm'] else 0.
                        error = abs(norm-r['state_norm'][arm])
                        maximum_norm_error = max(maximum_norm_error, error)
                        assert error < 2e-8 and r['state_norm'][arm] <= .05000002
                        count[arm] += int(r['state_norm'][arm] > 0)
                        applied[arm] += int(r['state_norm'][arm] > 0 and r['apply_count'] > 0)
    assert count == done['update_counts'] and applied == done['applied_update_counts']
    assert records == read(output/'per_shot.json')
    for arm, r in records.items():
        for section, selected in (('full360', r), ('cold20', r[:20]), ('suffix340', r[20:]), ('high_proxy90', r[-90:])):
            assert summary(selected, threshold) == done['results'][arm][section]
        rescued, harmed = [], []
        for base, changed in zip(records['cold'], r, strict=True):
            old = base['timely'] if base['is_disrupt'] else not base['alarm']
            new = changed['timely'] if changed['is_disrupt'] else not changed['alarm']
            if new and not old:
                rescued.append(base['shot'])
            if old and not new:
                harmed.append(base['shot'])
        assert dict(rescued=rescued, harmed=harmed) == done['comparisons'][arm]
    result = dict(saved_array_audit=True, complete_sha256=sha(output/'complete.json'),
                  shots=360, arms=len(arms), exact_decision_records=360*len(arms),
                  exact_summary_groups=len(arms)*4, update_counts=count, applied_update_counts=applied,
                  max_state_norm_error=maximum_norm_error, changed_timepoints=changed_points,
                  all_inputs_and_parents_unchanged=True, neural_replay=False)
    _durable_json(output/'independent_audit.json', result)
    print(result, flush=True)


def replay(output, shots):
    import torch
    from .. import simple_data
    from ..attention_basis_20261029.learning import MatrixAttention
    from ..causal_input import apply_input_contract
    from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
    from ..warmstart_alignment_20261022.alignment import transform
    from .learning import predict

    protocol = read(output/'protocol.json')
    if not shots or len(set(shots)) != len(shots) or not set(shots) <= set(protocol['order'][20:]):
        raise ValueError('distinct suffix shot IDs required for neural replay')
    if not (output/'independent_audit.json').is_file():
        raise ValueError('saved-array audit must precede neural replay')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    reference = Path(protocol['reference'])
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    state = torch.load(output/'feedback.pt', map_location='cpu', weights_only=True)
    pairs, refs = state['pairs'], state['references']
    results = []
    for sid in shots:
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k: raw[k] for k in ('values', 'available')}, read(reference/'input_map.json'))
        curves, info = predict(net, signal, slow, pairs['real'], pairs['misplaced'], refs['real'], refs['misplaced'])
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
