"""Enumerate every parent error, then cache one declared diagnostic endpoint."""
import h5py
import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..delayed_recurrent_memory_20261025.learning import batch
from ..delayed_recurrent_memory_20261025.model import patch_signal
from ..east_context_operator_20261031.run import read
from ..lifecycle_stream_20261025.audit import classify
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform


def choose_endpoint(times, scores, disruptive, down):
    times, scores = np.asarray(times), np.asarray(scores)
    if (times.ndim != 1 or not len(times) or scores.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(scores).all()):
        raise ValueError('matching finite nonempty endpoint arrays required')
    if disruptive:
        if down is None or not np.isfinite(down):
            raise ValueError('known disruption time required for oracle selection')
        eligible = np.flatnonzero((np.arange(len(times)) >= 16)
                                 & (np.round((down-times)*1000, 9) > 30.))
    else:
        eligible = np.arange(len(times))
    return int(eligible[np.argmax(scores[eligible])]) if len(eligible) else None


def enumerate_errors(parent):
    order = [r['shot'] for r in read(parent/'order.json')]
    p = read(parent/'protocol.json')
    saved = read(parent/'per_shot.json')['cold']
    hashes = read(parent/'input_hashes.json')
    declared_d = set(read(io23.ROOT/'shot_list/TEST/Disruption_EAST_Test.json'))
    declared_s = set(read(io23.ROOT/'shot_list/TEST/Non-disruption_EAST_Test.json'))
    assert set(order) <= declared_d | declared_s and not declared_d & declared_s
    errors = []
    for i, sid in enumerate(order):
        path = io23.path_for('EAST', sid)
        assert sha(path) == hashes[str(sid)]
        with h5py.File(path, 'r') as h:
            ds = h['data/ip']
            times = float(ds.attrs['StartTime'])+np.arange(len(ds))/float(ds.attrs['SampleRate'])
            flag = bool(h['meta/IsDisrupt'][()])
            down = float(h['meta/DownTime'][()]) if flag else None
        curve_path = parent/'shots'/f'{i+1:04d}_{sid}.npz'
        assert sha(curve_path) == read(parent/'shots'/f'{i+1:04d}_{sid}_seal.json')['sha256']
        with np.load(curve_path, allow_pickle=False) as a:
            assert np.array_equal(a['times'], times)
            scores = a['cold']
            prediction = classify(times, scores, flag, down, p['threshold'])
            assert dict(shot=sid, is_disrupt=flag, **prediction) == saved[i]
            error = not prediction['timely'] if flag else prediction['alarm']
            if not error:
                continue
            row = dict(shot=sid, rank=i+1, disruptive=flag, kind='FN_or_TARDY' if flag else 'FP',
                       label_disagreement=flag != (sid in declared_d), raw_sha256=hashes[str(sid)],
                       prediction_sha256=sha(curve_path))
            if i < 20:
                row.update(status='startup_cost_not_retroactively_probed', endpoint=None)
            else:
                endpoint = choose_endpoint(times, scores, flag, down)
                hits = np.flatnonzero(scores >= p['threshold']) if not flag else []
                first = int(hits[0]) if len(hits) else None
                row.update(status='probe' if endpoint is not None else 'no_eligible_endpoint',
                           endpoint=endpoint, first_alarm_index=first,
                           prior_alarm_before_first_effect=bool(first is not None and first < 16))
                if endpoint is not None:
                    row.update(selected_time_seconds=float(times[endpoint]),
                               selected_lead_ms=float((down-times[endpoint])*1000) if flag else None,
                               cold_probability=float(scores[endpoint]), enabled=endpoint >= 16,
                               original_batch_start=endpoint//32*32,
                               original_batch_stop=min(endpoint//32*32+32, len(times)))
            errors.append(row)
    assert len(errors) == 88 and sum(r['rank'] <= 20 for r in errors) == 10
    cases = [r for r in errors if r['status'] == 'probe']
    assert len(cases) == 78 and sum(r['disruptive'] for r in cases) == 36
    return errors, cases


@torch.no_grad()
def prepare_cases(models, parent, cases, mapping):
    hs, vs = [], []
    operator_data = {name:dict(basis=[], actual_context=[], sealed_actual_probability=[])
                     for name in models}
    reference = models['source_operator']
    device = reference.base.center.device
    receipts = []
    for case in cases:
        raw = apply_input_contract(simple_data.read_shot('EAST', case['shot'], rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        ends = range(case['original_batch_start'], case['original_batch_stop'])
        x, mask = batch(signal, ends, device)
        h, valid = reference.tokens(x, mask)
        position = case['endpoint']-case['original_batch_start']
        baseline = reference.risk(h, valid, h.new_zeros(len(h), 92)).sigmoid()[position]
        assert float(baseline) == case['cold_probability'], ('baseline original batch', case['shot'])
        hs.append(h[position].clone())
        vs.append(valid[position].clone())
        means, complete = patch_signal(signal, reference.base.center, reference.base.scale)
        clean = torch.where(complete, means, 0.)
        index = case['endpoint']//8-1
        detail = dict(shot=case['shot'], original_batch_zero_probability_exact=True, operators={})
        for name, model in models.items():
            trace = model.memory.trace(means, complete, mode='memory')
            keys = torch.cat((model.memory.key(model.memory.key_features(clean, complete)).tanh(),
                              means.new_ones(len(means), 1)), -1)
            bases = torch.einsum('nk,khfc->nhfc', keys, model.memory.kernel).reshape(-1, 92, 8)
            b = bases[index] if index >= 0 else means.new_zeros(92, 8)
            c = trace['context'][index] if index >= 0 else means.new_zeros(8)
            delta = b @ c
            original_delta = trace['delta'][index] if index >= 0 else means.new_zeros(92)
            assert torch.allclose(delta, original_delta, atol=1e-6, rtol=1e-5)
            d = h.new_zeros(len(h), 92)
            d[position] = delta
            actual = model.risk(h, valid, d).sigmoid()[position]
            folder = parent if name == 'source_operator' else io23.ROOT/'probe_outputs/east_operator360_v1'
            curve = 'intra' if name == 'source_operator' else 'learned_rule'
            with np.load(folder/'shots'/f"{case['rank']:04d}_{case['shot']}.npz", allow_pickle=False) as a:
                expected = float(a[curve][case['endpoint']])
            error = abs(float(actual)-expected)
            if error > 2e-6:
                raise AssertionError(('actual context reconstruction', case['shot'], name, error))
            operator_data[name]['basis'].append(b.clone())
            operator_data[name]['actual_context'].append(c.clone())
            operator_data[name]['sealed_actual_probability'].append(expected)
            detail['operators'][name] = dict(actual_probability_error=error,
                                             actual_context_norm=float(c.norm()))
        receipts.append(detail)
    for values in operator_data.values():
        for key in ('basis', 'actual_context'):
            values[key] = torch.stack(values[key])
    return dict(h=torch.stack(hs), valid=torch.stack(vs),
                labels=torch.tensor([r['disruptive'] for r in cases], device=device, dtype=torch.float32),
                enabled=torch.tensor([r['enabled'] for r in cases], device=device, dtype=torch.bool),
                operators=operator_data), receipts
