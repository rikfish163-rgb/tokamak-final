import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.arrival_stream_20261030 import learning as arrival
from experiments.risk_aligned_online.operational_transport_20261030 import run, stream


D = 8192


def signal(n=210):
    values = np.zeros((n, 23), dtype=np.float32)
    values[:, 0] = np.arange(n, dtype=np.float32) / 200
    values[:, 2] = np.arange(n, dtype=np.float32)
    return {"values": values, "available": np.ones_like(values, dtype=bool)}


def rules():
    keys = torch.zeros((2, D), dtype=torch.float64)
    keys[0, 0] = keys[1, 1] = 1
    coefficients = keys.clone()
    mean = torch.zeros(D, dtype=torch.float64)
    mean[2] = .4
    op = dict(keys=keys, coefficients=coefficients, center=torch.zeros(D, dtype=torch.float64),
              mean=mean, ridge=.1)
    point = copy.deepcopy(op)
    point["coefficients"] = -coefficients
    shuffled = copy.deepcopy(op)
    shuffled["coefficients"] = coefficients.flip(0)
    return dict(operational=op, point=point, shuffled=shuffled)


def gradients(g=(1., 0.), reverse=(0., 1.), risk=(1., 1.)):
    result = []
    for values in (g, reverse, risk):
        v = torch.zeros((1, D))
        v[0, :2] = torch.tensor(values)
        result.append(v)
    return result


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.register_buffer("center", torch.zeros(23))
        self.weight = nn.Parameter(torch.tensor(1.))
        self.calls = []

    def forward(self, x, available, state, slow, mode, *, forecast=False):
        assert mode == "qk" and not forecast
        self.calls.append((x[:, -1, 2].detach().clone(), state.detach().clone()))
        return {"logit": self.weight * (x[:, -1, 0] + state[:, 0] + state[:, 1])}


@pytest.fixture
def observed(monkeypatch):
    calls = []

    def fake(net, prefix, origin, slow):
        assert set(prefix) == {"values", "available"}
        assert len(prefix["values"]) == origin + 26
        calls.append(origin)
        g, rg, risk = gradients(g=(1., float(prefix["values"][-1, 0])),
                                reverse=(0., 1.), risk=(1., 1.))
        count = int(prefix["available"][origin + 1:origin + 26, 0].sum())
        return g, rg, risk, dict(observed=count, signal_loss=1., reverse_loss=2.,
                                 signal_norm=float(g.norm()), risk_norm=float(risk.norm()),
                                 origin_logit=0.)

    monkeypatch.setattr(stream, "observed_gradients", fake)
    return calls


def state_at(net, arm, point):
    for ends, states in net.calls:
        ends = ends.reshape(len(stream.NAMES), 32)
        states = states.reshape(len(stream.NAMES), 32, D)
        matches = torch.nonzero(ends[stream.NAMES.index(arm)] == point).flatten()
        if len(matches):
            return states[stream.NAMES.index(arm), matches[0]]
    raise AssertionError("point not found")


def test_mapping_controls_match_declared_directions_and_norms():
    g, rg, risk = gradients()
    states, receipt = stream.make_states(g, rg, risk, 25, rules())
    by_name = dict(zip(stream.NAMES, states))
    assert stream.NAMES == ("cold", "operational", "point", "reverse", "shuffled",
                            "constant", "risk_axis", "signal_axis", "identical")
    assert torch.count_nonzero(by_name["cold"]) == torch.count_nonzero(by_name["identical"]) == 0
    assert not torch.equal(by_name["operational"], by_name["point"])
    assert not torch.equal(by_name["operational"], by_name["reverse"])
    assert not torch.equal(by_name["operational"], by_name["shuffled"])
    expected_constant = torch.zeros(D)
    expected_constant[2] = -.02
    torch.testing.assert_close(by_name["constant"], expected_constant)
    other_g, _, _ = gradients(g=(0., 2.))
    other, _ = stream.make_states(other_g, rg, risk, 25, rules())
    torch.testing.assert_close(other[stream.NAMES.index('constant')], by_name['constant'])
    opnorm = by_name["operational"].norm()
    assert opnorm <= .050001
    torch.testing.assert_close(by_name["risk_axis"].norm(), opnorm)
    torch.testing.assert_close(by_name["signal_axis"].norm(), opnorm)
    assert torch.dot(by_name["risk_axis"], risk[0]) < 0
    torch.testing.assert_close(by_name["signal_axis"], -opnorm * g[0] / g.norm())
    assert receipt["state_norm"]["operational"] == pytest.approx(float(opnorm))
    assert receipt["max_bound"] <= .050001


def test_axis_controls_preserve_non_saturated_operational_norm():
    policy = rules()
    policy['operational']['coefficients'] *= .01
    policy['operational']['mean'].zero_()
    g, rg, risk = gradients(risk=(-1., 0.))
    states, _ = stream.make_states(g, rg, risk, 25, policy)
    op, axis, signal_axis = (states[stream.NAMES.index(name)] for name in
                             ('operational', 'risk_axis', 'signal_axis'))
    assert 0 < op.norm() < .001
    torch.testing.assert_close(axis.norm(), op.norm())
    torch.testing.assert_close(signal_axis.norm(), op.norm())
    assert torch.dot(axis, risk[0]) > 0


@pytest.mark.parametrize("zero_gradient,observed", [(True, 25), (False, 0)])
def test_no_signal_or_no_observations_zero_every_arm(zero_gradient, observed):
    g, rg, risk = gradients(g=(0., 0.) if zero_gradient else (1., 0.))
    states, _ = stream.make_states(g, rg, risk, observed, rules())
    assert torch.count_nonzero(states) == 0


def test_degenerate_risk_axis_is_zero_and_signal_axis_stays_matched():
    g, rg, risk = gradients(risk=(0., 0.))
    states, _ = stream.make_states(g, rg, risk, 25, rules())
    assert states[stream.NAMES.index("risk_axis")].norm() == 0
    torch.testing.assert_close(states[stream.NAMES.index("signal_axis")].norm(),
                               states[stream.NAMES.index("operational")].norm())


def test_cadence_maturity_and_replacement_with_per_shot_reset(observed):
    net = ToyNet()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    curves, info = stream.predict(net, signal(), None, rules())
    assert set(curves) == set(stream.NAMES)
    assert [(r["origin"], r["maturity"], r["first_affected"], r["apply_count"]) for r in info] == [
        (127, 152, 153, 25), (152, 177, 178, 25), (177, 202, 203, 7)]
    assert observed == [127, 152, 177]
    np.testing.assert_array_equal(curves["cold"], curves["identical"])
    assert state_at(net, "operational", 152).norm() == 0
    first = state_at(net, "operational", 153)
    expected_probability = torch.sigmoid(torch.tensor(signal()["values"][153, 0]) + first[:2].sum())
    assert curves['operational'][153] == pytest.approx(float(expected_probability), abs=1e-7)
    torch.testing.assert_close(state_at(net, "operational", 177), first)
    second = state_at(net, "operational", 178)
    assert not torch.equal(first, second)
    assert second.norm() <= .050001  # replacement, not accumulated first + second
    net.calls.clear()
    second_curves, second_info = stream.predict(net, signal(160), None, rules())
    fresh_curves, fresh_info = stream.predict(ToyNet(), signal(160), None, rules())
    for name in stream.NAMES:
        np.testing.assert_array_equal(second_curves[name], fresh_curves[name])
    assert second_info == fresh_info
    assert state_at(net, "operational", 127).norm() == 0
    assert all(p.grad is None for p in net.parameters())
    assert all(torch.equal(v, before[k]) for k, v in net.state_dict().items())


@pytest.mark.parametrize("length,applied", [(1, []), (128, []), (152, []), (153, [0]),
                                           (154, [1]), (177, [24]), (178, [25, 0])])
def test_incomplete_horizons_and_tail_counts(observed, length, applied):
    curves, info = stream.predict(ToyNet(), signal(length), None, rules())
    assert all(len(v) == length for v in curves.values())
    assert [r["apply_count"] for r in info] == applied


def test_future_perturbation_and_short_prefix_do_not_rewrite_history(observed):
    original = signal(220)
    altered = copy.deepcopy(original)
    altered["values"][165:, 0] += 20
    altered["available"][165:, 0] = False
    full, receipts = stream.predict(ToyNet(), original, None, rules())
    changed, changed_receipts = stream.predict(ToyNet(), altered, None, rules())
    short, _ = stream.predict(ToyNet(), {k: v[:165] for k, v in original.items()}, None, rules())
    for arm in stream.NAMES:
        np.testing.assert_array_equal(full[arm][:165], changed[arm][:165])
        np.testing.assert_array_equal(full[arm][:165], short[arm])
    assert receipts[0] == changed_receipts[0]


def test_zero_observation_replaces_old_state_after_maturity(observed):
    s = signal(190)
    s["available"][153:178, 0] = False
    net = ToyNet()
    curves, info = stream.predict(net, s, None, rules())
    assert info[0]["observed"] == 25 and info[1]["observed"] == 0
    assert state_at(net, "operational", 177).norm() > 0
    assert state_at(net, "operational", 178).norm() == 0
    np.testing.assert_array_equal(curves["operational"][178:], curves["cold"][178:])


@pytest.mark.parametrize("bad", [
    lambda s: dict(s, outcome=True),
    lambda s: {**s, "available": s["available"].astype(int)},
    lambda s: {**s, "values": s["values"][:, :22]},
    lambda s: {k: v[:0] for k, v in s.items()},
])
def test_outcomes_and_invalid_signal_are_rejected(observed, bad):
    with pytest.raises(ValueError):
        stream.predict(ToyNet(), bad(signal(160)), None, rules())


def test_fixed_nine_arm_batch_matches_arrival_forward_on_cpu():
    s = signal(41)
    states = torch.zeros((9, D))
    for lo, hi in ((0, 32), (32, 41)):
        actual = stream._forward(ToyNet(), s, lo, hi, states, None)
        expected = arrival._forward(ToyNet(), s, lo, hi, states, None)
        np.testing.assert_array_equal(actual, expected)


def test_real_matrix_attention_recomputes_zero_state_and_matches_arrival_cold(monkeypatch):
    from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(41)
        net = MatrixAttention(Simple23(Config(), np.zeros(23), np.ones(23)))
        before = {name: value.clone() for name, value in net.state_dict().items()}
        forecast_states = []

        def observe_state(module, args, kwargs):
            if kwargs.get('forecast'):
                forecast_states.append(args[2].detach().clone())

        hook = net.register_forward_pre_hook(observe_state, with_kwargs=True)
        s, slow = signal(180), torch.zeros((64, 64))
        actual, receipts = stream.predict(net, s, slow, rules())
        hook.remove()
        assert len(forecast_states) == 2
        assert all(torch.count_nonzero(value) == 0 for value in forecast_states)
        assert [r['origin'] for r in receipts] == [127, 152]
        assert all(r['observed'] > 0 and r['signal_gradient_norm'] > 0 for r in receipts)
        assert all(r['max_bound'] <= .050001 for r in receipts)
        assert all(torch.equal(value, before[name]) for name, value in net.state_dict().items())
        assert all(p.grad is None for p in net.parameters())

        from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.learning import batch
        prefix = {k: v[:153] for k, v in s.items()}
        direct_g, reverse_g, risk_g, info = stream.observed_gradients(net, prefix, 127, slow)
        x, available = batch(prefix, [127], net.base.center.device)
        truth = torch.as_tensor(prefix['values'][128:153], dtype=x.dtype)[None]
        mask = torch.as_tensor(prefix['available'][128:153])[None].clone()
        mask &= torch.isfinite(truth) & available[:, -1:]
        mask[:, :, 10] = False
        zero = x.new_zeros((1, D), requires_grad=True)
        raw = net(x, available, zero, slow, 'qk', forecast=True)
        direct_loss = net.base.forecast_loss(raw['forecast'], truth, mask)
        reverse_loss = net.base.forecast_loss(raw['forecast'], truth.flip(1), mask.flip(1))
        expected_g, = torch.autograd.grad(direct_loss, zero, retain_graph=True)
        expected_reverse, = torch.autograd.grad(reverse_loss, zero, retain_graph=True)
        expected_risk, = torch.autograd.grad(raw['logit'].sum(), zero)
        torch.testing.assert_close(direct_g, expected_g, rtol=0, atol=0)
        torch.testing.assert_close(reverse_g, expected_reverse, rtol=0, atol=0)
        torch.testing.assert_close(risk_g, expected_risk, rtol=0, atol=0)
        assert not torch.equal(direct_g, reverse_g)
        assert info['observed'] == int(mask.sum()) == 25 * 22

        def anchor(net, signal, origin, slow):
            return {'origin': origin}

        def mature(net, signal, anchor, slow, *fits):
            return torch.zeros((9, D)), {'apply_count': 0}

        monkeypatch.setattr(arrival, '_anchor', anchor)
        monkeypatch.setattr(arrival, '_mature', mature)
        old, _ = arrival.predict(net, s, slow, None, None, None, None)
        np.testing.assert_array_equal(actual['cold'], old['cold'])
        np.testing.assert_array_equal(actual['cold'], actual['identical'])

        prefix, _ = stream.predict(net, {k: v[:165] for k, v in s.items()}, slow, rules())
        for name in stream.NAMES:
            np.testing.assert_array_equal(actual[name][:165], prefix[name])
    finally:
        torch.set_num_threads(threads)


def test_outcome_gate_blocks_suffix_until_every_curve_is_sealed():
    seen = []
    gate = run.OutcomeGate(list(range(360)), lambda d, s: seen.append(s) or (False, None))
    for sid in range(21):
        gate.seal(sid, 'a' * 64)
    with pytest.raises(AssertionError):
        gate.read_outcome('EAST', 20)
    assert seen == []
    for sid in range(21, 360):
        gate.seal(sid, 'b' * 64)
    assert gate.read_outcome('EAST', 20) == (False, None)
    assert gate.accesses[-1]['sealed_count'] == 360


def test_evaluation_reuses_fixed_threshold_and_strict_30ms(monkeypatch):
    seen = []
    monkeypatch.setattr(run.fixed_test_protocol, 'validate_test_ids',
                        lambda device, ids: seen.append((device, len(ids))) or
                        dict(expected_disrupt=180, expected_safe=180, lead_ms=30.))
    base = [dict(shot=i, times=np.array([.970]), scores=np.array([.8]),
                 is_disrupt=i < 180, down=1. if i < 180 else None) for i in range(360)]
    base[0]['times'] = np.array([.969])  # 31ms succeeds; exactly 30ms does not.
    rows = {name: copy.deepcopy(base) for name in (*stream.NAMES, 'archive')}
    results, predictions, comparison = run.evaluate(rows, .7)
    assert len(seen) == len(rows) and all(x == ('EAST', 360) for x in seen)
    assert results['operational']['full360']['tp'] == 1
    assert results['operational']['full360']['fp'] == 180
    assert results['operational']['full360']['threshold'] == .7
    assert predictions['cold'][0]['timely'] and not predictions['cold'][1]['timely']
    assert comparison['operational'] == dict(rescued=[], harmed=[])


def test_default_policy_is_v2_and_metadata_rejects_mismatch():
    assert run.POLICY.name == 'operational_transport_cold20_v2'
    ids = list(range(360))
    previous = dict(reference='reference', reference_hashes={'models/cold.pt': 'a' * 64},
                    threshold=.7, order=ids)
    protocol = dict(scope='ENDED_COLD20_MATCHED_QK_GRADIENT_TEACHING', warmup_ids=ids[:20],
                    reference='reference', reference_hashes=previous['reference_hashes'],
                    previous=str(run.PREVIOUS), previous_sha256=run.PREVIOUS_SHA,
                    source_sha256=run.PARENT_SHA256, threshold=.7, ridge=.1, radius=.05,
                    source_updates=0, suffix_labels_accessed=False, plan_sha256='b' * 64)
    complete = dict(complete=True, source_unchanged=True, shots=20, examples=2,
                    disruptive_examples=1, safe_examples=1, policy_sha256='c' * 64,
                    full_stream_run=False)
    payload = dict(protocol=copy.deepcopy(protocol), rules=rules(), pairs_sha256='d' * 64)
    run.validate_policy_metadata(complete, protocol, payload, previous, 'b' * 64)
    for mutation in ('warmup', 'embedded', 'threshold', 'complete', 'arms', 'tensor_shape', 'nan'):
        c, p, x = copy.deepcopy((complete, protocol, payload))
        if mutation == 'warmup':
            p['warmup_ids'][0] = -1
        elif mutation == 'embedded':
            x['protocol']['radius'] = 1.
        elif mutation == 'threshold':
            p['threshold'] = .8
        elif mutation == 'complete':
            c['complete'] = False
        elif mutation == 'arms':
            del x['rules']['shuffled']
        elif mutation == 'tensor_shape':
            x['rules']['point']['keys'] = x['rules']['point']['keys'][:1]
        else:
            x['rules']['point']['mean'][0] = float('nan')
        with pytest.raises(ValueError):
            run.validate_policy_metadata(c, p, x, previous, 'b' * 64)


def test_archive_cold_must_match_exactly_even_without_alarm_change():
    cold = np.array([.1, .8], dtype=np.float32)
    assert run.check_cold_archive(cold, cold.copy(), .7) == (0., 0)
    altered = cold.copy()
    altered[0] = np.nextafter(altered[0], np.float32(1.))
    with pytest.raises(ValueError, match='not exactly archived'):
        run.check_cold_archive(cold, altered, .7)


def test_run_snapshot_includes_gradient_regression_tests():
    assert run.io23.ROOT/'tests/test_operational_transport_gradients.py' in run.code_files()


def test_cpu_full_runner_seals_before_outcomes_and_reuses_cold20(tmp_path, monkeypatch):
    root = tmp_path
    outputs = root / 'probe_outputs'
    old = outputs / 'arrival_reference_east360_v1'
    reference = root / 'reference'
    (old / 'shots').mkdir(parents=True)
    reference.mkdir()
    plan = root / 'plan.md'
    plan.write_text('fixed plan')
    source = root / 'parent.pt'
    source.write_text('fixed source')
    policy = outputs / 'fixture_policy'
    policy.mkdir()
    (policy / 'policy.pt').write_text('fixed policy')
    raw_path = root / 'raw.bin'
    raw_path.write_text('fixed raw')
    ids = list(range(1, 361))
    times = np.array([0., .7, .97])
    cold = np.array([.1, .8, .9], dtype=np.float32)

    def write_json(path, value):
        path.write_text(json.dumps(value))

    write_json(reference / 'order.json', [{'shot': sid} for sid in ids])
    write_json(reference / 'input_map.json', {'calibration_ids': ids[:20]})
    write_json(reference / 'input_hashes.json', {str(sid): run.sha(raw_path) for sid in ids})
    ref_hashes = {name: run.sha(reference / name) for name in
                  ('order.json', 'input_map.json', 'input_hashes.json')}
    write_json(old / 'protocol.json', dict(order=ids, reference='reference',
                                            reference_hashes=ref_hashes, threshold=.7))
    write_json(old / 'complete.json', {'complete': True})
    seals = []
    for index, sid in enumerate(ids, 1):
        name = f'{index:04d}_{sid}'
        path = old / 'shots' / (name + '.npz')
        np.savez_compressed(path, times=times, cold=cold)
        digest = run.sha(path)
        write_json(old / 'shots' / (name + '_seal.json'), {'sha256': digest})
        seals.append(dict(shot=sid, sha256=digest))
    write_json(old / 'all_predictions_sealed.json', dict(count=360, seals=seals))

    class FakeNet:
        def cuda(self):
            return self

        def state_dict(self):
            return {'frozen': torch.ones(1)}

    monkeypatch.setattr(run.io23, 'ROOT', root)
    monkeypatch.setattr(run, 'PREVIOUS', old)
    monkeypatch.setattr(run, 'PREVIOUS_SHA', run.sha(old / 'complete.json'))
    monkeypatch.setattr(run, 'PARENT_SHA256', run.sha(source))
    monkeypatch.setattr(run, 'DEFAULT_PARENT', source)
    monkeypatch.setattr(run, 'PLAN', plan)
    monkeypatch.setattr(run, 'code_files', lambda: {plan})
    monkeypatch.setattr(run, 'load_policy', lambda path, previous:
                        ({}, {'policy.pt': run.sha(policy / 'policy.pt')}))
    monkeypatch.setattr(run, 'load_parent', lambda path: (FakeNet(), {'threshold': .7}))
    monkeypatch.setattr(run, 'MatrixAttention', lambda base: base)
    monkeypatch.setattr(run.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(run.torch.cuda, 'get_device_name', lambda: 'CPU fixture only')
    monkeypatch.setattr(run.torch, 'load', lambda *a, **k: {'slow': torch.zeros((64, 64))})
    monkeypatch.setattr(run.torch, 'set_num_threads', lambda value: None)
    monkeypatch.setattr(run.io23, 'path_for', lambda device, sid: raw_path)
    monkeypatch.setattr(run, 'apply_input_contract', lambda raw: raw)
    monkeypatch.setattr(run, 'transform', lambda signal, mapping: signal)
    monkeypatch.setattr(run.simple_data, 'read_shot', lambda *a, **k: dict(
        times=times, values=np.zeros((len(times), 23), dtype=np.float32),
        available=np.ones((len(times), 23), dtype=bool)))
    monkeypatch.setattr(run.fixed_test_protocol, 'validate_test_ids', lambda device, values:
                        dict(expected_disrupt=180, expected_safe=180, lead_ms=30.))
    output = outputs / 'result'
    reads, predictions = [], []

    def outcome(device, sid):
        barrier = json.loads((output / 'all_predictions_sealed.json').read_text())
        assert barrier['count'] == len(barrier['seals']) == 360
        reads.append(sid)
        return sid <= 180, 1. if sid <= 180 else None

    def predict(net, signal, slow, rules):
        assert set(signal) == {'values', 'available'}
        with pytest.raises(AssertionError):
            run.io23.read_outcome('EAST', 21)
        predictions.append(True)
        return {name: cold.copy() for name in stream.NAMES}, []

    monkeypatch.setattr(run.io23, 'read_outcome', outcome)
    monkeypatch.setattr(run, 'predict', predict)
    run.run(output, policy)
    assert len(predictions) == 340 and reads == ids
    for index, sid in enumerate(ids, 1):
        name = f'{index:04d}_{sid}'
        with np.load(output / 'shots' / (name + '.npz')) as saved:
            np.testing.assert_array_equal(saved['cold'], cold)
            if index <= 20:
                for arm in stream.NAMES:
                    np.testing.assert_array_equal(saved[arm], cold)
        receipt = json.loads((output / 'shots' / (name + '_seal.json')).read_text())
        assert receipt['startup_archived'] == (index <= 20)
        assert not receipt['outcomes_used_in_prediction']
    result = json.loads((output / 'complete.json').read_text())
    accesses = json.loads((output / 'accesses.json').read_text())
    assert result['complete'] and result['archive_max_abs_error'] == 0
    assert all(access['sealed_count'] == 360 for access in accesses)
