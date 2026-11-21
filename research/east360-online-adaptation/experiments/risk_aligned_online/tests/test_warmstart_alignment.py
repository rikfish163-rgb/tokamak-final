import numpy as np
import pytest

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import (
    fit_statistics, fit_prefix_map, transform, make_map,
)


def record(sid, device='J-TEXT', n=129):
    x = (np.linspace(1., 3., n)[:, None] * np.arange(1., 24.)[None]).astype(np.float32)
    return dict(shot=sid, device=device, values=x, available=np.ones_like(x, bool))


def test_statistics_weight_shots_equally_not_record_duration():
    a, b = record(1, n=10), record(2, n=1000)
    a['values'][:] = 1.; b['values'][:] = 9.
    stats = fit_statistics([a, b], device='J-TEXT', allowed_ids=[1, 2])
    np.testing.assert_allclose(stats['mean'], 5.)
    np.testing.assert_allclose(stats['std'], 4.)
    assert stats['record_counts'] == [2] * 23


@pytest.mark.parametrize('policy', ['moments_all', 'robust_all', 'density_moments'])
def test_positive_affine_shift_is_undone_in_source_input_coordinates(policy):
    src, dst = record(1), record(2, 'EAST')
    dst['values'] = src['values'] * 12 + 7
    ss = fit_statistics([src], device='J-TEXT', allowed_ids=[1])
    ts = fit_statistics([dst], device='EAST', allowed_ids=[2])
    mapping = make_map(ss, ts, policy)
    out = transform({k: dst[k] for k in ('values', 'available')}, mapping)
    mapped = [i for i, r in enumerate(mapping['channels']) if r['status'] == 'mapped']
    np.testing.assert_allclose(out['values'][:, mapped], src['values'][:, mapped], rtol=2e-6, atol=1e-5)
    np.testing.assert_array_equal(out['available'], dst['available'])
    if policy == 'density_moments': assert mapped == [io23.RAW23.index('ne_nG')]
    else: assert len(mapped) == 21


def test_density_gain_preserves_zero_and_relative_variations():
    src, dst = record(1), record(2, 'EAST')
    dst['values'] *= 1000.
    ss = fit_statistics([src], device='J-TEXT', allowed_ids=[1])
    ts = fit_statistics([dst], device='EAST', allowed_ids=[2])
    mapping = make_map(ss, ts, 'density_gain')
    c = io23.RAW23.index('ne_nG')
    assert mapping['offset'][c] == 0.
    dst['values'][0, c] = 0.
    out = transform({k: dst[k] for k in ('values', 'available')}, mapping)
    assert out['values'][0, c] == 0.
    np.testing.assert_allclose(out['values'][1:, c], src['values'][1:, c], rtol=2e-6)
    for j in range(23):
        if j != c: np.testing.assert_array_equal(out['values'][:, j], dst['values'][:, j])


def test_constant_or_missing_channel_is_not_silently_masked():
    src, dst = record(1), record(2, 'EAST')
    dst['values'][:, 0] = 3.
    dst['available'][:, 5] = False
    dst['values'][:, 5] = np.nan
    ss = fit_statistics([src], device='J-TEXT', allowed_ids=[1])
    ts = fit_statistics([dst], device='EAST', allowed_ids=[2])
    mapping = make_map(ss, ts, 'moments_all')
    assert mapping['channels'][0]['status'] == 'constant_unidentifiable_identity'
    assert mapping['channels'][5]['status'] == 'missing_unidentifiable_identity'
    out = transform({k: dst[k] for k in ('values', 'available')}, mapping)
    np.testing.assert_array_equal(out['values'][:, 0], dst['values'][:, 0])
    np.testing.assert_array_equal(out['available'], dst['available'])
    assert np.isnan(out['values'][:, 5]).all()


def test_zero_iqr_with_sparse_activity_is_not_reported_as_constant():
    src, dst = record(1, n=128), record(2, 'EAST', n=128)
    dst['values'][:, 3] = 0.
    dst['values'][-1, 3] = 10.
    ss = fit_statistics([src], device='J-TEXT', allowed_ids=[1])
    ts = fit_statistics([dst], device='EAST', allowed_ids=[2])
    assert ts['iqr'][3] == 0. and ts['std'][3] > 0.
    mapping = make_map(ss, ts, 'robust_all')
    assert mapping['channels'][3]['status'] == 'zero_iqr_unidentifiable_identity'
    out = transform({k:dst[k] for k in ('values', 'available')}, mapping)
    np.testing.assert_array_equal(out['values'][:,3], dst['values'][:,3])


def test_uncertified_mode_features_never_rescaled_or_masked():
    src, dst = record(1), record(2, 'EAST')
    ss = fit_statistics([src], device='J-TEXT', allowed_ids=[1])
    ts = fit_statistics([dst], device='EAST', allowed_ids=[2])
    for policy in ('moments_all', 'robust_all', 'density_moments', 'density_gain'):
        mapping = make_map(ss, ts, policy)
        for name in ('MNM', 'mode_number_n'):
            c = io23.RAW23.index(name)
            assert mapping['gain'][c] == 1. and mapping['offset'][c] == 0.


def test_only_exact_ended_prefix_can_fit_mapping():
    ss = fit_statistics([record(1)], device='J-TEXT', allowed_ids=[1])
    records = [record(i, 'EAST') for i in range(10, 13)]
    order = list(range(10, 20))
    mapping = fit_prefix_map(ss, records, order=order, count=3, policy='moments_all')
    assert mapping['calibration_ids'] == [10, 11, 12]
    assert mapping['first_affected_index'] == 3
    for bad in (records[::-1], records[:2], records+[record(13, 'EAST')]):
        with pytest.raises(ValueError):
            fit_prefix_map(ss, bad, order=order, count=3, policy='moments_all')


@pytest.mark.parametrize('key', ['y', 'down', 'is_disrupt', 'event_label', 'valid'])
def test_statistics_reject_outcome_and_event_fields(key):
    r = record(1); r[key] = None
    with pytest.raises(ValueError, match='signals only'):
        fit_statistics([r], device='J-TEXT', allowed_ids=[1])


def test_no_future_window_dependence_or_mutation():
    src, dst = record(1), record(2, 'EAST')
    ss = fit_statistics([src], device='J-TEXT', allowed_ids=[1])
    mapping = fit_prefix_map(ss, [dst], order=[2, 3], count=1, policy='moments_all')
    signal = {k: dst[k].copy() for k in ('values', 'available')}
    before = signal['values'].copy()
    full = transform(signal, mapping)
    signal['values'][64:] *= 100
    other = transform(signal, mapping)
    np.testing.assert_array_equal(full['values'][:64], other['values'][:64])
    np.testing.assert_array_equal(dst['values'], before)
    with pytest.raises(ValueError): transform(dict(signal, y=None), mapping)


def test_duplicate_or_unauthorized_shots_rejected():
    with pytest.raises(ValueError):
        fit_statistics([record(1), record(1)], device='J-TEXT', allowed_ids=[1])
    with pytest.raises(ValueError):
        fit_statistics([record(2)], device='J-TEXT', allowed_ids=[1])


def test_mapping_cannot_invent_unknown_policy():
    ss = fit_statistics([record(1)], device='J-TEXT', allowed_ids=[1])
    with pytest.raises(ValueError): make_map(ss, ss, 'pick_using_test_labels')


def test_batched_causal_windows_and_risk_only_path_match_source():
    import torch
    from experiments.risk_aligned_online.simple23 import Config, Simple23
    from experiments.risk_aligned_online.simple_data import window
    from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback, initial_encoder
    from experiments.risk_aligned_online.warmstart_alignment_20261022.run import predict_many
    torch.set_num_threads(1); torch.manual_seed(7)
    base = Simple23(Config(dim=8, heads=2, fast_site='tokens'), np.zeros(23), np.ones(23)).eval()
    model = PairedFeedback(base, np.ones(23)); model.set_stage('frozen')
    r = record(1, n=131); r['available'][10:19, 2] = False
    signal = {k: r[k] for k in ('values', 'available')}
    scores = predict_many(model, {'one': signal}, device='cpu', batch_size=11)['one']
    fast = initial_encoder(model)
    x, m = [torch.tensor(np.stack(v)) for v in zip(*[window(r, t, 128) for t in range(131)])]
    reference = model.predict_fast(x, m, fast)['logit'].sigmoid().detach().numpy()
    np.testing.assert_allclose(scores, reference, rtol=0, atol=2e-6)
    changed = {k: v.copy() for k, v in signal.items()}; changed['values'][60:] *= 100
    future = predict_many(model, {'one': changed}, device='cpu', batch_size=11)['one']
    np.testing.assert_array_equal(scores[:60], future[:60])


def test_adding_alignment_arms_cannot_change_static_numerical_path():
    import torch
    from experiments.risk_aligned_online.simple23 import Config, Simple23
    from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback
    from experiments.risk_aligned_online.warmstart_alignment_20261022.run import predict_many
    torch.set_num_threads(1); torch.manual_seed(131)
    base = Simple23(Config(dim=8, heads=2, fast_site='tokens'), np.zeros(23), np.ones(23)).eval()
    model = PairedFeedback(base, np.ones(23)); model.set_stage('frozen')
    r = record(1, n=97)
    s = {k: r[k] for k in ('values', 'available')}
    one = predict_many(model, {'static': s}, device='cpu')['static']
    many = predict_many(model, {'static': s, **{f'arm{i}': s for i in range(8)}}, device='cpu')
    for value in many.values(): np.testing.assert_array_equal(value, one)
