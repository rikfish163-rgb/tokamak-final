import numpy as np

from experiments.risk_aligned_online import commissioning_held_diagnostic_20261112 as trial


def coupled_data():
    rng = np.random.default_rng(131)
    s = rng.normal(size=(72, trial.BINS, 1))
    gain = np.linspace(.7, 1.4, len(trial.GROUP))
    return s * gain + rng.normal(scale=.025, size=(72, trial.BINS, len(gain)))


def test_held_family_values_are_not_used_to_predict_that_family():
    e = coupled_data()
    fit = trial.fit_observer(e[:48], np.arange(48) // 6)
    changed = e[48:].copy()
    take = trial.GROUP == 1
    changed[..., take] += 20.
    got = trial.observer(e[48:], fit)
    other = trial.observer(changed, fit)
    np.testing.assert_array_equal(got['aligned'][..., take], other['aligned'][..., take])
    np.testing.assert_array_equal(got['shifted'][..., take], other['shifted'][..., take])
    assert not np.array_equal(got['aligned'][..., ~take], other['aligned'][..., ~take])


def test_phase_control_preserves_per_channel_values_and_group_comembership():
    e = coupled_data()
    shifted = trial.shift_groups(e)
    np.testing.assert_array_equal(np.sort(e, axis=1), np.sort(shifted, axis=1))
    for g in range(trial.FAMILIES):
        take = trial.GROUP == g
        np.testing.assert_array_equal(shifted[..., take], np.roll(e[..., take], g + 1, axis=1))


def test_same_frame_predicts_injected_shared_motion_better_than_misalignment():
    e = coupled_data()
    fit = trial.fit_observer(e[:48], np.arange(48) // 6)
    prediction = trial.observer(e[48:], fit)
    losses = trial.losses(e[48:], prediction, fit[1])
    assert np.all(losses['aligned'].mean(0) < .1 * losses['shifted'].mean(0))


def test_release_requires_full_arrival_and_real_observations():
    q = np.array([127, 159, 191, 223])
    mask = np.ones((4, trial.H, len(trial.GROUP)), dtype=bool)
    mask[1, 10, 2] = False
    n = 191 + trial.H + 1
    np.testing.assert_array_equal(trial.ready(q, mask, n), [0])
    np.testing.assert_array_equal(trial.ready(q, mask, n + 1), [0, 2])


def test_fitting_gives_each_shot_equal_mass():
    e = coupled_data()
    short = e[:20]
    source = np.r_[np.zeros(10, int), np.ones(10, int)]
    longer = np.concatenate((short, np.tile(short[:10], (9, 1, 1))))
    longer_id = np.r_[source, np.zeros(90, int)]
    f1 = trial.fit_observer(short, source)
    f2 = trial.fit_observer(longer, longer_id)
    for k in ('aligned', 'shifted', 'marginal'):
        np.testing.assert_allclose(trial.observer(e[-10:], f1)[k],
                                   trial.observer(e[-10:], f2)[k], atol=1e-12)


def test_synthetic_bias_localization_works_when_only_one_group_changes():
    e = coupled_data()
    fit = trial.fit_observer(e[:48], np.arange(48) // 6)
    clean = np.broadcast_to(fit[0], (8, *fit[0].shape))
    scores = trial.synthetic_localization(clean, fit)
    for k in ('level', 'ramp'):
        assert scores[k]['marginal'].shape == (8, trial.FAMILIES)
        assert scores[k]['aligned'].mean() == 1.


def test_local_estimate_is_causal_and_does_not_reuse_the_held_measurement():
    e = coupled_data()
    fit = trial.fit_observer(e[:24], np.arange(24)//6)
    a = e[24:].copy()
    r = np.arange(len(a)) * trial.lag.STRIDE
    full = trial.local_observer(a, r, fit)
    cut = 14
    short = trial.local_observer(a[:cut], r[:cut], fit)
    for k in trial.LOCAL:
        np.testing.assert_array_equal(short[k], full[k][:cut])
    held = trial.GROUP==2
    a[cut, :, held] += 20.
    changed = trial.local_observer(a, r, fit)
    for k in trial.LOCAL:
        np.testing.assert_array_equal(changed[k][cut, :, held], full[k][cut, :, held])


def test_local_observer_does_not_bridge_long_observation_gaps():
    e = coupled_data()
    fit = trial.fit_observer(e[:24], np.arange(24)//6)
    a = e[24:27]
    spacing = (trial.HISTORY+1)*trial.lag.STRIDE
    pred = trial.local_observer(a, np.arange(len(a))*spacing, fit)
    for v in pred.values():
        np.testing.assert_array_equal(v, np.broadcast_to(fit[0], a.shape))


def test_local_conditional_fit_uses_joint_motion_beyond_a_local_offset():
    e = coupled_data()
    fit = trial.fit_observer(e[:24], np.arange(24)//6)
    a = e[24:]
    pred = trial.local_observer(a, np.arange(len(a))*trial.lag.STRIDE, fit)
    error = trial.losses(a, pred, fit[1])
    assert np.all(error['local_aligned'][10:].mean(0) < .2*error['local_bias'][10:].mean(0))
    assert np.all(error['local_aligned'][10:].mean(0) < .2*error['local_shifted'][10:].mean(0))


def test_past_validation_never_uses_the_current_held_measurement():
    loss = np.broadcast_to(np.array([6., 4., 5., 5., 3., 1., 2.])[None,:,None],
                           (7,7,trial.FAMILIES)).copy()
    r = np.arange(len(loss))*trial.lag.STRIDE
    bias, accepted = trial.past_gates(loss,r)
    assert (bias[:2]==1).all() and not accepted[:2].any()
    assert (bias[2:]==4).all() and accepted[2:].all()
    loss[2:,5] = 25.
    b,c = trial.past_gates(loss,r)
    np.testing.assert_array_equal(b[:3],bias[:3])
    np.testing.assert_array_equal(c[:3],accepted[:3])
    assert not c[-1].any()


def test_past_validation_expires_across_a_gap():
    loss = np.broadcast_to(np.array([6.,4.,5.,5.,3.,1.,2.])[None,:,None],
                           (3,7,trial.FAMILIES))
    t = np.array([0,1,trial.HISTORY+2])*trial.lag.STRIDE
    bias, selected = trial.past_gates(loss,t)
    assert (bias[-1]==1).all() and not selected[-1].any()
