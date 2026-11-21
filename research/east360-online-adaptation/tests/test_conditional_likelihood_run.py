import numpy as np
import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.conditional_likelihood_20261029.run import (
    encode_features, folds, signal_windows, training_rows, summarize, evaluation_labels,
)


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.register_buffer('center', torch.zeros(23))
        self.base.register_buffer('scale', torch.ones(23))
        self.base.forecast_head = nn.Linear(17*64, 25*23)

    def forward(self, x, available, state, slow, mode):
        h = x[:, -1, :1].expand(-1, 17*64)
        self.base.forecast_head(h)
        return {'logit': x[:, -1, 0]}


def signals(n=300):
    x = np.arange(n*23, dtype=np.float32).reshape(n, 23)/1000
    return dict(values=x, available=np.ones_like(x, dtype=bool))


def test_forecast_context_does_not_access_future():
    net = ToyNet()
    s = signals()
    a = encode_features(net, s, [127], None)
    s['values'][128:] = 1e20
    b = encode_features(net, s, [127], None)
    for first, second in zip(a, b):
        np.testing.assert_array_equal(first, second)


def test_direct_target_uses_exact_arrived_25_and_excludes_mode():
    net = ToyNet()
    s = signals()
    target, mask = signal_windows(net, s, [127])
    expected = np.arcsinh(s['values'][128:153])-np.arcsinh(s['values'][127])
    np.testing.assert_allclose(target[0], expected, atol=3e-7)
    assert mask.shape == (1, 23) and not mask[0, 10]
    s['available'][152, 4] = False
    assert not signal_windows(net, s, [127])[1][0, 4]
    s['available'][153, 5] = False
    assert signal_windows(net, s, [127])[1][0, 5]


def test_targets_reject_unarrived_and_outcome_fields():
    net = ToyNet()
    s = signals(152)
    with pytest.raises(ValueError):
        signal_windows(net, s, [127])
    s = signals()
    s['label'] = 1
    with pytest.raises(ValueError):
        signal_windows(net, s, [127])


def test_five_folds_are_disjoint_and_exhaustive():
    f = folds(list(range(20)))
    assert len(f) == 5
    assert sorted(s for row in f for s in row['held']) == list(range(20))
    for row in f:
        assert len(row['fit']) == 16 and len(row['held']) == 4
        assert not set(row['fit']) & set(row['held'])
    with pytest.raises(ValueError):
        folds([1]*20)


def test_training_selection_filters_only_authorized_fit_outcomes():
    data = dict(ids=np.array([1, 1, 2]), origin_time=np.array([.7, .9, .7]),
                arrival_time=np.array([.725, .925, .725]))
    take, labels = training_rows(data, {1: (True, .92)})
    np.testing.assert_array_equal(take, [True, False, False])
    np.testing.assert_array_equal(labels, [0])


def test_summary_is_shot_weighted_not_window_weighted():
    rows = [dict(shot=1, group='safe', arm='real', windows=100, delta_bce=-1.),
            dict(shot=2, group='safe', arm='real', windows=1, delta_bce=3.)]
    s = summarize(rows)['real']['safe']
    assert s['mean_delta_bce'] == 1. and s['shots'] == 2


def test_safe_and_disruptive_evaluation_labels_preserve_window_shape():
    times = np.array([[.8, .9, 1.], [.801, .901, 1.001]])
    for flag, down in ((False, None), (True, 1.)):
        y, valid = evaluation_labels(times, flag, down)
        assert y.shape == valid.shape == times.shape
    y, valid = evaluation_labels(times, False, None)
    assert not y.any() and valid.all()
    y, valid = evaluation_labels(times, True, 1.)
    np.testing.assert_array_equal(y, [[0, 1, 1], [0, 1, 0]])
    assert not valid[-1, -1]
def test_density_fitter_rejects_unknown_scale_mode():
    from experiments.risk_aligned_online.conditional_likelihood_20261029.run import density_fitter
    import pytest
    with pytest.raises(ValueError, match='density scale'):
        density_fitter('pick_best_on_test')


def test_original_density_fitter_is_unchanged():
    from experiments.risk_aligned_online.conditional_likelihood_20261029.run import density_fitter
    from experiments.risk_aligned_online.conditional_likelihood_20261029.density import fit_density
    assert density_fitter('fitted_residual') is fit_density


