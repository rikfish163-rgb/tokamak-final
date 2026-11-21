import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_patch_retention_20261114 as study
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_full_coordinates_are_reversible_and_dc_erases_only_within_patch_detail():
    x = np.random.default_rng(1).normal(size=(5, 16, 8, 8))
    wave = study.temporal_coordinates(x)
    np.testing.assert_allclose(np.einsum('...kc,kt->...tc', wave, study.BASIS), x, atol=1e-13)
    without_detail = np.einsum('...kc,kt->...tc', study.dc_only(wave), study.BASIS)
    np.testing.assert_allclose(without_detail, np.broadcast_to(x.mean(2, keepdims=True), x.shape), atol=1e-13)


def test_decoder_has_one_shared_slope_and_a_separate_intercept_by_patch_position():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(15, 4, 3))
    w = rng.normal(size=(3, 6))
    offset = rng.normal(size=(4, 6))
    fit = study.fit_decoder(x, x@w+offset, np.full(len(x), 1/len(x)), ridge=0.)
    unseen = rng.normal(size=(3, 4, 3))
    np.testing.assert_allclose(study.decode(unseen, fit), unseen@w+offset, atol=1e-12)


def test_actual_encoder_views_keep_position_time_and_ignore_later_samples():
    torch.manual_seed(3)
    rng = np.random.default_rng(4)
    base = Simple23(Config(fast_site='tokens'), torch.zeros(23), torch.ones(23)).eval()
    x = rng.normal(size=(169, 23)).astype(np.float32)
    m = np.ones_like(x, dtype=bool)
    m[0, study.COLS[0]] = False
    t = np.array([127, 131, 151])
    frames = study.encoder_views(base, dict(values=x, available=m), t)
    short = study.encoder_views(base, dict(values=x[:152], available=m[:152]), t)
    assert frames['supported'].tolist() == [False, True, True]
    assert frames['raw'].shape == (3, 16, 8, 8)
    assert frames['block2'].shape == (3, 16, 64)
    np.testing.assert_allclose(frames['base'][:, 0], frames['source'][:, 0], atol=1e-7)
    for key in frames:
        np.testing.assert_array_equal(frames[key], short[key])


def test_per_patch_l1_keeps_detail_and_radiation_separate():
    truth = np.zeros((3, 16, 8, 8))
    estimated = truth.copy()
    estimated[..., 2, study.RADIATION[0]] = 3.
    error = study.recovery_error(estimated, truth)
    np.testing.assert_array_equal(error[:, [0, 2]], 0.)
    np.testing.assert_allclose(error[:, 1], 3/(7*8))
    np.testing.assert_allclose(error[:, 3], 3/(7*4))
