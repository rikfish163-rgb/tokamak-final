import numpy as np
import pytest
import torch

from .direction import balanced_direction, peak_candidates


def test_direction_is_unit_and_sign_fixed_not_a_class_contrast():
    g = torch.tensor([[.6, .8], [.6, .8], [-.6, -.8]])
    v, audit = balanced_direction(g, [0, 0, 1])
    torch.testing.assert_close(v, torch.tensor([.6, .8]), atol=1e-6, rtol=0)
    assert audit['class_counts'] == [2, 1]
    assert audit['first_energy_fraction'] == pytest.approx(1.)
    assert abs(float(v.norm()) - 1) < 1e-6


def test_duplication_within_one_class_does_not_change_class_weight():
    g = torch.tensor([[1., 0.], [1., 1.]])
    v, _ = balanced_direction(g, [0, 1])
    repeat, _ = balanced_direction(torch.stack([g[0], g[0], g[0], g[1]]), [0, 0, 0, 1])
    torch.testing.assert_close(v, repeat)


@pytest.mark.parametrize('g,y', [([[0., 0.], [1., 0.]], [0, 1]),
                              ([[1., 0.], [1., 0.]], [0, 0]),
                              ([[1., 0.], [0., 1.]], [0, 1]),
                              ([[float('nan'), 0.], [1., 0.]], [0, 1])])
def test_invalid_or_degenerate_direction_rejected(g, y):
    with pytest.raises(ValueError):
        balanced_direction(torch.tensor(g), y)


def test_peak_candidates_strict30_no_upper125_and_original_indices():
    shot = dict(device='J-TEXT', times=np.array([0., .8, .96, .97, 1.]),
                down=1., valid=np.array([True, True, True, True, False]), is_disrupt=True)
    assert peak_candidates(shot, first_effect=1).tolist() == [1, 2]
    shot['is_disrupt'] = False
    assert peak_candidates(shot, first_effect=1).tolist() == [1, 2, 3]


def test_peak_candidates_no_opportunity_is_empty_not_fake_peak():
    shot = dict(device='J-TEXT', times=np.array([.97, .99]), down=1.,
                valid=np.ones(2, bool), is_disrupt=True)
    assert not len(peak_candidates(shot, first_effect=0))


def test_peak_candidates_reject_target_data():
    with pytest.raises(ValueError):
        peak_candidates(dict(device='EAST'), first_effect=6)
