import pytest
import torch

from experiments.risk_aligned_online.east_decoder_calibration_20261102.temporal_audit import (
    invariance_fixture,
    temporal_split,
)


def test_temporal_split_is_orthogonal_and_reports_parameter_not_physical_energy():
    generator = torch.Generator().manual_seed(8)
    delta = torch.randn(575, 4, generator=generator, dtype=torch.float64)
    dc, ac, summary = temporal_split(delta)
    torch.testing.assert_close(dc + ac, delta)
    torch.testing.assert_close(
        ac.reshape(25, 23, 4).mean(0), torch.zeros(23, 4, dtype=delta.dtype)
    )
    assert abs(float((dc * ac).sum())) < 1e-11
    assert summary["quantity"] == "squared_parameter_norm_not_physical_energy"
    assert abs(summary["dc_fraction"] + summary["ac_fraction"] - 1) < 1e-12
    zero = temporal_split(torch.zeros_like(delta))[2]
    assert zero["dc_fraction"] is None and zero["ac_fraction"] is None
    with pytest.raises(ValueError):
        temporal_split(torch.zeros(23, 4))
    delta[0, 0] = float("nan")
    with pytest.raises(ValueError):
        temporal_split(delta)


def test_actual_centered_loss_discards_state_dependent_dc_even_with_missingness():
    result = invariance_fixture()
    assert result["scope"] == "synthetic_algebra_not_neural_replay"
    assert set(result["masks"]) == {"complete", "partial"}
    for values in result["masks"].values():
        assert values["loss_abs_difference"] < 1e-12
        assert values["state_gradient_max_abs_difference"] < 1e-12
        assert values["uncentered_loss_abs_difference"] > 1e-3
