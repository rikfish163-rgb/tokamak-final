import numpy as np
import pytest
import torch

from layer_interaction_replay_20261105 import (
    MASK_ARMS,
    component_targets,
    require_curve_parity,
)


def test_fixed_masks_preserve_kept_parameters_and_norm_control():
    action = torch.linspace(-0.1, 0.1, 512).reshape(2, 256)
    before = action.clone()
    outputs = component_targets(action)
    assert tuple(outputs) == MASK_ARMS[1:]
    torch.testing.assert_close(outputs["moving_peak"], before)
    torch.testing.assert_close(outputs["layer0"] + outputs["layer1"], before)
    torch.testing.assert_close(outputs["gain"] + outputs["bias"], before)
    for i in range(2):
        arm = outputs[f"layer{i}_norm"]
        block = slice(i * 128, (i + 1) * 128)
        other = slice((1 - i) * 128, (2 - i) * 128)
        assert torch.count_nonzero(arm[:, other]) == 0
        torch.testing.assert_close(arm.norm(dim=1), before.norm(dim=1))
        torch.testing.assert_close(
            torch.nn.functional.normalize(arm[:, block]),
            torch.nn.functional.normalize(before[:, block]),
        )
    torch.testing.assert_close(action, before)


def test_empty_and_exact_zero_write_do_not_create_a_direction():
    for n in (0, 1):
        for value in component_targets(torch.zeros(n, 256)).values():
            assert value.shape == (n, 256)
            assert torch.count_nonzero(value) == 0
    with pytest.raises(ValueError, match="missing direction"):
        component_targets(torch.cat((torch.ones(1, 128), torch.zeros(1, 128)), dim=1))


def test_numerical_match_must_also_preserve_threshold_crossing():
    tau = 0.7258946895599365
    a = np.array([tau - 1e-7, tau + 1e-6])
    assert require_curve_parity(a + 1e-9, a, tau) < 4e-6
    with pytest.raises(ValueError, match="forecast parity"):
        require_curve_parity(a + 2e-7, a, tau)
    with pytest.raises(ValueError, match="forecast parity"):
        require_curve_parity(a + 0.01, a, tau)
    with pytest.raises(ValueError, match="clock alignment"):
        require_curve_parity(a[:1], a, tau)
