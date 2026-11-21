import copy

import torch

from experiments.risk_aligned_online.direct_decoder_meta_20261102 import trial


def test_all_arms_share_absolute_maintenance_and_same_anchor():
    risk = dict(mean=torch.tensor(-0.1), worst=torch.tensor(0.2))
    absolute = torch.tensor(2.0)
    anchor = torch.tensor(0.5)
    assert torch.isclose(
        trial.combine("forecast_only", absolute, risk, anchor), torch.tensor(0.40005)
    )
    assert torch.isclose(
        trial.combine("mean_meta", absolute, risk, anchor), torch.tensor(-0.59995)
    )
    assert torch.isclose(
        trial.combine("worst_meta", absolute, risk, anchor), torch.tensor(2.40005)
    )


def test_gate_cannot_claim_meta_gain_if_forecast_only_is_as_good():
    value = dict(
        safe=dict(mean_delta_bce=-0.01),
        danger=dict(mean_delta_bce=-0.01),
        compatible_danger=dict(mean_delta_bce=-0.01),
        balanced=-0.01,
    )
    s = {
        task: {
            a + "_" + m: copy.deepcopy(value) for a in trial.ARMS for m in trial.MODES
        }
        for task in ("native", "no_MNM_CIII")
    }
    assert not trial.gates(s)["passed"]
    s["no_MNM_CIII"]["worst_meta_real"]["balanced"] = -0.02
    assert trial.gates(s)["passed"]
