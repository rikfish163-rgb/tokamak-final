from types import SimpleNamespace

import torch

from experiments.risk_aligned_online.armijo_feedback_20261102 import rejection_audit


def test_components_reconstruct_exact_old_objective_and_candidate(monkeypatch):
    model = SimpleNamespace(
        base=SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23))
    )
    pack = dict(
        phi=torch.eye(4).repeat(4, 1)[None], valid=torch.ones(1, 17, dtype=torch.bool)
    )
    state = torch.ones(4, 64) * 0.1
    gradient = torch.zeros_like(state)
    gradient[0, 0] = 0.01
    truth = torch.zeros(1, 25, 23)
    pattern = torch.linspace(-1, 1, 25).reshape(1, 25, 1).expand(1, 25, 23)

    def auxiliary(_model, _pack, w, _coefficient):
        return pattern * w[0, 0]

    monkeypatch.setattr(rejection_audit, "auxiliary", auxiliary)
    value = rejection_audit.components(
        model,
        pack,
        state,
        gradient,
        truth,
        torch.ones_like(truth, dtype=torch.bool),
        torch.zeros(25, 23),
    )
    expected = state + rejection_audit.metric_direction(pack, gradient)
    assert torch.equal(value["proposal"], expected)
    assert value["total_before"] == float(
        (value["data_before_tensor"] + value["anchor_before_tensor"]).detach()
    )
    assert value["total_proposal"] == float(
        value["data_proposal_tensor"] + value["anchor_proposal_tensor"]
    )


def test_component_classification_does_not_claim_total_loss_equals_data_loss():
    classify = rejection_audit.reason_class
    assert classify(False, -0.01, 0.02) == "anchor_overrides_data_improvement"
    assert classify(False, 0.02, -0.01) == "data_term_increase"
    assert classify(False, 0.01, 0.01) == "both_terms_increase"
    assert classify(False, 0.0, 0.0) == "flat_or_roundoff"
    assert classify(True, 0.01, -0.02) == "accepted_while_data_fit_worsens"
    assert classify(True, -0.01, 0.001) == "accepted_with_data_improvement"
