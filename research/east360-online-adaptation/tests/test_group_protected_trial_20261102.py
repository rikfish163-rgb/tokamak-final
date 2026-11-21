import copy

import torch

from experiments.risk_aligned_online.group_protected_meta_20261102 import trial


def row(shot, group, lead, delta):
    return dict(shot=shot, group=group, lead_ms=lead, delta_bce=delta)


def test_compatible_window_does_not_relabel_source_early_or_accept_30ms():
    r = [
        row(1, "safe", None, -0.1),
        row(2, "danger", 30.0, -0.9),
        row(2, "danger", 31.0, -0.2),
        row(3, "danger", 126.0, 0.8),
        row(4, "early", 50.0, -0.7),
    ]
    s = trial.summarize(r)
    assert s["compatible_danger"]["n_points"] == 1
    assert s["compatible_danger"]["n_shots"] == 1
    assert s["compatible_danger"]["mean_delta_bce"] == -0.2


def test_gate_cannot_be_met_by_only_better_negative_windows_or_other_arm():
    value = dict(
        safe=dict(mean_delta_bce=-0.01),
        danger=dict(mean_delta_bce=-0.01),
        compatible_danger=dict(mean_delta_bce=-0.01),
        balanced=-0.01,
    )
    s = {
        task: {
            a + "_" + m: copy.deepcopy(value)
            for a in trial.ARMS
            for m in ("real", "reverse", "hold")
        }
        for task in trial.TASKS
    }
    old = {"no_MNM_CIII": {"true_real": copy.deepcopy(value)}}
    assert not trial.gates(s, old)["passed"]
    s["no_MNM_CIII"]["worst_risk_real"]["balanced"] = -0.02
    assert trial.gates(s, old)["passed"]
    s["no_MNM_CIII"]["worst_risk_real"]["compatible_danger"]["mean_delta_bce"] = 0.01
    assert not trial.gates(s, old)["passed"]


def test_pair_forward_keeps_both_update_graphs_and_detaches_reference(monkeypatch):
    table = torch.ones(6, 25, 23, requires_grad=True)
    seen = []

    def adapted(_model, _data, i, parameter, arm, *, outer):
        seen.append((i, arm, outer))
        return (
            parameter.mean().expand(4, 64),
            dict(accepted=True),
            torch.full((6,), 1 / 6),
        )

    def predicted(_model, pack, state):
        return dict(logit=pack["h"].flatten() + state.mean())

    monkeypatch.setattr(trial, "adapt", adapted)
    monkeypatch.setattr(trial, "forward", predicted)
    cache = dict(
        later=dict(h=torch.tensor([[2.0], [3.0]])),
        records=[dict(label=0.0), dict(label=1.0)],
    )
    after, before, labels, _ = trial.pair_forward(None, cache, [0, 1], table)
    assert seen == [(0, "true", True), (1, "true", True)]
    assert after.requires_grad and not before.requires_grad
    torch.testing.assert_close(after, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(before, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(labels, torch.tensor([0.0, 1.0]))
    after.sum().backward()
    torch.testing.assert_close(table.grad, torch.full_like(table, 2 / table.numel()))
