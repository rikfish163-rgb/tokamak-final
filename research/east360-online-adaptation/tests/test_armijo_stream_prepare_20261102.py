"""CPU-only functional contracts; no HDF5 reads, CUDA calls, or output files."""

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.armijo_feedback_20261102 import core, prepare, stream
from experiments.risk_aligned_online.critical_shot_meta_20261102.run import export_traces
from experiments.risk_aligned_online.dual_history_ttt_20261101.run import digest_array
from experiments.risk_aligned_online.dual_history_ttt_20261101.stream import targets


def functional_runtime(monkeypatch, accepted=(True, False, False)):
    model = torch.nn.Module()
    model.base = torch.nn.Module()
    model.base.register_buffer("center", torch.zeros(23, dtype=torch.float64))
    model.base.register_buffer("scale", torch.ones(23, dtype=torch.float64))
    model.register_parameter("frozen", torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64), requires_grad=False))
    values = np.repeat(np.arange(448, dtype=np.float64)[:, None], 23, axis=1)
    signal = dict(values=values, available=np.ones_like(values, dtype=bool))
    encoded, ordering = [], []

    def encode(_model, signal, ends, _basis):
        encoded.append(tuple(int(i) for i in ends))
        return dict(index=torch.as_tensor(ends), value=torch.as_tensor(signal["values"][ends, 0]))

    def forward(_model, pack, state):
        n = len(pack["index"])
        return dict(
            logit=pack["value"] * 0.001 + state[0, 0] + model.frozen * 0.01,
            rms=torch.zeros(n),
            capped=torch.zeros(n, dtype=torch.bool),
        )

    def auxiliary(_model, pack, state, coefficient):
        ordering.append(("forecast", int(pack["index"][0])))
        return (state[0, 0] + coefficient[0, 0]).expand(len(pack["index"]), 25, 23)

    def loss(_model, prediction, truth, mask):
        return (prediction[mask] - truth[mask]).square().mean() if mask.any() else prediction.sum() * 0

    def propose(_model, point, state, truth, mask, coefficient, *, create_graph=False):
        origin = int(point["index"][0])
        step = (origin - 134) // 128
        ordering.append(("update", origin))
        current = state.clone() if create_graph else state.detach().clone()
        current.requires_grad_(True)
        take_step = accepted[step] and bool(mask.any())
        delta = torch.zeros_like(current)
        delta[0, 0] = coefficient[0, 0]
        after = current + delta if take_step else current
        if not create_graph:
            after = after.detach()
        gradient = torch.zeros_like(state)
        gradient[0, 0] = 1
        evaluations = 2 if take_step else 11
        receipt = dict(
            accepted=take_step,
            reason="armijo" if take_step else "budget_exhausted",
            alpha=0.5 if take_step else 0.0,
            evaluations=evaluations,
            loss_before=2.0,
            loss_after=1.0 if take_step else 2.0,
            armijo_bound=1.5 if take_step else None,
            trials=[
                dict(alpha=2.0**-i, loss=1.0 if take_step and i == 1 else 3.0, armijo_bound=1.5)
                for i in range(evaluations)
            ],
            gradient_norm=1.0,
            step_norm=float((after.detach() - current.detach()).norm()),
            gradient256=gradient.flatten().tolist(),
            state256_after=after.detach().flatten().tolist(),
        )
        return after, receipt

    for module in (core, stream, prepare):
        for name, function in (
            ("auxiliary", auxiliary),
            ("innovation_loss", loss),
            ("forward", forward),
            ("propose", propose),
        ):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, function)
    monkeypatch.setattr(stream, "encode", encode)
    monkeypatch.setattr(io23, "read_outcome", lambda *args: pytest.fail("no outcome access"))
    coefficient = torch.full((25, 23), 0.2, dtype=torch.float64, requires_grad=True)
    pack = dict(index=torch.arange(448), value=torch.as_tensor(values[:, 0]))
    example = dict(shot=69854, times=np.arange(448) / 1000, signal=signal, pack=pack, disruptive=False, down=None)
    return model, example, coefficient, encoded, ordering


def test_actual_clock_seals_common_support_and_cache_matches_without_source_changes(monkeypatch):
    model, example, b, encoded, ordering = functional_runtime(monkeypatch, (True, True, True))
    example["signal"]["available"][145, 0] = False
    original = {k: v.clone() for k, v in model.state_dict().items()}
    heads = {
        a: (b.detach(), mode)
        for a, mode in (("meta", "real"), ("forecast", "real"), ("reverse", "reverse"), ("hold", "hold"))
    }
    original_targets = stream.targets

    def sealed_targets(signal, origin, end):
        assert ordering[-len(heads) :] == [("forecast", origin)] * len(heads)
        ordering.append(("targets", end))
        return original_targets(signal, origin, end)

    monkeypatch.setattr(stream, "targets", sealed_targets)
    actual = stream.simulate(model, example["signal"], None, heads)
    encoding_calls = len(encoded)
    cached = stream.simulate(model, example["signal"], None, heads, cached=example["pack"])
    assert len(encoded) == encoding_calls
    for arm, (_, mode) in heads.items():
        np.testing.assert_array_equal(actual["logits"][arm], cached["logits"][arm])
        np.testing.assert_array_equal(actual["probability"][arm], cached["probability"][arm])
        assert [(t["origin"], t["maturity"], t["first_affected"]) for t in actual["tapes"][arm]] == [
            (134, 159, 160),
            (262, 287, 288),
            (390, 415, 416),
        ]
        for left, right in zip(actual["tapes"][arm], cached["tapes"][arm], strict=True):
            assert torch.equal(left["before"], right["before"]) and torch.equal(left["after"], right["after"])
            assert left["receipt"] == right["receipt"] and "loss_proposal" not in left["receipt"]
            truth, mask = targets(example["signal"], left["origin"], left["maturity"])
            assert left["target_sha256"] == digest_array(truth[mode])
            assert left["mask_sha256"] == digest_array(mask) and left["observed"] == int(mask.sum())
            assert len(left["forecast_sha256"]) == 64 and left["receipt"]["evaluations"] <= 11
        assert actual["actions"][arm]["endpoints"] == 448
    for group in zip(*(actual["tapes"][a] for a in heads), strict=True):
        assert len({t["mask_sha256"] for t in group}) == 1
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert model.frozen.grad is None
    arrays, info = export_traces(actual["tapes"], actual["actions"])
    assert arrays["meta_gradient"].shape == (3, 256)
    assert info["updates"]["meta"][0]["loss_after"] == 1.0
    assert info["updates"]["meta"][0]["alpha"] == 0.5


def test_matured_updates_and_changed_future_never_rewrite_prefix(monkeypatch):
    model, example, b, _, _ = functional_runtime(monkeypatch, (True, True, True))
    heads = {"meta": (b.detach(), "real"), "forecast": (b.detach(), "real")}
    full = stream.simulate(model, example["signal"], None, heads)
    baseline = (example["pack"]["value"] * 0.001 + model.frozen * 0.01).numpy()
    for arm in heads:
        np.testing.assert_array_equal(full["logits"][arm][:160], baseline[:160])
    prefix = {k: v[:416].copy() for k, v in example["signal"].items()}
    truncated = stream.simulate(model, prefix, None, heads)
    changed = {k: v.copy() for k, v in example["signal"].items()}
    changed["values"][416:] *= 100
    repeated = stream.simulate(model, changed, None, heads)
    for arm in heads:
        np.testing.assert_array_equal(full["logits"][arm][:416], truncated["logits"][arm])
        np.testing.assert_array_equal(full["logits"][arm][:416], repeated["logits"][arm][:416])


@pytest.mark.parametrize("field", ["disruptive", "down", "labels"])
def test_simulate_rejects_outcome_bearing_payload(monkeypatch, field):
    model, example, b, _, _ = functional_runtime(monkeypatch)
    with pytest.raises(ValueError):
        stream.simulate(model, {**example["signal"], field: True}, None, {"meta": (b.detach(), "real")})


@pytest.mark.parametrize("arm", ["meta", "forecast"])
def test_full_history_survives_rejected_last_and_signal_detaches_incoming_state(monkeypatch, arm):
    model, example, b, _, _ = functional_runtime(monkeypatch, (True, False, False))
    simulation = stream.simulate(model, example["signal"], None, {arm: (b.detach(), "real")}, cached=example["pack"])
    result = prepare.replay_query(model, example, simulation, arm, b, 416)
    assert [r["accepted"] for r in result["receipts"]] == [True, False, False]
    assert torch.equal(result["state"].detach(), simulation["tapes"][arm][-1]["after"])
    assert float(result["after"].detach()) == float(simulation["logits"][arm][416])
    assert not result["before"].requires_grad
    assert float(result["before"]) == float(result["after"].detach())
    (gradient,) = torch.autograd.grad(result["after"], b, retain_graph=True)
    assert gradient[0, 0] == 1
    (signal_gradient,) = torch.autograd.grad(result["signal"], b)
    truth, mask = targets(example["signal"], 390, 415)
    constant = simulation["tapes"][arm][-1]["before"][0, 0] + b[0, 0].detach()
    expected = 2 * (constant - torch.as_tensor(truth["real"])[torch.as_tensor(mask)]).mean()
    torch.testing.assert_close(signal_gradient[0, 0], expected)


def test_all_rejected_risk_path_has_legitimate_none_gradient(monkeypatch):
    model, example, b, _, _ = functional_runtime(monkeypatch, (False, False, False))
    simulation = stream.simulate(model, example["signal"], None, {"meta": (b.detach(), "real")}, cached=example["pack"])
    result = prepare.replay_query(model, example, simulation, "meta", b, 416)
    (gradient,) = torch.autograd.grad(result["after"], b, allow_unused=True)
    assert gradient is None and all(not r["accepted"] for r in result["receipts"])


@pytest.mark.parametrize("query,count", [(159, 0), (160, 1), (287, 1), (288, 2), (415, 2), (416, 3)])
def test_replay_only_consumes_strictly_matured_prefix(monkeypatch, query, count):
    model, example, b, _, _ = functional_runtime(monkeypatch)
    simulation = stream.simulate(model, example["signal"], None, {"meta": (b.detach(), "real")}, cached=example["pack"])
    result = prepare.replay_query(model, example, simulation, "meta", b, query)
    assert result is None if count == 0 else len(result["receipts"]) == count


@pytest.mark.parametrize("arm", ["full_operational", "metric_operational", "operational", "unknown"])
def test_old_recipe_arm_names_are_not_silently_accepted(monkeypatch, arm):
    model, example, b, _, _ = functional_runtime(monkeypatch)
    with pytest.raises((ValueError, AssertionError)):
        prepare.replay_query(model, example, {"tapes": {}, "logits": {}}, arm, b, 160)


@pytest.mark.parametrize("corruption", ["state", "receipt", "risk"])
def test_replay_keeps_exact_state_receipt_and_original_batch_risk_checks(monkeypatch, corruption):
    model, example, b, _, _ = functional_runtime(monkeypatch)
    simulation = stream.simulate(model, example["signal"], None, {"meta": (b.detach(), "real")}, cached=example["pack"])
    if corruption == "state":
        simulation["tapes"]["meta"][0]["after"] = simulation["tapes"]["meta"][0]["after"] + 0.01
    elif corruption == "receipt":
        simulation["tapes"]["meta"][0]["receipt"]["alpha"] = 0.25
    else:
        simulation["logits"]["meta"][416] += 1e-5
    with pytest.raises((ValueError, AssertionError)):
        prepare.replay_query(model, example, simulation, "meta", b, 416)


def test_early_critical_is_not_replaced_and_forecast_has_only_signal_regularization(monkeypatch):
    model, example, b, _, _ = functional_runtime(monkeypatch)
    example.update(disruptive=True, down=10.0)
    heads = {arm: (b.detach(), "real") for arm in ("meta", "forecast")}
    simulation = stream.simulate(model, example["signal"], None, heads, cached=example["pack"])
    objectives = {}
    for arm in heads:
        simulation["logits"][arm][50] = 10
        objective, receipt = prepare.episode(model, example, simulation, arm, b, 0.7)
        assert receipt["query"] == 50 and not receipt["supported"]
        assert receipt["label"] == 1  # Operational shot label, not a 125ms point label.
        assert receipt["risk_before"] == receipt["risk_after"]
        truth, mask = targets(example["signal"], 134, 159)
        signal = (b[0, 0] - torch.as_tensor(truth["real"])[torch.as_tensor(mask)]).square().mean()
        expected = (0.20 if arm == "meta" else 1.0) * signal + 1e-4 * b.square().mean()
        torch.testing.assert_close(objective, expected)
        objectives[arm] = objective
    assert objectives["forecast"] > objectives["meta"]


def test_supported_forecast_objective_excludes_nonzero_risk_change(monkeypatch):
    model, example, b, _, _ = functional_runtime(monkeypatch, (True, True, True))
    heads = {arm: (b.detach(), "real") for arm in ("meta", "forecast")}
    simulation = stream.simulate(model, example["signal"], None, heads, cached=example["pack"])
    assert tuple(prepare.ARMS) == ("meta", "forecast")
    for arm in heads:
        objective, receipt = prepare.episode(model, example, simulation, arm, b, 0.7)
        assert receipt["supported"] and receipt["query"] == 447
        replay = prepare.replay_query(model, example, simulation, arm, b, 447)
        before = prepare.margin_loss(replay["before"], 0.7, 0).detach()
        after = prepare.margin_loss(replay["after"], 0.7, 0)
        assert float((after - before).detach()) != 0
        expected = replay["signal"] + 1e-4 * b.square().mean()
        if arm == "meta":
            expected = (after - before) / 0.1 + 0.20 * replay["signal"] + 1e-4 * b.square().mean()
        torch.testing.assert_close(objective, expected)


def test_unobserved_stream_has_no_measured_gradient_or_solver_evaluations(monkeypatch):
    actual_propose = core.propose
    model, example, b, _, _ = functional_runtime(monkeypatch)
    monkeypatch.setattr(core, "propose", actual_propose)
    monkeypatch.setattr(stream, "propose", actual_propose)
    example["signal"]["available"][:] = False
    result = stream.simulate(model, example["signal"], None, {"meta": (b.detach(), "real")}, cached=example["pack"])
    arrays, info = export_traces(result["tapes"], result["actions"])
    for event in result["tapes"]["meta"]:
        assert event["observed"] == 0 and torch.equal(event["before"], event["after"])
        assert event["receipt"]["reason"] == "no_target" and event["receipt"]["evaluations"] == 0
        assert "gradient256" not in event["receipt"]
    assert all(not row["gradient_computed"] for row in info["updates"]["meta"])
    assert not arrays["meta_gradient"].any()


def test_no_target_core_branch_preserves_earlier_coefficient_graph():
    b = torch.ones(25, 23, requires_grad=True)
    state = torch.ones(4, 64) * b[0, 0]
    truth = torch.zeros(1, 25, 23)
    after, receipt = core.propose(
        None, {}, state, truth, torch.zeros_like(truth, dtype=torch.bool), b, create_graph=True
    )
    (gradient,) = torch.autograd.grad(after.sum(), b)
    assert gradient[0, 0] == 256
    assert receipt["reason"] == "no_target" and receipt["evaluations"] == 0
    assert not receipt["accepted"] and "gradient256" not in receipt


def test_existing_output_fails_before_cuda_or_input_loading(monkeypatch):
    monkeypatch.setattr(prepare, "configure", lambda: pytest.fail("no CUDA setup"), raising=False)
    monkeypatch.setattr(prepare, "load_inputs", lambda: pytest.fail("no input loading"), raising=False)
    with pytest.raises(FileExistsError):
        prepare.main(Path("."), preflight=False)
