"""CPU-only contracts for the explicitly retrospective endpoint diagnostic."""

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from .. import io23, simple_data
from ..fast_update import normalized_step
from ..simple23 import Config, Simple23
from . import fast_capacity_probe as probe
from .embedding_model import EmbeddingModel


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class ToySource(Simple23):
    """Keep the real wrapper/fast-origin API, with an analytical CPU logit."""

    def __init__(self, *, label=None, saturated=False, constant=False):
        super().__init__(Config(fast_site="tokens", fast_rule="normalized"),
                         torch.zeros(23), torch.ones(23))
        self.curved_label = label
        self.saturated = saturated
        self.constant = constant
        with torch.no_grad():
            self.fast_origin.copy_(torch.linspace(-.2, .3, 128))
            self.risk_head.bias.fill_(0.)

    def forward(self, values, available, fast=None):
        delta = fast - self.initial_fast(len(values))
        if self.constant:
            logit = self.risk_head.bias.expand(len(values))
        elif self.curved_label is not None:
            sign = 1. if self.curved_label else -1.
            logit = -sign * 200. * (delta[:, 0] - .025).square()
        else:
            logit = delta[:, 0] + self.risk_head.bias
            if self.saturated:
                logit = logit + 80.
        return {"logit": logit, "forecast": values.new_zeros(len(values), 25, 23)}


def case(**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        model = EmbeddingModel(ToySource(**kwargs)).eval().requires_grad_(False)
    values = torch.zeros(1, 128, 23)
    available = torch.ones_like(values, dtype=torch.bool)
    return model, values, available


@pytest.mark.parametrize("label", [0, 1])
def test_label_direction_fixed_update_projection_and_threshold(label):
    model, values, available = case()
    threshold = .4 if label == 0 else .6
    result = probe.probe_point(model, values, available, label, threshold)
    assert result["diagnostic"] == "POSTHOC_ORACLE_POINT_DIAGNOSTIC"
    assert result["step_size"] == .10 and result["l2_radius"] == 1.
    assert result["completed_steps"] == 32
    assert len(result["trace"]) == 33
    assert result["initial_probability"] == pytest.approx(.5)
    direction = 1 if label else -1
    assert direction * result["best_logit_change"] > .99
    assert direction * (result["best_probability"] - .5) > .23
    assert result["best_threshold_reached"]
    assert result["final_threshold_reached"]
    assert result["trace"][0]["fast_norm"] == 0.
    assert result["trace"][1]["fast_norm"] == pytest.approx(.1, abs=1e-7)
    assert all(row["fast_norm"] <= 1. + 1e-7 for row in result["trace"])
    # Exact source update, including its epsilon and projection, not an SGD alias.
    current = torch.zeros(1, 128)
    gradient = torch.zeros_like(current)
    gradient[0, 0] = -direction
    for row in result["trace"][1:]:
        current = normalized_step(current, torch.zeros_like(current), gradient,
                                  step=.10, radius=1.)
        assert row["fast_norm"] == pytest.approx(float(current.double().norm()), abs=1e-7)
    assert result["immutability"]["risk_parameters_bitwise_unchanged"]


def test_source_and_wrapper_are_immutable_including_registration_and_grads():
    model, values, available = case()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    registered = {name: (tensor, tensor.data_ptr(), tensor._version)
                  for name, tensor in [*model.named_parameters(), *model.named_buffers()]}
    original_values, original_available = values.clone(), available.clone()
    probe.probe_point(model, values, available, 1, .6)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.reshape(-1).view(torch.uint8),
                           before[name].reshape(-1).view(torch.uint8))
    after = dict([*model.named_parameters(), *model.named_buffers()])
    for name, (tensor, pointer, version) in registered.items():
        assert after[name] is tensor and after[name].data_ptr() == pointer
        assert after[name]._version == version
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not module.training for module in model.modules())
    assert torch.equal(values, original_values) and torch.equal(available, original_available)


@pytest.mark.parametrize("label", [0, 1])
def test_best_includes_initial_and_need_not_equal_final(label):
    model, values, available = case(label=label)
    result = probe.probe_point(model, values, available, label, .5, steps=1)
    assert result["best_step"] == 0
    assert result["best_probability"] == result["initial_probability"]
    assert result["best_logit_change"] == 0.
    assert result["best_fast_norm"] == 0.
    assert result["final_probability"] != result["best_probability"]
    sign = 1 if label else -1
    assert sign * result["final_logit_change"] < 0.


def test_best_is_signed_logit_not_rounded_probability():
    model, values, available = case(saturated=True)
    result = probe.probe_point(model, values, available, 1, .6, steps=2)
    assert all(row["probability"] == 1. for row in result["trace"])
    assert result["best_step"] == 2 and result["best_logit_change"] > .19


@pytest.mark.parametrize("constant,steps", [(False, 0), (True, 32)])
def test_initial_only_and_fast_independent_logit(constant, steps):
    model, values, available = case(constant=constant)
    result = probe.probe_point(model, values, available, 0, .5, steps=steps)
    assert result["completed_steps"] == 0 and len(result["trace"]) == 1
    assert result["initial_probability"] == result["best_probability"] == result["final_probability"]
    # >= is an alarm: label 0 at equality has NOT reached its endpoint target.
    assert not result["best_threshold_reached"]


@pytest.mark.parametrize("field,value", [
    ("label", -1), ("label", 2), ("label", .5), ("label", True),
    ("threshold", float("nan")), ("threshold", -.1), ("threshold", 1.1),
    ("steps", -1), ("steps", 33), ("steps", 1.5), ("steps", True),
])
def test_invalid_scalar_inputs(field, value):
    model, values, available = case()
    kwargs = {"label": 1, "threshold": .6, "steps": 32, field: value}
    with pytest.raises(ValueError):
        probe.probe_point(model, values, available, **kwargs)


@pytest.mark.parametrize("bad", ["batch", "length", "channels", "dtype", "mask", "empty", "nan"])
def test_invalid_windows(bad):
    model, values, available = case()
    if bad == "batch":
        values, available = values.repeat(2, 1, 1), available.repeat(2, 1, 1)
    elif bad == "length":
        values, available = values[:, :127], available[:, :127]
    elif bad == "channels":
        values, available = values[:, :, :22], available[:, :, :22]
    elif bad == "dtype":
        values = values.double()
    elif bad == "mask":
        available = available.float()
    elif bad == "empty":
        available.fill_(False)
    else:
        values[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        probe.probe_point(model, values, available, 1, .6)


@pytest.mark.parametrize("bad", ["unfrozen", "training", "raw_rule", "not_wrapper"])
def test_only_frozen_eval_normalized_wrapper_is_accepted(bad):
    model, values, available = case()
    if bad == "unfrozen":
        model.delta_bias.requires_grad_(True)
    elif bad == "training":
        model.train()
    elif bad == "raw_rule":
        from dataclasses import replace
        model.base.cfg = replace(model.cfg, fast_rule="raw")
    else:
        model = model.base
    with pytest.raises(ValueError):
        probe.probe_point(model, values, available, 1, .6)


def test_bitwise_guard_detects_signed_zero_parameter_mutation():
    model, values, available = case()
    with torch.no_grad():
        model.base.risk_head.bias.fill_(-0.)
    original = model.base.forward

    def mutate(*args, **kwargs):
        with torch.no_grad():
            model.base.risk_head.bias.fill_(0.)
        return original(*args, **kwargs)

    model.base.forward = mutate
    with pytest.raises(AssertionError, match="immutability"):
        probe.probe_point(model, values, available, 1, .6, steps=0)


def trace_and_outcome(disruptive=True):
    times = np.asarray([0., .050, .125, .130, .131, .156])
    trace = dict(shot=54017, times=times, scores=np.asarray([.99, .9, .4, .98, .5, .99]))
    y, valid = io23.labels_at(times, disruptive, .160 if disruptive else None, 125.)
    outcome = dict(shot=54017, device="EAST", times=times, is_disrupt=disruptive,
                   down=.160 if disruptive else None, horizon=125., y=y, valid=valid)
    return trace, outcome


def test_disruptive_endpoint_uses_strict_timely_positive_interval_only():
    trace, outcome = trace_and_outcome()
    point = probe.select_endpoint(trace, outcome)
    assert point["end_index"] == 1 and point["label"] == 1
    assert point["lead_ms"] == 110. and point["sealed_probability"] == .9
    # Exactly 125 ms is eligible; exactly 30 ms is not. Ties use earliest index.
    trace["times"] = outcome["times"] = np.asarray([.035, .050, .125, .130, .131, .156])
    trace["scores"][0] = .9
    outcome["y"], outcome["valid"] = io23.labels_at(trace["times"], True, .160, 125.)
    assert probe.select_endpoint(trace, outcome)["end_index"] == 0


def test_safe_endpoint_uses_only_valid_maximum_and_no_lead():
    trace, outcome = trace_and_outcome(False)
    outcome["valid"][0] = outcome["valid"][-1] = False
    point = probe.select_endpoint(trace, outcome)
    assert point["end_index"] == 3 and point["label"] == 0
    assert point["lead_ms"] is None


def test_missing_timely_interval_is_an_explicit_failure():
    trace, outcome = trace_and_outcome()
    outcome["valid"].fill(False)
    with pytest.raises(ValueError, match="eligible"):
        probe.select_endpoint(trace, outcome)


def memory_run(monkeypatch):
    """Mock the existing seal/summary schema; create no on-disk run or output."""
    root = Path("/posthoc-test-static-run")
    threshold = .6
    predictions = [dict(shot=54017, is_disrupt=True, alarm=False, timely=False, lead_ms=None)]
    summary = dict(arm="static", observed_count=1, roster_count=1,
                   metrics=dict(threshold=threshold, predictions=predictions))
    protocol = dict(persistent_site="embedding", source_offline_only=True,
                    source_gate=dict(qualified=True), final_score_correction_head=False,
                    alarm_tuning=False, east_label_ms=125., threshold=threshold,
                    args=dict(limit=0), source_sha256="a" * 64, prior_sha256="b" * 64,
                    model_config=Config(fast_site="tokens", fast_rule="normalized").to_dict())
    receipt_path = root / "static/0001_54017_seal.json"
    receipt = dict(shot=54017, arm="static", threshold=threshold,
                   npz="0001_54017_sealed.npz", sha256="c" * 64)
    docs = {root / "protocol.json": protocol, root / "summary.json": {"static": summary},
            root / "static/summary.json": summary, root / "order.json": [dict(shot=54017)],
            receipt_path: receipt, root / "static/0001_54017_audit.json":
            dict(shot=54017, intra=False, persistent=None, prediction=predictions[0],
                 seal_sha256="c" * 64)}
    archive = dict(shot=np.asarray(54017), times=np.asarray([0., .05, .125]),
                   scores=np.asarray([.5, .5, .5], np.float32))
    monkeypatch.setattr(probe, "_read_json", lambda path: deepcopy(docs[Path(path)]))
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter([receipt_path]))
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(probe, "_hash", lambda path: "c" * 64 if str(path).endswith(".npz") else "d" * 64)
    monkeypatch.setattr(probe.np, "load", lambda *args, **kwargs: nullcontext(archive))
    return root, docs, archive


def test_completed_static_loader_checks_hash_identity_and_outcome_summary(monkeypatch):
    root, _, _ = memory_run(monkeypatch)
    loaded = probe.load_completed_static(root, [54017])
    assert loaded["threshold"] == .6 and len(loaded["records"]) == 1
    assert loaded["records"][0]["prediction"]["is_disrupt"] is True
    monkeypatch.setattr(probe, "_hash", lambda path: "e" * 64)
    with pytest.raises(ValueError, match="SHA"):
        probe.load_completed_static(root, [54017])


@pytest.mark.parametrize("bad", ["incomplete", "identity", "threshold", "nonstatic", "missing_shot"])
def test_completed_static_loader_rejects_bad_artifacts(monkeypatch, bad):
    root, docs, archive = memory_run(monkeypatch)
    ids = [54017]
    if bad == "incomplete":
        docs[root / "order.json"].append(dict(shot=54244))
    elif bad == "identity":
        archive["shot"] = np.asarray(54244)
    elif bad == "threshold":
        docs[root / "static/0001_54017_seal.json"]["threshold"] = .61
    elif bad == "nonstatic":
        docs[root / "static/0001_54017_audit.json"]["intra"] = True
    else:
        ids = [54244]
    with pytest.raises(ValueError):
        probe.load_completed_static(root, ids)


def test_cli_mock_reconstructs_window_freezes_all_and_never_writes_model(monkeypatch):
    from . import run

    model, _, _ = case()
    trace, outcome = trace_and_outcome()
    trace["scores"][:] = .5
    shot = dict(outcome, values=np.zeros((6, 23), np.float32),
                available=np.ones((6, 23), bool))
    loaded = dict(protocol=dict(source_sha256="a" * 64, prior_sha256="b" * 64,
                                model_config=model.cfg.to_dict()), threshold=.6,
                  records=[dict(trace=trace, prediction=dict(is_disrupt=True, alarm=False, timely=False))],
                  input_sha256={})
    events = []
    captured = []
    monkeypatch.setattr(probe, "load_completed_static", lambda *args: (events.append("seals") or loaded))
    monkeypatch.setattr(probe, "_hash", lambda path: "a" * 64 if str(path).endswith("source.pt") else "b" * 64)
    monkeypatch.setattr(run, "load_parent", lambda *args, **kwargs:
                        (model, dict(threshold=.6, model_config=model.cfg.to_dict()),
                         dict(qualified=True, checkpoint_sha256="a" * 64), None))
    monkeypatch.setattr(simple_data, "read_shot", lambda device, sid: shot)
    monkeypatch.setattr(simple_data, "attach_outcome", lambda *args:
                        (events.append("outcome") or outcome))
    monkeypatch.setattr(probe, "write_report", lambda path, report: captured.append((path, report)))
    monkeypatch.setattr(Path, "exists", lambda self: False)
    result = probe.main(["--run", "/static-run", "--checkpoint", "/source.pt",
                         "--prior", "/prior.pt", "--shots", "54017", "--device", "cpu",
                         "--steps", "2", "--output", "/user-output.json"])
    assert events == ["seals", "outcome"]
    assert len(captured) == 1 and captured[0][0] == Path("/user-output.json")
    assert result["diagnostic"] == "POSTHOC_ORACLE_POINT_DIAGNOSTIC"
    assert result["online_metrics_reported"] is False
    assert result["points"][0]["static_zero_delta_matches_seal"]
    assert result["points"][0]["initial_probability"] == .5
    assert result["points"][0]["end_index"] == 1
    assert all(not p.requires_grad for p in model.parameters())
