from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.full_path_meta_20261102 import prepare


ARM = "full_operational"


def fake_example(monkeypatch, accepted=(True, False, False)):
    model = SimpleNamespace(base=SimpleNamespace(center=torch.zeros(23, dtype=torch.float64)))
    coefficient = torch.full((25, 23), 0.2, dtype=torch.float64, requires_grad=True)
    values = np.repeat(np.arange(448, dtype=np.float64)[:, None], 23, axis=1)
    example = dict(
        shot=1,
        times=np.arange(448) / 1000,
        disruptive=False,
        down=None,
        signal=dict(values=values, available=np.ones_like(values, bool)),
        pack=dict(index=torch.arange(448)),
    )

    def advance(_model, pack, state, truth, mask, b, *, create_graph):
        origin = int(pack["index"][0])
        step = (origin - 134) // 128
        assert int(truth[0, -1, 0]) == origin + 25
        assert mask.dtype == torch.bool and create_graph in (True, False)
        current = state.clone() if create_graph else state.detach().clone()
        current.requires_grad_(True)
        after = current + b[0, 0] * (step + 1) if accepted[step] else current
        receipt = dict(accepted=accepted[step], reason="improved" if accepted[step] else "not_improved")
        return (after if create_graph else after.detach()), receipt

    def forward(_model, pack, state):
        return dict(logit=pack["index"].to(torch.float64) * 0.001 + state[0, 0])

    def auxiliary(_model, _pack, state, b):
        return (state[0, 0] + b[0, 0]).expand(1, 25, 23)

    monkeypatch.setattr(prepare, "advance", advance)
    monkeypatch.setattr(prepare, "forward", forward)
    monkeypatch.setattr(prepare, "auxiliary", auxiliary)
    monkeypatch.setattr(prepare, "innovation_loss", lambda _m, p, _t, _a: p.square().mean())
    tapes, state = [], coefficient.new_zeros(4, 64)
    logits = []
    for lo in range(0, 448, 32):
        logits.append(forward(model, prepare.take(example["pack"], lo, lo + 32), state)["logit"].detach().numpy())
        end = lo + 31
        if end < 159 or (end - 159) % 128:
            continue
        origin = end - 25
        observed, available = prepare.targets(example["signal"], origin, end)
        before = state.clone()
        state, receipt = advance(
            model,
            prepare.take(example["pack"], origin, origin + 1),
            state,
            torch.tensor(observed["real"]),
            torch.tensor(available),
            coefficient.detach(),
            create_graph=False,
        )
        tapes.append(
            dict(
                origin=origin,
                maturity=end,
                first_affected=end + 1,
                before=before.detach(),
                after=state.detach(),
                receipt=receipt,
            )
        )
    simulation = dict(tapes={ARM: tapes}, logits={ARM: np.concatenate(logits)})
    return model, example, simulation, coefficient


def test_full_replay_retains_earlier_path_when_last_update_rejected(monkeypatch):
    model, example, simulation, b = fake_example(monkeypatch)
    result = prepare.replay_full_query(model, example, simulation, ARM, b, 416)
    assert len(result["receipts"]) == 3
    assert [r["accepted"] for r in result["receipts"]] == [True, False, False]
    assert torch.equal(result["state"].detach(), simulation["tapes"][ARM][-1]["after"])
    (gradient,) = torch.autograd.grad(result["after"], b, retain_graph=True)
    assert gradient[0, 0] == 1
    assert not result["before"].requires_grad
    assert float(result["after"].detach()) == float(simulation["logits"][ARM][416])


def test_diagnostic_regularizer_does_not_gain_full_state_gradient(monkeypatch):
    model, example, simulation, b = fake_example(monkeypatch, (True, True, False))
    result = prepare.replay_full_query(model, example, simulation, ARM, b, 416)
    (risk_gradient,) = torch.autograd.grad(result["after"], b, retain_graph=True)
    (signal_gradient,) = torch.autograd.grad(result["signal"], b)
    assert risk_gradient[0, 0] == 3
    assert float(signal_gradient[0, 0]) == pytest.approx(1.6)


def test_all_rejected_updates_have_no_coefficient_to_risk_path(monkeypatch):
    model, example, simulation, b = fake_example(monkeypatch, (False, False, False))
    result = prepare.replay_full_query(model, example, simulation, ARM, b, 416)
    (gradient,) = torch.autograd.grad(result["after"], b, allow_unused=True)
    assert gradient is None
    assert not any(r["accepted"] for r in result["receipts"])


@pytest.mark.parametrize("query,count", [(159, 0), (160, 1), (287, 1), (288, 2), (415, 2), (416, 3)])
def test_replay_uses_only_updates_strictly_before_fixed_query(monkeypatch, query, count):
    model, example, simulation, b = fake_example(monkeypatch)
    result = prepare.replay_full_query(model, example, simulation, ARM, b, query)
    assert result is None if not count else len(result["receipts"]) == count


@pytest.mark.parametrize("corruption", ["state", "receipt", "risk", "clock"])
def test_replay_fails_on_any_numeric_or_receipt_mismatch(monkeypatch, corruption):
    model, example, simulation, b = fake_example(monkeypatch)
    if corruption == "state":
        simulation["tapes"][ARM][0]["after"] = simulation["tapes"][ARM][0]["after"] + 0.01
    elif corruption == "receipt":
        simulation["tapes"][ARM][0]["receipt"]["reason"] = "changed"
    elif corruption == "risk":
        simulation["logits"][ARM][416] += 1e-5
    else:
        simulation["tapes"][ARM].pop(0)
    with pytest.raises(AssertionError):
        prepare.replay_full_query(model, example, simulation, ARM, b, 416)


def test_early_critical_query_is_not_replaced_and_uses_first_signal(monkeypatch):
    model, example, simulation, b = fake_example(monkeypatch)
    simulation["logits"][ARM][50] = 10
    objective, receipt = prepare.episode(model, example, simulation, ARM, b, 0.7)
    assert receipt["query"] == 50 and not receipt["supported"]
    assert receipt["risk_before"] == receipt["risk_after"]
    assert receipt["signal"] == pytest.approx(0.04)
    expected = 0.2 * b[0, 0].square() + 1e-4 * b.square().mean()
    torch.testing.assert_close(objective, expected)


def test_existing_output_rejected_before_gpu_setup(monkeypatch):
    monkeypatch.setattr(prepare, "configure", lambda: pytest.fail("no GPU setup"))
    with pytest.raises(FileExistsError):
        prepare.main(Path("."))


def test_snapshot_only_uses_entry_plan_and_imported_modules(monkeypatch):
    saved, archived = {}, []
    entry = Path(prepare.__file__)
    imported = entry.parent / "core.py"
    monkeypatch.setattr(
        prepare,
        "sys",
        SimpleNamespace(
            modules={
                "experiments.risk_aligned_online.full_path_meta_20261102.core": SimpleNamespace(__file__=str(imported)),
                "not_an_experiment": SimpleNamespace(__file__="/unrelated.py"),
            }
        ),
    )
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(Path, "glob", lambda *args: pytest.fail("no package glob"))
    monkeypatch.setattr(prepare, "sha", lambda path: "hash")
    monkeypatch.setattr(prepare, "write", lambda path, value: saved.update({path.name: value}))

    class Archive:
        def __init__(self, *args):
            assert args[1] == "x"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, path, name):
            archived.append(path)

    monkeypatch.setattr(prepare.zipfile, "ZipFile", Archive)
    prepare.snapshot(
        Path("/no-files-created"),
        dict(ids=list(range(20)), source_sha256="source", input_map_sha256="map"),
        Path("/cold"),
        "ENDED20_FULL_PATH_RISK_META",
    )
    assert set(archived) == {entry, imported, prepare.PLAN}
    protocol = saved["protocol.json"]
    assert protocol["scope"] == "ENDED20_FULL_PATH_RISK_META"
    assert protocol["heads"] == {ARM: "metric"}
    assert protocol["diagnostic_incoming_state_detached"] is True


def test_preflight_distinguishes_all_rejected_from_rejected_last(monkeypatch):
    model = SimpleNamespace(base=SimpleNamespace(center=torch.zeros(23)))
    tape = dict(before=torch.zeros(4, 64), after=torch.zeros(4, 64), receipt={})
    simulation = dict(logits={ARM: np.zeros(448)}, probability={ARM: np.zeros(448)}, tapes={ARM: [tape, tape, tape]})
    monkeypatch.setattr(prepare, "read_example", lambda *args: {"signal": {}, "pack": {}})
    monkeypatch.setattr(prepare, "simulate", lambda *args, **kwargs: simulation)

    def replay(model, example, simulation, arm, coefficient, query):
        count = 1 if query < 288 else (2 if query < 416 else 3)
        receipts = [dict(accepted=v) for v in (False, True, False)[:count]]
        after = coefficient[0, 0] * 2 if count > 1 else torch.tensor(0.0, requires_grad=True)
        return dict(after=after, receipts=receipts, maturity=159 + (count - 1) * 128)

    monkeypatch.setattr(prepare, "replay_full_query", replay)
    result = prepare.run_preflight(model, None, {}, {})
    assert [c["head_gradient_norm"] for c in result["checks"]] == [None, None, 2.0, 2.0, 2.0]
    assert result["checks"][-1]["last_accepted"] is False
    assert result["checks"][-1]["accepted_path_updates"] == 1


def test_training_uses_one_head_exact_128_paired_seeded_batches_without_io(monkeypatch):
    model = torch.nn.Module()
    model.base = torch.nn.Module()
    model.base.register_buffer("center", torch.zeros(23))
    ids, reads, saved, simulated_heads = list(range(1, 21)), [], {}, []
    parent = dict(ids=ids, source_sha256="source", input_map_sha256="map")
    monkeypatch.setattr(prepare, "configure", lambda: None)
    monkeypatch.setattr(prepare, "load_inputs", lambda: (model, {"threshold": 0.7}, Path("/cold"), parent, {}))
    monkeypatch.setattr(prepare, "source_basis", lambda model: None)
    monkeypatch.setattr(prepare, "snapshot", lambda *args: None)
    monkeypatch.setattr(prepare, "sha", lambda path: "hash")
    monkeypatch.setattr(prepare, "write", lambda path, value: saved.update({path.name: value}))

    def read_example(model, basis, mapping, parent, sid):
        reads.append(sid)
        return dict(
            shot=sid, disruptive=sid <= 8, down=1.0 if sid <= 8 else None, times=np.arange(2), signal={}, pack={}
        )

    def simulate(model, signal, basis, heads, *, cached):
        simulated_heads.append(heads)
        assert set(heads) == {ARM} and heads[ARM][1:] == ("real", "metric")
        assert heads[ARM][0].shape == (25, 23) and not heads[ARM][0].requires_grad
        return {}

    def episode(model, example, simulation, arm, coefficient, threshold):
        assert arm == ARM and threshold == 0.7
        return (coefficient[0, 0] - 1).square(), dict(supported=True)

    monkeypatch.setattr(prepare, "read_example", read_example)
    monkeypatch.setattr(prepare, "simulate", simulate)
    monkeypatch.setattr(prepare, "episode", episode)
    prepare.main(Path("/no-files-created-full-path-training"))
    assert reads == ids and len(simulated_heads) == 256
    history, done = saved["history.json"], saved["completed.json"]
    assert len(history) == done["steps"] == done["head_optimizer_steps"] == 128
    assert done["number_heads"] == 1 and done["suffix_shots_read"] == 0
    assert done["support_counts"] == {ARM: 256} and done["source_tensors_unchanged"]
    assert set(saved["heads.json"]) == {ARM}
    assert np.asarray(saved["heads.json"][ARM]).shape == (25, 23)
    assert [r["shot"] for r in saved["training_cohort.json"]] == ids
    rng = np.random.default_rng(131)
    for index, row in enumerate(history, 1):
        expected = [ids[int(rng.integers(8))], ids[8 + int(rng.integers(12))]]
        assert row["step"] == index and row["sample_shots"] == expected
        assert set(row["gradient_norms"]) == {ARM}
