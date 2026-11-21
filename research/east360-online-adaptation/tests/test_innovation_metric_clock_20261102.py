from types import SimpleNamespace

import numpy as np
import torch

from experiments.risk_aligned_online.innovation_metric_20261102 import stream


def test_actual_clock_seals_and_never_revises_prior_points(monkeypatch):
    model = SimpleNamespace(base=SimpleNamespace(center=torch.zeros(1)))

    def encode(_model, signal, ends, _basis):
        return dict(
            index=torch.tensor(ends), value=torch.tensor(signal["values"][ends, 0])
        )

    def forward(_model, pack, state):
        return dict(
            logit=pack["value"] * 0.001 + state[0, 0],
            rms=torch.zeros(len(pack["index"])),
            capped=torch.zeros(len(pack["index"]), dtype=torch.bool),
        )

    def auxiliary(_model, pack, state, _head):
        return torch.zeros(1, 25, 23) + pack["index"][0] + state[0, 0]

    def update(_model, point, state, truth, hold, mask, _head, geometry):
        origin = int(point["index"][0])
        assert float(truth[0, -1, 0]) == origin + 25
        assert float(hold[0, 0, 0]) == origin
        assert geometry == "metric"
        changed = state.clone()
        changed[0, 0] += 1
        return changed, dict(accepted=True)

    for name, function in [
        ("encode", encode),
        ("forward", forward),
        ("auxiliary", auxiliary),
        ("update", update),
    ]:
        monkeypatch.setattr(stream, name, function)
    values = np.repeat(np.arange(448, dtype=np.float32)[:, None], 23, axis=1)
    signal = dict(values=values, available=np.ones_like(values, dtype=bool))
    heads = dict(
        cold=(None, "real", "metric"), active=(torch.zeros(25, 23), "real", "metric")
    )
    full = stream.simulate(model, signal, None, heads)
    shift = full["logits"]["active"] - full["logits"]["cold"]
    np.testing.assert_allclose(
        shift[[159, 160, 287, 288, 415, 416]], [0, 1, 1, 2, 2, 3], atol=3e-7
    )
    assert [
        (t["origin"], t["maturity"], t["first_affected"])
        for t in full["tapes"]["active"]
    ] == [(134, 159, 160), (262, 287, 288), (390, 415, 416)]
    truncated = stream.simulate(
        model, {k: v[:416].copy() for k, v in signal.items()}, None, heads
    )
    changed = {k: v.copy() for k, v in signal.items()}
    changed["values"][416:] *= 2
    perturbed = stream.simulate(model, changed, None, heads)
    for a in heads:
        np.testing.assert_array_equal(full["logits"][a][:416], truncated["logits"][a])
        np.testing.assert_array_equal(
            full["logits"][a][:416], perturbed["logits"][a][:416]
        )
