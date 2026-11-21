from types import SimpleNamespace

import numpy as np
import torch

from experiments.risk_aligned_online.critical_shot_meta_20261102 import stream


def test_sealed_points_are_not_revised_by_matured_updates_or_later_suffix(monkeypatch):
    model = SimpleNamespace(base=SimpleNamespace(center=torch.zeros(1)))

    def encode(_model, signal, ends, _basis):
        return {
            "index": torch.tensor(ends),
            "value": torch.tensor(signal["values"][ends, 0]),
        }

    def forward(_model, pack, state):
        return dict(
            logit=pack["value"] * 0.001 + state[0, 0],
            rms=torch.zeros(len(pack["index"])),
            capped=torch.zeros(len(pack["index"]), dtype=torch.bool),
        )

    def auxiliary(_model, pack, state, head):
        return torch.zeros(1, 25, 23) + pack["index"][0] + state[0, 0]

    def propose(_model, point, state, truth, hold, mask, head):
        origin = int(point["index"][0])
        assert float(truth[0, -1, 0]) == origin + 25
        assert float(hold[0, 0, 0]) == origin
        changed = state.clone()
        changed[0, 0] += 1
        return changed, dict(accepted=True)

    for name, function in [
        ("encode", encode),
        ("forward", forward),
        ("auxiliary", auxiliary),
        ("propose", propose),
    ]:
        monkeypatch.setattr(stream, name, function)
    values = np.repeat(np.arange(448, dtype=np.float32)[:, None], 23, axis=1)
    signal = dict(values=values, available=np.ones_like(values, dtype=bool))
    heads = {"cold": (None, "real"), "active": (torch.zeros(25, 23), "real")}
    full = stream.simulate(model, signal, None, heads)
    shift = full["logits"]["active"] - full["logits"]["cold"]
    np.testing.assert_allclose(
        shift[[159, 160, 287, 288, 415, 416]], [0, 1, 1, 2, 2, 3], atol=3e-7
    )
    assert [
        (t["origin"], t["maturity"], t["first_affected"])
        for t in full["tapes"]["active"]
    ] == [(134, 159, 160), (262, 287, 288), (390, 415, 416)]
    prefix = {k: v[:416].copy() for k, v in signal.items()}
    truncated = stream.simulate(model, prefix, None, heads)
    for arm in heads:
        np.testing.assert_array_equal(
            full["logits"][arm][:416], truncated["logits"][arm]
        )
    # Only perturb rows after all tested sealed points. Even the source
    # representation is allowed to see new rows once their own time arrives.
    changed = {k: v.copy() for k, v in signal.items()}
    changed["values"][416:] *= 2
    repeated = stream.simulate(model, changed, None, heads)
    for arm in heads:
        np.testing.assert_array_equal(
            full["logits"][arm][:416], repeated["logits"][arm][:416]
        )
