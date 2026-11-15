import numpy as np

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.event_conditional_feedback_20261102 import (
    context_fit,
)


def test_event_context_rebinds_cached_outcome_to_hdf_without_changing_features(
    monkeypatch,
):
    data = dict(
        shot=np.array([11, 11, 12]),
        outcome=np.array([0, 0, 1]),
        target=np.array([2, 2, 3]),
        raw=np.ones((3, 115)),
    )
    calls = []

    def outcome(device, sid):
        calls.append((device, sid))
        return sid == 11, 2.0 if sid == 11 else None

    monkeypatch.setattr(io23, "read_outcome", outcome)
    updated, receipt = context_fit.hdf_authoritative_outcomes(data)
    np.testing.assert_array_equal(updated["outcome"], [1, 1, 0])
    np.testing.assert_array_equal(data["outcome"], [0, 0, 1])
    for name in ("shot", "target", "raw"):
        np.testing.assert_array_equal(updated[name], data[name])
    assert calls == [("J-TEXT", 11), ("J-TEXT", 12)]
    assert receipt["changed_rows"] == 3
    assert receipt["changed_shots"] == [11, 12]
    assert receipt["authority"] == "HDF5 meta/IsDisrupt"


def test_event_probe_extract_no_longer_copies_csv_hard_outcome(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    from experiments.risk_aligned_online import event_prefix_probe

    monkeypatch.setattr(
        event_prefix_probe,
        "read_events",
        lambda _: (
            {11: [event_prefix_probe.Event(11, 1, 2, 0.0, 0.003)]},
            {11: {"outcome": 0}},
        ),
    )
    monkeypatch.setattr(
        event_prefix_probe.simple_data, "source_roles", lambda: {"train": [11]}
    )
    monkeypatch.setattr(
        event_prefix_probe.simple_data,
        "read_shot",
        lambda *_: dict(
            values=np.ones((3, 23), np.float32),
            available=np.ones((3, 23), bool),
            times=np.array([0.0, 0.001, 0.002]),
        ),
    )
    monkeypatch.setattr(io23, "read_outcome", lambda *_: (True, 1.0))
    monkeypatch.setattr(
        event_prefix_probe,
        "frozen_features",
        lambda _, x, a: (torch.zeros(len(x), 192), torch.zeros(len(x), 1)),
    )
    monkeypatch.setattr(event_prefix_probe, "sha256", lambda _: "fake_fixture_hash")
    model = SimpleNamespace(
        cfg=SimpleNamespace(window=4), center=torch.zeros(23), scale=torch.ones(23)
    )
    cache = event_prefix_probe.extract_cache(
        model, None, tmp_path, stride=1, batch_size=4, device="cpu"
    )
    np.testing.assert_array_equal(cache["outcome"], [1, 1, 1])
    np.testing.assert_array_equal(cache["target"], [2, 2, 2])
