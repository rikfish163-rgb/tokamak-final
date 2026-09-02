from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grid_search as grid  # noqa: E402


def _shot(value: float, *, disruptive: bool) -> dict:
    # The value identifies which shot was selected by the sampler.
    return {
        "shot": int(value),
        "v": np.full((420, 17), value, dtype=np.float32),
        "a": np.ones((420, 17), dtype=bool),
        "is_disrupt": disruptive,
        "down": 0.299 if disruptive else None,
        "start": 0.0,
        "rate": 1000.0,
    }


def test_v3c_historical_sampler_is_shot_first_and_deterministic() -> None:
    """The registered V3C support sampler draws a shot before its class.

    A safe shot has no positive windows.  Therefore a class-first sampler and
    the historical shot-first sampler produce different, observable shot
    sequences even with the same seed.  This locks the parent-model lineage
    contract without depending on a large HDF5 fixture.
    """

    shots = [_shot(1.0, disruptive=True), _shot(2.0, disruptive=False)]
    sampler = grid.build_sampler(
        shots,
        None,
        125.0,
        0.0,
        seed=131,
        sampling_mode="shot_first",
    )

    observed = []
    labels = []
    for _ in range(8):
        window, label = sampler(0.5)
        observed.append(float(window[0, 0]))
        labels.append(float(label))

    # Independent reference implementation of the registered draw order.
    rng = np.random.default_rng(131)
    positive = [list(range(174, 300)), []]
    negative = [list(range(127, 174)), list(range(127, 420))]
    expected = []
    expected_labels = []
    while len(expected) < 8:
        index = int(rng.integers(len(shots)))
        if positive[index] and rng.random() < 0.5:
            endpoint = positive[index][int(rng.integers(len(positive[index])))]
            expected_labels.append(1.0)
        else:
            if not negative[index]:
                continue
            endpoint = negative[index][int(rng.integers(len(negative[index])))]
            expected_labels.append(0.0)
        expected.append(float(shots[index]["v"][endpoint, 0]))

    assert observed == pytest.approx(expected)
    assert labels == pytest.approx(expected_labels)


def test_unknown_sampler_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="sampling mode"):
        grid.build_sampler(
            [_shot(1.0, disruptive=True)],
            None,
            125.0,
            0.0,
            seed=131,
            sampling_mode="future_oracle",
        )
