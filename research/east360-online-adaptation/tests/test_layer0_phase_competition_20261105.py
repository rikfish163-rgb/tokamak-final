from unittest.mock import Mock

import numpy as np
import torch

import layer0_local_memory_20261105 as memory
from run_layer0_phase_competition_20261105 import phase_learn
from verify_layer0_query_memory_20261105 import expected_early_anchors


def test_verifier_uses_actual_anchor_times_and_does_not_fill_missing_positions():
    clock = np.arange(500, dtype=np.float64) / 1000
    assert expected_early_anchors(clock, 0.699, 3, "all").tolist() == [135, 167, 199]
    assert expected_early_anchors(clock, 0.699, 2, "matched").tolist() == [135, 199]
    assert not len(expected_early_anchors(clock, 0.699, 4, "matched"))


def test_phase_policy_keeps_safe_labels_only_as_guards():
    delegate = Mock(side_effect=AssertionError("safe query must not reach optimizer"))
    assert phase_learn(delegate, "all")(
        None, (9, None, (None, False, None)), (), (), 0.5, {}, None, 2
    ) == (None, {"decision": "guard_only_safe_outcome"})
    delegate.assert_not_called()


def test_phase_policy_records_no_early_bank_instead_of_guessing():
    t = np.arange(250, dtype=np.float64) / 1000
    y = np.zeros(len(t))
    delegate = Mock(side_effect=AssertionError("cannot fit without historical bank"))
    case = (None, None, torch.zeros(4, 192))
    _, note = phase_learn(delegate, "all")(
        None,
        (7, case, (t, True, 0.23)),
        (),
        (),
        0.5,
        {"both": y, "inter": y},
        None,
        2,
    )
    assert note["decision"] == "no_eligible_early_reference"
    assert note["early_anchor_indices"] == []
    delegate.assert_not_called()


def test_fitting_callable_is_scoped_to_one_query_and_actual_anchors(monkeypatch):
    import run_layer0_phase_competition_20261105 as driver

    times = np.arange(950, dtype=np.float64) / 1000
    bank = torch.arange(26 * 192, dtype=torch.float32).reshape(26, 192)
    trace = np.zeros(len(times))
    sentinel = object()

    def probe(*args):
        early = args[-1]
        actual_early = 135 + np.arange(6) * 32
        torch.testing.assert_close(early, bank[(actual_early - 135) // 32])
        return sentinel, {"steps": 20}

    def delegate(base, case, guard, episodes, bandwidth, sealed, output, block):
        assert memory.fit_episode is not previous
        q = driver.protocol.teaching_indices(
            times, sealed, True, 0.8, driver.r.THRESHOLD
        )
        out, fit = memory.fit_episode(
            base, None, None, bank, episodes, bandwidth, q, True
        )
        return out, {"fit": fit}

    previous = memory.fit_episode
    monkeypatch.setattr(driver.kernel, "fit_phase", probe)
    fit, receipt = phase_learn(delegate, "all")(
        None,
        (17, (None, None, bank), (times, True, 0.8)),
        (),
        (),
        0.5,
        {"both": trace, "inter": trace},
        None,
        2,
    )
    assert fit is sentinel
    assert receipt["early_anchor_indices"] == [135, 167, 199, 231, 263, 295]
    assert memory.fit_episode is previous
