import pytest

import run_layer0_query_memory_20261105 as core
from run_layer0_disruptive_feedback_20261105 import disruptive_feedback


def test_known_safe_outcome_is_reserved_for_guard_not_optimizer(monkeypatch):
    def fit(*args, **kwargs):
        pytest.fail("safe outcomes do not create proposals in this policy")

    monkeypatch.setattr(core, "learn", fit)
    call = disruptive_feedback(fit)
    new, receipt = call(None, (8, None, (None, False, None)))
    assert new is None
    assert receipt["decision"] == "guard_only_safe_outcome"


def test_disruptive_query_calls_original_learn_only_once():
    seen = []

    def original(*a, **k):
        seen.append((a, k))
        return ("state", {"decision": "accepted"})

    guarded = disruptive_feedback(original)
    c = (81, None, (None, True, 0.8))
    assert guarded(0, c, 1, 2) == ("state", {"decision": "accepted"})
    assert seen == [((0, c, 1, 2), {})]
    with pytest.raises(ValueError):
        guarded(0, (81, None, (None, None, None)))
