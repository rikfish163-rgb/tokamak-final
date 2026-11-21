"""Outcome-free tests for the conditional replay roster contract."""

import numpy as np
import pytest

import conditional_confirmation_roster_20261105 as roster


def _record(n=2):
    return {
        "times": np.arange(n, dtype=float) * 0.001,
        "values": np.zeros((n, 23), dtype=np.float32),
        "available": np.zeros((n, 23), dtype=bool),
    }


def _synthetic_inputs():
    return {
        "full": {"shot_ids": [9, 8, 7, 6, 5, 4, 3, 2, 1]},
        "pilot": {"shot_ids": [9, 20]},
        "fixed360": {"test": {"disruption_shots": [8], "non_disruption_shots": [7]}},
        "paper131_stream": {"stream": {"shot_ids": [6]}},
        "support": {"train": {"disruption_shots": [5], "non_disruption_shots": [4]}},
        "formal51": {"D_stream": {"shot_ids": [3]}},
    }


def test_identity_subtraction_uses_every_specified_id_source_and_sorts():
    assert roster.extract_candidates(_synthetic_inputs()) == [1, 2]


def test_duplicate_full_ids_and_non_integer_ids_fail_closed():
    inputs = _synthetic_inputs()
    inputs["full"]["shot_ids"].append(1)
    with pytest.raises(ValueError, match="duplicate"):
        roster.extract_candidates(inputs)
    inputs = _synthetic_inputs()
    inputs["pilot"]["shot_ids"].append(2.5)
    with pytest.raises(ValueError, match="integer"):
        roster.extract_candidates(inputs)


def test_real_named_inputs_reproduce_exact_candidate_count():
    inputs = roster.load_input_payloads()
    candidates = roster.extract_candidates(inputs)
    assert len(inputs["full"]["shot_ids"]) == 2953
    assert len(candidates) == 835
    assert candidates == sorted(set(candidates))
    assert set(candidates).isdisjoint(roster.fixed_ids(inputs))


def test_proxy_cutoff_uses_all_first_100_parent_ids_including_fit():
    parent = {"order": [{"shot": 1000}] + [{"shot": x} for x in range(1, 100)]
              + [{"shot": 9999}]}
    assert roster.parent_cutoff(parent) == 1000
    assert roster.later_proxy_ids([2000, 1000, 1001], 1000) == [1001, 2000]
    with pytest.raises(ValueError, match="100"):
        roster.parent_cutoff({"order": parent["order"][:99]})


def test_real_parent_is_same_locked_360_ids():
    inputs = roster.load_input_payloads()
    roster.check_parent_matches_fixed(inputs)
    assert roster.parent_cutoff(inputs["parent"]) == 95930


def test_short_nonempty_all_masked_signal_is_valid():
    assert roster.signal_problem(_record(1)) is None


@pytest.mark.parametrize(
    ("alter", "reason"),
    [
        (lambda r: r.update(times=np.empty((1, 1))), "time grid must be one-dimensional"),
        (lambda r: r.update(times=np.array([])), "time grid must be nonempty"),
        (lambda r: r.update(times=np.array([0.0, np.nan])), "time grid contains non-finite values"),
        (lambda r: r.update(times=np.array([0.0, 0.0])), "time grid must be strictly increasing"),
        (lambda r: r.update(times=np.array([1.0, 0.0])), "time grid must be strictly increasing"),
        (lambda r: r.update(values=np.zeros((2, 22))), "values shape"),
        (lambda r: r.update(available=np.zeros((1, 23))), "available shape"),
    ],
)
def test_invalid_signal_contracts_report_the_reason(alter, reason):
    record = _record()
    alter(record)
    assert reason in roster.signal_problem(record)


def test_partition_accounts_for_every_candidate_and_keeps_exact_failure_reason():
    records = {1: _record(1), 2: _record(2), 3: _record(2), 4: _record(2)}
    records[2]["times"][1] = np.nan

    def load(sid):
        if sid == 3:
            raise ValueError("unaligned channel length: EAST 3 Bt")
        return records[sid]

    result = roster.screen_candidates(
        [4, 3, 2, 1], read_shot=load, hash_shot=lambda sid: f"{sid:064x}"
    )
    assert result["ids"] == [1, 4]
    assert result["excluded"] == [
        {"sid": 2, "reason": "time grid contains non-finite values"},
        {"sid": 3, "reason": "ValueError: unaligned channel length: EAST 3 Bt"},
    ]
    assert result["samples"] == {"1": 1, "4": 2}
    assert set(result["raw_sha256"]) == {"1", "2", "3", "4"}
    assert set(result["ids"]).isdisjoint(x["sid"] for x in result["excluded"])
    assert set(result["ids"]) | {x["sid"] for x in result["excluded"]} == {1, 2, 3, 4}


def test_unknown_failures_and_hash_errors_abort_instead_of_excluding():
    def broken(_sid):
        raise ValueError("unexpected implementation error")

    with pytest.raises(ValueError, match="unexpected implementation error"):
        roster.screen_candidates([1], read_shot=broken, hash_shot=lambda _: "digest")
    with pytest.raises(OSError, match="storage"):
        roster.screen_candidates(
            [1], read_shot=lambda _: _record(),
            hash_shot=lambda _: (_ for _ in ()).throw(OSError("storage")),
        )


def test_nominal_east_wrapper_never_calls_outcome_reader(monkeypatch):
    from experiments.risk_aligned_online import io23

    calls = []
    source = _record(1)

    def signals(device, sid, *, rate_contract):
        calls.append((device, sid, rate_contract))
        return {"v": source["values"], "a": source["available"],
                "times": source["times"], "planning_ip": np.zeros(1),
                "planning_Bt": np.zeros(1)}

    def forbidden(*_args, **_kwargs):
        pytest.fail("outcome reader must not be called")

    monkeypatch.setattr(io23, "read_signals", signals)
    monkeypatch.setattr(io23, "read_outcome", forbidden)
    assert roster.signal_problem(roster.read_nominal(42)) is None
    assert calls == [("EAST", 42, "nominal")]
