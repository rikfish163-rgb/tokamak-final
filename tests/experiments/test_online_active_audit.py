from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "experiments/transfer_methods_2026/online_active_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("online_active_audit", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_causal_innovations_do_not_read_a_modified_future() -> None:
    module = load_module()
    names = module.RETAINED_NAMES
    values = np.zeros((40, len(names)), dtype=np.float64)
    available = np.ones_like(values, dtype=bool)
    values[:, names.index("ip_A")] = np.linspace(1.0e5, 8.0e4, len(values))
    values[:, names.index("Bt")] = 2.0
    values[:, names.index("v_loop")] = np.linspace(0.2, 0.8, len(values))
    values[:, names.index("dx_a")] = np.linspace(0.0, 0.1, len(values))
    values[:, names.index("dy_a")] = np.linspace(0.0, 0.2, len(values))

    original = module.causal_innovations(values, available, sample_rate_hz=1000.0, lag=8)
    changed = values.copy()
    changed[25:] = 1.0e9
    modified = module.causal_innovations(changed, available, sample_rate_hz=1000.0, lag=8)

    for name in original.metrics:
        np.testing.assert_array_equal(original.metrics[name][:25], modified.metrics[name][:25])
        np.testing.assert_array_equal(original.available[name][:25], modified.available[name][:25])


def test_physics_support_is_unknown_with_fewer_than_two_families() -> None:
    module = load_module()
    family_scores = {
        "electrical": np.asarray([0.8, 0.7]),
        "position": np.asarray([0.0, 0.0]),
    }
    family_available = {
        "electrical": np.asarray([True, True]),
        "position": np.asarray([False, False]),
    }

    support, known, count = module.combine_family_scores(
        family_scores, family_available, activity_threshold=0.5, min_families=2
    )

    np.testing.assert_array_equal(known, np.asarray([False, False]))
    np.testing.assert_array_equal(count, np.asarray([1, 1]))
    np.testing.assert_array_equal(support, np.asarray([0.0, 0.0]))


def test_query_components_separate_fp_and_fn_contradictions() -> None:
    module = load_module()
    threshold = 0.25
    fp_like = module.query_components(
        risk=np.asarray([0.8] * 200),
        physics_support=np.asarray([0.05] * 200),
        physics_known=np.asarray([True] * 200),
        threshold=threshold,
    )
    fn_like = module.query_components(
        risk=np.asarray([0.02] * 180 + [0.08] * 20),
        physics_support=np.asarray([0.1] * 180 + [0.95] * 20),
        physics_known=np.asarray([True] * 200),
        threshold=threshold,
    )

    assert fp_like["fp_contradiction"] > 0.7
    assert fp_like["fp_contradiction"] > fp_like["fn_contradiction"]
    assert fn_like["fn_contradiction"] > 0.7
    assert fn_like["fn_contradiction"] > fn_like["fp_contradiction"]


def test_block_budget_selector_uses_exact_causal_quota() -> None:
    module = load_module()
    selector = module.BlockBudgetSelector(cutoff=0.9, block_size=20, quota=4)
    decisions = [selector.decide(score=0.0, stream_index=i) for i in range(40)]

    assert [i for i, selected in enumerate(decisions[:20]) if selected] == [16, 17, 18, 19]
    assert [i for i, selected in enumerate(decisions[20:], 20) if selected] == [36, 37, 38, 39]
    assert selector.total_selected == 8


def test_causal_ledger_selector_consumes_one_current_shot_at_a_time() -> None:
    module = load_module()
    ledger = [
        {"stream_index": index, "shot": 100 + index, "query_score": score}
        for index, score in enumerate([0.1, 0.9, 0.2, 0.3])
    ]
    selector = module.CausalLedgerQuerySelector(
        ledger, cutoff=0.8, block_size=4, quota=1
    )

    decisions = [selector.decide(shot=100 + index, stream_index=index) for index in range(4)]

    assert decisions == [False, True, False, False]
    assert selector.selected_shots == [101]
    assert selector.consumed == 4


def test_causal_ledger_selector_rejects_future_access_and_label_columns() -> None:
    module = load_module()
    with pytest.raises(ValueError, match="label-bearing"):
        module.CausalLedgerQuerySelector(
            [{"stream_index": 0, "shot": 10, "query_score": 0.5, "is_disrupt": True}],
            cutoff=0.5, block_size=1, quota=1,
        )

    selector = module.CausalLedgerQuerySelector(
        [
            {"stream_index": 0, "shot": 10, "query_score": 0.5},
            {"stream_index": 1, "shot": 20, "query_score": 0.5},
        ],
        cutoff=0.5, block_size=2, quota=1,
    )
    with pytest.raises(RuntimeError, match="next causal ledger row"):
        selector.decide(shot=20, stream_index=1)


def test_outcome_vault_fails_closed_before_query_or_final_evaluation() -> None:
    module = load_module()
    vault = module.OutcomeVault({10: False, 20: True})

    with pytest.raises(PermissionError):
        vault.reveal_for_query(10, selected=False)
    assert vault.reveal_for_query(20, selected=True) is True
    with pytest.raises(PermissionError):
        vault.all_labels()
    vault.open_final_evaluation()
    assert vault.all_labels() == {10: False, 20: True}


def test_shot_outcome_vault_seals_downtime_together_with_label() -> None:
    module = load_module()
    vault = module.ShotOutcomeVault({
        10: {"is_disrupt": False, "down_time_s": None},
        20: {"is_disrupt": True, "down_time_s": 3.25},
    })

    with pytest.raises(PermissionError):
        vault.reveal_for_query(20, selected=False)
    assert vault.reveal_for_query(20, selected=True) == {
        "is_disrupt": True, "down_time_s": 3.25,
    }
    with pytest.raises(PermissionError):
        vault.all_outcomes()
    vault.open_final_evaluation()
    assert vault.all_outcomes()[10]["down_time_s"] is None


def test_lazy_outcome_oracle_reads_only_current_selected_shot_until_final() -> None:
    module = load_module()
    reads = []
    truth = {
        10: {"is_disrupt": False, "down_time_s": None},
        20: {"is_disrupt": True, "down_time_s": 3.25},
    }
    oracle = module.LazyShotOutcomeOracle([10, 20], lambda shot: reads.append(shot) or truth[shot])

    assert reads == []
    oracle.seal_current(10)
    oracle.discard_unqueried(10)
    assert reads == []
    oracle.seal_current(20)
    assert oracle.reveal_for_query(20, selected=True)["is_disrupt"] is True
    assert reads == [20]
    with pytest.raises(PermissionError):
        oracle.all_outcomes()
    oracle.open_final_evaluation()
    assert oracle.all_outcomes() == truth
    assert reads == [20, 10, 20]


def test_prequential_audit_sorts_shots_and_seals_before_reveal() -> None:
    module = load_module()
    events: list[tuple[str, int]] = []
    shots = [{"shot": 30}, {"shot": 10}, {"shot": 20}]
    vault = module.OutcomeVault({10: False, 20: True, 30: False})
    selector = module.BlockBudgetSelector(cutoff=0.0, block_size=3, quota=1)

    records = module.run_prequential_audit(
        shots=shots,
        predict=lambda shot: np.asarray([0.1 * (shot["shot"] // 10)]),
        summarize=lambda shot, risk: {"query_score": float(risk.max())},
        selector=selector,
        outcomes=vault,
        on_seal=lambda record: events.append(("seal", record["shot"])),
        on_reveal=lambda record: events.append(("reveal", record["shot"])),
    )

    assert [record["shot"] for record in records] == [10, 20, 30]
    selected = [record for record in records if record["queried"]]
    assert len(selected) == 1
    selected_shot = selected[0]["shot"]
    assert events.index(("seal", selected_shot)) < events.index(("reveal", selected_shot))


def test_confusion_and_query_capture_reports_error_enrichment() -> None:
    module = load_module()
    records = [
        {"shot": 1, "base_max_risk": 0.9, "queried": True},   # FP
        {"shot": 2, "base_max_risk": 0.8, "queried": False},  # TP
        {"shot": 3, "base_max_risk": 0.1, "queried": True},   # FN
        {"shot": 4, "base_max_risk": 0.2, "queried": False},  # TN
    ]
    result = module.confusion_and_query_capture(
        records, {1: False, 2: True, 3: True, 4: False}, threshold=0.5
    )

    assert result["confusion"] == {"TP": 1, "FP": 1, "FN": 1, "TN": 1}
    assert result["queried_by_group"] == {"TP": 0, "FP": 1, "FN": 1, "TN": 0}
    assert result["query_error_precision"] == 1.0
    assert result["error_enrichment_vs_stream"] == 2.0


def test_safe_local_residual_requires_two_fp_shots_and_exploration() -> None:
    module = load_module()
    residual = module.SafeLocalResidual(threshold=0.25, radius=1.0, max_logit_drop=1.0)
    context = np.asarray([0.0, 0.0])

    assert residual.correct(0.8, context) == 0.8
    assert residual.add_confirmed_fp(1, context, 0.8, exploration=False, protected_tp=[]) is True
    assert residual.correct(0.8, context) == 0.8
    assert residual.add_confirmed_fp(2, context, 0.7, exploration=True, protected_tp=[]) is True
    assert residual.correct(0.8, context) < 0.8
    assert residual.correct(0.2, context) == 0.2


def test_safe_local_residual_rejects_prototype_that_would_flip_known_tp() -> None:
    module = load_module()
    residual = module.SafeLocalResidual(threshold=0.25, radius=1.0, max_logit_drop=2.0)
    context = np.asarray([0.0, 0.0])
    assert residual.add_confirmed_fp(1, context, 0.9, exploration=True, protected_tp=[]) is True
    accepted = residual.add_confirmed_fp(
        2, context, 0.9, exploration=False,
        protected_tp=[(0.26, context)],
    )

    assert accepted is False
    assert residual.prototype_shots == [1]


def test_stratified_selector_reserves_base_and_auxiliary_slots() -> None:
    module = load_module()
    selector = module.StratifiedBlockBudgetSelector(
        block_size=10, base_quota=3, auxiliary_quota=1,
        base_cutoff=1.0, auxiliary_cutoff=1.0,
    )
    reasons = []
    for index in range(10):
        base = 2.0 if index < 3 else 0.0
        auxiliary = 2.0 if index in (0, 3) else 0.0
        reasons.append(selector.decide(
            base_score=base, auxiliary_score=auxiliary, stream_index=index
        ))

    assert reasons[:4] == ["base", "base", "base", "auxiliary"]
    assert sum(reason is not None for reason in reasons) == 4
    assert selector.total_selected == 4


def test_stratified_selector_forced_fill_keeps_exact_budget() -> None:
    module = load_module()
    selector = module.StratifiedBlockBudgetSelector(
        block_size=10, base_quota=3, auxiliary_quota=1,
        base_cutoff=99.0, auxiliary_cutoff=99.0,
    )
    reasons = [selector.decide(base_score=0.0, auxiliary_score=0.0, stream_index=i) for i in range(20)]

    assert sum(reason is not None for reason in reasons[:10]) == 4
    assert sum(reason is not None for reason in reasons[10:]) == 4
    assert selector.total_selected == 8
