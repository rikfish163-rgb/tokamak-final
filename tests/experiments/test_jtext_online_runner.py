from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
MODULE = HERE / "run_jtext_online_active.py"
CONFIG = HERE / "paper_grade_package" / "configs" / "jtext_online_active_v1.json"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jtext_online_adapter as adapter
import jtext_online_io as jio


def load_runner():
    assert MODULE.is_file(), "J-TEXT online runner production module is missing"
    sys.modules.pop("run_jtext_online_active", None)
    return importlib.import_module("run_jtext_online_active")


def make_controller(*, seed: int = 41):
    torch.manual_seed(seed)
    residual = adapter.JTextOnlineResidual(
        feature_count=17, hidden_size=8, max_logit_shift=0.5,
    )
    return adapter.JTextOnlineController(
        residual,
        intra_lr=0.03,
        inter_lr=0.05,
        loss_weights={
            "next": 1.0,
            "teacher": 0.5,
            "risk_anchor": 0.1,
            "physics": 0.0,
            "supervised": 1.0,
            "replay": 0.5,
        },
    )


def make_signal(shot: int, *, suffix_after: int | None = None, delta: float = 0.0):
    rows = 6
    values = np.stack([
        np.linspace(0.05 * (channel + 1), 0.2 * (channel + 1), rows)
        for channel in range(17)
    ], axis=1).astype(np.float32)
    if suffix_after is not None:
        values[int(suffix_after) + 1 :] += np.float32(delta)
    availability = np.ones_like(values, dtype=bool)
    times = np.arange(rows, dtype=np.float64) / 1000.0
    return jio.JTextSignalShot(
        shot_id=shot,
        values=values,
        availability=availability,
        times_s=times,
        sample_rate_hz=1000.0,
        start_time_s=0.0,
    )


def source_trace(signal: jio.JTextSignalShot) -> np.ndarray:
    # Pointwise rebuilt17 source surrogate: endpoint k reads no row after k.
    return np.asarray(
        0.08 * signal.values[:, 0] + 0.01 * signal.values[:, 1],
        dtype=np.float32,
    )


class TrackingOracle:
    def __init__(
        self, roster: tuple[int, ...], outcomes: dict[int, jio.JTextRevealedOutcome],
        reads: list[int],
    ) -> None:
        self.roster = roster
        self.outcomes = outcomes
        self.reads = reads
        self.cursor = 0
        self.sealed: int | None = None

    def seal_current(self, shot_id: int, *, prediction_sha256: str) -> None:
        assert self.roster[self.cursor] == shot_id
        assert len(prediction_sha256) == 64
        assert self.sealed is None
        self.sealed = shot_id

    def reveal_for_query(
        self, shot_id: int, *, selected: bool,
    ) -> jio.JTextRevealedOutcome:
        assert selected is True
        assert self.sealed == shot_id
        self.reads.append(shot_id)
        outcome = self.outcomes[shot_id]
        self.sealed = None
        self.cursor += 1
        return outcome

    def discard_unqueried(self, shot_id: int) -> None:
        assert self.sealed == shot_id
        self.sealed = None
        self.cursor += 1


def outcomes(first_shot: int, second_shot: int, *, first_disruptive: bool = True):
    return {
        first_shot: jio.JTextRevealedOutcome(
            shot_id=first_shot,
            is_disrupt=first_disruptive,
            down_time_s=0.005 if first_disruptive else None,
            advanced_time_ms=70.0 if first_disruptive else None,
            positive_horizon_ms=70.0 if first_disruptive else None,
        ),
        second_shot: jio.JTextRevealedOutcome(
            shot_id=second_shot,
            is_disrupt=False,
            down_time_s=None,
            advanced_time_ms=None,
            positive_horizon_ms=None,
        ),
    }


def run_two_shots(
    module,
    tmp_path: Path,
    *,
    first_disruptive: bool = True,
    first_suffix_delta: float = 0.0,
    arm: str = "full_active",
    output_name: str = "run",
    stop_after: int | None = None,
    resume_state: dict[str, Any] | None = None,
    controller_seed: int = 41,
):
    first, second = 1051600, 1051601
    signals = {
        first: make_signal(first, suffix_after=1, delta=first_suffix_delta),
        second: make_signal(second),
    }
    reads: list[int] = []
    static_seen: list[bool] = []
    output = tmp_path / output_name

    def oracle_factory(remaining: tuple[int, ...]):
        static_seen.append((output / "static_seal_manifest.json").is_file())
        return TrackingOracle(
            remaining,
            outcomes(first, second, first_disruptive=first_disruptive),
            reads,
        )

    result = module.run_jtext_online_stream(
        ordered_shots=(first, second),
        signal_loader=signals.__getitem__,
        source_trace_builder=source_trace,
        oracle_factory=oracle_factory,
        controller=make_controller(seed=controller_seed),
        arm=arm,
        budget_percent=20,
        seed=131,
        output=output,
        query_block_size=2,
        query_quota=1,
        active_cutoff=0.0,
        stop_after=stop_after,
        resume_state=resume_state,
    )
    return result, reads, static_seen


def test_config_cli_and_arm_matrix_keep_diagnostic_and_source_isolation() -> None:
    module = load_runner()
    assert CONFIG.is_file()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert module.ARMS == (
        "static", "intra_only", "inter_random", "inter_active",
        "full_random", "full_active", "frontload_equal_labels",
        "equal_compute_random",
    )
    assert payload["schema_version"] == "jtext-online-active-v1"
    assert payload["claim_status"] == "TEST_DRIVEN_EXPLORATION"
    assert payload["research_claim"] == "JTEXT_TEST220_ORDERED_PREQUENTIAL_DIAGNOSTIC"
    assert payload["arms"] == list(module.ARMS)
    assert payload["budgets_percent"] == [10, 20, 30]
    assert payload["primary_budget"]["percent"] == 20
    assert payload["primary_budget"]["query_count"] == 44
    assert payload["source_isolation"]["east_transfer_parent"] == (
        "raw_static_jtext_source_checkpoint_only"
    )
    assert payload["source_isolation"]["jtext_online_state_transfer_allowed"] is False
    assert payload["forbidden_inputs"] == [
        "EAST_raw8", "EAST_support_normalization", "EAST_thresholds",
    ]

    args = module.build_parser().parse_args([
        "--source-checkpoint", "/tmp/source.pt",
        "--arm", "full_active",
        "--budget", "20",
        "--seed", "131",
        "--data-root", "/tmp/data",
        "--output", "/tmp/output",
    ])
    assert args.arm == "full_active"
    assert args.budget == 20
    assert args.seed == 131
    assert not any(name.startswith("east") for name in vars(args))


def test_query_value_is_exact_immutable_and_zero_novelty_before_reveal() -> None:
    module = load_runner()
    value = module.JTextQueryValue(
        w=0.7, eta=0.1,
        uncertainty_scale=1.0,
        contradiction_scale=1.0,
        novelty_scale=1.0,
    )
    memory = module.JTextRevealedPhenotypeMemory()

    receipt = value.evaluate(
        uncertainty=0.8,
        contradiction=0.2,
        phenotype=np.asarray([1.0, 1.0]),
        memory=memory,
    )

    assert receipt["uncertainty"] == pytest.approx(0.8)
    assert receipt["contradiction"] == pytest.approx(0.2)
    assert receipt["novelty"] == 0.0
    assert receipt["value"] == pytest.approx(0.7 * 0.8 + 0.3 * 0.2)
    with pytest.raises(FrozenInstanceError):
        value.w = 0.1
    with pytest.raises(FrozenInstanceError):
        value.eta = 0.9
    with pytest.raises(FrozenInstanceError):
        value.novelty_scale = 2.0


def test_query_novelty_reads_only_strictly_prior_revealed_phenotypes() -> None:
    module = load_runner()
    value = module.JTextQueryValue(
        w=0.5, eta=0.2,
        uncertainty_scale=1.0,
        contradiction_scale=1.0,
        novelty_scale=1.0,
    )
    memory = module.JTextRevealedPhenotypeMemory()
    current = np.asarray([1.0, 1.0])

    before = value.evaluate(
        uncertainty=0.4, contradiction=0.6, phenotype=current, memory=memory,
    )
    memory.add_after_reveal(10, np.asarray([0.0, 0.0]))
    after_prior_reveal = value.evaluate(
        uncertainty=0.4, contradiction=0.6, phenotype=current, memory=memory,
    )

    assert before["novelty"] == 0.0
    assert after_prior_reveal["novelty"] == pytest.approx(1.0)
    assert after_prior_reveal["value"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.6 + 0.2)
    with pytest.raises(RuntimeError, match="strictly increasing"):
        memory.add_after_reveal(9, current)


def test_budget_sensitivities_and_random_manifests_are_exact_and_label_free() -> None:
    module = load_runner()
    shots = tuple(range(220))
    assert {percent: module.budget_quota(percent) for percent in (10, 20, 30)} == {
        10: 2, 20: 4, 30: 6,
    }

    manifest = module.build_random_query_manifest(
        shots, budget_percent=20, seed=131,
    )

    assert len(manifest) == 220
    assert sum(row["selected"] for row in manifest) == 44
    for block in range(11):
        rows = manifest[block * 20 : (block + 1) * 20]
        assert sum(row["selected"] for row in rows) == 4
    assert all(
        set(row) == {"stream_index", "shot", "block", "selected", "route"}
        for row in manifest
    )
    forbidden = {"label", "outcome", "is_disrupt", "DownTime", "AdvancedTime"}
    assert all(forbidden.isdisjoint(row) for row in manifest)


def test_c_mut_05_realized_220_active_random_ledgers_match_all_compute_inputs() -> None:
    module = load_runner()
    shots = tuple(range(220))
    scores = np.linspace(0.01, 0.99, 220)
    active = module.build_active_query_manifest(
        shots, scores=scores, budget_percent=20, cutoff=0.8,
    )
    random = module.build_random_query_manifest(
        shots, budget_percent=20, seed=137,
    )
    caps = module.JTextComputeCaps(
        current_windows=8,
        replay_windows=8,
        guard_windows=8,
        optimizer_attempts=2,
    )
    active_compute = module.realize_compute_ledger(
        active, caps=caps, online_parameter_count=321,
    )
    random_compute = module.realize_compute_ledger(
        random, caps=caps, online_parameter_count=321,
    )
    audit = module.audit_matched_controls(
        active, random, active_compute, random_compute,
    )

    assert audit["matched"] is True
    assert audit["active_queries"] == audit["random_queries"] == 44
    assert audit["queries_per_block"] == {
        "active": [4] * 11, "random": [4] * 11,
    }
    assert audit["optimizer_attempts"] == {"active": 88, "random": 88}
    assert audit["current_windows"] == {"active": 352, "random": 352}
    assert audit["replay_windows"] == {"active": 344, "random": 344}
    assert audit["guard_windows"] == {"active": 352, "random": 352}
    assert [row["replay_size"] for row in active_compute] == [
        row["replay_size"] for row in random_compute
    ]
    assert [row["flop_inputs"] for row in active_compute] == [
        row["flop_inputs"] for row in random_compute
    ]


def test_frontload_evaluation_excludes_every_training_shot() -> None:
    module = load_runner()
    plan = module.build_frontload_plan(tuple(range(220)), budget_percent=20)

    assert len(plan["training_shots"]) == 44
    assert plan["suffix_start_index"] == 44
    assert len(plan["evaluation_shots"]) == 176
    assert set(plan["training_shots"]).isdisjoint(plan["evaluation_shots"])
    assert all(not plan["evaluation_mask"][index] for index in range(44))
    assert all(plan["evaluation_mask"][index] for index in range(44, 220))


@pytest.mark.parametrize(
    ("tpr", "fpr", "reason"),
    [(0.59, 0.10, "TPR_BELOW_0.60"), (0.80, 0.41, "FPR_ABOVE_0.40")],
)
def test_abnormal_full_arm_stops_with_machine_readable_audit(
    tpr: float, fpr: float, reason: str,
) -> None:
    module = load_runner()
    receipt = module.abnormal_run_gate(
        arm="full_active", operational_tpr=tpr, fpr=fpr,
    )

    assert receipt["triggered"] is True
    assert receipt["status"] == "STOPPED_ABNORMAL_RUN"
    assert reason in receipt["reasons"]
    assert receipt["tuning_allowed"] is False
    assert receipt["required_audits"] == [
        "label", "threshold", "alarm_index", "scaler", "order", "output",
    ]


def test_selected_event_order_unselected_is_unread_and_static_seal_precedes_oracle(
    tmp_path: Path,
) -> None:
    module = load_runner()
    result, reads, static_seen = run_two_shots(module, tmp_path)
    first, second = 1051600, 1051601

    assert static_seen == [True]
    assert result["phase_receipts"][0]["event"] == "static_seal"
    assert result["phase_receipts"][1]["event"] == "online_reset"
    selected = [row for row in result["events"] if row["shot"] == first]
    order = [row["event"] for row in selected]
    assert order == ["predict", "seal", "query", "reveal", "update"]
    assert [row["sequence"] for row in selected] == sorted(
        row["sequence"] for row in selected
    )
    unselected = [row for row in result["events"] if row["shot"] == second]
    assert [row["event"] for row in unselected] == [
        "predict", "seal", "query", "discard",
    ]
    assert reads == [first]
    assert result["shot_receipts"][1]["persistent_before_sha256"] == (
        result["shot_receipts"][1]["persistent_after_sha256"]
    )
    assert all(row["claim_status"] == "TEST_DRIVEN_EXPLORATION" for row in result["events"])


def test_intra_receipts_are_post_endpoint_seal_and_strictly_ordered(tmp_path: Path) -> None:
    module = load_runner()
    result, _reads, _static_seen = run_two_shots(
        module, tmp_path, arm="intra_only",
    )

    intra = [row for row in result["endpoint_receipts"] if row["event"] == "intra_update"]
    seals = {
        (row["shot"], row["endpoint"]): row["sequence"]
        for row in result["endpoint_receipts"]
        if row["event"] == "endpoint_seal"
    }
    assert intra
    for row in intra:
        assert row["source_endpoint"] < row["observed_endpoint"]
        assert seals[(row["shot"], row["observed_endpoint"])] < row["sequence"]
        assert set(row["losses"]) == {"next", "teacher", "risk_anchor", "physics"}
        assert not {
            "label", "outcome", "DownTime", "AdvancedTime", "TTD", "horizon",
        }.intersection(row)


def test_future_suffix_mutation_leaves_prior_endpoint_receipts_byte_identical(
    tmp_path: Path,
) -> None:
    module = load_runner()
    original, _reads, _static = run_two_shots(
        module, tmp_path, arm="intra_only", output_name="original",
        first_suffix_delta=0.0,
    )
    mutated, _reads2, _static2 = run_two_shots(
        module, tmp_path, arm="intra_only", output_name="mutated",
        first_suffix_delta=99.0,
    )
    first = 1051600

    def prefix_bytes(result):
        rows = [
            row for row in result["endpoint_receipts"]
            if row["shot"] == first
            and row.get("endpoint", row.get("observed_endpoint", 999)) <= 1
        ]
        return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()

    assert prefix_bytes(original) == prefix_bytes(mutated)


def test_c_mut_02_postseal_outcome_mutation_preserves_current_artifacts_only(
    tmp_path: Path,
) -> None:
    module = load_runner()
    disruptive, _reads, _static = run_two_shots(
        module, tmp_path, output_name="disruptive", first_disruptive=True,
    )
    safe, _reads2, _static2 = run_two_shots(
        module, tmp_path, output_name="safe", first_disruptive=False,
    )
    first, second = 1051600, 1051601

    def canonical(rows):
        return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()

    assert canonical([
        row for row in disruptive["endpoint_receipts"] if row["shot"] == first
    ]) == canonical([
        row for row in safe["endpoint_receipts"] if row["shot"] == first
    ])
    assert canonical([
        row for row in disruptive["events"]
        if row["shot"] == first and row["event"] in {"predict", "seal", "query"}
    ]) == canonical([
        row for row in safe["events"]
        if row["shot"] == first and row["event"] in {"predict", "seal", "query"}
    ])
    assert disruptive["query_receipts"][0] == safe["query_receipts"][0]
    disruptive_later = next(
        row for row in disruptive["events"]
        if row["shot"] == second and row["event"] == "predict"
    )
    safe_later = next(
        row for row in safe["events"]
        if row["shot"] == second and row["event"] == "predict"
    )
    assert disruptive_later["prediction_sha256"] != safe_later["prediction_sha256"]


def _shot_path(data_root: Path, shot: int) -> Path:
    return data_root / "J-TEXT/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"


def write_hdf5(data_root: Path, shot: int, *, disruptive: bool) -> None:
    path = _shot_path(data_root, shot)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for index, name in enumerate(jio.SOURCE_RAW_SIGNAL_NAMES):
            values = np.linspace(0.1, 1.0, 8, dtype=np.float32) + index * 0.1
            signal = data.create_dataset(name, data=values)
            if name == "ip":
                signal.attrs["SampleRate"] = 1000.0
                signal.attrs["StartTime"] = 0.0
        meta = handle.create_group("meta")
        meta.create_dataset("IsDisrupt", data=np.asarray([disruptive], np.bool_))
        meta.create_dataset("DownTime", data=np.asarray([0.007], np.float64))


def test_two_shot_cpu_static_full_active_resume_is_byte_identical(tmp_path: Path) -> None:
    module = load_runner()
    first, second = 1051700, 1051701
    data_root = tmp_path / "data"
    write_hdf5(data_root, first, disruptive=True)
    write_hdf5(data_root, second, disruptive=False)
    advanced = tmp_path / "AdvancedTime.json"
    advanced.write_text(json.dumps({str(first): 55.0}), encoding="utf-8")

    def oracle_factory(remaining: tuple[int, ...]):
        return jio.JTextLazyOutcomeOracle(
            data_root=data_root,
            ordered_roster=remaining,
            advanced_time_path=advanced,
        )

    common = {
        "ordered_shots": (first, second),
        "signal_loader": lambda shot: jio.load_jtext_signal_only(data_root, shot),
        "source_trace_builder": source_trace,
        "oracle_factory": oracle_factory,
        "arm": "full_active",
        "budget_percent": 20,
        "seed": 131,
        "query_block_size": 2,
        "query_quota": 1,
        "active_cutoff": 0.0,
    }
    uninterrupted = module.run_jtext_online_stream(
        **common, controller=make_controller(seed=73),
        output=tmp_path / "uninterrupted",
    )
    partial = module.run_jtext_online_stream(
        **common, controller=make_controller(seed=73),
        output=tmp_path / "resumed",
        stop_after=1,
    )
    resume_path = tmp_path / "runner_resume.pt"
    module.save_runner_resume(resume_path, partial["resume_state"])
    restored = module.load_runner_resume(resume_path)
    resumed = module.run_jtext_online_stream(
        **common, controller=make_controller(seed=73),
        output=tmp_path / "resumed",
        resume_state=restored,
    )

    assert partial["summary"]["completed"] is False
    assert resumed["summary"] == uninterrupted["summary"]
    artifact_names = (
        "static_predictions.jsonl",
        "static_seal_manifest.json",
        "online_events.jsonl",
        "endpoint_receipts.jsonl",
        "query_ledger.jsonl",
        "update_ledger.jsonl",
        "summary.json",
    )
    for name in artifact_names:
        assert (tmp_path / "resumed" / name).read_bytes() == (
            tmp_path / "uninterrupted" / name
        ).read_bytes()
