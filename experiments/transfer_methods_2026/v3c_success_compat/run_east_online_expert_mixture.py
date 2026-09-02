#!/usr/bin/env python3
"""Run the frozen-expert EAST online-active development stream.

This runner consumes only artifacts that were already produced by the
registered max(25 ms, AdvancedTime) J-TEXT source -> EAST support110 transfer
mainline.  It never retrains those models.  balanced360 is replayed in strict
shot order with the protocol

    predict -> durable seal -> query -> HDF5 reveal (selected only) -> update

and all balanced360 outcomes are withheld from final scoring until the online
stream has completed.  The output remains TEST_DRIVEN_EXPLORATION because the
same balanced360 stream was used to select this architecture and its range.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_CONFIG = (
    HERE / "paper_grade_package/configs/east_online_expert_mixture_v1.json"
)
CANONICAL_SUPPORT_ROSTER = ROOT / "config/splits/east_support110_label_free_roster.jsonl"
CANONICAL_BALANCED360_ROSTER = ROOT / "config/splits/east_balanced360_label_free_roster.jsonl"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import east_online_expert_mixture as mixture  # noqa: E402
import selected_mainline_contract as selected_contract  # noqa: E402
import unified_online_value as online_value  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def support_contract(selection: Mapping[str, Any]) -> str:
    """Describe support110 from HDF5-authoritative receipt counts.

    The legacy roster directory name encodes 10D/100N, but HDF5 truth is the
    only label authority.  Building this string from the aligned threshold
    receipts prevents that historical name from leaking into result metadata.
    """
    total = int(selection["support_count"])
    disruptive = int(selection["support_disruptive"])
    safe = int(selection["support_safe"])
    if disruptive < 0 or safe < 0 or disruptive + safe != total:
        raise RuntimeError("support receipt class counts do not sum to support_count")
    return f"fixed support{total}, HDF5 {disruptive}D/{safe}N"


def load_label_free_roster(path: Path) -> list[int]:
    rows = load_jsonl(Path(path))
    expected = list(range(len(rows)))
    declared = [int(row.get("stream_index", -1)) for row in rows]
    shots = [int(row["shot"]) for row in rows]
    if declared != expected:
        raise RuntimeError(f"label-free roster stream_index is not contiguous: {path}")
    if len(shots) != len(set(shots)):
        raise RuntimeError(f"label-free roster contains duplicate shots: {path}")
    return shots


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(f"JSONL row {line_number} is not an object: {path}")
                rows.append(row)
    return rows


class DurableJsonlWriter:
    """Append one event and fsync it before a later reveal is permitted."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None

    def __enter__(self) -> "DurableJsonlWriter":
        self._handle = self.path.open("x", encoding="utf-8")
        return self

    def append(self, row: Mapping[str, Any]) -> str:
        if self._handle is None:
            raise RuntimeError("durable JSONL writer is not open")
        body = dict(row)
        encoded = json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        self._handle.write(encoded + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class OutcomeAccessGate:
    """Fail closed around balanced360 outcome access.

    Static predictions must all be sealed before online replay starts.  For the
    online replay, each shot must be sealed and receive a query decision before
    its outcome can be opened.  Unselected outcomes remain inaccessible until
    every online prediction is complete and final evaluation begins.
    """

    def __init__(
        self, roster: Sequence[int],
        outcome_reader: Callable[[int], Mapping[str, Any]],
    ) -> None:
        self.roster = [int(shot) for shot in roster]
        if not self.roster or self.roster != sorted(self.roster):
            raise ValueError("outcome gate roster must be non-empty and chronological")
        if len(self.roster) != len(set(self.roster)):
            raise ValueError("outcome gate roster must contain unique shots")
        self._reader = outcome_reader
        self._static_index = 0
        self._online_index = 0
        self._phase = "static"
        self._authorized: dict[int, bool] = {}
        self._revealed: dict[int, dict[str, Any]] = {}
        self._pre_final_reader_calls: list[int] = []
        self._final_reader_calls: list[int] = []

    def seal_static(self, shot: int) -> None:
        if self._phase != "static" or self._static_index >= len(self.roster):
            raise RuntimeError("static seal phase is closed")
        expected = self.roster[self._static_index]
        if int(shot) != expected:
            raise RuntimeError("static predictions must be sealed in roster order")
        self._static_index += 1

    def begin_online(self) -> None:
        if self._phase != "static" or self._static_index != len(self.roster):
            raise RuntimeError("all static predictions must be sealed before online replay")
        self._phase = "online"

    def seal_online(self, shot: int) -> None:
        if self._phase != "online" or self._online_index >= len(self.roster):
            raise RuntimeError("online seal phase is closed")
        expected = self.roster[self._online_index]
        if int(shot) != expected:
            raise RuntimeError("online predictions must be sealed in roster order")
        self._online_index += 1

    def authorize_query(self, shot: int, *, selected: bool) -> None:
        shot = int(shot)
        if self._phase != "online" or shot not in self.roster[:self._online_index]:
            raise RuntimeError("prediction must be sealed before a query decision")
        if shot in self._authorized:
            raise RuntimeError("query decision was already recorded for shot")
        if self.roster[self._online_index - 1] != shot:
            raise RuntimeError("query decision must immediately follow its sealed prediction")
        self._authorized[shot] = bool(selected)

    def reveal_selected(self, shot: int) -> dict[str, Any]:
        shot = int(shot)
        if self._phase != "online" or not self._authorized.get(shot, False):
            raise RuntimeError("outcome reveal is forbidden for a shot that was not selected")
        if shot in self._revealed:
            raise RuntimeError("selected outcome was already revealed")
        payload = dict(self._reader(shot))
        self._revealed[shot] = copy.deepcopy(payload)
        self._pre_final_reader_calls.append(shot)
        return payload

    def begin_final_evaluation(self) -> None:
        if (
            self._phase != "online"
            or self._online_index != len(self.roster)
            or len(self._authorized) != len(self.roster)
        ):
            raise RuntimeError("all online predictions and queries must finish before scoring")
        self._phase = "final_evaluation"

    def read_for_final_evaluation(self, shot: int) -> dict[str, Any]:
        shot = int(shot)
        if self._phase != "final_evaluation" or shot not in self.roster:
            raise RuntimeError("full outcomes are unavailable before final evaluation")
        if shot not in self._revealed:
            self._revealed[shot] = dict(self._reader(shot))
            self._final_reader_calls.append(shot)
        return copy.deepcopy(self._revealed[shot])

    def audit(self) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "static_sealed": self._static_index,
            "online_sealed": self._online_index,
            "query_decisions": len(self._authorized),
            "selected_reveals_before_final": len(self._pre_final_reader_calls),
            "selected_reveal_shots": list(self._pre_final_reader_calls),
            "unselected_outcomes_opened_before_final": False,
            "outcome_reader_calls_after_final_started": len(self._final_reader_calls),
        }


class RandomBlockBudgetSelector:
    """Outcome- and score-independent exact random budget control."""

    def __init__(self, *, block_size: int, quota: int, seed: int) -> None:
        if int(block_size) <= 0 or not 0 <= int(quota) <= int(block_size):
            raise ValueError("require 0 <= quota <= block_size")
        self.block_size = int(block_size)
        self.quota = int(quota)
        self.seed = int(seed)
        self._next_index = 0
        self._positions: dict[int, set[int]] = {}
        self.total_selected = 0

    def _block_positions(self, block: int) -> set[int]:
        if block not in self._positions:
            rng = np.random.default_rng(self.seed + 1_000_003 * int(block))
            selected = rng.choice(
                self.block_size, size=self.quota, replace=False,
            ) if self.quota else np.empty(0, dtype=np.int64)
            self._positions[block] = {int(value) for value in selected}
        return self._positions[block]

    def decide(self, *, score: float, stream_index: int) -> dict[str, Any]:
        if int(stream_index) != self._next_index:
            raise RuntimeError("random query decisions must be contiguous and ordered")
        if not np.isfinite(float(score)):
            raise ValueError("query score must be finite")
        block = int(stream_index) // self.block_size
        position = int(stream_index) % self.block_size
        selected = position in self._block_positions(block)
        self.total_selected += int(selected)
        self._next_index += 1
        return {
            "selected": bool(selected),
            "route": "random_precommitted" if selected else None,
            "score": float(score),
            "block": block,
            "position": position,
            "precommitted_positions": sorted(self._block_positions(block)),
        }


def operational_metrics(
    predictions: Sequence[Mapping[str, Any]],
    outcomes: Mapping[int, Mapping[str, Any]], *, minimum_lead_ms: float = 30.0,
) -> dict[str, Any]:
    disruptive = safe = true_alarm = false_alarm = timely = 0
    leads: list[float] = []
    for row in predictions:
        shot = int(row["shot"])
        outcome = outcomes[shot]
        alarmed = bool(row["alarmed"])
        if bool(outcome["is_disrupt"]):
            disruptive += 1
            if alarmed:
                true_alarm += 1
                lead = (
                    float(outcome["down_time_s"]) - float(row["alarm_time_s"])
                ) * 1000.0
                leads.append(lead)
                timely += int(lead >= float(minimum_lead_ms) - 1e-9)
        else:
            safe += 1
            false_alarm += int(alarmed)
    tpr = true_alarm / max(disruptive, 1)
    timely_tpr = timely / max(disruptive, 1)
    fpr = false_alarm / max(safe, 1)
    return {
        "TP": true_alarm,
        "FP": false_alarm,
        "FN": disruptive - true_alarm,
        "TN": safe - false_alarm,
        "tpr": tpr,
        "tpr_at_30ms": timely_tpr,
        "fpr": fpr,
        "j": timely_tpr - fpr,
        "median_lead_ms": float(np.median(leads)) if leads else None,
        "over_3000ms_count": int(sum(lead > 3000.0 for lead in leads)),
        "minimum_lead_ms": float(minimum_lead_ms),
    }


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "east-online-frozen-expert-mixture-v1":
        raise ValueError("EAST expert-mixture config schema mismatch")
    if payload.get("claim_status") != "TEST_DRIVEN_EXPLORATION":
        raise ValueError("balanced360 expert mixture must remain development-only")
    if payload.get("expert_seeds") != [131, 133, 137]:
        raise ValueError("registered frozen-expert roster is 131/133/137")
    query = payload.get("query", {})
    if int(query.get("block_size", 0)) != 20:
        raise ValueError("registered online budget block size is 20 shots")
    return payload


def _hdf5_path(data_root: Path, shot: int) -> Path:
    return (
        Path(data_root) / "EAST/unified_hdf5"
        / str(int(shot) // 100 * 100) / f"{int(shot)}.hdf5"
    )


def read_timing_only(data_root: Path, shot: int, *, expected_length: int) -> np.ndarray:
    """Read only observable signal timing; never touch ``meta`` labels."""
    path = _hdf5_path(data_root, shot)
    with h5py.File(path, "r") as handle:
        ip = handle["data/ip"]
        sample_rate = float(np.asarray(
            ip.attrs.get("SampleRate", 1000.0),
        ).reshape(-1)[0])
        start_time = float(np.asarray(
            ip.attrs.get("StartTime", 0.0),
        ).reshape(-1)[0])
        signal_length = int(len(ip))
    if sample_rate <= 0.0 or signal_length - 128 + 1 != int(expected_length):
        raise RuntimeError("HDF5 timing and frozen risk trace length mismatch")
    return start_time + (
        127 + np.arange(int(expected_length), dtype=np.float64)
    ) / sample_rate


def read_hdf5_outcome(data_root: Path, shot: int) -> dict[str, Any]:
    path = _hdf5_path(data_root, shot)
    with h5py.File(path, "r") as handle:
        disruptive = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
        down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
    if disruptive and not down.size:
        raise RuntimeError(f"disruptive EAST shot {shot} has no HDF5 DownTime")
    return {
        "is_disrupt": disruptive,
        "down_time_s": float(down[0]) if disruptive else None,
    }


def _safe_torch_load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"frozen parent is not a mapping: {path}")
    return payload


def load_frozen_family(parent_root: Path, seeds: Sequence[int]) -> dict[str, Any]:
    roots = [Path(parent_root) / f"seed{int(seed)}" for seed in seeds]
    support_roster = load_label_free_roster(CANONICAL_SUPPORT_ROSTER)
    balanced_roster = load_label_free_roster(CANONICAL_BALANCED360_ROSTER)
    if set(support_roster) & set(balanced_roster):
        raise RuntimeError("canonical EAST support110 and balanced360 overlap")
    support_sha = sha256_json(support_roster)
    balanced_sha = sha256_json(balanced_roster)
    members = []
    reference_roster: list[int] | None = None
    reference_test_sha: str | None = None
    for seed, root in zip(seeds, roots):
        parent_path = root / "east-parent/east_fixed_parent.pt"
        ledger_path = root / "east-static/static_predictions.jsonl"
        if not parent_path.is_file() or not ledger_path.is_file():
            raise FileNotFoundError(f"missing frozen expert artifacts for seed {seed}")
        parent_sha = sha256_file(parent_path)
        parent = _safe_torch_load(parent_path)
        if (
            parent.get("source_label_policy") != "advanced_floor"
            or parent.get("selected_label_arm")
            != "source_advanced_floor__east_fixed"
            or parent.get("east_target_label_policy") != "fixed125"
            or not math.isclose(float(parent.get("east_target_horizon_ms", -1)), 125.0)
        ):
            raise RuntimeError(
                f"seed {seed} is not the registered advanced-floor source parent"
            )
        if int(parent.get("seed", -1)) != int(seed):
            raise RuntimeError(f"seed {seed} parent metadata seed mismatch")
        contract_fields = selected_contract.SelectedMainlineContract.__dataclass_fields__
        try:
            parent_contract = selected_contract.SelectedMainlineContract(**{
                name: parent[name] for name in contract_fields
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"seed {seed} selected-parent contract is incomplete") from exc
        selected_contract.validate_selected_parent(parent, parent_contract)
        if (
            str(parent.get("east_support_roster_sha256")) != support_sha
            or str(parent.get("east_test_roster_sha256")) != balanced_sha
        ):
            raise RuntimeError(f"seed {seed} canonical EAST roster SHA256 mismatch")
        receipt = selected_contract.validate_threshold_receipt(
            parent.get("east_threshold_receipt", {}),
            expected_scope="EAST support110",
            expected_roster=support_roster,
            expected_threshold=float(parent["east_threshold"]),
        )
        rows = load_jsonl(ledger_path)
        roster = [int(row["shot"]) for row in rows]
        if roster != balanced_roster:
            raise RuntimeError(f"seed {seed} static ledger is not chronological balanced360")
        if reference_roster is None:
            reference_roster = roster
            reference_test_sha = str(parent["east_test_roster_sha256"])
        elif roster != reference_roster:
            raise RuntimeError("frozen expert balanced360 rosters do not align")
        elif str(parent["east_test_roster_sha256"]) != reference_test_sha:
            raise RuntimeError("frozen expert test-roster contracts do not align")
        members.append({
            "seed": int(seed),
            "root": root,
            "parent_path": parent_path,
            "parent_sha256": parent_sha,
            "source_checkpoint_sha256": str(parent["source_checkpoint_sha256"]),
            "threshold": float(parent["east_threshold"]),
            "threshold_logit": float(mixture._logit(float(parent["east_threshold"]))),
            "threshold_receipt": receipt,
            "static_ledger_path": ledger_path,
            "static_ledger_sha256": sha256_file(ledger_path),
            "static_rows": {int(row["shot"]): row for row in rows},
        })
        del parent
    assert reference_roster is not None
    return {
        "members": members,
        "roster": reference_roster,
        "roster_sha256": sha256_json(reference_roster),
        "contract_test_roster_sha256": reference_test_sha,
        "canonical_support_roster_sha256": support_sha,
        "canonical_balanced360_roster_sha256": balanced_sha,
        "canonical_support_balanced_overlap": 0,
    }


class FrozenTraceRepository:
    """Verify and cache outcome-free expert traces for one balanced360 replay."""

    def __init__(self, family: Mapping[str, Any]) -> None:
        self.family = family
        self._cache: dict[int, dict[str, Any]] = {}

    def load(self, shot: int) -> dict[str, Any]:
        shot = int(shot)
        if shot in self._cache:
            return self._cache[shot]
        margins = []
        physics_rows = []
        trace_hashes = []
        for member in self.family["members"]:
            row = member["static_rows"][shot]
            relative = Path(str(row["risk_trace_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("static trace path escapes the EAST static seal")
            trace_path = member["root"] / "east-static" / relative
            actual_sha = sha256_file(trace_path)
            if actual_sha != str(row["risk_trace_sha256"]):
                raise RuntimeError(f"seed {member['seed']} static trace SHA256 mismatch")
            with np.load(trace_path) as payload:
                sealed_shot = int(np.asarray(payload["shot_id"]).reshape(-1)[0])
                if sealed_shot != shot:
                    raise RuntimeError("static trace shot identity mismatch")
                logit = np.asarray(payload["static_logit"], dtype=np.float64)
                physics = np.asarray(payload["physics_support"], dtype=np.float64)
            if (
                logit.ndim != 1 or physics.shape != logit.shape
                or not np.isfinite(logit).all() or not np.isfinite(physics).all()
                or np.any((physics < 0.0) | (physics > 1.0))
            ):
                raise RuntimeError("sealed static logit/physics trace is invalid")
            margins.append(logit - float(member["threshold_logit"]))
            trace_hashes.append(actual_sha)
            physics_rows.append(physics)
        shapes = {array.shape for array in margins + physics_rows}
        if len(shapes) != 1 or any(array.ndim != 1 for array in margins + physics_rows):
            raise RuntimeError(f"frozen expert trace shapes do not align for shot {shot}")
        for physics in physics_rows[1:]:
            if not np.array_equal(physics_rows[0], physics):
                raise RuntimeError(f"observable physics trace differs across experts for shot {shot}")
        result = {
            "expert_margin_traces": np.stack(margins),
            "physics_support": physics_rows[0],
            "static_trace_sha256": trace_hashes,
            "physics_array_sha256": hashlib.sha256(
                physics_rows[0].astype("<f8", copy=False).tobytes()
            ).hexdigest(),
        }
        self._cache[shot] = result
        return result


def _selector(
    *, mode: str, block_size: int, quota: int, history_size: int,
    quantile: float, seed: int,
):
    if mode == "active":
        return online_value.AdaptiveBlockBudgetSelector(
            block_size=int(block_size), quota=int(quota),
            history_size=int(history_size), quantile=float(quantile),
            exploration_offset=3 + int(seed) % 5, exploration_stride=7,
        )
    if mode == "random":
        return RandomBlockBudgetSelector(
            block_size=int(block_size), quota=int(quota), seed=int(seed),
        )
    raise ValueError("selector mode must be active or random")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    parent_root = Path(args.parent_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    family = load_frozen_family(parent_root, config["expert_seeds"])
    candidates = mixture.simplex_grid(
        members=len(config["expert_seeds"]),
        step=float(config["weight_grid_step"]),
    )
    support_selection = mixture.select_support_weights(
        [member["threshold_receipt"] for member in family["members"]],
        candidates=candidates,
    )
    initial_weights = np.asarray(support_selection["weights"], dtype=np.float64)
    budget_percent = int(args.budget_percent)
    block_size = int(config["query"]["block_size"])
    if budget_percent not in config["query"]["allowed_budget_percent"]:
        raise ValueError("budget is outside the registered robustness range")
    quota = int(round(block_size * budget_percent / 100.0))
    if quota <= 0 or quota > block_size:
        raise ValueError("budget does not map to a valid exact block quota")

    repo = FrozenTraceRepository(family)
    gate = OutcomeAccessGate(
        family["roster"], lambda shot: read_hdf5_outcome(data_root, shot),
    )
    static_rows: list[dict[str, Any]] = []
    static_path = output / "static_predictions.jsonl"
    with DurableJsonlWriter(static_path) as writer:
        for stream_index, shot in enumerate(family["roster"]):
            traces = repo.load(shot)
            times = read_timing_only(
                data_root, shot,
                expected_length=traces["expert_margin_traces"].shape[1],
            )
            decision = mixture.mixture_alarm(
                traces["expert_margin_traces"], traces["physics_support"],
                initial_weights, config["alarm_parameters"],
            )
            index = decision["alarm_index"]
            row = {
                "stream_index": stream_index,
                "shot": shot,
                "alarmed": bool(decision["alarmed"]),
                "alarm_index": None if index is None else int(index),
                "alarm_time_s": None if index is None else float(times[int(index)]),
                "route": decision["route"],
                "weights": initial_weights.tolist(),
                "expert_static_trace_sha256": traces["static_trace_sha256"],
                "physics_array_sha256": traces["physics_array_sha256"],
            }
            row["row_sha256"] = writer.append(row)
            static_rows.append(row)
            gate.seal_static(shot)
    static_seal = {
        "schema_version": "east-expert-mixture-static-seal-v1",
        "prediction_count": len(static_rows),
        "roster_sha256": family["roster_sha256"],
        "predictions_sha256": sha256_file(static_path),
        "weights_source": "EAST support110 receipts only",
        "weights": initial_weights.tolist(),
        "balanced360_outcomes_opened": False,
    }
    (output / "static_seal_manifest.json").write_text(
        json.dumps(static_seal, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    gate.begin_online()

    controller = mixture.CausalExpertWeightController(
        initial_weights=initial_weights, candidates=candidates,
        minimum_per_class=int(args.minimum_per_class),
        minimum_lead_ms=float(config["minimum_lead_ms"]),
        maximum_expert_weight=float(np.max(initial_weights)),
        alarm_parameters=config["alarm_parameters"],
        uncertainty_weight=float(config["query"]["uncertainty_weight"]),
        disagreement_weight=float(config["query"]["disagreement_weight"]),
        disagreement_scale=float(config["query"]["disagreement_scale"]),
    )
    selector = _selector(
        mode=args.mode, block_size=block_size, quota=quota,
        history_size=int(config["query"]["history_size"]),
        quantile=float(args.quantile), seed=int(args.selector_seed),
    )
    online_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    reveal_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    with (
        DurableJsonlWriter(output / "online_predictions.jsonl") as prediction_writer,
        DurableJsonlWriter(output / "query_decisions.jsonl") as query_writer,
        DurableJsonlWriter(output / "reveals.jsonl") as reveal_writer,
        DurableJsonlWriter(output / "updates.jsonl") as update_writer,
    ):
        for stream_index, shot in enumerate(family["roster"]):
            traces = repo.load(shot)
            times = read_timing_only(
                data_root, shot,
                expected_length=traces["expert_margin_traces"].shape[1],
            )
            prediction = controller.predict_and_seal(
                shot,
                expert_margin_traces=traces["expert_margin_traces"],
                physics_support=traces["physics_support"],
                endpoint_times_s=times,
            )
            prediction["stream_index"] = stream_index
            prediction["expert_static_trace_sha256"] = traces["static_trace_sha256"]
            prediction["physics_array_sha256"] = traces["physics_array_sha256"]
            prediction["durable_row_sha256"] = prediction_writer.append(prediction)
            online_rows.append(prediction)
            gate.seal_online(shot)

            query = selector.decide(
                score=float(prediction["query_value"]["score"]),
                stream_index=stream_index,
            )
            query_row = {
                **query,
                "stream_index": stream_index,
                "shot": shot,
                "prediction_sha256": prediction["prediction_sha256"],
                "selector_mode": args.mode,
                "selector_seed": int(args.selector_seed),
            }
            query_row["durable_row_sha256"] = query_writer.append(query_row)
            query_rows.append(query_row)
            gate.authorize_query(shot, selected=bool(query["selected"]))
            if bool(query["selected"]):
                outcome = gate.reveal_selected(shot)
                reveal = {
                    "stream_index": stream_index,
                    "shot": shot,
                    "is_disrupt": bool(outcome["is_disrupt"]),
                    "down_time_s": outcome["down_time_s"],
                    "query_row_sha256": query_row["durable_row_sha256"],
                }
                reveal["durable_row_sha256"] = reveal_writer.append(reveal)
                reveal_rows.append(reveal)
                update = controller.update_after_reveal(
                    shot, selected=True,
                    is_disrupt=bool(outcome["is_disrupt"]),
                    down_time_s=outcome["down_time_s"],
                )
                update["stream_index"] = stream_index
                update["reveal_row_sha256"] = reveal["durable_row_sha256"]
                update["durable_row_sha256"] = update_writer.append(update)
                update_rows.append(update)
            else:
                controller.discard_without_reveal(shot)

    gate.begin_final_evaluation()
    outcomes = {
        shot: gate.read_for_final_evaluation(shot) for shot in family["roster"]
    }
    static_metrics = operational_metrics(
        static_rows, outcomes, minimum_lead_ms=float(config["minimum_lead_ms"]),
    )
    online_metrics = operational_metrics(
        online_rows, outcomes, minimum_lead_ms=float(config["minimum_lead_ms"]),
    )
    parent_after = {
        str(member["seed"]): sha256_file(member["parent_path"])
        for member in family["members"]
    }
    parent_before = {
        str(member["seed"]): member["parent_sha256"]
        for member in family["members"]
    }
    if parent_before != parent_after:
        raise RuntimeError("a frozen transfer expert changed during online replay")
    query_count = sum(int(row["selected"]) for row in query_rows)
    expected_queries = quota * (len(family["roster"]) // block_size)
    if query_count != expected_queries or len(reveal_rows) != query_count:
        raise RuntimeError("online query/reveal budget mismatch")
    summary = {
        "schema_version": "east-online-frozen-expert-mixture-result-v1",
        "claim_status": "TEST_DRIVEN_EXPLORATION",
        "independent_future_confirmation": False,
        "claim_boundary": (
            "balanced360 was used for architecture and hyperparameter development; "
            "the result is causal within the replay but is not an untouched future test"
        ),
        "method": {
            "name": "support-calibrated frozen transfer-expert mixture",
            "expert_seeds": list(config["expert_seeds"]),
            "neural_parameter_updates": 0,
            "online_trainable_state": "three convex weights only",
            "raw_candidate_weight_count": len(candidates),
            "candidate_weight_count": len(controller.candidates),
            "support_anchored_maximum_expert_weight": float(
                controller.maximum_expert_weight
            ),
            "exposure_cap_source": "maximum component of support110-selected initial weights",
            "initial_weight_selection": support_selection,
            "final_weights": controller.weights.tolist(),
            "minimum_per_class": int(args.minimum_per_class),
        },
        "selector": {
            "mode": args.mode,
            "seed": int(args.selector_seed),
            "budget_percent": budget_percent,
            "block_size": block_size,
            "quota_per_block": quota,
            "query_count": query_count,
            "quantile": float(args.quantile),
        },
        "metrics": {"static": static_metrics, "online": online_metrics},
        "paired_delta": {
            "tpr_at_30ms": online_metrics["tpr_at_30ms"] - static_metrics["tpr_at_30ms"],
            "fpr": online_metrics["fpr"] - static_metrics["fpr"],
            "j": online_metrics["j"] - static_metrics["j"],
        },
        "normality": {
            "tpr_below_0_60": online_metrics["tpr_at_30ms"] < 0.60,
            "fpr_above_0_40": online_metrics["fpr"] > 0.40,
        },
        "causal_access_audit": gate.audit(),
        "lineage": {
            "source_label_policy": "max(25 ms, AdvancedTime); missing -> 25 ms",
            "east_support": support_contract(support_selection),
            "east_stream": "chronological balanced360, 180D/180N",
            "east_target_label_policy": "fixed125",
            "minimum_lead_ms": float(config["minimum_lead_ms"]),
            "over_3000ms_counts_as_tpr": True,
            "roster_sha256": family["roster_sha256"],
            "contract_test_roster_sha256": family["contract_test_roster_sha256"],
            "parent_sha256_before": parent_before,
            "parent_sha256_after": parent_after,
            "source_checkpoint_sha256": {
                str(member["seed"]): member["source_checkpoint_sha256"]
                for member in family["members"]
            },
            "config_path": str(Path(args.config).resolve()),
            "config_sha256": sha256_file(Path(args.config)),
            "python": sys.executable,
        },
        "artifacts": {
            "static_predictions_sha256": sha256_file(static_path),
            "online_predictions_sha256": sha256_file(output / "online_predictions.jsonl"),
            "query_decisions_sha256": sha256_file(output / "query_decisions.jsonl"),
            "reveals_sha256": sha256_file(output / "reveals.jsonl"),
            "updates_sha256": sha256_file(output / "updates.jsonl"),
        },
    }
    (output / "final_controller_state.json").write_text(
        json.dumps(controller.state_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("active", "random"), default="active")
    parser.add_argument("--selector-seed", type=int, default=131)
    parser.add_argument("--budget-percent", type=int, default=30)
    parser.add_argument("--quantile", type=float, default=0.70)
    parser.add_argument("--minimum-per-class", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({
        "output": str(Path(sys.argv[sys.argv.index('--output') + 1]).resolve())
        if "--output" in sys.argv else None,
        "metrics": summary["metrics"],
        "paired_delta": summary["paired_delta"],
        "query_count": summary["selector"]["query_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "OutcomeAccessGate", "RandomBlockBudgetSelector", "operational_metrics",
    "run",
]
