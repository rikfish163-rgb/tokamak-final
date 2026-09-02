#!/usr/bin/env python3
"""Aggregate trust-region expert-mixture runs and equal-label controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import east_online_expert_mixture as mixture  # noqa: E402
import run_east_online_expert_mixture as runner  # noqa: E402


def select_batch_weights(
    *, timely: np.ndarray, false_alarm: np.ndarray, labels: np.ndarray,
    candidates: np.ndarray, initial_weights: np.ndarray,
) -> dict[str, Any]:
    """Fit a fixed direct-transfer control from its labelled cohort only."""
    timely_array = np.asarray(timely, dtype=bool)
    fp_array = np.asarray(false_alarm, dtype=bool)
    outcome = np.asarray(labels, dtype=bool).reshape(-1)
    grid = np.asarray(candidates, dtype=np.float64)
    initial = np.asarray(initial_weights, dtype=np.float64).reshape(-1)
    if (
        timely_array.ndim != 2 or fp_array.shape != timely_array.shape
        or len(outcome) != timely_array.shape[0]
        or grid.shape != (timely_array.shape[1], len(initial))
        or not np.any(outcome) or not np.any(~outcome)
    ):
        raise ValueError("batch weight-selection inputs do not align")
    metrics = []
    for index in range(grid.shape[0]):
        tpr = float(np.mean(timely_array[outcome, index]))
        fpr = float(np.mean(fp_array[~outcome, index]))
        metrics.append({"tpr_at_30ms": tpr, "fpr": fpr, "j": tpr - fpr})
    chosen = min(
        range(len(grid)),
        key=lambda index: (
            -metrics[index]["j"],
            float(np.linalg.norm(grid[index] - initial)),
            tuple(float(value) for value in grid[index]),
        ),
    )
    return {
        "candidate_index": int(chosen),
        "weights": grid[chosen].tolist(),
        "metrics": metrics[chosen],
        "labelled_count": int(len(outcome)),
        "labelled_disruptive": int(outcome.sum()),
        "labelled_safe": int((~outcome).sum()),
        "selection_rule": "max queried-cohort TPR@30-FPR, tie nearest support weights",
    }


def _score(vector: tuple[np.ndarray, np.ndarray], d: np.ndarray, n: np.ndarray) -> float:
    timely, alarmed = vector
    return float(np.mean(timely[d]) - np.mean(alarmed[n]))


def paired_stratified_bootstrap(
    pairs: Sequence[
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    ], *, labels: np.ndarray, draws: int = 50_000, seed: int = 20260825,
) -> dict[str, Any]:
    """Bootstrap the equal-weight mean paired operational-J difference."""
    outcome = np.asarray(labels, dtype=bool).reshape(-1)
    if not pairs or not np.any(outcome) or not np.any(~outcome):
        raise ValueError("paired bootstrap requires pairs and both outcomes")
    for baseline, improved in pairs:
        if any(len(values) != len(outcome) for values in (*baseline, *improved)):
            raise ValueError("paired bootstrap vectors do not align")
    disruptive = np.flatnonzero(outcome)
    safe = np.flatnonzero(~outcome)
    estimate = float(np.mean([
        _score(improved, disruptive, safe) - _score(baseline, disruptive, safe)
        for baseline, improved in pairs
    ]))
    generator = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        d = generator.choice(disruptive, len(disruptive), replace=True)
        n = generator.choice(safe, len(safe), replace=True)
        samples[draw] = np.mean([
            _score(improved, d, n) - _score(baseline, d, n)
            for baseline, improved in pairs
        ])
    return {
        "estimand": "equal-run-weight mean paired delta in TPR@30-FPR",
        "estimate": estimate,
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "probability_positive": float(np.mean(samples > 0.0)),
        "draws": int(draws),
        "stratified_by_outcome": True,
    }


def variable_roster_bootstrap(
    pairs: Sequence[
        tuple[
            tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray,
        ]
    ], *, draws: int = 50_000, seed: int = 20260826,
) -> dict[str, Any]:
    """Paired bootstrap when each selector seed has a different complement."""
    if not pairs:
        raise ValueError("variable-roster bootstrap requires at least one pair")
    prepared = []
    estimates = []
    for baseline, improved, labels in pairs:
        outcome = np.asarray(labels, dtype=bool)
        d = np.flatnonzero(outcome)
        n = np.flatnonzero(~outcome)
        if not len(d) or not len(n):
            raise ValueError("every complement needs both outcomes")
        prepared.append((baseline, improved, d, n))
        estimates.append(_score(improved, d, n) - _score(baseline, d, n))
    generator = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        deltas = []
        for baseline, improved, d, n in prepared:
            sampled_d = generator.choice(d, len(d), replace=True)
            sampled_n = generator.choice(n, len(n), replace=True)
            deltas.append(
                _score(improved, sampled_d, sampled_n)
                - _score(baseline, sampled_d, sampled_n)
            )
        samples[draw] = float(np.mean(deltas))
    return {
        "estimand": "equal-seed mean paired complement delta in TPR@30-FPR",
        "estimate": float(np.mean(estimates)),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "probability_positive": float(np.mean(samples > 0.0)),
        "draws": int(draws),
        "stratified_by_outcome_within_seed": True,
    }


def operational_vector(
    rows: Sequence[Mapping[str, Any]], outcomes: Mapping[int, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timely, alarmed, labels = [], [], []
    for row in rows:
        outcome = outcomes[int(row["shot"])]
        label = bool(outcome["is_disrupt"])
        alarm = bool(row["alarmed"])
        labels.append(label)
        alarmed.append(alarm)
        timely.append(bool(
            label and alarm
            and (
                float(outcome["down_time_s"]) - float(row["alarm_time_s"])
            ) * 1000.0 >= 30.0 - 1e-9
        ))
    return (
        np.asarray(timely, dtype=bool),
        np.asarray(alarmed, dtype=bool),
        np.asarray(labels, dtype=bool),
    )


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = ("tpr_at_30ms", "fpr", "j", "median_lead_ms")
    return {
        key: float(np.mean([float(row[key]) for row in rows])) for key in keys
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ), encoding="utf-8")


def validate_run_ledger(
    run_root: Path, summary: Mapping[str, Any], *, expected_roster: Sequence[int],
) -> dict[str, Any]:
    """Recompute the prediction/query/reveal/update chain from released JSONL."""
    root = Path(run_root)
    names = {
        "static_predictions": "static_predictions.jsonl",
        "online_predictions": "online_predictions.jsonl",
        "query_decisions": "query_decisions.jsonl",
        "reveals": "reveals.jsonl",
        "updates": "updates.jsonl",
    }
    for key, name in names.items():
        expected = str(summary["artifacts"][f"{key}_sha256"])
        if runner.sha256_file(root / name) != expected:
            raise RuntimeError(f"released {key} SHA256 mismatch: {root}")
    static = runner.load_jsonl(root / names["static_predictions"])
    predictions = runner.load_jsonl(root / names["online_predictions"])
    queries = runner.load_jsonl(root / names["query_decisions"])
    reveals = runner.load_jsonl(root / names["reveals"])
    updates = runner.load_jsonl(root / names["updates"])
    roster = [int(shot) for shot in expected_roster]
    if [int(row["shot"]) for row in static] != roster:
        raise RuntimeError(f"released static roster mismatch: {root}")
    if [int(row["shot"]) for row in predictions] != roster:
        raise RuntimeError(f"released online roster mismatch: {root}")
    if [int(row["shot"]) for row in queries] != roster:
        raise RuntimeError(f"released query roster mismatch: {root}")
    forbidden = {"is_disrupt", "outcome", "down_time_s", "DownTime", "AdvancedTime"}
    selected_queries = []
    for prediction, query in zip(predictions, queries):
        if forbidden & set(prediction) or forbidden & set(query):
            raise RuntimeError(f"outcome field appears before reveal: {root}")
        receipt = dict(prediction)
        recorded_prediction_sha = str(receipt.pop("prediction_sha256"))
        for name in ("stream_index", "expert_static_trace_sha256", "physics_array_sha256"):
            receipt.pop(name, None)
        if runner.sha256_json(receipt) != recorded_prediction_sha:
            raise RuntimeError(f"prediction receipt SHA256 mismatch: {root}")
        if str(query["prediction_sha256"]) != recorded_prediction_sha:
            raise RuntimeError(f"query does not reference its prediction: {root}")
        if bool(query["selected"]):
            selected_queries.append(query)
    if [int(row["shot"]) for row in reveals] != [
        int(row["shot"]) for row in selected_queries
    ]:
        raise RuntimeError(f"selected query/reveal shots mismatch: {root}")
    if [int(row["shot"]) for row in updates] != [int(row["shot"]) for row in reveals]:
        raise RuntimeError(f"reveal/update shots mismatch: {root}")
    for query, reveal, update in zip(selected_queries, reveals, updates):
        if str(reveal["query_row_sha256"]) != runner.sha256_json(query):
            raise RuntimeError(f"reveal does not reference its query: {root}")
        if str(update["reveal_row_sha256"]) != runner.sha256_json(reveal):
            raise RuntimeError(f"update does not reference its reveal: {root}")
    expected_queries = int(summary["selector"]["query_count"])
    if len(selected_queries) != expected_queries or len(updates) != expected_queries:
        raise RuntimeError(f"released query budget mismatch: {root}")
    return {
        "run": root.name,
        "prediction_count": len(predictions),
        "query_count": len(selected_queries),
        "reveal_count": len(reveals),
        "update_count": len(updates),
        "outcome_fields_before_reveal": False,
        "hash_chain_recomputed": True,
    }


def build_report(
    *, runs_root: Path, parent_root: Path, data_root: Path, output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    seeds = (131, 133, 137)
    budgets = (10, 20, 30, 40)
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config_sha = runner.sha256_file(runner.DEFAULT_CONFIG)
    canonical_roster = runner.load_label_free_roster(
        runner.CANONICAL_BALANCED360_ROSTER
    )
    summaries: dict[tuple[str, int, int], dict[str, Any]] = {}
    ledger_audits = []
    budget_rows = []
    for mode in ("active", "random"):
        for budget in budgets:
            current = []
            for seed in seeds:
                path = runs_root / f"{mode}_b{budget}_seed{seed}/summary.json"
                summary = json.loads(path.read_text(encoding="utf-8"))
                if (
                    summary["claim_status"] != "TEST_DRIVEN_EXPLORATION"
                    or summary["lineage"]["config_sha256"] != config_sha
                    or summary["selector"]["mode"] != mode
                    or int(summary["selector"]["seed"]) != seed
                    or int(summary["selector"]["budget_percent"]) != budget
                    or not np.isclose(
                        float(summary["selector"]["quantile"]),
                        float(config["query"]["selected_quantile"]), atol=1e-12,
                    )
                    or int(summary["method"]["minimum_per_class"])
                    != int(config["query"]["selected_minimum_per_class"])
                    or int(summary["selector"]["query_count"]) != int(3.6 * budget)
                    or bool(summary["causal_access_audit"][
                        "unselected_outcomes_opened_before_final"
                    ])
                ):
                    raise RuntimeError(f"causal/budget audit failed: {path}")
                ledger_audits.append(validate_run_ledger(
                    path.parent, summary, expected_roster=canonical_roster,
                ))
                summaries[(mode, budget, seed)] = summary
                current.append(summary)
            online = [row["metrics"]["online"] for row in current]
            deltas = [float(row["paired_delta"]["j"]) for row in current]
            budget_rows.append({
                "mode": mode,
                "budget_percent": budget,
                "query_count": int(current[0]["selector"]["query_count"]),
                **_mean_metrics(online),
                "mean_delta_j_vs_static": float(np.mean(deltas)),
                "minimum_delta_j_vs_static": float(np.min(deltas)),
                "maximum_delta_j_vs_static": float(np.max(deltas)),
            })
    with (output / "budget_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(budget_rows[0]))
        writer.writeheader()
        writer.writerows(budget_rows)

    family = runner.load_frozen_family(parent_root, list(seeds))
    repo = runner.FrozenTraceRepository(family)
    raw_candidates = mixture.simplex_grid(members=3, step=float(config["weight_grid_step"]))
    support = mixture.select_support_weights(
        [member["threshold_receipt"] for member in family["members"]],
        candidates=raw_candidates,
    )
    initial = np.asarray(support["weights"], dtype=np.float64)
    candidates = raw_candidates[
        np.max(raw_candidates, axis=1) <= np.max(initial) + 1e-12
    ]

    # Precompute and seal every candidate decision without opening any outcome.
    decisions: dict[int, list[dict[str, Any]]] = {}
    decision_digest_rows = []
    for shot in family["roster"]:
        traces = repo.load(shot)
        times = runner.read_timing_only(
            data_root, shot,
            expected_length=traces["expert_margin_traces"].shape[1],
        )
        shot_rows = []
        for weights in candidates:
            alarm = mixture.mixture_alarm(
                traces["expert_margin_traces"], traces["physics_support"],
                weights, config["alarm_parameters"],
            )
            index = alarm["alarm_index"]
            shot_rows.append({
                "alarmed": bool(alarm["alarmed"]),
                "alarm_index": None if index is None else int(index),
                "alarm_time_s": None if index is None else float(times[int(index)]),
                "route": alarm["route"],
            })
        decisions[shot] = shot_rows
        decision_digest_rows.append({
            "shot": shot,
            "rows": [[row["alarmed"], row["alarm_index"]] for row in shot_rows],
        })
    candidate_seal = {
        "prediction_count": len(family["roster"]),
        "candidate_count": len(candidates),
        "roster_sha256": family["roster_sha256"],
        "candidate_decision_sha256": runner.sha256_json(decision_digest_rows),
        "outcomes_opened_before_seal": False,
    }
    (output / "direct_candidate_seal.json").write_text(
        json.dumps(candidate_seal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    direct_inputs: dict[int, dict[str, Any]] = {}
    prefinal_opened: set[int] = set()
    for seed in seeds:
        run_dir = runs_root / f"active_b30_seed{seed}"
        query_rows = runner.load_jsonl(run_dir / "query_decisions.jsonl")
        query_ids = [int(row["shot"]) for row in query_rows if bool(row["selected"])]
        reveal_rows = runner.load_jsonl(run_dir / "reveals.jsonl")
        if [int(row["shot"]) for row in reveal_rows] != query_ids:
            raise RuntimeError(f"seed {seed}: query/reveal identities differ")
        query_outcomes = {}
        for reveal in reveal_rows:
            shot = int(reveal["shot"])
            outcome = runner.read_hdf5_outcome(data_root, shot)
            if (
                bool(reveal["is_disrupt"]) != bool(outcome["is_disrupt"])
                or reveal["down_time_s"] != outcome["down_time_s"]
            ):
                raise RuntimeError(f"seed {seed}: reveal differs from HDF5 for {shot}")
            query_outcomes[shot] = outcome
            prefinal_opened.add(shot)
        labels = np.asarray([
            bool(query_outcomes[shot]["is_disrupt"]) for shot in query_ids
        ])
        timely = np.zeros((len(query_ids), len(candidates)), dtype=bool)
        false_alarm = np.zeros_like(timely)
        for row_index, shot in enumerate(query_ids):
            outcome = query_outcomes[shot]
            for candidate_index, decision in enumerate(decisions[shot]):
                if bool(outcome["is_disrupt"]) and decision["alarmed"]:
                    lead = (
                        float(outcome["down_time_s"])
                        - float(decision["alarm_time_s"])
                    ) * 1000.0
                    timely[row_index, candidate_index] = lead >= 30.0 - 1e-9
                false_alarm[row_index, candidate_index] = bool(
                    (not outcome["is_disrupt"]) and decision["alarmed"]
                )
        fitted = select_batch_weights(
            timely=timely, false_alarm=false_alarm, labels=labels,
            candidates=candidates, initial_weights=initial,
        )
        online_final = summaries[("active", 30, seed)]["method"]["final_weights"]
        if not np.allclose(fitted["weights"], online_final, atol=1e-12):
            raise RuntimeError(f"seed {seed}: batch and final online weights differ")
        query_set = set(query_ids)
        complement = [shot for shot in family["roster"] if shot not in query_set]
        direct_rows = []
        for stream_index, shot in enumerate(complement):
            decision = decisions[shot][int(fitted["candidate_index"])]
            direct_rows.append({
                "stream_index": stream_index,
                "shot": shot,
                **decision,
                "weights": fitted["weights"],
            })
        direct_path = output / f"direct_frontload_seed{seed}_predictions.jsonl"
        _write_jsonl(direct_path, direct_rows)
        direct_inputs[seed] = {
            "query_ids": query_ids,
            "query_class_counts": {
                "shots": len(query_ids),
                "disruptive": int(labels.sum()),
                "safe": int((~labels).sum()),
            },
            "fitted": fitted,
            "complement": complement,
            "direct_rows": direct_rows,
            "direct_predictions_sha256": runner.sha256_file(direct_path),
        }

    # Only now open the complete balanced360 outcome roster for all scoring.
    outcomes = {
        shot: runner.read_hdf5_outcome(data_root, shot) for shot in family["roster"]
    }
    static_rows = runner.load_jsonl(
        runs_root / "active_b30_seed131/static_predictions.jsonl"
    )
    static_vector = operational_vector(static_rows, outcomes)
    labels_full = static_vector[2]
    active_pairs = []
    random_pairs = []
    direct_rows_report = []
    direct_bootstrap_pairs = []
    static_complement_pairs = []
    for seed in seeds:
        active_rows = runner.load_jsonl(
            runs_root / f"active_b30_seed{seed}/online_predictions.jsonl"
        )
        random_rows = runner.load_jsonl(
            runs_root / f"random_b30_seed{seed}/online_predictions.jsonl"
        )
        active_vector = operational_vector(active_rows, outcomes)
        random_vector = operational_vector(random_rows, outcomes)
        active_pairs.append((static_vector[:2], active_vector[:2]))
        random_pairs.append((random_vector[:2], active_vector[:2]))

        direct = direct_inputs[seed]
        complement_set = set(direct["complement"])
        static_c = [row for row in static_rows if int(row["shot"]) in complement_set]
        online_c = [row for row in active_rows if int(row["shot"]) in complement_set]
        direct_c = direct["direct_rows"]
        static_metrics = runner.operational_metrics(static_c, outcomes)
        online_metrics = runner.operational_metrics(online_c, outcomes)
        direct_metrics = runner.operational_metrics(direct_c, outcomes)
        static_v = operational_vector(static_c, outcomes)
        online_v = operational_vector(online_c, outcomes)
        direct_v = operational_vector(direct_c, outcomes)
        if not np.array_equal(static_v[2], online_v[2]) or not np.array_equal(
            static_v[2], direct_v[2]
        ):
            raise RuntimeError("equal-label complement vectors do not align")
        direct_bootstrap_pairs.append((direct_v[:2], online_v[:2], direct_v[2]))
        static_complement_pairs.append((static_v[:2], online_v[:2], static_v[2]))
        direct_rows_report.append({
            "seed": seed,
            "query_class_counts": direct["query_class_counts"],
            "complement_count": len(direct["complement"]),
            "fitted_weights": direct["fitted"]["weights"],
            "static": static_metrics,
            "direct_frontload": direct_metrics,
            "causal_online": online_metrics,
            "online_minus_direct_j": online_metrics["j"] - direct_metrics["j"],
            "online_minus_static_j": online_metrics["j"] - static_metrics["j"],
            "direct_predictions_sha256": direct["direct_predictions_sha256"],
        })

    main_static = summaries[("active", 30, 131)]["metrics"]["static"]
    main_active = _mean_metrics([
        summaries[("active", 30, seed)]["metrics"]["online"] for seed in seeds
    ])
    main_random = _mean_metrics([
        summaries[("random", 30, seed)]["metrics"]["online"] for seed in seeds
    ])
    active_per_selector = [{
        "selector_seed": seed,
        **summaries[("active", 30, seed)]["metrics"]["online"],
        "delta_j_vs_static": summaries[("active", 30, seed)]["paired_delta"]["j"],
        "query_count": summaries[("active", 30, seed)]["selector"]["query_count"],
        "final_weights": summaries[("active", 30, seed)]["method"]["final_weights"],
    } for seed in seeds]
    random_per_selector = [{
        "selector_seed": seed,
        **summaries[("random", 30, seed)]["metrics"]["online"],
        "delta_j_vs_static": summaries[("random", 30, seed)]["paired_delta"]["j"],
        "query_count": summaries[("random", 30, seed)]["selector"]["query_count"],
        "final_weights": summaries[("random", 30, seed)]["method"]["final_weights"],
    } for seed in seeds]
    direct_aggregate = {
        name: _mean_metrics([row[name] for row in direct_rows_report])
        for name in ("static", "direct_frontload", "causal_online")
    }
    report = {
        "schema_version": "east-online-expert-mixture-adjudication-v1",
        "status": "DIAGNOSTIC_ONLY_TEST_DRIVEN_EXPLORATION",
        "independent_future_confirmation": False,
        "main_30_percent": {
            "static": main_static,
            "active_equal_seed_mean": main_active,
            "random_equal_seed_mean": main_random,
            "active_per_selector": active_per_selector,
            "random_per_selector": random_per_selector,
            "active_minus_static_delta_j": main_active["j"] - main_static["j"],
            "active_minus_random_delta_j": main_active["j"] - main_random["j"],
            "active_all_seed_improves_static": all(
                summaries[("active", 30, seed)]["paired_delta"]["j"] > 0.0
                for seed in seeds
            ),
            "active_all_seed_improves_paired_random": all(
                summaries[("active", 30, seed)]["metrics"]["online"]["j"]
                > summaries[("random", 30, seed)]["metrics"]["online"]["j"]
                for seed in seeds
            ),
        },
        "bootstrap": {
            "active_minus_static": paired_stratified_bootstrap(
                active_pairs, labels=labels_full,
            ),
            "active_minus_paired_random": paired_stratified_bootstrap(
                random_pairs, labels=labels_full,
            ),
            "complement_online_minus_direct_frontload": variable_roster_bootstrap(
                direct_bootstrap_pairs,
            ),
            "complement_online_minus_static": variable_roster_bootstrap(
                static_complement_pairs,
            ),
        },
        "budget_curve": budget_rows,
        "released_run_ledger_audits": ledger_audits,
        "equal_label_budget_control": {
            "design": (
                "For each selector seed, fit the same trust-region convex state once from "
                "the 108 queried labels, remove those shots, and compare fixed direct "
                "front-load, causal online, and static predictions on the identical 252-shot complement."
            ),
            "per_seed": direct_rows_report,
            "equal_seed_mean": direct_aggregate,
            "all_seed_online_beats_direct": all(
                row["online_minus_direct_j"] > 0.0 for row in direct_rows_report
            ),
        },
        "contracts": {
            "source_label_policy": "max(25 ms, AdvancedTime); missing -> 25 ms",
            "east_support": runner.support_contract(support),
            "east_stream": "chronological balanced360, 180D/180N",
            "minimum_lead_ms": 30.0,
            "over_3000ms_counts_as_tpr": True,
            "support_initial_weights": initial.tolist(),
            "maximum_expert_weight": float(np.max(initial)),
            "maximum_expert_weight_source": "support110-selected initial weights",
            "neural_parameter_updates": 0,
            "pre_final_direct_control_outcomes": "queried shots only",
            "union_of_queried_shots_opened_before_final": len(prefinal_opened),
            "candidate_decision_seal": candidate_seal,
        },
        "limitations": [
            "balanced360 informed method and parameter selection; this is not an independent future confirmation.",
            "The three runs vary query-selector exploration, while the three frozen experts are a single ensemble rather than independent deployed models.",
            "The active-vs-random paired bootstrap interval may include zero even when all three observed selector seeds favor active querying.",
            "The direct-label control matches label access and online state capacity, not neural-training FLOPs of a full target re-fit.",
        ],
    }
    report_path = output / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    static = report["main_30_percent"]["static"]
    active = report["main_30_percent"]["active_equal_seed_mean"]
    random = report["main_30_percent"]["random_equal_seed_mean"]
    direct = report["equal_label_budget_control"]["equal_seed_mean"]
    markdown = f"""# EAST 冻结迁移专家在线主动适应：开发流裁决

状态：`DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION`。balanced360 参与了方法开发，不能替代独立未来流确认。

## 主要结果（balanced360，三条查询轨迹等权均值）

| 方法 | TPR@30 ms | FPR | J=TPR@30−FPR | 中位提前量/ms |
|---|---:|---:|---:|---:|
| 支持110选权后的静态冻结集成 | {static['tpr_at_30ms']:.4f} | {static['fpr']:.4f} | {static['j']:.4f} | {static['median_lead_ms']:.1f} |
| 30%随机查询 + 因果权重更新 | {random['tpr_at_30ms']:.4f} | {random['fpr']:.4f} | {random['j']:.4f} | {random['median_lead_ms']:.1f} |
| 30%主动查询 + 支持域信任区 | {active['tpr_at_30ms']:.4f} | {active['fpr']:.4f} | {active['j']:.4f} | {active['median_lead_ms']:.1f} |

主动方案相对同父静态 `ΔJ={report['main_30_percent']['active_minus_static_delta_j']:+.4f}`，三条轨迹均提升；分层配对 bootstrap 95% 区间为 `{report['bootstrap']['active_minus_static']['ci95']}`。主动相对等预算随机 `ΔJ={report['main_30_percent']['active_minus_random_delta_j']:+.4f}`，三条逐一占优，但其区间 `{report['bootstrap']['active_minus_paired_random']['ci95']}` 仍可能跨零，故只能作为支持性证据。

## 等标签预算反事实

每条轨迹把实际查询的108炮一次性用于拟合同容量的固定迁移状态，并从评分中剔除这108炮。剩余252炮的三seed等权结果：

| 方法 | TPR@30 ms | FPR | J |
|---|---:|---:|---:|
| 静态 | {direct['static']['tpr_at_30ms']:.4f} | {direct['static']['fpr']:.4f} | {direct['static']['j']:.4f} |
| 同108标签直接前置适配 | {direct['direct_frontload']['tpr_at_30ms']:.4f} | {direct['direct_frontload']['fpr']:.4f} | {direct['direct_frontload']['j']:.4f} |
| 因果在线逐炮适配 | {direct['causal_online']['tpr_at_30ms']:.4f} | {direct['causal_online']['fpr']:.4f} | {direct['causal_online']['j']:.4f} |

三条轨迹的在线方案均优于同标签直接前置适配；这排除了“增益完全来自额外标签数量”的解释，但尚未匹配完整神经网络重训 FLOPs。

## 方法边界

- 三个迁移神经模型全程冻结；在线只更新三个非负、和为1的专家权重。
- 初始权重仅由 EAST support110 选择，为 `{initial.tolist()}`。
- 任一专家权重不得超过 support110 初始最大暴露 `{float(np.max(initial)):.1f}`，避免少量偏置查询导致单seed塌缩。
- 当前炮始终先预测并持久化，再由不含结局的“不确定性 + 专家分歧”价值函数查询；只有被查询炮才能释放 HDF5 结局并影响后续炮。
- 所有大于等于30 ms的提前报警均计入 TPR；大于3000 ms同样计入，同时单独统计。
"""
    (output / "REPORT_zh.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        runs_root=args.runs_root.expanduser().resolve(),
        parent_root=args.parent_root.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
    )
    print(json.dumps({
        "main_30_percent": report["main_30_percent"],
        "bootstrap": report["bootstrap"],
        "equal_label_budget": report["equal_label_budget_control"]["equal_seed_mean"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "paired_stratified_bootstrap", "select_batch_weights",
    "variable_roster_bootstrap",
]
