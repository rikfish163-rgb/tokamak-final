from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_archived_parent_replay_uses_ordered_unique_roster_manifest_without_a1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The runner imports h5py for the later replay phase.  This test stops
    # before that phase, so keep the regression independent of the optional
    # HDF5/GPU runtime and ensure no HDF5 API can be reached accidentally.
    monkeypatch.setitem(sys.modules, "h5py", types.ModuleType("h5py"))
    fake_torch = types.ModuleType("torch")
    fake_torch_nn = types.ModuleType("torch.nn")
    fake_torch_functional = types.ModuleType("torch.nn.functional")
    fake_torch.nn = fake_torch_nn
    fake_torch_nn.functional = fake_torch_functional
    for name, value in (
        ("torch", fake_torch),
        ("torch.nn", fake_torch_nn),
        ("torch.nn.functional", fake_torch_functional),
    ):
        monkeypatch.setitem(sys.modules, name, value)
    for name in (
        "dual_threshold_alarm", "dynamic_hazard_adapter",
        "east_private23_sidecar", "east_private_continual",
        "evaluate_dynamic_positive_gate", "online_active_audit",
        "online_auxiliary", "online_episode_value_v3",
        "run_dynamic_hazard_adapter", "run_east_private23_online",
        "run_online_active_a1", "run_online_active_a3",
        "run_online_protected_residual", "run_two_timescale_online",
        "shot_alarm", "two_timescale_online_adapter", "unified_online_value",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    runner = load_module(
        "run_unified_online_active_archived_parent_test",
        HERE / "run_unified_online_active.py",
    )
    roster_manifest = tmp_path / "primary_static_roster.jsonl"
    roster_manifest.write_text(
        "".join(
            json.dumps({"shot": shot, "static_risk": 0.1}) + "\n"
            for shot in (84821, 84834, 84899)
        )
    )
    sealed_parent = tmp_path / "archived" / "frontloaded_target_state.pt"
    sealed_parent.parent.mkdir()
    sealed_parent.write_bytes(b"placeholder sealed parent")
    missing_a1 = tmp_path / "a1-does-not-exist"
    missing_adapter = tmp_path / "adapter-does-not-exist"
    captured: dict[str, Any] = {}
    original_resolve = runner.timescale_ablation.resolve_online_roster

    def capture_roster(
        base_roster: list[int], *, order_manifest: Path | None,
        allow_diagnostic_reorder: bool,
    ) -> Any:
        resolution = original_resolve(
            base_roster,
            order_manifest=order_manifest,
            allow_diagnostic_reorder=allow_diagnostic_reorder,
        )
        captured["base_roster"] = list(base_roster)
        captured["resolved_roster"] = list(resolution.roster)
        captured["order_manifest"] = order_manifest
        captured["allow_diagnostic_reorder"] = allow_diagnostic_reorder
        return resolution

    monkeypatch.setattr(
        runner.timescale_ablation, "resolve_online_roster", capture_roster,
    )
    runner.alarm.DEV = "cpu"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unified_online_active.py",
            "--seed", "131",
            "--out", str(tmp_path / "runner-output"),
            "--sealed-parent-state", str(sealed_parent),
            "--roster-manifest", str(roster_manifest),
            "--a1-dir", str(missing_a1),
            "--adapter-dir", str(missing_adapter),
        ],
    )

    with pytest.raises(RuntimeError, match="requires rl_env CUDA"):
        runner.main()

    assert captured == {
        "base_roster": [84821, 84834, 84899],
        "resolved_roster": [84821, 84834, 84899],
        "order_manifest": None,
        "allow_diagnostic_reorder": False,
    }
    assert not (missing_a1 / "config.json").exists()
    assert not (missing_a1 / "seals.jsonl").exists()


def test_random_query_plan_has_no_target_commands_and_three_online_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = load_module(
        "run_causal_random_query_baseline_test",
        HERE / "paper_grade_package/scripts/run_causal_random_query_baseline.py",
    )
    seeds = (131, 133, 137)
    sealed_root = tmp_path / "sealed-parents"
    primary_root = tmp_path / "primary"
    for seed in seeds:
        parent_dir = sealed_root / f"seed{seed}" / "support_only"
        parent_dir.mkdir(parents=True)
        (parent_dir / "frontloaded_target_state.pt").write_bytes(b"sealed")
        (parent_dir / "parent_fidelity.json").write_text("{}\n")

    manifest_paths: dict[int, Path] = {}

    def fake_build_manifests(
        *, seed: int, output: Path, replicates: int,
    ) -> list[Path]:
        assert replicates == 1
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"seed{seed}_random_query_r0.json"
        path.write_text(json.dumps({"shots": [1, 2]}) + "\n")
        manifest_paths[seed] = path
        return [path]

    monkeypatch.setattr(baseline, "SEALED_PARENTS", sealed_root)
    monkeypatch.setattr(baseline, "PRIMARY", primary_root)
    monkeypatch.setattr(baseline, "build_manifests", fake_build_manifests)
    output_root = tmp_path / "random-query-plan"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_causal_random_query_baseline.py",
            "--output-root", str(output_root),
            "--seeds", *(str(seed) for seed in seeds),
            "--replicates", "1",
            "--stage", "plan",
        ],
    )

    baseline.main()

    plan = json.loads((output_root / "run_plan.json").read_text())
    assert [entry["seed"] for entry in plan["commands"]] == list(seeds)
    for entry in plan["commands"]:
        seed = int(entry["seed"])
        assert entry["target"] == []
        assert len(entry["online"]) == 1
        command = entry["online"][0]
        for option in (
            "--sealed-parent-state",
            "--roster-manifest",
            "--fixed-query-manifest",
        ):
            assert option in command
        sealed_index = command.index("--sealed-parent-state")
        assert Path(command[sealed_index + 1]) == (
            sealed_root / f"seed{seed}/support_only/frontloaded_target_state.pt"
        )
        roster_index = command.index("--roster-manifest")
        assert Path(command[roster_index + 1]) == (
            primary_root / f"seed{seed}/static_predictions.jsonl"
        )
        query_index = command.index("--fixed-query-manifest")
        assert Path(command[query_index + 1]) == manifest_paths[seed]


def test_random_query_report_discloses_realized_compute_mediation(
    tmp_path: Path,
) -> None:
    report_module = load_module(
        "build_causal_random_query_report_test",
        HERE / "paper_grade_package/scripts/build_causal_random_query_report.py",
    )
    package = tmp_path / "package"
    random_root = tmp_path / "random"
    hashes = {
        "source_checkpoint": "source",
        "base_before_after": "base",
        "phase2_initial_private": "private",
        "phase2_initial_shared": "shared",
        "static_predictions": "static",
    }

    def summary(*, tpr: float, fpr: float, queries: int, inter: int) -> dict[str, Any]:
        return {
            "query_count": queries,
            "hashes": hashes,
            "results": {"online_unified": {
                "tpr_at_30ms": tpr, "fpr": fpr,
                "tpr": tpr, "median_lead_ms": 100.0,
            }},
            "counts": {
                "intra_attempts": 100,
                "shared_inter_accepted": inter,
                "shared_inter_rollback": 0,
                "private_inter_accepted": inter,
                "private_inter_rollback": 0,
            },
        }

    for seed in (131, 133, 137):
        active_dir = package / f"evidence/primary_online_artifacts/seed{seed}"
        active_dir.mkdir(parents=True)
        (active_dir / "summary.json").write_text(
            json.dumps(summary(tpr=0.85, fpr=0.15, queries=30, inter=12))
        )
        for replicate in range(2):
            run = random_root / f"seed{seed}/causal_random_query_r{replicate}"
            run.mkdir(parents=True)
            (run / "summary.json").write_text(
                json.dumps(summary(tpr=0.82, fpr=0.16, queries=30, inter=6))
            )

    report = report_module.build_report(
        package=package, random_root=random_root,
        seeds=[131, 133, 137], replicates=2,
    )
    aggregate = report["aggregate"]
    assert aggregate["active_minus_random_utility_mean"] == pytest.approx(0.04)
    assert aggregate["seeds_active_better"] == 3
    assert aggregate["exact_one_sided_sign_test_p_if_all_three_positive"] == 0.125
    assert aggregate["active_shared_inter_attempts_equal_seed_mean"] == 12
    assert aggregate["random_shared_inter_attempts_equal_seed_mean"] == 6
    assert aggregate["realized_inter_attempt_ratio_active_over_random"] == 2
    assert any("not an equal-FLOP" in text for text in report["limitations"])
