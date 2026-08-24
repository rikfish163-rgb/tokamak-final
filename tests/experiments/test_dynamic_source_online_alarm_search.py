from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
MODULE = HERE / "run_dynamic_source_online_alarm_search.py"
CONFIG = (
    HERE / "paper_grade_package/configs/"
    "dynamic_source_online_alarm_search_v1.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_dynamic_source_online_alarm_search", MODULE,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_search_config_changes_only_registered_alarm_initialization() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["claim_status"] == "TEST_DRIVEN_EXPLORATION"
    assert payload["selection_stream"] == "EAST balanced360"
    assert payload["method_change_scope"] == "alarm_state_initialization_only"
    assert payload["selected_candidate"] == {
        "initial_high_threshold": -0.1,
        "initial_physics_weight": 6.0,
        "initial_trend_weight": 13.5,
        "initial_early_threshold": 7.5,
    }


def test_replay_command_reuses_sealed_source_parent_and_full_online_method(
    tmp_path: Path,
) -> None:
    module = load_module()
    parent_root = tmp_path / "parents"
    seed_root = parent_root / "seed131"
    for relative in (
        "source-train/source_checkpoint.pt",
        "east-parent/east_fixed_parent.pt",
        "east-parent/balanced360_roster.jsonl",
        "run_contract.json",
    ):
        path = seed_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    command, receipt = module.build_replay_command(
        seed=131,
        parent_root=parent_root,
        output=tmp_path / "out",
        python=Path(sys.executable),
        search_config_path=CONFIG,
    )

    assert "--static-only" not in command
    assert command[command.index("--max-shots") + 1] == "360"
    assert command[command.index("--initial-high-threshold") + 1] == "-0.1"
    assert command[command.index("--initial-physics-weight") + 1] == "6.0"
    assert command[command.index("--initial-trend-weight") + 1] == "13.5"
    assert command[command.index("--initial-early-threshold") + 1] == "7.5"
    assert "--strict-window-active" in command
    assert "--typed-endpoint-memory" in command
    assert "--episode-value-v3" in command
    assert receipt["reused_parent_artifacts"] is True
    assert receipt["source_training_repeated"] is False
    assert receipt["east_support_training_repeated"] is False
