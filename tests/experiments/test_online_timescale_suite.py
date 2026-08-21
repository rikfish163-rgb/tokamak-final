from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))


def load_module():
    path = HERE / "run_online_timescale_ablation_suite.py"
    spec = importlib.util.spec_from_file_location("online_timescale_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_suite_command_isolates_modes_and_prices_only_label_queries(tmp_path: Path) -> None:
    module = load_module()
    common = module.SuitePaths(
        source_checkpoint=tmp_path / "source.pt",
        a1_dir=tmp_path / "a1",
        adapter_dir=tmp_path / "adapter",
        out_root=tmp_path / "out",
    )
    intra = module.build_runner_command(
        seed=131, max_shots=360, mode="intra_only", paths=common,
        query_cost=0.35, query_quota=2,
    )
    inter = module.build_runner_command(
        seed=131, max_shots=360, mode="inter_only", paths=common,
        query_cost=0.35, query_quota=2,
    )
    joint = module.build_runner_command(
        seed=131, max_shots=360, mode="intra_inter", paths=common,
        query_cost=0.35, query_quota=2,
        query_score_mode="gradient_blend", query_quantile=0.80,
        episode_physics_escape=0.80,
    )

    assert intra[intra.index("--query-cost") + 1] == "0.0"
    assert inter[inter.index("--query-cost") + 1] == "0.35"
    assert joint[joint.index("--query-cost") + 1] == "0.35"
    assert joint[joint.index("--query-score-mode") + 1] == "gradient_blend"
    assert joint[joint.index("--query-quantile") + 1] == "0.8"
    assert joint[joint.index("--episode-physics-escape") + 1] == "0.8"
    for command, mode in ((intra, "intra_only"), (inter, "inter_only"), (joint, "intra_inter")):
        assert command[command.index("--timescale-mode") + 1] == mode
        assert command[command.index("--query-quota") + 1] == "2"
        assert "--development-search" in command
        assert "--episode-isolate-learning-feedback" in command


def test_suite_forwards_the_sealed_parent_state(tmp_path: Path) -> None:
    module = load_module()
    state = tmp_path / "sealed_parent.pt"
    common = module.SuitePaths(
        source_checkpoint=tmp_path / "source.pt",
        a1_dir=tmp_path / "a1",
        adapter_dir=tmp_path / "adapter",
        out_root=tmp_path / "out",
        sealed_parent_state=state,
    )

    command = module.build_runner_command(
        seed=131, max_shots=360, mode="inter_only", paths=common,
        query_cost=0.0, query_quota=6,
    )

    index = command.index("--sealed-parent-state")
    assert Path(command[index + 1]) == state.resolve()


def test_suite_forwards_causal_error_memory_weight(tmp_path: Path) -> None:
    module = load_module()
    common = module.SuitePaths(
        source_checkpoint=tmp_path / "source.pt",
        a1_dir=tmp_path / "a1",
        adapter_dir=tmp_path / "adapter",
        out_root=tmp_path / "out",
    )

    command = module.build_runner_command(
        seed=131, max_shots=360, mode="intra_inter", paths=common,
        query_cost=0.35, query_quota=2, error_memory_weight=0.60,
    )

    index = command.index("--error-memory-weight")
    assert command[index + 1] == "0.6"


def test_suite_forwards_temporal_risk_forecast_contract(tmp_path: Path) -> None:
    module = load_module()
    common = module.SuitePaths(
        source_checkpoint=tmp_path / "source.pt",
        a1_dir=tmp_path / "a1",
        adapter_dir=tmp_path / "adapter",
        out_root=tmp_path / "out",
    )

    command = module.build_runner_command(
        seed=131, max_shots=360, mode="intra_inter", paths=common,
        query_cost=0.35, query_quota=2,
        adaptation_events=256,
        intra_steps=2,
        intra_lr=5e-4,
        private_intra_lr=4e-4,
        intra_physics_weight=1.0,
        intra_observed_physics_weight=1.0,
        intra_parameter_scope="representation_only",
        intra_temporal_risk_weight=1.0,
        intra_temporal_target_max_delta=0.5,
        intra_temporal_target_gain=4.0,
    )

    shared_lr = command.index("--intra-lr")
    steps = command.index("--intra-steps")
    events = command.index("--adaptation-events")
    private_lr = command.index("--private-intra-lr")
    physics_weight = command.index("--intra-physics-weight")
    observed_physics_weight = command.index(
        "--intra-observed-physics-weight"
    )
    parameter_scope = command.index("--intra-parameter-scope")
    weight = command.index("--intra-temporal-risk-weight")
    delta = command.index("--intra-temporal-target-max-delta")
    gain = command.index("--intra-temporal-target-gain")
    assert command[shared_lr + 1] == "0.0005"
    assert command[steps + 1] == "2"
    assert command[events + 1] == "256"
    assert command[private_lr + 1] == "0.0004"
    assert command[physics_weight + 1] == "1.0"
    assert command[observed_physics_weight + 1] == "1.0"
    assert command[parameter_scope + 1] == "representation_only"
    assert command[weight + 1] == "1.0"
    assert command[delta + 1] == "0.5"
    assert command[gain + 1] == "4.0"


def test_order_diagnostic_suite_separates_fixed_query_and_oracle_query_ceilings(
    tmp_path: Path,
) -> None:
    path = HERE / "run_online_order_diagnostic_suite.py"
    spec = importlib.util.spec_from_file_location("online_order_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    specs = module.order_diagnostic_specs(tmp_path / "manifests")
    assert [item.name for item in specs] == [
        "random_fixed_q", "chronological_oracle_q",
        "oracle_order_fixed_q", "oracle_order_plus_q",
    ]
    assert specs[0].query_manifest.name == "chronological_queries.json"
    assert specs[1].query_manifest.name == "chronological_oracle_error_queries.json"
    assert specs[2].query_manifest.name == "chronological_queries.json"
    assert specs[3].query_manifest.name == "oracle_error_queries.json"
    assert specs[0].label_oracle_used is False
    assert specs[1].label_oracle_used is True
    assert specs[2].label_oracle_used is True
    assert specs[3].label_oracle_used is True
