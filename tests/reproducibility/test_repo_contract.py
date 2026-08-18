from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_module(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_project_paths_are_checkout_relative_and_data_can_be_rebound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = load_module("reproducibility/paths.py", "reproducibility_paths")
    monkeypatch.delenv("FUSION_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("FUSION_DATA_ROOT", raising=False)
    assert paths.project_root() == ROOT
    assert paths.data_root() == ROOT / "data"

    external = tmp_path / "tokamak-data"
    monkeypatch.setenv("FUSION_DATA_ROOT", str(external))
    assert paths.data_root() == external.resolve()


def test_data_audit_fails_closed_without_required_layout(tmp_path: Path) -> None:
    contract = load_module(
        "reproducibility/data_contract.py", "reproducibility_data_contract"
    )
    with pytest.raises(contract.DataContractError, match="J-TEXT/unified_hdf5"):
        contract.audit_data(tmp_path, full=False)


def test_python_sources_do_not_embed_original_machine_path() -> None:
    forbidden = "Fuison/experiments/transfer_methods_2026"
    offenders = []
    for path in (ROOT / "experiments/transfer_methods_2026").glob("*.py"):
        if forbidden in path.read_text(errors="replace"):
            offenders.append(path.name)
    assert offenders == []


def test_github_repository_has_no_file_at_or_above_100_mb() -> None:
    if not (ROOT / ".git").is_dir():
        pytest.skip("tracked-file size check runs after local git initialization")
    offenders = []
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw in tracked:
        if not raw:
            continue
        path = ROOT / raw.decode()
        if path.is_file() and path.stat().st_size >= 100_000_000:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_reference_verifier_accepts_committed_multiseed_result() -> None:
    verifier = load_module(
        "reproducibility/verify_results.py", "reproducibility_verify_results"
    )
    path = ROOT / "results/reference/multiseed_summary.json"
    payload = json.loads(path.read_text())
    verdict = verifier.compare_multiseed_results(payload, payload)
    assert verdict["passed"] is True
    assert verdict["checked_seeds"] == [131, 133, 137]


def test_unified_runner_accepts_explicit_stage_inputs() -> None:
    source = (
        ROOT / "experiments/transfer_methods_2026/run_unified_online_active.py"
    ).read_text()
    assert 'parser.add_argument("--a1-dir"' in source
    assert 'parser.add_argument("--adapter-dir"' in source


def test_reproduction_commands_connect_all_three_stages(tmp_path: Path) -> None:
    runner = load_module("reproducibility/reproduce.py", "reproducibility_reproduce")
    commands = runner.build_seed_commands(
        seed=131, output_root=tmp_path, python_executable="python",
    )
    assert len(commands) == 3
    assert commands[0][-2:] == ["--out", str(tmp_path / "seed131/a1")]
    assert "--a1" in commands[1]
    assert "--a1-dir" in commands[2]
    assert "--adapter-dir" in commands[2]
    assert "--locked-best-v8" in commands[2]


def test_reproduction_reference_can_be_scoped_to_requested_seeds() -> None:
    runner = load_module("reproducibility/reproduce.py", "reproducibility_reproduce_subset")
    reference = json.loads(
        (ROOT / "results/reference/multiseed_summary.json").read_text()
    )
    scoped = runner._reference_for_seeds(reference, (131,))
    assert [row["seed"] for row in scoped["seeds"]] == [131]
    verdict = runner.compare_multiseed_results(scoped, scoped)
    assert verdict["passed"] is True
