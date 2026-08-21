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


def test_content_hash_is_byte_exact(tmp_path: Path) -> None:
    contract = load_module(
        "reproducibility/data_contract.py", "reproducibility_content_hash"
    )
    payload = tmp_path / "shot.hdf5"
    payload.write_bytes(b"tokamak-content-v1\n")
    assert contract._sha256_file(payload) == (
        "a4ff12f3b490e99ed58b8c03477f6e26973cc38d692139530e1041ec7dfca36b"
    )


def test_portable_audit_never_serializes_machine_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    contract = load_module(
        "reproducibility/data_contract.py", "reproducibility_portable_audit"
    )
    for device in ("J-TEXT", "EAST"):
        (tmp_path / device / "unified_hdf5").mkdir(parents=True)
    monkeypatch.setattr(contract, "_jtext_ids", lambda _root: [])
    monkeypatch.setattr(contract, "_east_ids", lambda _root: [])
    result = contract.audit_data(tmp_path, portable=True)
    assert result["data_root"] == "<external-data-root>"
    assert str(tmp_path) not in json.dumps(result)


def test_data_contract_enforces_canonical_jtext_role_counts_and_no_overlap() -> None:
    contract = load_module(
        "reproducibility/data_contract.py", "reproducibility_jtext_split_contract"
    )

    roles = contract._jtext_roles(ROOT)

    assert {name: len(values) for name, values in roles.items()} == {
        "train": 1353,
        "val": 160,
        "test": 220,
    }
    assert len(set().union(*map(set, roles.values()))) == 1733


def test_python_sources_do_not_embed_original_machine_path() -> None:
    forbidden = "Fuison/experiments/transfer_methods_2026"
    offenders = []
    for path in (ROOT / "experiments/transfer_methods_2026").glob("*.py"):
        if forbidden in path.read_text(errors="replace"):
            offenders.append(path.name)
    assert offenders == []


def test_manifest_labels_never_override_hdf5_outcomes() -> None:
    common = (
        ROOT / "experiments/transfer_methods_2026/common.py"
    ).read_text()
    rebuilt = (
        ROOT / "experiments/transfer_methods_2026/rebuilt_harness.py"
    ).read_text()

    assert 'sh["is_disrupt"] = lab' not in common
    assert "isd = bool(label_map" not in rebuilt


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
    path = ROOT / "results/reference/paper_v3c_multiseed_summary.json"
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
    assert len(commands) == 1
    command = commands[0]
    assert command[0] == "python"
    assert command[1].endswith(
        "paper_grade_package/scripts/reproduce_primary_chain.py"
    )
    assert command[command.index("--seed") + 1] == "131"
    assert command[command.index("--stage") + 1] == "all"
    assert Path(command[command.index("--output-root") + 1]) == tmp_path
    assert "--locked-best-v8" not in command


def test_reproduction_reference_can_be_scoped_to_requested_seeds() -> None:
    runner = load_module("reproducibility/reproduce.py", "reproducibility_reproduce_subset")
    reference = json.loads(
        (ROOT / "results/reference/paper_v3c_multiseed_summary.json").read_text()
    )
    scoped = runner._reference_for_seeds(reference, (131,))
    assert [row["seed"] for row in scoped["seeds"]] == [131]
    verdict = runner.compare_multiseed_results(scoped, scoped)
    assert verdict["passed"] is True


def test_readme_distinguishes_cpu_audit_from_cuda_and_checkout_cli() -> None:
    readme = (ROOT / "README.md").read_text()

    assert "uv sync --frozen --extra dev" not in readme
    assert "uv sync --frozen --only-group ci --no-install-project" in readme
    assert "not a standalone wheel" in readme.lower()
    assert "FUSION_PROJECT_ROOT" in readme
