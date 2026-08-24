from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "experiments/transfer_methods_2026/paper_grade_package/scripts"
    / "validate_dynamic_mainline_package.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "validate_dynamic_mainline_package", SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _one_page_pdf(path: Path) -> None:
    from matplotlib.figure import Figure

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(1, 1))
    figure.text(0.2, 0.5, "dynamic mainline")
    figure.savefig(path, format="pdf")


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    repo = tmp_path / "repo"
    package = repo / "experiments/transfer_methods_2026/paper_grade_package"
    evidence = package / "evidence/dynamic_mainline_v1/report.json"
    figures = package / "figures_dynamic/figure_manifest_v2.json"
    config = package / "configs/primary_chain_dynamic_source_v1.json"
    pdf = package / "technical_report_dynamic/main.pdf"
    code_manifest = repo / "config/mainline_code_manifest.json"
    code = repo / "model.py"

    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("VALUE = 1\n", encoding="utf-8")
    _json(code_manifest, {
        "schema_version": "jtext-east-mainline-code-manifest-v1",
        "tracked_files": ["model.py"],
    })
    _json(evidence, {
        "schema_version": "dynamic-mainline-evidence-v1",
        "current_results": {"operational_tpr": 0.733333, "fpr": 0.220370},
    })
    _json(figures, {
        "schema_version": "figure-manifest-v2",
        "figures": [],
    })
    _json(config, {
        "schema_version": "primary-chain-dynamic-source-v1",
        "runtime": {"data_root": "${FUSION_DATA_ROOT}"},
    })
    _one_page_pdf(pdf)

    paths = {
        "evidence_report": evidence,
        "figure_manifest": figures,
        "report_pdf": pdf,
        "effective_config": config,
        "code_manifest": code_manifest,
    }
    artifacts = {
        role: {
            "path": str(path.relative_to(repo)),
            "sha256": _sha(path),
        }
        for role, path in paths.items()
    }
    artifacts["report_pdf"]["input_sha256"] = {
        role: artifacts[role]["sha256"]
        for role in (
            "evidence_report", "figure_manifest", "effective_config",
            "code_manifest",
        )
    }
    manifest = {
        "schema_version": "dynamic-mainline-package-manifest-v1",
        "artifacts": artifacts,
        "code_files": [{"path": "model.py", "sha256": _sha(code)}],
    }
    manifest_path = package / "technical_report_dynamic/build_manifest.json"
    _json(manifest_path, manifest)
    return repo, manifest_path, manifest


def _refresh_artifact(
    manifest_path: Path, manifest: dict, role: str, artifact: Path,
) -> None:
    digest = _sha(artifact)
    manifest["artifacts"][role]["sha256"] = digest
    if role != "report_pdf":
        manifest["artifacts"]["report_pdf"]["input_sha256"][role] = digest
    _json(manifest_path, manifest)


def test_valid_hash_closed_package_passes_and_pdf_has_pages(tmp_path: Path) -> None:
    validator = _module()
    repo, manifest, _ = _fixture(tmp_path)

    result = validator.validate_package(repo_root=repo, manifest_path=manifest)

    assert result["passed"] is True
    assert result["pdf"]["pages"] == 1
    assert result["failures"] == []


def test_missing_or_hash_drifted_input_fails_closed(tmp_path: Path) -> None:
    validator = _module()
    repo, manifest, _ = _fixture(tmp_path)
    evidence = repo / (
        "experiments/transfer_methods_2026/paper_grade_package/"
        "evidence/dynamic_mainline_v1/report.json"
    )
    evidence.write_text("{}\n", encoding="utf-8")

    result = validator.validate_package(repo_root=repo, manifest_path=manifest)

    assert result["passed"] is False
    assert any("hash mismatch" in failure for failure in result["failures"])


def test_portable_manifests_reject_home_or_media_absolute_paths(
    tmp_path: Path,
) -> None:
    validator = _module()
    repo, manifest_path, manifest = _fixture(tmp_path)
    figure_manifest = repo / manifest["artifacts"]["figure_manifest"]["path"]
    _json(figure_manifest, {
        "schema_version": "figure-manifest-v2",
        "source": "/home/user/private/run.json",
    })
    _refresh_artifact(manifest_path, manifest, "figure_manifest", figure_manifest)

    result = validator.validate_package(
        repo_root=repo, manifest_path=manifest_path,
    )

    assert result["passed"] is False
    assert any("machine absolute path" in item for item in result["failures"])


def test_legacy_v3c_operating_point_cannot_be_current_result(
    tmp_path: Path,
) -> None:
    validator = _module()
    repo, manifest_path, manifest = _fixture(tmp_path)
    evidence = repo / manifest["artifacts"]["evidence_report"]["path"]
    _json(evidence, {
        "schema_version": "dynamic-mainline-evidence-v1",
        "current_results": {"operational_tpr": 0.8407, "fpr": 0.1667},
    })
    _refresh_artifact(manifest_path, manifest, "evidence_report", evidence)

    result = validator.validate_package(
        repo_root=repo, manifest_path=manifest_path,
    )

    assert result["passed"] is False
    assert any("legacy v3c metric" in item for item in result["failures"])


def test_corrupt_pdf_and_stale_pdf_binding_fail_closed(tmp_path: Path) -> None:
    validator = _module()
    repo, manifest_path, manifest = _fixture(tmp_path)
    pdf = repo / manifest["artifacts"]["report_pdf"]["path"]
    pdf.write_bytes(b"%PDF-not-parseable")
    _refresh_artifact(manifest_path, manifest, "report_pdf", pdf)

    corrupt = validator.validate_package(
        repo_root=repo, manifest_path=manifest_path,
    )
    assert corrupt["passed"] is False
    assert any("PDF" in item for item in corrupt["failures"])

    repo, manifest_path, manifest = _fixture(tmp_path / "stale")
    manifest["artifacts"]["report_pdf"]["input_sha256"][
        "effective_config"
    ] = "0" * 64
    _json(manifest_path, manifest)
    stale = validator.validate_package(
        repo_root=repo, manifest_path=manifest_path,
    )
    assert stale["passed"] is False
    assert any("stale PDF binding" in item for item in stale["failures"])
