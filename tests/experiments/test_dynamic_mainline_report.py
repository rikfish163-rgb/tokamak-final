from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "experiments/transfer_methods_2026/paper_grade_package"
SCRIPT = PACKAGE / "scripts/build_dynamic_report.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_dynamic_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamic_report_contract_is_current_and_hash_bound(tmp_path: Path) -> None:
    manifest = tmp_path / "report_manifest.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check-only", "--manifest", str(manifest)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "dynamic-mainline-report-manifest-v1"
    assert payload["claim_status"] == "TEST_DRIVEN_EXPLORATION"
    assert payload["independent_confirmation"] is False
    assert payload["retired_v3c_token_hits"] == []
    assert payload["headline_metrics"]["jtext_static"]["tpr"] == 0.866667
    assert payload["headline_metrics"]["east_static"]["fpr"] == 0.142593
    assert payload["headline_metrics"]["east_online"]["j"] == 0.681481
    assert all(len(item["sha256"]) == 64 for item in payload["artifacts"])


def test_retired_v3c_numbers_are_rejected_from_active_documents() -> None:
    module = _module()

    assert module.find_retired_v3c_tokens({"README.md": "TPR 0.8515; FPR 0.2121"})
    assert module.find_retired_v3c_tokens({"README.md": "TPR 0.866667; FPR 0.130303"}) == []


def test_latex_project_declares_glossary_and_formula_code_mapping() -> None:
    tex = (PACKAGE / "technical_report_dynamic/main.tex").read_text(encoding="utf-8")

    assert "符号与缩写" in tex
    assert "公式与代码映射" in tex
    assert "TEST\\_DRIVEN\\_EXPLORATION" in tex
    assert "max(25" in tex
