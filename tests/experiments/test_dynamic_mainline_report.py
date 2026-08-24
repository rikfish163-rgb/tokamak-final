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
    assert "README.md" not in {item["path"] for item in payload["artifacts"]}
    assert "README_CN.md" not in {item["path"] for item in payload["artifacts"]}


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


def test_root_readme_preserves_original_v3c_and_budget_controls() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # Original paper-v3c lineage remains the repository headline result.
    assert "0.840741" in readme
    assert "0.166667" in readme
    assert "0.848148" in readme
    assert "0.151852" in readme
    assert "102/90/99" in readme

    # Query-budget curve and the same-30-label direct-frontload control remain visible.
    for token in ("16", "35", "66", "97", "0.830189", "0.140351",
                  "0.836478", "0.134503", "0.937107", "0.274854"):
        assert token in readme

    # Dynamic-label reruns are retained only as a separately named appendix.
    assert "AdvancedTime 动态标签重跑附录" in readme
    assert "不得覆盖原 paper-v3c 主结果" in readme


def test_root_headline_matches_sealed_v3c_and_q30_evidence() -> None:
    evidence = PACKAGE / "evidence"
    primary = json.loads((evidence / "three_stage_results.json").read_text())
    lineage = primary["primary_lineage"]
    static = lineage["east_static360"]["aggregate"]
    online = lineage["east_online360"]["aggregate"]
    assert static["tpr_at_30ms"]["mean"] == 0.8407407407407407
    assert static["fpr"]["mean"] == 0.16666666666666666
    assert online["tpr_at_30ms"]["mean"] == 0.8481481481481481
    assert online["fpr"]["mean"] == 0.15185185185185185

    q30 = json.loads((
        evidence / "causal_timescale_equal_budget_seed131_20260821/report.json"
    ).read_text())
    complement = q30["same_label_disjoint_330_complement"]
    assert complement["online_intra_inter"]["utility_tpr30_minus_fpr"] == 0.7019750634447754
    assert complement["direct_frontload_support110_recalibrated"]["utility_tpr30_minus_fpr"] == 0.6676597153260511
    assert q30["equal_label_budget_verdict"]["compute_matched"] is False
