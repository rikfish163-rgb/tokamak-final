#!/usr/bin/env python3
"""Validate and compile the dynamic-source Chinese technical report.

The script binds the reader-facing report to the independently recomputed,
post-seal static-versus-online evidence object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Mapping


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
LATEX_DIR = PACKAGE / "technical_report_dynamic"

HEADLINE_METRICS = {
    "jtext_static": {"tpr": 0.866667, "fpr": 0.130303, "auc": 0.967906, "j": 0.736364},
    "east_static": {"tpr": 0.820370, "fpr": 0.157407, "j": 0.662963},
    "east_online": {"tpr": 0.824074, "fpr": 0.146296, "j": 0.677778},
}

# These values belong to the separate paper-v3c lineage.  They may remain in
# the repository headline README, but must not be copied into this dynamic-label
# appendix as though both reports shared one checkpoint and label contract.
RETIRED_V3C_TOKENS = (
    "0.8515", "0.2121", "0.9219", "0.8407", "0.1667", "0.8481", "0.1519",
)

ACTIVE_ARTIFACTS = (
    Path("experiments/transfer_methods_2026/paper_grade_package/DYNAMIC_MAINLINE_TECHNICAL_REPORT_zh.md"),
    Path("experiments/transfer_methods_2026/paper_grade_package/technical_report_dynamic/main.tex"),
    Path("experiments/transfer_methods_2026/paper_grade_package/evidence_dynamic/dynamic_source_online_gain/report.json"),
    Path("experiments/transfer_methods_2026/paper_grade_package/configs/dynamic_source_online_alarm_search_v1.json"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_retired_v3c_tokens(texts: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {"path": str(path), "token": token}
        for path, text in texts.items()
        for token in RETIRED_V3C_TOKENS
        if token in text
    ]


def audit_repository(repo: Path = REPO) -> dict:
    repo = Path(repo).resolve()
    paths = [repo / relative for relative in ACTIVE_ARTIFACTS]
    missing = [str(path.relative_to(repo)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"dynamic report artifacts missing: {missing}")
    texts = {
        str(path.relative_to(repo)): path.read_text(encoding="utf-8") for path in paths
    }
    retired_hits = find_retired_v3c_tokens(texts)
    if retired_hits:
        raise ValueError(f"retired v3c metrics leaked into active documents: {retired_hits}")
    merged = "\n".join(texts.values())
    required = (
        "TEST_DRIVEN_EXPLORATION",
        "DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION",
        "max(25",
        "/meta/IsDisrupt",
        "/meta/DownTime",
        "0.866667",
        "0.130303",
        "0.820370",
        "0.157407",
        "0.824074",
        "0.146296",
        "0.677778",
        "296",
    )
    absent = [token for token in required if token not in merged]
    if absent:
        raise ValueError(f"dynamic report contract tokens missing: {absent}")
    return {
        "schema_version": "dynamic-mainline-report-manifest-v2",
        "claim_status": "TEST_DRIVEN_EXPLORATION",
        "evidence_status": "DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION",
        "independent_confirmation": False,
        "metrics_origin": "post-seal static and online predictions independently joined to HDF5 truth by validate_dynamic_source_online_gain.py",
        "headline_metrics": HEADLINE_METRICS,
        "retired_v3c_token_hits": retired_hits,
        "artifacts": [
            {
                "path": str(path.relative_to(repo)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
    }


def write_manifest(payload: dict, path: Path) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compile_report(*, output_dir: Path) -> Path:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk is required to compile the dynamic report")
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    command = [
        latexmk,
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={target}",
        "main.tex",
    ]
    subprocess.run(command, cwd=LATEX_DIR, check=True)
    pdf = target / "main.pdf"
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("XeLaTeX did not produce a valid PDF")
    return pdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--manifest", type=Path,
        default=LATEX_DIR / "output/report_manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=LATEX_DIR / "output/pdf",
    )
    args = parser.parse_args()

    payload = audit_repository(REPO)
    write_manifest(payload, args.manifest)
    if not args.check_only:
        pdf = compile_report(output_dir=args.output_dir)
        payload["compiled_pdf"] = {
            "path": str(pdf.relative_to(REPO)) if pdf.is_relative_to(REPO) else pdf.name,
            "sha256": sha256_file(pdf),
            "bytes": pdf.stat().st_size,
        }
        write_manifest(payload, args.manifest)
        print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
