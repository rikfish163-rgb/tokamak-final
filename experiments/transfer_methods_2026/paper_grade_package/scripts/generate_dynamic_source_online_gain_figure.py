#!/usr/bin/env python3
"""Generate the hash-bound paired EAST static/online development figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from publication_style import (
    OKABE_ITO,
    apply_nature_style,
    minimum_configured_font_size,
    save_publication_figure,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_rows(report: dict) -> list[dict]:
    validation = report.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "PASS":
        raise ValueError("figure requires a validator-passed evidence report")
    if report.get("claim_status") != "TEST_DRIVEN_EXPLORATION":
        raise ValueError("balanced360 figure must retain the development claim")
    rows = []
    for row in report["seed_results"]:
        rows.append({
            "seed": int(row["seed"]),
            "static_tpr": float(row["metrics"]["static"]["tpr_at_30ms"]),
            "static_fpr": float(row["metrics"]["static"]["fpr"]),
            "online_tpr": float(row["metrics"]["online"]["tpr_at_30ms"]),
            "online_fpr": float(row["metrics"]["online"]["fpr"]),
            "delta_j": float(row["metrics"]["paired_delta"]["j"]),
            "queries": int(row["query_count"]),
        })
    if [row["seed"] for row in rows] != [131, 133, 137]:
        raise ValueError("figure requires the registered three seeds")
    return rows


def generate(report_path: Path, output_base: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = extract_rows(report)
    apply_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), layout="constrained")
    x = np.arange(3)
    labels = [str(row["seed"]) for row in rows]
    series = (
        ("Static", OKABE_ITO[4], "o", "-"),
        ("Online active", OKABE_ITO[1], "s", "--"),
    )
    for ax, metric, ylabel, limits in (
        (axes[0], "tpr", "Operational TPR (lead ≥30 ms)", (0.72, 0.90)),
        (axes[1], "fpr", "False-positive rate", (0.09, 0.21)),
    ):
        for name, color, marker, linestyle in series:
            prefix = "static" if name == "Static" else "online"
            values = [row[f"{prefix}_{metric}"] for row in rows]
            ax.plot(
                x, values, label=name, color=color, marker=marker,
                linestyle=linestyle, markerfacecolor="white",
                markeredgewidth=1.0,
            )
        ax.set_xticks(x, labels)
        ax.set_xlabel("Training seed")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*limits)
        ax.legend(loc="best")
    deltas = [row["delta_j"] for row in rows]
    axes[2].bar(
        x, deltas, width=0.58, color=OKABE_ITO[2], edgecolor="black",
        linewidth=0.6, hatch="///",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.7)
    axes[2].axhline(
        float(np.mean(deltas)), color=OKABE_ITO[5], linewidth=1.0,
        linestyle="--", label="Mean ΔJ",
    )
    for index, row in enumerate(rows):
        axes[2].text(
            index, row["delta_j"] + 0.0012,
            f"{row['delta_j']:.4f}\nq={row['queries']}",
            ha="center", va="bottom", fontsize=6,
        )
    axes[2].set_xticks(x, labels)
    axes[2].set_xlabel("Training seed")
    axes[2].set_ylabel("Paired ΔJ (online − static)")
    axes[2].set_ylim(0.0, 0.0275)
    axes[2].legend(loc="upper right")
    for label, ax in zip("abc", axes):
        ax.text(
            -0.16, 1.04, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom",
        )
    if minimum_configured_font_size() < 6.0:
        raise RuntimeError("configured publication font is smaller than 6 pt")
    outputs = save_publication_figure(fig, output_base)
    plt.close(fig)
    manifest = {
        "schema_version": "dynamic-source-online-gain-figure-v1",
        "claim_status": "TEST_DRIVEN_EXPLORATION",
        "cohort": "same EAST balanced360 roster (180 disruptive / 180 safe)",
        "replication_unit": "three independently trained model seeds",
        "uncertainty": (
            "exact per-seed paired proportions; no inferential error bars because "
            "the three seeds are not independent physical-shot cohorts"
        ),
        "input": str(report_path.resolve()),
        "input_sha256": sha256_file(report_path),
        "rows": rows,
        "outputs": {
            path.suffix.removeprefix("."): {
                "path": str(path), "sha256": sha256_file(path),
            }
            for path in outputs
        },
    }
    manifest_path = output_base.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.report, args.output)
    print(json.dumps(manifest["outputs"], sort_keys=True))


if __name__ == "__main__":
    main()
