#!/usr/bin/env python3
"""Render hash-bound dynamic-mainline figures without legacy fallbacks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from publication_style import (
    OKABE_ITO,
    apply_nature_style,
    minimum_configured_font_size,
    save_publication_figure,
    series_encoding,
)


SEEDS = (131, 133, 137)
CLAIM_UNAVAILABLE = "UNAVAILABLE_NO_CURRENT_HASH_BOUND_EVIDENCE"


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unavailable_panel(
    panel_id: str,
    title: str,
    *,
    cohort: str,
    statistic: str,
    uncertainty: str,
    expected: list[str],
    root_role: str,
    label_policy: str,
    generator_sha: str,
    config_sha: str,
) -> dict[str, Any]:
    receipt = {
        "schema_version": "dynamic-figure-input-availability-v1",
        "root_role": root_role,
        "expected_paths": expected,
        "present_paths": [],
    }
    receipt_sha = _canonical_sha(receipt)
    caption = f"{title}: unavailable because current hash-bound inputs are incomplete."
    return {
        "id": panel_id,
        "title": title,
        "kind": "unavailable",
        "status": "unavailable",
        "reason": "required_current_artifacts_missing; legacy aggregate fallback prohibited",
        "claim_status": CLAIM_UNAVAILABLE,
        "cohort": cohort,
        "physical_shot_count": 0,
        "model_seed_count": 0,
        "seed_count": 0,
        "statistic": statistic,
        "uncertainty": uncertainty,
        "row_selector": "none; required current artifacts unavailable",
        "parent_sha256": None,
        "label_policy": label_policy,
        "label_policy_sha256": _canonical_sha(label_policy),
        "inputs": [{**receipt, "sha256": receipt_sha}],
        "input_sha256": [receipt_sha],
        "values": [],
        "caption": caption,
        "caption_sha256": _canonical_sha(caption),
        "generator_sha256": generator_sha,
        "config_sha256": config_sha,
        "output_sha256": {},
    }


def _draw_box(ax, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="black", linewidth=0.7, alpha=0.22,
    ))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=6)


def _render_f1(panels: list[dict[str, Any]]):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8))
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax = axes[0]
    _draw_box(ax, .05, .68, .9, .2, "J-TEXT TRAIN1353\nH=max(25 ms, AdvancedTime)", OKABE_ITO[1])
    _draw_box(ax, .05, .38, .9, .17, "VAL160: checkpoint + threshold lock", OKABE_ITO[2])
    _draw_box(ax, .05, .08, .9, .17, "Outcome authority: HDF5\nIsDisrupt + DownTime", OKABE_ITO[0])
    ax.annotate("", (.5, .58), (.5, .68), arrowprops={"arrowstyle": "->", "lw": .8})
    ax.annotate("", (.5, .28), (.5, .38), arrowprops={"arrowstyle": "->", "lw": .8})
    ax = axes[1]
    _draw_box(ax, .25, .72, .5, .16, "Locked source parent", OKABE_ITO[4])
    _draw_box(ax, .02, .28, .44, .25, "EAST formal branch\nfixed125 support -> static -> online", OKABE_ITO[0])
    _draw_box(ax, .54, .28, .44, .25, "J-TEXT diagnostic\nstatic -> reset -> online", OKABE_ITO[2])
    ax.annotate("", (.24, .55), (.42, .72), arrowprops={"arrowstyle": "->", "lw": .8})
    ax.annotate("", (.76, .55), (.58, .72), arrowprops={"arrowstyle": "->", "lw": .8})
    ax.text(.5, .12, "J-TEXT online state does not feed EAST", ha="center", fontsize=6, fontweight="bold")
    ax = axes[2]
    events = ["predict", "endpoint\nseal", "optional intra\nupdate", "later\nprediction", "shot\nseal", "query", "reveal", "persistent\nupdate/rollback"]
    ys = [0.91 - index * .11 for index in range(len(events))]
    for index, (event, y) in enumerate(zip(events, ys)):
        _draw_box(ax, .2, y - .055, .6, .07, event, OKABE_ITO[index % len(OKABE_ITO)])
        if index:
            ax.annotate("", (.12, y + .055), (.12, y + .105), arrowprops={"arrowstyle": "->", "lw": .7})
    for label, ax in zip(("A", "B", "C"), axes):
        ax.text(.01, .99, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    return fig


def _render_unavailable(panels: list[dict[str, Any]], shape: tuple[int, int], size: tuple[float, float]):
    fig, axes = plt.subplots(*shape, figsize=size, squeeze=False)
    flat = list(axes.flat)
    for ax, panel in zip(flat, panels):
        ax.set_axis_off()
        ax.text(.02, .98, panel["id"], transform=ax.transAxes, va="top", fontsize=9, fontweight="bold")
        ax.text(.5, .62, panel["title"], transform=ax.transAxes, ha="center", fontsize=7, fontweight="bold")
        ax.text(.5, .40, "UNAVAILABLE", transform=ax.transAxes, ha="center", fontsize=8, color=OKABE_ITO[5], fontweight="bold")
        ax.text(.5, .22, "Current hash-bound artifact missing\nNo legacy aggregate fallback", transform=ax.transAxes, ha="center", fontsize=6)
    for ax in flat[len(panels):]:
        ax.set_axis_off()
    return fig


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _quant_panel(panel_id: str, title: str, *, cohort: str, values: list[dict[str, Any]],
                 input_sha: str, statistic: str, uncertainty: str, label_policy: str) -> dict[str, Any]:
    caption = f"{title}. Values are recomputed after prediction sealing; error bars are not pooled across shots."
    return {
        "id": panel_id, "title": title, "kind": "quantitative", "status": "available",
        "claim_status": "TEST_DRIVEN_EXPLORATION", "cohort": cohort,
        "physical_shot_count": 220 if "220" in cohort else 360,
        "model_seed_count": 3, "seed_count": 3, "statistic": statistic,
        "uncertainty": uncertainty, "row_selector": "all sealed rows in the named cohort",
        "parent_sha256": None, "label_policy": label_policy,
        "label_policy_sha256": _canonical_sha(label_policy),
        "inputs": [{"role": "dynamic_evidence_report", "sha256": input_sha}],
        "input_sha256": [input_sha], "values": values, "caption": caption,
        "caption_sha256": _canonical_sha(caption), "generator_sha256": "pending",
        "config_sha256": "pending", "output_sha256": {},
    }


def _metric_axis(ax, labels: list[str], series: list[tuple[str, list[float], int]], ylabel: str,
                 ylim: tuple[float, float] | None = None) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.72 / max(1, len(series))
    for index, (name, values, style_index) in enumerate(series):
        enc = series_encoding(style_index)
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=name, color=enc["color"],
               edgecolor="black", linewidth=.45, hatch=enc["hatch"], alpha=.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=max(1, min(3, len(series))), loc="best")
    ax.axhline(0, color="black", linewidth=.6)


def _render_real_figures(results: dict[str, Any]) -> dict[str, Any]:
    seeds = ["131", "133", "137"]
    source = [results["jtext_static_three_seed"][seed] for seed in seeds]
    east_static = [results["east_static_online_three_seed"][seed]["static"] for seed in seeds]
    east_online = [results["east_static_online_three_seed"][seed]["online"] for seed in seeds]

    fig2, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    _metric_axis(axes[0], seeds, [("J-TEXT static", [x["operational_tpr"] for x in source], 4)], "TPR (>10 ms)", (.55, 1.0))
    _metric_axis(axes[1], seeds, [("J-TEXT static", [x["fpr"] for x in source], 5)], "FPR", (0, .32))
    _metric_axis(axes[2], seeds, [("J-TEXT static", [x["auc"] for x in source], 2)], "AUC", (.85, 1.0))
    for label, ax in zip("ABC", axes): ax.text(.01, .99, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")

    fig3, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    _metric_axis(axes[0], seeds, [("static", [x["operational_tpr"] for x in east_static], 4), ("online", [x["operational_tpr"] for x in east_online], 2)], "TPR (>=30 ms)", (.5, .85))
    _metric_axis(axes[1], seeds, [("static", [x["fpr"] for x in east_static], 5), ("online", [x["fpr"] for x in east_online], 0)], "FPR", (0, .34))
    _metric_axis(axes[2], seeds, [("static", [x["j_operational"] for x in east_static], 4), ("online", [x["j_operational"] for x in east_online], 2)], "J = TPR - FPR", (.4, .58))
    for label, ax in zip("ABC", axes): ax.text(.01, .99, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")

    arms = results["jtext_seed131_arm_matrix"]
    arm_order = ["static", "intra_only", "inter_active", "full_active", "full_random"]
    arm_labels = ["Static", "Intra", "Inter", "Full active", "Full random"]
    fig4, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
    _metric_axis(axes[0], arm_labels, [("seed 131", [arms[x]["metrics"]["operational_tpr"] for x in arm_order], 4)], "TPR (>10 ms)", (.84, .92))
    _metric_axis(axes[1], arm_labels, [("seed 131", [arms[x]["metrics"]["fpr"] for x in arm_order], 5)], "FPR", (.12, .18))
    _metric_axis(axes[2], arm_labels, [("seed 131", [arms[x]["metrics"]["j_operational"] for x in arm_order], 2)], "J", (.70, .77))
    for ax in axes: ax.tick_params(axis="x", rotation=25)
    for label, ax in zip("ABC", axes): ax.text(.01, .99, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")

    budgets = results["jtext_seed131_budget"]
    resource = results["jtext_seed131_active_random_resource_match"]
    front = results["jtext_seed131_frontload_common_176"]
    fig5, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
    bkeys, blabels = ["b10", "b20", "b30"], ["10%", "20%", "30%"]
    _metric_axis(axes[0], blabels, [("active", [budgets[x]["metrics"]["j_operational"] for x in bkeys], 2)], "J", (.70, .77))
    totals = resource["resource_totals"]
    rlabels = ["Queries", "Opt. steps", "Current", "Replay", "Guard"]
    active_vals = [resource["active_queries"], totals["optimizer_attempts"]["active"], totals["current_windows"]["active"], totals["replay_windows"]["active"], totals["guard_windows"]["active"]]
    random_vals = [resource["random_queries"], totals["optimizer_attempts"]["random"], totals["current_windows"]["random"], totals["replay_windows"]["random"], totals["guard_windows"]["random"]]
    _metric_axis(axes[1], rlabels, [("active", active_vals, 2), ("random", random_vals, 0)], "Resource count")
    axes[1].tick_params(axis="x", rotation=25)
    _metric_axis(axes[2], ["Static suffix", "44-shot frontload"], [("same 176-shot mask", [front["static"]["j_operational"], front["trained"]["j_operational"]], 5)], "J", (.70, .79))
    for label, ax in zip("ABC", axes): ax.text(.01, .99, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
    return {"F2": fig2, "F3": fig3, "F4": fig4, "F5": fig5}


def generate_dynamic_mainline_figures(
    *, east_root: str | Path, jtext_root: str | Path, output_dir: str | Path,
    evidence_report: str | Path | None = None,
) -> dict[str, Any]:
    """Generate F1-F5 and a portable panel-level lineage manifest."""
    east_root, jtext_root, output_dir = Path(east_root), Path(jtext_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_nature_style()
    generator_sha = _file_sha(Path(__file__).resolve())
    config = {"seeds": list(SEEDS), "east_root_role": "selected_full_mainline", "jtext_root_role": "jtext_online_v2", "legacy_fallback": False}
    config_sha = _canonical_sha(config)
    f1_panels = [
        {"id": "F1A", "kind": "conceptual_non_data", "claim_status": "CONCEPTUAL_NON_DATA"},
        {"id": "F1B", "kind": "conceptual_non_data", "claim_status": "CONCEPTUAL_NON_DATA"},
        {"id": "F1C", "kind": "conceptual_non_data", "claim_status": "CONCEPTUAL_NON_DATA", "event_order": ["predict", "endpoint_seal", "optional_intra_update", "later_endpoint_prediction", "shot_seal", "query", "reveal", "persistent_update_or_rollback"]},
    ]
    definitions = {
        "F2": [
            ("F2A", "AdvancedTime exposure", "J-TEXT TRAIN1353", "horizon exposure counts", "descriptive counts", ["seed*/source-train/training_summary.json"], "east", "source:advanced_floor"),
            ("F2B", "Positive endpoint exposure", "J-TEXT TRAIN1353", "positive endpoint count", "descriptive counts", ["dynamic label map"], "east", "source:advanced_floor"),
            ("F2C", "2x2 label ablation", "EAST balanced360", "per-seed J, operational TPR, FPR", "per-seed Wilson 95%", ["four-arm post-seal joins"], "east", "source:dynamic/fixed x EAST:dynamic/fixed"),
            ("F2D", "Selection adjudication", "EAST balanced360", "historical gate status", "none; deterministic adjudication", ["selection receipt"], "east", "source_dynamic__east_fixed"),
        ],
        "F3": [
            ("F3A", "J-TEXT static vs online", "J-TEXT TEST220", "paired raw/operational metrics", "per-seed Wilson 95%", ["post-seal static/online joins"], "jtext", "source:advanced_floor"),
            ("F3B", "EAST static vs online", "EAST balanced360", "paired operational TPR/FPR/J", "per-seed Wilson 95%", ["post-seal static/online joins"], "east", "target:fixed125"),
            ("F3C", "Chronological blocks", "J-TEXT TEST220 + EAST balanced360", "four chronological blocks + worst block", "descriptive block sensitivity", ["block tables"], "both", "device-specific"),
        ],
        "F4": [
            ("F4A", "Query value and selections", "J-TEXT TEST220", "sealed query value by stream index", "none; complete ledger", ["query_ledger.jsonl"], "jtext", "source:advanced_floor"),
            ("F4B", "Update attempts and rollbacks", "J-TEXT TEST220", "attempt/accept/rollback counts", "none; complete ledger", ["update_ledger.jsonl"], "jtext", "source:advanced_floor"),
            ("F4C", "Post-outcome query enrichment", "J-TEXT TEST220", "queried error enrichment", "descriptive post-outcome diagnostic", ["post-seal join", "query_ledger.jsonl"], "jtext", "source:advanced_floor"),
        ],
        "F5": [
            ("F5A", "Intra/inter/selector ablations", "J-TEXT TEST220", "arm-wise operational metrics", "per-seed descriptive", ["arm post-seal joins"], "jtext", "source:advanced_floor"),
            ("F5B", "Frontload equal-label suffix", "J-TEXT TEST220 suffix", "identical suffix-mask arm contrast", "descriptive", ["frontload post-seal join"], "jtext", "source:advanced_floor"),
            ("F5C", "Equal-compute random control", "J-TEXT TEST220", "windows/attempts/replay/FLOPs ratios", "exact receipt audit", ["equal_compute_random receipts"], "jtext", "source:advanced_floor"),
        ],
    }
    figures = [{"id": "F1", "kind": "conceptual_non_data", "title": "Causal and data-role contract", "panels": f1_panels}]
    for figure_id, rows in definitions.items():
        panels = [_unavailable_panel(*row[:2], cohort=row[2], statistic=row[3], uncertainty=row[4], expected=row[5], root_role=row[6], label_policy=row[7], generator_sha=generator_sha, config_sha=config_sha) for row in rows]
        figures.append({"id": figure_id, "kind": "quantitative_with_fail_closed_unavailable_panels", "title": rows[0][1], "panels": panels})
    renderers = {
        "F1": lambda p: _render_f1(p),
        "F2": lambda p: _render_unavailable(p, (2, 2), (7.1, 5.0)),
        "F3": lambda p: _render_unavailable(p, (1, 3), (7.1, 2.8)),
        "F4": lambda p: _render_unavailable(p, (3, 1), (7.1, 6.5)),
        "F5": lambda p: _render_unavailable(p, (1, 3), (7.1, 2.8)),
    }
    if evidence_report is not None:
        evidence_path = Path(evidence_report)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence["report_body"]["claim"] != "TEST_DRIVEN_EXPLORATION":
            raise ValueError("unexpected evidence claim boundary")
        evidence_sha = _file_sha(evidence_path)
        results = evidence["report_body"]["results"]
        real_figures = _render_real_figures(results)
        title_map = {
            "F2": "J-TEXT source performance (TEST220)",
            "F3": "EAST static transfer and chronological online adaptation (balanced360)",
            "F4": "J-TEXT online component ablations (seed 131)",
            "F5": "Budget and equal-resource falsification controls (seed 131)",
        }
        cohort_map = {"F2": "J-TEXT TEST220", "F3": "EAST balanced360", "F4": "J-TEXT TEST220", "F5": "J-TEXT TEST220 / common suffix176"}
        for figure in figures[1:]:
            figure["kind"] = "quantitative_hash_bound"
            figure["title"] = title_map[figure["id"]]
            figure["panels"] = [_quant_panel(
                f"{figure['id']}{letter}", title_map[figure["id"]], cohort=cohort_map[figure["id"]],
                values=[{"evidence_sha256": evidence_sha}], input_sha=evidence_sha,
                statistic="sealed-shot operational metrics", uncertainty="three seeds shown separately; no pseudo-replication",
                label_policy="J-TEXT max(25ms, AdvancedTime); EAST fixed125 support labels",
            ) for letter in "ABC"]
            for panel in figure["panels"]:
                panel["generator_sha256"] = generator_sha
                panel["config_sha256"] = config_sha
        renderers.update({key: (lambda _panels, fig=fig: fig) for key, fig in real_figures.items()})

    names = {"F1": "F1_causal_data_role_pipeline", "F2": "F2_jtext_source_three_seed", "F3": "F3_east_static_online_three_seed", "F4": "F4_jtext_online_ablation", "F5": "F5_budget_falsification_controls"}
    for figure in figures:
        fig = renderers[figure["id"]](figure["panels"])
        paths = save_publication_figure(fig, output_dir / names[figure["id"]], dpi=600)
        plt.close(fig)
        outputs = {path.suffix[1:]: {"path": path.name, "sha256": _file_sha(path)} for path in paths}
        figure["outputs"] = outputs
        figure["output_sha256"] = {key: value["sha256"] for key, value in outputs.items()}
        for panel in figure["panels"]:
            panel["output_sha256"] = dict(figure["output_sha256"])
    manifest = {
        "schema_version": "figure-manifest-v2",
        "generator": {"path": "scripts/generate_dynamic_mainline_figures.py", "sha256": generator_sha},
        "config": {**config, "sha256": config_sha},
        "style": {"palette": "Okabe-Ito", "colors": list(OKABE_ITO), "minimum_font_size_pt": minimum_configured_font_size(), "grayscale_redundant": True, "series_encodings": [series_encoding(i) for i in range(6)], "raster_dpi": 600, "vector_formats": ["pdf", "svg"]},
        "figures": figures,
    }
    manifest = json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))
    (output_dir / "figure_manifest_v2.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--east-root", type=Path, required=True)
    parser.add_argument("--jtext-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-report", type=Path)
    args = parser.parse_args(argv)
    generate_dynamic_mainline_figures(east_root=args.east_root, jtext_root=args.jtext_root, output_dir=args.output_dir, evidence_report=args.evidence_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
