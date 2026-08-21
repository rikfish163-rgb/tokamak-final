#!/usr/bin/env python3
"""Generate traceable, publication-ready figures for the three-stage study.

The script intentionally reads only machine evidence already produced by the
registered evaluators.  It does not fit models, change thresholds, or inspect
unqueried labels during the online stream.  Every exported figure is linked to
its input and output SHA-256 hashes in ``figures/figure_manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).expanduser().resolve()
PACKAGE = ROOT / "experiments/transfer_methods_2026/paper_grade_package"
EVIDENCE_PATH = PACKAGE / "evidence/three_stage_results.json"
FIGURE_DIR = Path(
    os.environ.get("FUSION_FIGURE_DIR", PACKAGE / "figures")
).expanduser().resolve()
PRIMARY_FALLBACK = ROOT / "experiments/transfer_methods_2026/online_episode_value_v3_runs"
ROBUSTNESS_PATH = (
    PACKAGE / "evidence/dual_threshold_alarm_robustness_summary.json"
)
TRACE_PATH = PACKAGE / "evidence/case_study_seed131_shot73790.npz"
EQUAL_BUDGET_PATH = (
    PACKAGE
    / "evidence/equal_label_budget_frontload_v2/common_mask_report/summary.json"
)
CAUSAL_RANDOM_PATH = (
    PACKAGE / "evidence/causal_random_query_online_v1/comparison.json"
)
EAST_TARGET_ONLY_PATH = PACKAGE / "evidence/east_target_only_v1/comparison.json"
from publication_style import apply_nature_style, save_publication_figure  # noqa: E402


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#777777"
LIGHT_GREY = "#E6E6E6"
SEEDS = (131, 133, 137)


def primary_run(seed: int) -> Path:
    bundled = PACKAGE / f"evidence/primary_online_artifacts/seed{int(seed)}"
    if bundled.is_dir():
        return bundled
    return PRIMARY_FALLBACK / f"final_seed{int(seed)}_episode_v3"


def manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_panel_labels(axes: Any) -> None:
    flat = np.asarray(axes, dtype=object).reshape(-1)
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", flat, strict=False):
        axis.text(
            -0.14,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
        )


def finish(fig: plt.Figure, stem: str) -> list[Path]:
    files = save_publication_figure(
        fig,
        FIGURE_DIR / stem,
        formats=("pdf", "svg", "png"),
        dpi=600,
        pad_inches=0.03,
    )
    plt.close(fig)
    return files


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "white",
    fontsize: float = 7.0,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="white" if facecolor != LIGHT_GREY else "black",
        linespacing=1.2,
    )


def draw_module(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    lines: tuple[str, ...],
    *,
    accent: str,
) -> None:
    """Draw a restrained paper-style module, not an infographic card."""
    x, y = xy
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            linewidth=1.0,
            facecolor="#FAFAFA",
            edgecolor="#333333",
        )
    )
    ax.plot([x, x + width], [y + height, y + height], color=accent, linewidth=2.0)
    ax.text(x + 0.018, y + height - 0.075, title, ha="left", va="top", fontsize=7.2, fontweight="bold")
    ax.text(
        x + 0.018,
        y + height - 0.145,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=6.1,
        color="#333333",
        linespacing=1.35,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color="#333333",
        )
    )


def figure_0_end_to_end_architecture() -> list[Path]:
    """Render the actual source-transfer-online computation and update boundary."""
    fig, axes = plt.subplots(3, 1, figsize=(7.205, 5.65))
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    ax = axes[0]
    draw_module(
        ax, (0.015, 0.23), 0.16, 0.56, "J-TEXT causal input",
        ("rebuilt17 + masks", "128-point endpoint", "TRAIN1353 only"), accent=BLUE,
    )
    draw_module(
        ax, (0.215, 0.23), 0.14, 0.56, "Patch tokens",
        ("patch 8 / stride 8", "linear embed", "d = 64"), accent=BLUE,
    )
    draw_module(
        ax, (0.395, 0.23), 0.18, 0.56, "Source backbone",
        ("2 Transformer blocks", "4 attention heads", "causal endpoint view"), accent=BLUE,
    )
    draw_module(
        ax, (0.615, 0.23), 0.18, 0.56, "Training heads",
        ("risk / horizon", "onset supervision", "heads not target inputs"), accent=PURPLE,
    )
    draw_module(
        ax, (0.835, 0.23), 0.15, 0.56, "Source seal",
        ("VAL160 threshold", "TEST220 report*", "checkpoint SHA"), accent=BLUE,
    )
    for left, right in ((0.175, 0.215), (0.355, 0.395), (0.575, 0.615), (0.795, 0.835)):
        arrow(ax, (left + 0.004, 0.51), (right - 0.004, 0.51))
    ax.text(0.015, 0.91, "a  Source representation learning", fontsize=7.5, fontweight="bold", ha="left")

    ax = axes[1]
    draw_module(
        ax, (0.015, 0.23), 0.15, 0.56, "Frozen source",
        ("shared rebuilt17", "low-level encoder", "source SHA fixed"), accent=BLUE,
    )
    draw_module(
        ax, (0.205, 0.23), 0.15, 0.56, "EAST support110",
        ("15D / 95N by HDF5", "0–125 ms targets", "fit + threshold"), accent=ORANGE,
    )
    draw_module(
        ax, (0.395, 0.23), 0.17, 0.56, "Target shared path",
        ("adapt-high temporal layer", "dynamic hazard residual", "support-bounded"), accent=ORANGE,
    )
    draw_module(
        ax, (0.605, 0.23), 0.16, 0.56, "EAST-private path",
        ("raw8 diagnostics", "causal summaries", "private residual head"), accent=GREEN,
    )
    draw_module(
        ax, (0.805, 0.23), 0.18, 0.56, "Static target alarm",
        ("shared + private logits", "physics-supported state", "first alarm; ≥30 ms"), accent=VERMILLION,
    )
    # Route cross-module dependencies through whitespace so no edge obscures text.
    ax.plot([0.165, 0.185, 0.375], [0.70, 0.84, 0.84], color="#333333", linewidth=1.0)
    arrow(ax, (0.375, 0.84), (0.395, 0.70))
    arrow(ax, (0.359, 0.50), (0.391, 0.50))
    ax.plot([0.28, 0.28, 0.585], [0.23, 0.16, 0.16], color="#333333", linewidth=1.0)
    arrow(ax, (0.585, 0.16), (0.605, 0.23))
    ax.plot([0.565, 0.585, 0.785], [0.70, 0.84, 0.84], color="#333333", linewidth=1.0)
    arrow(ax, (0.785, 0.84), (0.805, 0.70))
    arrow(ax, (0.765, 0.50), (0.801, 0.50))
    ax.text(0.015, 0.91, "b  Few-shot target adaptation", fontsize=7.5, fontweight="bold", ha="left")
    ax.text(0.29, 0.10, "support labels", fontsize=5.8, color=ORANGE, ha="center")
    ax.text(0.49, 0.10, "target-trainable", fontsize=5.8, color=ORANGE, ha="center")
    ax.text(0.68, 0.10, "target-private", fontsize=5.8, color=GREEN, ha="center")

    ax = axes[2]
    modules = (
        (0.015, 0.12, "Causal prefix", ("current + past", "raw17 + raw8"), BLUE),
        (0.165, 0.13, "Intra-shot", ("label-free", "bounded update"), BLUE),
        (0.325, 0.13, "Predict", ("risk trace", "physics support"), GREEN),
        (0.485, 0.12, "Seal", ("alarm + state", "outcome hidden"), GREEN),
        (0.635, 0.12, "Query", ("uncertainty", "+ physics value"), PURPLE),
        (0.785, 0.09, "Reveal", ("selected only", "HDF5 outcome"), VERMILLION),
        (0.905, 0.08, "Update", ("error typed", "guard / replay"), ORANGE),
    )
    for x, width, title, lines, accent in modules:
        draw_module(ax, (x, 0.26), width, 0.49, title, lines, accent=accent)
    for left, right in ((0.135, 0.165), (0.295, 0.325), (0.455, 0.485), (0.605, 0.635), (0.755, 0.785), (0.875, 0.905)):
        arrow(ax, (left + 0.004, 0.505), (right - 0.004, 0.505))
    gate = 0.77
    ax.plot([gate, gate], [0.17, 0.88], color=VERMILLION, linestyle="--", linewidth=1.0)
    ax.text(gate - 0.008, 0.87, "no outcome access", ha="right", va="top", fontsize=5.6, color=VERMILLION)
    ax.text(0.945, 0.13, "affects later shots only", ha="center", fontsize=5.5, color=ORANGE)
    ax.text(0.015, 0.91, "c  Ordered online-active development stream", fontsize=7.5, fontweight="bold", ha="left")

    fig.text(
        0.5, 0.008,
        "Blue: frozen/shared source structure  |  Orange: target-supervised update  |  "
        "Green: EAST-private/deployed path  |  Purple: query control",
        ha="center", va="bottom", fontsize=5.8,
    )
    return finish(fig, "fig0_end_to_end_model_architecture")


def figure_1_protocol() -> list[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(7.205, 4.15), gridspec_kw={"height_ratios": [1.08, 0.92]})
    ax, bx = axes
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    draw_module(
        ax, (0.015, 0.30), 0.185, 0.49, "J-TEXT data",
        ("TRAIN1353  model fit", "VAL160  threshold", "TEST220*  report"), accent=BLUE,
    )
    draw_module(
        ax, (0.245, 0.30), 0.175, 0.49, "Source encoder",
        ("causal patch Transformer", "rebuilt17 inputs", "frozen for transfer"), accent=BLUE,
    )
    draw_module(
        ax, (0.465, 0.30), 0.205, 0.49, "Target adaptation",
        ("support110: 15D / 95N", "fit + threshold selection", "adapt-high + private raw8"), accent=ORANGE,
    )
    draw_module(
        ax, (0.715, 0.30), 0.27, 0.49, "Ordered development stream",
        ("balanced360: 180D / 180N", "static baseline → causal online", "method and parameter development"), accent=GREEN,
    )
    for start, end in ((0.200, 0.245), (0.420, 0.465), (0.670, 0.715)):
        arrow(ax, (start + 0.004, 0.545), (end - 0.004, 0.545))
    ax.text(0.4425, 0.84, "shared 17", ha="center", va="bottom", fontsize=5.8, color=BLUE)
    ax.text(0.6925, 0.84, "target state", ha="center", va="bottom", fontsize=5.8, color=ORANGE)
    ax.text(
        0.015, 0.12,
        "* Count-aligned balanced comparator; not a chronological or prospective test split.",
        ha="left", va="center", fontsize=6.0, color="#444444",
    )

    bx.plot([0.055, 0.945], [0.54, 0.54], color="#333333", linewidth=1.0)
    positions = np.linspace(0.07, 0.93, 6)
    titles = (
        "Predict",
        "Intra-shot adapt",
        "Seal",
        "Query",
        "Reveal",
        "Inter-shot update",
    )
    subtitles = (
        "causal endpoint",
        "past windows only",
        "prediction + state",
        "without outcome",
        "selected shots only",
        "for later shots",
    )
    accents = (BLUE, BLUE, GREEN, PURPLE, VERMILLION, ORANGE)
    for index, (xpos, title, subtitle, accent) in enumerate(zip(positions, titles, subtitles, accents, strict=True), start=1):
        bx.scatter([xpos], [0.54], s=165, facecolor="white", edgecolor=accent, linewidth=1.3, zorder=3)
        bx.text(xpos, 0.54, str(index), ha="center", va="center", fontsize=6.0, fontweight="bold", color=accent)
        above = index % 2 == 1
        ypos = 0.74 if above else 0.33
        bx.text(xpos, ypos, title, ha="center", va="bottom" if above else "top", fontsize=6.7, fontweight="bold")
        bx.text(xpos, ypos - 0.055 if above else ypos - 0.065, subtitle, ha="center", va="top", fontsize=5.7, color="#555555")
    gate_x = (positions[3] + positions[4]) / 2
    bx.plot([gate_x, gate_x], [0.14, 0.90], color=VERMILLION, linestyle="--", linewidth=1.0)
    bx.text(gate_x - 0.012, 0.90, "outcome unavailable", ha="right", va="top", fontsize=5.8, color=VERMILLION)
    bx.text(gate_x + 0.012, 0.90, "post-shot reveal permitted", ha="left", va="top", fontsize=5.8, color=VERMILLION)
    bx.text(
        0.50, 0.06,
        "Current-shot outcome and future windows are excluded before the sealed prediction.",
        ha="center", va="center", fontsize=6.2,
    )
    add_panel_labels(axes)
    return finish(fig, "fig1_protocol_and_causal_contract")


def figure_2_source_shortcut(evidence: dict[str, Any]) -> list[Path]:
    old = evidence["primary_lineage"]["source_test220"]
    pad = evidence["padfix_ablation"]
    old_audit = old["shortcut_audit"]["seeds"]
    pad_audit = pad["validation_shortcut_audit"]
    old_metrics = old["seeds"]
    pad_metrics = pad["source_test220"]["seeds"]
    pad_static = pad["east_static360"]["seeds"]
    pad_online = pad["east_online360"]["seeds"]

    fig, axes = plt.subplots(2, 2, figsize=(7.205, 5.2))
    probe_old = [old_audit[str(s)]["audit"]["paired_context_probe_macro_auc"] for s in SEEDS]
    probe_pad = [pad_audit[str(s)]["paired_context_probe_macro_auc"] for s in SEEDS]
    risk_old = [old_audit[str(s)]["audit"]["paired_context_mean_abs_risk_difference"] for s in SEEDS]
    risk_pad = [pad_audit[str(s)]["paired_context_mean_abs_risk_difference"] for s in SEEDS]

    ax = axes[0, 0]
    for index, (seed, color) in enumerate(zip(SEEDS, [BLUE, ORANGE, GREEN], strict=True)):
        ax.plot([0, 1], [probe_old[index], probe_pad[index]], color=color, marker="o", linewidth=1.2, label=f"seed {seed}")
    ax.axhline(0.65, color="black", linestyle="--", linewidth=1.0, label="gate")
    ax.set_ylabel("padding probe macro-AUC")
    ax.set_xticks([0, 1], ["original", "PADFIX"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.55, 1.0)
    ax.legend(ncol=2, loc="best")

    ax = axes[0, 1]
    for index, color in enumerate([BLUE, ORANGE, GREEN]):
        ax.plot([0, 1], [risk_old[index], risk_pad[index]], color=color, marker="o", linewidth=1.2)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("same-endpoint mean |Δrisk|")
    ax.set_xticks([0, 1], ["original", "PADFIX"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 0.08)

    ax = axes[1, 0]
    for seed, color in zip(SEEDS, [BLUE, ORANGE, GREEN], strict=True):
        ax.plot(
            [0, 1],
            [old_metrics[str(seed)]["timely_tpr"], pad_metrics[str(seed)]["metrics"]["timely_tpr"]],
            color=color,
            marker="o",
            linewidth=1.2,
        )
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks([0, 1], ["original", "PADFIX"])
    ax.set_ylabel("J-TEXT operational TPR (>10 ms)")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.78, 0.92)

    ax = axes[1, 1]
    for seed, color in zip(SEEDS, [BLUE, ORANGE, GREEN], strict=True):
        ax.plot(
            [0, 1],
            [pad_static[str(seed)]["fpr"], pad_online[str(seed)]["fpr"]],
            color=color,
            marker="o",
            linewidth=1.2,
        )
    ax.axhline(0.20, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks([0, 1], ["static", "online"])
    ax.set_ylabel("EAST FPR with PADFIX parent")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.13, 0.22)
    add_panel_labels(axes)
    return finish(fig, "fig3_source_shortcut_and_padfix_tradeoff")


def figure_3_three_stage(evidence: dict[str, Any]) -> list[Path]:
    source = evidence["official_results"]["source_test220"]["seeds"]
    static = evidence["official_results"]["east_static360"]["seeds"]
    online = evidence["official_results"]["east_online360"]["seeds"]
    bootstrap = evidence["official_results"]["bootstrap"]

    fig, axes = plt.subplots(2, 2, figsize=(7.205, 5.1))
    colors = [BLUE, ORANGE, GREEN]

    ax = axes[0, 0]
    offsets = np.asarray([-0.08, 0.0, 0.08])
    for index, (seed, color) in enumerate(zip(SEEDS, colors, strict=True)):
        ax.scatter(
            np.asarray([0.0, 1.0]) + offsets[index],
            [source[str(seed)]["operational_tpr"], source[str(seed)]["fpr"]],
            color=color,
            marker=["o", "s", "^"][index],
            label=f"seed {seed}",
            zorder=3,
        )
    ax.plot([-0.18, 0.18], [0.90, 0.90], color="black", linestyle="--", linewidth=1.0)
    ax.plot([0.82, 1.18], [0.15, 0.15], color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks([0, 1], ["operational TPR\n(>10 ms)", "FPR"])
    ax.set_ylabel("J-TEXT TEST220 rate")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0.10, 0.96)
    ax.legend(ncol=3, loc="center", bbox_to_anchor=(0.52, 0.57))

    ax = axes[0, 1]
    for seed, color in zip(SEEDS, [BLUE, ORANGE, GREEN], strict=True):
        ax.plot(
            [0, 1],
            [static[str(seed)]["operational_tpr"], online[str(seed)]["operational_tpr"]],
            marker="o",
            color=color,
            linewidth=1.2,
        )
    ax.axhline(0.80, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks([0, 1], ["static", "online"])
    ax.set_ylabel("EAST operational TPR (≥30 ms)")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.78, 0.88)

    ax = axes[1, 0]
    for seed, color in zip(SEEDS, colors, strict=True):
        ax.plot(
            [0, 1],
            [static[str(seed)]["fpr"], online[str(seed)]["fpr"]],
            marker="o",
            color=color,
            linewidth=1.2,
        )
    ax.axhline(0.20, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks([0, 1], ["static", "online"])
    ax.set_ylabel("EAST FPR")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.10, 0.22)

    ax = axes[1, 1]
    names = ["ΔTPR", "ΔFPR"]
    keys = ["delta_tpr", "delta_fpr"]
    med = np.asarray([100.0 * bootstrap[key]["median"] for key in keys])
    lo = np.asarray([100.0 * bootstrap[key]["ci95"][0] for key in keys])
    hi = np.asarray([100.0 * bootstrap[key]["ci95"][1] for key in keys])
    y = np.arange(2)
    ax.errorbar(
        med,
        y,
        xerr=[med - lo, hi - med],
        fmt="o",
        color="#333333",
        ecolor=GREY,
        capsize=2,
    )
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y, names)
    ax.set_xlabel("online − static (percentage points)")
    ax.invert_yaxis()
    add_panel_labels(axes)
    return finish(fig, "fig2_same_lineage_metrics_and_block_resampling")


def figure_4_case_study() -> list[Path]:
    run = primary_run(131)
    static_rows = {int(row["shot"]): row for row in read_jsonl(run / "static_predictions.jsonl")}
    online_rows = {int(row["shot"]): row for row in read_jsonl(run / "predictions.jsonl")}
    query_rows = {int(row["shot"]): row for row in read_jsonl(run / "query_decisions.jsonl")}
    shot = 73790
    trace = np.load(TRACE_PATH)
    static = np.asarray(trace["static_risk"], dtype=float)
    online = np.asarray(trace["online_risk"], dtype=float)
    physics = np.asarray(trace["physics_support"], dtype=float)
    endpoint = np.asarray(trace["endpoint_memory_logit_residual"], dtype=float)
    x = np.arange(len(static))
    candidate = int(static_rows[shot]["alarm_index"])
    left, right = max(0, candidate - 420), min(len(x), candidate + 520)

    fig, axes = plt.subplots(3, 1, figsize=(7.205, 5.3), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]})
    ax = axes[0]
    ax.plot(x[left:right], static[left:right], label="static risk", color=GREY, linewidth=1.0)
    ax.plot(x[left:right], online[left:right], label="online risk", color=BLUE, linewidth=1.2)
    ax.axvline(candidate, color=VERMILLION, linestyle="--", linewidth=1.0)
    ax.set_ylabel("risk")
    ax.legend(loc="upper left", ncol=2)
    ax.text(candidate + 12, 0.94 * ax.get_ylim()[1], "base alarm", color=VERMILLION, fontsize=7)

    axes[1].plot(x[left:right], physics[left:right], color=GREEN)
    axes[1].set_ylabel("physics support")
    axes[1].axvline(candidate, color=VERMILLION, linestyle="--", linewidth=1.0)
    axes[2].plot(x[left:right], endpoint[left:right], color=PURPLE)
    axes[2].set_ylabel("memory logit\nresidual")
    axes[2].set_xlabel("causal endpoint index (1 ms sampling)")
    axes[2].axvline(candidate, color=VERMILLION, linestyle="--", linewidth=1.0)

    episode = online_rows[shot]["episode_confirmation"]
    query = query_rows[shot]
    annotation = f"safe shot {shot}  |  static FP → online TN  |  queried={query['selected']}"
    axes[0].text(
        0.99,
        0.965,
        annotation,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        bbox={"boxstyle": "square,pad=0.18", "facecolor": "white", "edgecolor": LIGHT_GREY},
    )
    add_panel_labels(axes)
    return finish(fig, "fig6_causal_false_alarm_rescue_shot73790")


def figure_5_chronology_and_queries(evidence: dict[str, Any]) -> list[Path]:
    blocks = evidence["time_blocks"]
    queries = evidence["primary_lineage"]["east_online360"]["query_analysis"]
    fig, axes = plt.subplots(2, 2, figsize=(7.205, 5.1))

    for metric, axis, ylabel in (
        ("fpr", axes[0, 0], "FPR"),
        ("tpr_at_30ms", axes[0, 1], "operational TPR (≥30 ms)"),
    ):
        for phase, color, marker in (("static", GREY, "o"), ("online", BLUE, "s")):
            means = []
            lows = []
            highs = []
            for block in range(1, 5):
                values = [row[metric] for row in blocks if row["phase"] == phase and row["block"] == block]
                means.append(float(np.mean(values)))
                lows.append(float(np.min(values)))
                highs.append(float(np.max(values)))
            axis.errorbar(
                np.arange(1, 5),
                means,
                yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)],
                color=color,
                marker=marker,
                capsize=2,
                label=phase,
            )
        axis.set_xticks(range(1, 5))
        axis.set_xlabel("chronological block")
        axis.set_ylabel(ylabel)
    axes[0, 0].axhline(0.20, color=VERMILLION, linestyle="--", linewidth=1.0)
    axes[0, 0].legend(ncol=2)

    ax = axes[1, 0]
    x = np.arange(len(SEEDS))
    enrich = [queries[str(seed)]["error_enrichment"] for seed in SEEDS]
    all_rate = [queries[str(seed)]["operational_error_rate_all"] for seed in SEEDS]
    selected_rate = [queries[str(seed)]["operational_error_rate_queried"] for seed in SEEDS]
    width = 0.35
    ax.bar(x - width / 2, all_rate, width, color=GREY, label="all shots")
    ax.bar(x + width / 2, selected_rate, width, color=PURPLE, label="queried")
    ax.set_xticks(x, SEEDS)
    ax.set_ylabel("operational error rate")
    ax.legend(ncol=2)
    for index, value in enumerate(enrich):
        ax.text(index, selected_rate[index] + 0.015, f"{value:.2f}×", ha="center", fontsize=6.5)

    ax = axes[1, 1]
    for seed, color in zip(SEEDS, [BLUE, ORANGE, GREEN], strict=True):
        run = primary_run(seed)
        static = read_jsonl(run / "static_predictions.jsonl")
        online = read_jsonl(run / "predictions.jsonl")
        cumulative = np.cumsum(
            [int(a["alarmed"]) - int(b["alarmed"]) for a, b in zip(static, online, strict=True)]
        )
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color=color, label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xlabel("stream position")
    ax.set_ylabel("cumulative alarms removed")
    ax.legend(ncol=3, loc="upper left")
    add_panel_labels(axes)
    return finish(fig, "fig4_chronological_blocks_and_active_query_value")


def figure_6_controls_and_robustness(evidence: dict[str, Any]) -> list[Path]:
    controls = evidence["negative_controls"]["results"]
    robustness = read_json(ROBUSTNESS_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(7.205, 3.0))

    order = ["no_labels", "feature_lag", "label_lag", "label_flip", "temporal_zero", "raw8_zero", "real_final_v3c"]
    labels = ["no labels", "feature lag", "label lag", "label flip", "temporal zero", "raw8 zero", "real v3c"]
    fp = [controls[name]["total_fp_reduction"] for name in order]
    tp_loss = [controls[name]["total_operational_tp_loss"] for name in order]
    y = np.arange(len(order))
    for index, (fp_value, loss_value) in enumerate(zip(fp, tp_loss, strict=True)):
        axes[0].hlines(
            index,
            min(-loss_value, 0),
            max(fp_value, 0),
            color=LIGHT_GREY,
            linewidth=1.2,
            zorder=1,
        )
    axes[0].scatter(
        fp,
        y,
        color=[GREY] * 6 + [BLUE],
        marker="s",
        label="FP reduction",
        zorder=3,
    )
    axes[0].scatter(
        [-value for value in tp_loss],
        y,
        color=VERMILLION,
        marker="x",
        label="operational TP loss (left)",
        zorder=3,
    )
    axes[0].axvline(0, color="black", linewidth=1.0)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("three-seed shot-model count")
    axes[0].invert_yaxis()
    axes[0].legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    results = robustness["results"]
    fpr = np.asarray([row["metrics"]["fpr"] for row in results])
    tpr30 = np.asarray([row["metrics"]["tpr_lead_ge_30ms"] for row in results])
    passed = np.asarray([row["gate"]["passed"] for row in results])
    axes[1].scatter(fpr[passed], tpr30[passed], color=BLUE, label=f"pass ({passed.sum()}/37)", s=18)
    axes[1].scatter(fpr[~passed], tpr30[~passed], color=VERMILLION, marker="x", label="fail", s=22)
    axes[1].axvline(0.20, color="black", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.75, color="black", linestyle="--", linewidth=1.0)
    registered = results[0]["metrics"]
    axes[1].scatter([registered["fpr"]], [registered["tpr_lead_ge_30ms"]], facecolor="none", edgecolor="black", s=70, linewidth=1.0, label="registered")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("operational TPR (≥30 ms)")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    add_panel_labels(axes)
    return finish(fig, "fig7_negative_controls_and_alarm_robustness")


def figure_7_alarm_parameter_trajectories(evidence: dict[str, Any]) -> list[Path]:
    """Show that the deployed alarm is an adaptive multi-parameter state machine."""
    audit = evidence["primary_lineage"]["alarm_state_machine"]
    parameters = audit["adaptive_parameters"]
    labels = {
        "high_threshold": "high threshold",
        "physics_weight": "physics weight",
        "trend_weight": "trend weight",
        "early_threshold": "early threshold",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.205, 5.0), sharex=True)
    colors = [BLUE, ORANGE, GREEN]
    for axis, name in zip(np.asarray(axes).reshape(-1), parameters, strict=True):
        for seed, color in zip(SEEDS, colors, strict=True):
            rows = read_jsonl(primary_run(seed) / "predictions.jsonl")
            values = np.asarray([float(row["parameters"][name]) for row in rows])
            axis.step(
                np.arange(1, len(values) + 1), values,
                where="post", color=color, linewidth=1.0, label=f"seed {seed}",
            )
        initial = float(audit["seeds"][str(SEEDS[0])]["initial_parameters"][name])
        axis.axhline(initial, color=GREY, linestyle="--", linewidth=1.0)
        axis.set_ylabel(labels[name])
        axis.set_xlabel("chronological shot position")
    axes[0, 0].legend(ncol=3, loc="best")
    add_panel_labels(axes)
    return finish(fig, "fig5_adaptive_alarm_state_machine")


def figure_8_equal_label_budget() -> list[Path]:
    """Contrast causal online use of labels with front-loaded target refitting."""

    report = read_json(EQUAL_BUDGET_PATH)
    direct = report["direct_selected_complement"]["comparisons"]
    stage_keys = (
        "static",
        "selected_frontload_frozen_threshold",
        "online",
    )
    stage_labels = ("static", "query-label\nfront-load", "causal\nonline")
    seed_colors = (BLUE, ORANGE, GREEN)
    seed_markers = ("o", "s", "^")
    seed_styles = ("-", "--", "-.")

    fig, axes = plt.subplots(2, 2, figsize=(7.205, 5.0))

    def paired_panel(
        axis: plt.Axes,
        metric: str,
        ylabel: str,
        ylim: tuple[float, float],
    ) -> None:
        x = np.arange(len(stage_keys), dtype=float)
        seed_values = []
        for row, color, marker, linestyle in zip(
            direct, seed_colors, seed_markers, seed_styles, strict=True,
        ):
            values = [float(row["arms"][key][metric]) for key in stage_keys]
            seed_values.append(values)
            axis.plot(
                x,
                values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.2,
                label=f"seed {row['seed']}",
                zorder=2,
            )
        means = np.mean(np.asarray(seed_values, dtype=float), axis=0)
        axis.scatter(
            x,
            means,
            marker="D",
            s=28,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=4,
            label="seed mean",
        )
        axis.set_xticks(x, stage_labels)
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.25, len(stage_keys) - 0.75)
        axis.set_ylim(*ylim)

    paired_panel(axes[0, 0], "tpr_at_30ms", "operational TPR (≥30 ms)", (0.82, 1.0))
    axes[0, 0].legend(ncol=2, loc="lower left")
    paired_panel(axes[0, 1], "fpr", "FPR", (0.0, 0.30))
    paired_panel(axes[1, 0], "tpr30_minus_fpr", "TPR$_{30}$ − FPR", (0.65, 0.90))

    ax = axes[1, 1]
    comparisons = report["comparisons"]
    delta_keys = (
        "selected_frontload_minus_static",
        "random_frontload_minus_support_only_refit",
        "online_minus_static",
    )
    delta_labels = ("query-label\nfront-load", "random-label\nfront-load", "causal\nonline")
    category_colors = (ORANGE, GREY, BLUE)
    for x_index, (key, color) in enumerate(zip(delta_keys, category_colors, strict=True)):
        values = []
        for row in comparisons:
            seed_index = SEEDS.index(int(row["seed"]))
            replicate = int(row["random_replicate"])
            jitter = (replicate - 1) * 0.055 + (seed_index - 1) * 0.012
            value = float(row["primary_deltas"][key])
            values.append(value)
            ax.scatter(
                [x_index + jitter],
                [value],
                color=color,
                marker=seed_markers[seed_index],
                s=18,
                alpha=0.55,
                linewidth=0,
                zorder=2,
            )
        seed_means = []
        for seed in SEEDS:
            per_seed = [
                float(row["primary_deltas"][key])
                for row in comparisons if int(row["seed"]) == seed
            ]
            seed_means.append(float(np.mean(per_seed)))
        ax.vlines(
            x_index,
            min(seed_means),
            max(seed_means),
            color=color,
            linewidth=1.0,
            zorder=3,
        )
        ax.scatter(
            [x_index],
            [float(np.mean(seed_means))],
            marker="D",
            s=34,
            facecolor="white",
            edgecolor=color,
            linewidth=1.3,
            zorder=4,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(np.arange(3), delta_labels)
    ax.set_ylabel("Δ(TPR$_{30}$ − FPR) vs matched static")
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-0.20, 0.08)
    for seed, marker in zip(SEEDS, seed_markers, strict=True):
        ax.scatter([], [], color="#555555", marker=marker, label=f"seed {seed}")
    ax.legend(ncol=3, loc="lower left")

    add_panel_labels(axes)
    return finish(fig, "fig8_equal_label_budget_frontload_ablation")


def figure_9_causal_random_query() -> list[Path]:
    """Compare active querying with causal block-matched random schedules."""

    report = read_json(CAUSAL_RANDOM_PATH)
    per_seed = report["per_seed"]
    x = np.arange(len(SEEDS), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(7.205, 2.65))

    ax = axes[0]
    for index, seed in enumerate(SEEDS):
        row = per_seed[str(seed)]
        random_values = np.asarray([
            float(item["utility_tpr30_minus_fpr"])
            for item in row["random_replicates"]
        ])
        jitter = np.asarray([-0.08, 0.0, 0.08])
        ax.scatter(
            index + jitter, random_values, marker="o", s=22,
            facecolor="white", edgecolor=GREY, linewidth=1.0,
            label="random replicate" if index == 0 else None,
        )
        ax.plot(
            [index - 0.12, index + 0.12],
            [float(np.mean(random_values))] * 2,
            color=GREY, linewidth=1.3,
            label="random mean" if index == 0 else None,
        )
        ax.scatter(
            index, float(row["active"]["utility_tpr30_minus_fpr"]),
            marker="D", s=32, color=BLUE, edgecolor="white", linewidth=1.0,
            zorder=4, label="active" if index == 0 else None,
        )
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set_xlabel("model seed")
    ax.set_ylabel("TPR$_{30}$ − FPR")
    ax.set_ylim(0.62, 0.73)
    ax.legend(loc="lower left", fontsize=6)

    ax = axes[1]
    deltas = np.asarray([
        float(per_seed[str(seed)]["active_minus_random_mean_utility"])
        for seed in SEEDS
    ])
    ax.bar(x, deltas, width=0.58, color=GREEN, edgecolor="none")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set_xlabel("model seed")
    ax.set_ylabel("active − random mean")
    ax.set_ylim(-0.002, 0.024)
    for index, value in enumerate(deltas):
        ax.text(index, value + 0.0007, f"{value:.3f}", ha="center", va="bottom", fontsize=6)

    ax = axes[2]
    active_attempts = np.asarray([
        float(per_seed[str(seed)]["active"]["shared_inter_attempts"])
        for seed in SEEDS
    ])
    random_attempts = np.asarray([
        float(per_seed[str(seed)]["random_mean_within_seed"]["shared_inter_attempts"])
        for seed in SEEDS
    ])
    width = 0.34
    ax.bar(x - width / 2, active_attempts, width, color=BLUE, label="active")
    ax.bar(x + width / 2, random_attempts, width, color=GREY, label="random mean")
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set_xlabel("model seed")
    ax.set_ylabel("realized inter-shot\nupdate attempts")
    ax.legend(loc="upper left", fontsize=6)
    ax.text(
        0.98, 0.04,
        f"equal-seed ratio = {report['aggregate']['realized_inter_attempt_ratio_active_over_random']:.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
    )

    add_panel_labels(axes)
    return finish(fig, "fig9_active_vs_causal_random_query")


def figure_10_target_only_baselines() -> list[Path]:
    """Separate target supervision, scaler choice and transferred weights."""

    report = read_json(EAST_TARGET_ONLY_PATH)
    arms = (
        report["target_only_arms"]["target_only_matched17"]["per_seed"],
        report["target_only_arms"]["target_only_local17"]["per_seed"],
        report["transfer_static"]["per_seed"],
    )
    labels = ("target-only\nsource scaler", "strict EAST-only\nlocal scaler", "J-TEXT → EAST\ntransfer-static")
    colors = (GREY, ORANGE, BLUE)
    markers = ("o", "s", "^")
    metrics = (
        ("operational_tpr", "operational TPR (≥30 ms)", (0.68, 0.89)),
        ("fpr", "FPR", (0.07, 0.22)),
        ("utility_tpr30_minus_fpr", "TPR$_{30}$ − FPR", (0.58, 0.72)),
        ("operational_median_lead_ms", "operational median lead (ms)", (80, 285)),
    )
    seed_colors = (BLUE, ORANGE, GREEN)
    fig, axes = plt.subplots(2, 2, figsize=(7.205, 5.0))
    for axis, (metric, ylabel, ylim) in zip(
        np.asarray(axes).reshape(-1), metrics, strict=True,
    ):
        for seed, seed_color in zip(SEEDS, seed_colors, strict=True):
            values = [float(arm[str(seed)][metric]) for arm in arms]
            axis.plot(
                np.arange(3), values, color=seed_color, linewidth=1.0,
                marker="o", markersize=3.5, label=f"seed {seed}",
            )
        means = [
            float(np.mean([float(arm[str(seed)][metric]) for seed in SEEDS]))
            for arm in arms
        ]
        for index, (mean, color, marker) in enumerate(zip(means, colors, markers, strict=True)):
            axis.scatter(
                index, mean, marker=marker, s=36, facecolor="white",
                edgecolor=color, linewidth=1.4, zorder=4,
            )
        axis.set_xticks(np.arange(3), labels)
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.set_xlim(-0.25, 2.25)
    axes[0, 0].legend(ncol=3, loc="lower right", fontsize=6)
    add_panel_labels(axes)
    return finish(fig, "fig10_target_only_vs_transfer")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    apply_nature_style()
    evidence = read_json(EVIDENCE_PATH)

    figures: dict[str, dict[str, Any]] = {}
    builders = (
        ("fig0", figure_0_end_to_end_architecture, []),
        ("fig1", figure_1_protocol, []),
        ("fig2", figure_3_three_stage, [evidence]),
        ("fig3", figure_2_source_shortcut, [evidence]),
        ("fig4", figure_5_chronology_and_queries, [evidence]),
        ("fig5", figure_7_alarm_parameter_trajectories, [evidence]),
        ("fig6", figure_4_case_study, []),
        ("fig7", figure_6_controls_and_robustness, [evidence]),
        ("fig8", figure_8_equal_label_budget, []),
        ("fig9", figure_9_causal_random_query, []),
        ("fig10", figure_10_target_only_baselines, []),
    )
    for name, builder, arguments in builders:
        outputs = builder(*arguments)
        figures[name] = {
            "outputs": {path.name: sha256_file(path) for path in outputs},
            "width_mm": 183,
            "raster_dpi": 600,
        }

    inputs = [
        EVIDENCE_PATH,
        ROBUSTNESS_PATH,
        TRACE_PATH,
        EQUAL_BUDGET_PATH,
        CAUSAL_RANDOM_PATH,
        EAST_TARGET_ONLY_PATH,
        Path(__file__).resolve(),
        Path(__file__).with_name("publication_style.py"),
        PACKAGE / "configs/primary_chain_v3c.json",
        ROOT / "experiments/transfer_methods_2026/grid_search.py",
        ROOT / "experiments/transfer_methods_2026/rebuilt_harness.py",
        ROOT / "config/splits/east_paper131_split_all_in.json",
        ROOT / "config/splits/east_hdf5_authoritative_balanced360_v1.json",
        ROOT / "splits/train.txt",
        ROOT / "splits/val.txt",
        ROOT / "splits/test.txt",
    ]
    for seed in SEEDS:
        run = primary_run(seed)
        inputs.extend([run / "static_predictions.jsonl", run / "predictions.jsonl", run / "query_decisions.jsonl"])
        inputs.append(
            PACKAGE / f"evidence/source_lineage/seed{seed}/source_checkpoint.pt"
        )
    manifest = {
        "schema_version": "paper-figure-manifest-v1",
        "style": "Nature-like, 183 mm double-column, colorblind-safe, vector plus 600 dpi PNG",
        "quantitative_figures_are_data_backed": True,
        "generative_images_used": False,
        "claim_boundary": evidence["claim_boundary"],
        "inputs": {manifest_path(path): sha256_file(path) for path in inputs},
        "figures": figures,
    }
    (FIGURE_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"figure_count": len(figures), "manifest": str(FIGURE_DIR / 'figure_manifest.json')}, indent=2))


if __name__ == "__main__":
    main()
