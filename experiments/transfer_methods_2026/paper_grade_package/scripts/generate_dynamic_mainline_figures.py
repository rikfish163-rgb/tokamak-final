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
            ax.annotate("", (.5, y + .065), (.5, y + .105), arrowprops={"arrowstyle": "->", "lw": .7})
    for label, ax in zip(("A", "B", "C"), axes):
        ax.text(-.02, 1.0, label, fontsize=9, fontweight="bold", va="top")
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


def generate_dynamic_mainline_figures(
    *, east_root: str | Path, jtext_root: str | Path, output_dir: str | Path,
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
    names = {"F1": "F1_causal_data_role_pipeline", "F2": "F2_advancedtime_label_ablation", "F3": "F3_static_online_three_seed", "F4": "F4_query_update_audit", "F5": "F5_falsification_controls"}
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
    args = parser.parse_args(argv)
    generate_dynamic_mainline_figures(east_root=args.east_root, jtext_root=args.jtext_root, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
