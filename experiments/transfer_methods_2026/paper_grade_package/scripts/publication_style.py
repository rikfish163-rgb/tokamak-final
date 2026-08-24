"""Self-contained publication figure style/export helpers.

Vendored into the paper package so regeneration does not depend on a
user-specific Codex skill installation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


OKABE_ITO = (
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
)

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<")
LINE_STYLES = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
)
HATCHES = ("", "///", "\\\\", "xx", "..", "++", "oo", "--")
_FONT_SIZE_KEYS = (
    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
)


def series_encoding(index: int) -> dict[str, object]:
    """Return color plus grayscale-redundant encodings for one series."""

    normalized = int(index) % len(OKABE_ITO)
    return {
        "color": OKABE_ITO[normalized],
        "marker": MARKERS[normalized],
        "linestyle": LINE_STYLES[normalized],
        "hatch": HATCHES[normalized],
    }


def minimum_configured_font_size() -> float:
    """Return the smallest publication text size currently configured."""

    return min(float(mpl.rcParams[key]) for key in _FONT_SIZE_KEYS)


def apply_nature_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "font.size": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 1.0,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "jtext-east-paper-v1",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_publication_figure(
    fig: plt.Figure,
    filename: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
    dpi: int = 600,
    pad_inches: float = 0.03,
) -> list[Path]:
    base = Path(filename)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in formats:
        output = base.with_suffix(f".{extension}")
        metadata = None
        if extension == "pdf":
            metadata = {
                "Creator": "J-TEXT-EAST paper-grade package",
                "Producer": "Matplotlib",
                "CreationDate": None,
                "ModDate": None,
            }
        elif extension == "svg":
            metadata = {
                "Creator": "J-TEXT-EAST paper-grade package",
                "Date": "2026-08-20",
            }
        fig.savefig(
            output,
            format=extension,
            dpi=min(dpi, 300) if extension in {"pdf", "svg", "eps"} else dpi,
            bbox_inches="tight",
            pad_inches=pad_inches,
            facecolor="white",
            edgecolor="none",
            metadata=metadata,
        )
        outputs.append(output)
        print(f"saved: {output}")
    return outputs


__all__ = [
    "HATCHES",
    "LINE_STYLES",
    "MARKERS",
    "OKABE_ITO",
    "apply_nature_style",
    "minimum_configured_font_size",
    "save_publication_figure",
    "series_encoding",
]
