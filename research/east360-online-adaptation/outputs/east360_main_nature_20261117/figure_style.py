"""Shared typography, colours and physical dimensions for all four panels."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
INK = "#22313A"
MUTED = "#707B85"
RULE = "#9AA3AA"
BLUE = "#547FA3"
TEAL = "#007F86"
PALE = "#BBC7CF"
WARM = "#C27A38"
PAPER_WARM = "#FBF3E9"
DPI = 600
PT_PER_MM = 72 / 25.4

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "Arial", "DejaVu Sans"],
    "font.size": 7.2,
    "axes.labelsize": 7.4,
    "axes.titlesize": 8.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})


def figure(width_mm, height_mm=68):
    return plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")


def heading(fig, letter, title, subtitle=None, left=0.13):
    fig.text(left - 0.09, 0.956, letter, fontsize=10, weight="bold", va="top")
    fig.text(left, 0.956, title, fontsize=8.8, weight="bold", va="top")
    if subtitle:
        fig.text(left, 0.873, subtitle, color=MUTED, fontsize=6.7, va="top")


def legend(fig, left=0.135, top=0.863):
    handles = [
        Line2D([], [], color=MUTED, lw=1.1, ls=(0, (3, 2)), label="前四步"),
        Line2D([], [], color=TEAL, lw=1.3, label="再加第5步"),
        Line2D([], [], color=RULE, lw=0.8, ls=":", label="报警线"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, top),
               ncol=3, borderaxespad=0, handlelength=1.8,
               columnspacing=1.1, handletextpad=0.5)


def save_panel(fig, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"panel_{name}.{extension}", dpi=DPI)
    plt.close(fig)


def draw_scores(ax, x, parent, active, alarm):
    ax.plot(x, active, color=TEAL, lw=1.2, zorder=2)
    ax.plot(x, parent, color=MUTED, lw=1.05, ls=(0, (3, 2)), zorder=3)
    ax.axhline(alarm, color=RULE, lw=0.85, ls=":", zorder=1)
    ax.set(ylim=(0, 0.72), yticks=(0, 0.2, 0.4, 0.6))


def first_point(ax, x, y, colour, marker="o"):
    ax.scatter(x, y, s=17, facecolors="white", edgecolors=colour,
               linewidths=1.05, marker=marker, zorder=5)
