"""Unsmoothed first-alarm traces for disrupted shot 70320."""

import numpy as np

from figure_data import curves, metadata
from figure_style import MUTED, PAPER_WARM, RULE, TEAL, draw_scores, figure, first_point, heading, legend, save_panel


WIDTH_MM = 91.5
HEIGHT_MM = 68


def build():
    data, meta = curves(70320), metadata()
    case = meta["shot70320"]
    x = (data["time_s"] - case["down_s"]) * 1000
    fig = figure(WIDTH_MM)
    heading(fig, "c", "70320：第5步提前首报")
    legend(fig, left=0.13, top=0.865)
    ax = fig.add_axes((0.13, 0.185, 0.825, 0.59))
    ax.axvspan(-30, 0, color=PAPER_WARM, zorder=0)
    ax.axvline(-30, color=RULE, lw=0.65, ls=(0, (2, 3)), zorder=1)
    draw_scores(ax, x, data["before_step5"], data["with_step5"], meta["threshold"])
    for arm, colour, offset, alignment in (
        ("with_step5", TEAL, (0, 10), "center"),
        ("before_step5", MUTED, (-11, 12), "right"),
    ):
        i = int(np.flatnonzero(data[arm] >= meta["threshold"])[0])
        assert abs(data["time_s"][i] - case["firsts"][arm]) < 1e-9
        first_point(ax, x[i], data[arm][i], colour, "o" if arm == "before_step5" else "D")
        ax.annotate(f"{abs(x[i]):.0f} ms", (x[i], data[arm][i]), xytext=offset,
                    textcoords="offset points", fontsize=7.3, ha=alignment, color=colour)
    ax.set(xlim=(-310, 4), xticks=(-300, -200, -100, -30, 0),
           xlabel="相对破裂时刻（ms）", ylabel="报警分数")
    return fig


if __name__ == "__main__":
    save_panel(build(), "c")
