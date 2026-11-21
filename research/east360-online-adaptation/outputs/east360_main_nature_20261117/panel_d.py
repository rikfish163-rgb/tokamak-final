"""First false alarm precedes the first step-5 update in safe shot 67900."""

import numpy as np

from figure_data import curves, metadata
from figure_style import INK, MUTED, RULE, draw_scores, figure, first_point, heading, save_panel


WIDTH_MM = 91.5
HEIGHT_MM = 68


def build():
    data, meta = curves(67900), metadata()
    case = meta["shot67900"]
    x = (data["time_s"] - case["signal_start_s"]) * 1000
    write_ms = (case["first_step5_write_s"] - case["signal_start_s"]) * 1000
    fig = figure(WIDTH_MM)
    heading(fig, "d", "67900：误报早于首次修正")
    ax = fig.add_axes((0.13, 0.185, 0.825, 0.59))
    draw_scores(ax, x, data["before_step5"], data["with_step5"], meta["threshold"])
    ax.axvline(write_ms, color=RULE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(write_ms - 5, 0.692, f"{write_ms:.0f} ms 首次修正", color=MUTED,
            fontsize=7, ha="right", va="top")
    indices = {arm: int(np.flatnonzero(data[arm] >= meta["threshold"])[0])
               for arm in ("before_step5", "with_step5")}
    assert indices["before_step5"] == indices["with_step5"]
    i = indices["with_step5"]
    assert abs(data["time_s"][i] - case["firsts"]["with_step5"]) < 1e-9
    first_point(ax, x[i], data["with_step5"][i], INK)
    ax.annotate(f"{x[i]:.0f} ms 误报", (x[i], data["with_step5"][i]),
                xytext=(-5, 11), textcoords="offset points", color=INK,
                fontsize=7.3, ha="center")
    ax.set(xlim=(0, 365), xticks=(0, 100, 200, 300),
           xlabel="记录开始后的时间（ms）", ylabel="报警分数")
    return fig


if __name__ == "__main__":
    save_panel(build(), "d")
