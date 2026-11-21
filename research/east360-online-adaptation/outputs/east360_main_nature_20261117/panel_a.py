"""Cumulative timely-warning gains and false alarms on the fixed EAST360 stream."""

import numpy as np

from figure_data import metrics
from figure_style import BLUE, INK, MUTED, PALE, RULE, TEAL, WARM, figure, heading, save_panel


WIDTH_MM = 105
HEIGHT_MM = 68


def build():
    rows = [r for r in metrics() if r["cohort"] == "EAST360"]
    steps = np.array([r["step"] for r in rows])
    tp = np.array([r["tp"] for r in rows])
    fp = np.array([r["fp"] for r in rows])
    fig = figure(WIDTH_MM)
    heading(fig, "a", "EAST360：逐步增加的及时预警", "180 破裂 / 180 未破裂；前20炮计入评分")

    ax = fig.add_axes((0.13, 0.48, 0.81, 0.28))
    ax.plot(steps, tp, color=PALE, lw=1.25, zorder=1)
    ax.plot(steps[-2:], tp[-2:], color=TEAL, lw=1.6, zorder=2)
    colours = [MUTED, BLUE, BLUE, BLUE, BLUE, TEAL]
    for i, (step, count, colour) in enumerate(zip(steps, tp, colours, strict=True)):
        ax.scatter(step, count, s=25 if i == 5 else 18, facecolors=colour,
                   edgecolors="white", linewidths=0.65, zorder=3)
        ax.annotate(str(count), (step, count), xytext=(0, 6),
                    textcoords="offset points", ha="center", color=colour,
                    fontsize=8, weight="bold" if i == 5 else "normal")
    ax.set(xlim=(-0.25, 5.35), ylim=(130, 150), yticks=(130, 140, 150),
           ylabel="及时检出（炮）", xticks=steps)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.spines["bottom"].set_visible(False)

    af = fig.add_axes((0.13, 0.23, 0.81, 0.175), sharex=ax)
    af.axhline(36, color=WARM, ls=(0, (3, 2)), lw=0.85)
    af.text(2.65, 38.2, "20% 上限", color=WARM, fontsize=6.7, va="bottom")
    af.plot(steps, fp, color=RULE, lw=1, zorder=1)
    for i, (step, count, colour) in enumerate(zip(steps, fp, colours, strict=True)):
        af.scatter(step, count, s=20 if i == 5 else 15, facecolors="white",
                   edgecolors=colour, linewidths=1, zorder=3)
        if i in (0, 1, 2, 5):
            af.annotate(str(count), (step, count), xytext=(0, -9),
                        textcoords="offset points", ha="center", color=colour,
                        fontsize=7.2, weight="bold" if i == 5 else "normal")
    af.set(ylim=(19, 42), yticks=(20, 30, 40), ylabel="误报（炮）", xticks=steps)
    af.set_xticklabels(("静态", "1", "2", "3", "4", "5"))
    af.set_xlabel("累加处理步骤", labelpad=4)
    fig.text(0.13, 0.045, "完整方法：及时 81.11%  /  误报 18.89%", fontsize=7.1, color=INK)
    return fig


if __name__ == "__main__":
    save_panel(build(), "a")
