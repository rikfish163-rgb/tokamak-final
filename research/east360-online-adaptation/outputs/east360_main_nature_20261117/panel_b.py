"""False alarms exceed the primary budget in the frozen historical extension."""

from figure_data import metrics
from figure_style import BLUE, INK, MUTED, PALE, PAPER_WARM, TEAL, WARM, figure, heading, save_panel


WIDTH_MM = 78
HEIGHT_MM = 68


def build():
    rows = [r for r in metrics() if r["cohort"] == "EAST2593"]
    fig = figure(WIDTH_MM)
    heading(fig, "b", "2,593炮扩展：误报超限", "1,018 破裂 / 1,575 未破裂", left=0.16)
    ax = fig.add_axes((0.16, 0.27, 0.79, 0.48))
    ax.axvspan(20, 40, color=PAPER_WARM, zorder=0)
    ax.axvline(20, color=WARM, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(20.6, -0.54, "20% 上限", fontsize=6.7, color=WARM, va="bottom")
    for i, (row, colour) in enumerate(zip(rows, (PALE, BLUE, TEAL), strict=True)):
        percent = row["fpr"] * 100
        ax.barh(i, percent, height=0.39, color=colour, zorder=2)
        ax.text(1.2, i, f'{row["fp"]} / {row["n_safe"]:,}', va="center",
                color=INK if i == 0 else "white", fontsize=6.9, zorder=3)
        ax.text(percent + 1.2, i, f"{percent:.2f}%", va="center", color=colour if i else MUTED,
                fontsize=7.7, weight="bold" if i == 2 else "normal")
    ax.set(xlim=(0, 40), ylim=(2.55, -0.65), yticks=(0, 1, 2),
           yticklabels=("静态", "前四步", "完整"), xticks=(0, 10, 20, 30),
           xlabel="误报率（%）")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    current = rows[-1]
    fig.text(0.16, 0.055, f'完整方法及时：{current["tp"]}/{current["n_disruptive"]:,}（{current["tpr"] * 100:.2f}%）',
             fontsize=6.9, color=INK)
    return fig


if __name__ == "__main__":
    save_panel(build(), "b")
