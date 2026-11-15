"""Draw a reader's guide and two detailed figures from the verified trace."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns


HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#14283b", "#45576a", "#e6ebf1"
LATEST, BASE, OLDER, NOWAIT = "#006bb0", "#626e7b", "#87577e", "#b75615"
THRESH, LIGHT, WAIT = "#40576c", "#e5f1fa", "#f1f3f6"
ERR = sns.blend_palette(["#315a92", "#f8f9fc", "#aa501a"], as_cmap=True)
CH = {
    "n=1 amplitude": "磁扰动 n=1", "dx_a": "水平位移", "dy_a": "垂直位移",
    "v_loop": "环电压", "sxr_var": "软 X 射线方差", "xuv_var": "极紫外方差",
    "sxr_skew": "软 X 射线偏度", "xuv_skew": "极紫外偏度",
}


def style():
    cjk = "Noto Sans CJK SC"
    try:
        mpl.font_manager.findfont(cjk, fallback_to_default=False)
    except ValueError:
        # Matplotlib can index only the first face of some font collections.
        cjk = "Noto Sans CJK JP"
    sns.set_theme(style="ticks", context="paper", font=cjk, palette=[LATEST, BASE, OLDER, NOWAIT])
    path = HERE.parents[2] / "Fuison/.agents/skills/figure-style/kernel.py"
    spec = spec_from_file_location("figure_style", path)
    if path.is_file() and spec and spec.loader:
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.apply_figure_style(font=cjk, sizes=(9, 8, 7))
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [cjk, "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#8793a0",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.unicode_minus": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
        "legend.frameon": False, "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": None,
        "figure.dpi": 130, "savefig.dpi": 400,
    })


def title(fig, main, sub):
    fig.text(.055, .982, main, va="top", weight="bold", fontsize=14)
    fig.text(.055, .948, sub, va="top", color=MUTED, fontsize=9)


def label(ax, letter, value, *, x=0, y=1.11):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, weight="bold", color=LATEST)
    ax.annotate(value, (x, y), xycoords="axes fraction", xytext=(13, 0),
                textcoords="offset points", ha="left", va="bottom", fontsize=9)


def guides(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=.6)


def series(ax, x, y, *, color, ls="-", lw=1.35, **kwargs):
    """Draw each supplied time point; no averaging, interpolation, or interval."""
    return sns.lineplot(x=x, y=y, ax=ax, color=color, linestyle=ls, linewidth=lw,
                        sort=False, estimator=None, errorbar=None, legend=False, **kwargs)


def time_panel(ax, data, meta):
    tr, t = meta["trace"], data["t70320"]
    origin, release = tr["issue_index"], tr["release_index"]
    start = release + tr["wait_samples"]
    done = start + tr["patch_width"]
    end = meta["cases"]["70320"]["scores"]["current_5"]["first_index"]
    ix = np.arange(origin-2, end+5)
    x = 1000 * (t[ix] - t[origin])
    ax.set(xlim=(x[0], x[-1]), ylim=(-.2, 2.75), yticks=[])
    ax.set_xlabel(f"距发出预测的时间（ms；0 = {t[origin]:.3f} s）", labelpad=2)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 58])
    ax.spines["left"].set_visible(False)
    ax.hlines(.15, 0, end-origin, color="#aab6c0", lw=.7, zorder=0)
    ax.broken_barh([(1, release-origin-1)], (1.19, .48), facecolors=LIGHT)
    ax.text((release-origin)/2, 1.43, f"25 个实测点 · {t[origin+1]:.3f}–{t[release-1]:.3f} s",
            ha="center", va="center", color=LATEST, fontsize=8)
    ax.broken_barh([(release-origin, start-release)], (1.19, .48), facecolors=WAIT)
    ax.annotate("额外等 5 点", xy=((release+start)/2-origin, 1.42), xytext=(29, 2.16),
                textcoords="data", ha="center", fontsize=8, color=MUTED,
                arrowprops={"arrowstyle": "-", "lw": .6, "color": "#9ba7b4"})
    handoff = np.clip((ix - start)/(done-start), 0., 1.)
    use = (ix >= release) & (ix <= end)
    series(ax, x[use], .15 + .82*handoff[use], lw=1.7, color=LATEST)
    ax.text((start+done)/2-origin+5.5, 1.49, f"8 个间隔平滑切换\n{t[start]:.3f}–{t[done]:.3f} s",
            ha="center", va="center", fontsize=8, color=LATEST)
    for xi, yi in ((0, .15), (release-origin, .15), (end-origin, .97)):
        ax.scatter(xi, yi, color=LATEST if xi == end-origin else BASE, s=12, zorder=3)
    ax.text(.2, .35, "发预测", fontsize=8)
    ax.text(release-origin, .28, f"{t[release]:.3f} s 误差可用", fontsize=8,
            ha="right", va="bottom")
    ax.text(end-origin, 1.22, f"{t[end]:.3f} s\n首报", fontsize=8,
            ha="center", va="bottom", color=LATEST, weight="bold")
    label(ax, "a", "先发出预测，等实测到齐才允许修正", y=1.0)


def forecast_panel(ax, data, meta):
    t = data["t70320"]
    idx = data["target_index"]
    c = int(np.flatnonzero(data["channels"] == "dy_a")[0])
    x = 1000*(t[idx] - t[meta["trace"]["issue_index"]])
    for n in range(5):
        if n % 2 == 0:
            ax.axvspan(.5+5*n, 5.5+5*n, color="#f2f6fa", zorder=0, lw=0)
    series(ax, x, data["forecast"][:, c], ls="--", color=BASE, lw=1.5, label="发出时的预测")
    series(ax, x, data["observation"][:, c], marker="o", ms=2.0, lw=1.45, color=LATEST,
           label="后来到达的实测")
    guides(ax)
    ax.set(xlim=(.1, 25.9), ylim=(-3.2, 3.75), xticks=np.arange(3, 26, 5),
           xticklabels=[f"{i}–{i+4}" for i in range(1, 26, 5)],
           yticks=[-3, -1, 1, 3], ylabel="垂直位移代理\n（模型坐标，无量纲）",
           xlabel="预测的 25 个点（每 5 点为一段）")
    ax.legend(loc="lower right", ncol=2, handlelength=1.7, columnspacing=1.15)
    label(ax, "b", "垂直位移实测先低于预测，后高于预测", y=1.075)


def error_panel(ax, error, channels, headline, *, left):
    norm = TwoSlopeNorm(0, -1.5, 1.5)
    sns.heatmap(error.T, cmap=ERR, norm=norm, cbar=False, ax=ax,
                xticklabels=[f"{i}–{i+4}" for i in range(1, 26, 5)],
                yticklabels=[CH[c] for c in channels] if left else False)
    ax.set_title(headline)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    ax.tick_params(axis="y", length=0, pad=5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for row, c in enumerate(channels):
        for col in range(5):
            v = error[col, row]
            ax.text(col+.5, row+.5, f"{v:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=7,
                    color="white" if abs(v) >= 1.17 else INK)
        if c == "dy_a":
            ax.add_patch(Rectangle((.01, row+.03), 4.98, .94, fill=False, lw=.9, ec=INK))


def write_panel(ax, data, meta):
    # Display one actual feature with a large change, not a projected or invented signal.
    a = data["token_delta"][0]
    _, dim = np.unravel_index(np.argmax(abs(a)), a.shape)
    y = a[:, dim]
    xpos = np.arange(1, len(y)+1)
    guides(ax)
    ax.axhline(0, color=BASE, lw=.6)
    ax.bar(xpos, y, width=.68, color=np.where(y >= 0, LATEST, "#ba6832"))
    ax.set(xlim=(.3, len(y)+.65), ylim=(-.38, .39),
           xticks=[1, 4, 8, 12, 16], xlabel="8 点一块（从早到晚）",
           ylabel="内部数值的改变量")
    label(ax, "d", "改写第一层看到的时间块", y=1.22)
    ax.text(0, 1.07, f"例：报警时，第 {dim+1} 维（改动最大的一维）",
            transform=ax.transAxes, va="bottom", fontsize=8, color=MUTED)


def readout_panel(ax, data, meta):
    logit = data["logits_at_alarm"]
    threshold = meta["threshold"]
    needed = np.log(threshold/(1-threshold))-logit[0]
    old, latest = logit[[2, 1]]-logit[0]
    ax.axvline(0, color="#a2abb5", lw=.7)
    ax.axvline(needed, color=THRESH, lw=.85, ls=(0, (3, 2)))
    ax.barh([1, 0], [latest, old], height=.40, color=[LATEST, OLDER])
    ax.set(ylim=(-.42, 1.54), xlim=(-.095, .176), yticks=[0, 1], yticklabels=["较早", "最新"],
           xticks=[-.05, 0, .05, .10, .15], xlabel="对换算前风险分数的改变量（logit）")
    ax.tick_params(axis="y", length=0, pad=4)
    for row, dx, col in ((1, latest, LATEST), (0, old, OLDER)):
        ax.text(.006 if dx > 0 else -.008, row, f"{dx:+.4f}".replace("-", "−"),
                fontsize=8, va="center", ha="left" if dx > 0 else "right", color="white")
    label(ax, "e", "同样大小的改动，最后得分一升一降", y=1.22)
    ax.text(0, 1.07, f"虚线：越过报警线还差 {needed:.4f}", transform=ax.transAxes,
            va="bottom", fontsize=8, color=MUTED)


def mechanism(data, meta):
    fig = plt.figure(figsize=(183/25.4, 216/25.4))
    title(fig, "EAST 70320｜从波形预测误差到模型内部修正",
          "同一炮的实际回放。波形坐标、误差和第一层输入均取自复算。")
    ax = fig.add_axes([.14, .821, .81, .079])
    time_panel(ax, data, meta)
    ax = fig.add_axes([.14, .601, .81, .141])
    forecast_panel(ax, data, meta)
    left = fig.add_axes([.205, .282, .295, .209])
    right = fig.add_axes([.545, .282, .295, .209])
    label(left, "c", "从当前误差中扣掉发出预测时已能估计的偏差", y=1.182, x=-.22)
    t = data["t70320"]
    tr = meta["trace"]
    error_panel(left, data["new_error"], data["channels"],
                f"最新误差 · {t[tr['issue_index']]:.3f} s 发出", left=True)
    error_panel(right, data["preceding_error"], data["channels"],
                f"较早误差 · {t[tr['previous_issue_index']]:.3f} s 发出", left=False)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(0, -1.5, 1.5), cmap=ERR),
                      cax=fig.add_axes([.87, .305, .015, .16]), ticks=[-1.5, 0, 1.5])
    cb.ax.tick_params(length=2)
    cb.set_label("有符号误差")
    write_panel(fig.add_axes([.14, .061, .34, .142]), data, meta)
    readout_panel(fig.add_axes([.575, .061, .372, .142]), data, meta)
    return fig


CURVES = {
    "bias_only": (BASE, (0, (4, 2)), "第 4 行｜只保留原有四步"),
    "current_5": (LATEST, "-", "第 5 行｜最新误差，等 5 点"),
    "earlier_5": (OLDER, (0, (1.2, 2)), "第 6 行｜较早误差，等 5 点"),
    "current_0": (NOWAIT, (0, (5, 2, 1.2, 2)), "第 7 行｜最新误差，不额外等"),
}


def curve(ax, data, shot, k, ix, *, x=None, markers=False):
    c, ls, _ = CURVES[k]
    x = data[f"t{shot}"][ix] if x is None else x
    series(ax, x, data[f"s{shot}_{k}"][ix], color=c, ls=ls, lw=1.75 if k == "current_5" else 1.2,
           marker="o" if markers else None, ms=2.5, zorder=4 if k == "current_5" else 3)


def threshold_line(ax, threshold):
    ax.axhline(threshold, color=THRESH, ls=(0, (2, 2.5)), lw=.85, zorder=2)
    guides(ax)


def overview(ax, data, meta):
    d = meta["cases"]["70320"]
    s = d["scores"]
    t = data["t70320"]
    down = d["down_s"]
    ix = np.flatnonzero((t >= 3.61) & (t <= down))
    late = down-meta["timely_strictly_over_ms"]/1000
    threshold = meta["threshold"]
    ax.axvspan(late, down, color="#eceff3", lw=0)
    ax.axvline(down, color=INK, lw=.8)
    ax.axvspan(3.6935, 3.7065, color=LIGHT, lw=0)
    threshold_line(ax, threshold)
    for k in ("bias_only", "current_5"):
        curve(ax, data, 70320, k, ix)
        first = s[k]
        ax.scatter(first["first_s"], first["first_score"], color=CURVES[k][0],
                   s=22, zorder=7, edgecolor="white", linewidth=.55)
    ax.annotate("提前 225 ms", (s["current_5"]["first_s"], s["current_5"]["first_score"]),
                xytext=(3.716, .611), textcoords="data", color=LATEST, fontsize=8, weight="bold",
                arrowprops={"arrowstyle": "-", "lw": .75, "color": LATEST})
    ax.annotate("提前 12 ms", (s["bias_only"]["first_s"], s["bias_only"]["first_score"]),
                xytext=(3.876, .607), textcoords="data", color=BASE, fontsize=8, ha="center",
                arrowprops={"arrowstyle": "-", "lw": .7, "color": BASE})
    ax.text(late-.003, .071, "最后 30 ms", fontsize=8, color=MUTED, ha="right", va="center")
    ax.text(3.754, threshold+.021, "报警线", fontsize=8, color=THRESH, va="bottom")
    ax.text(down-.003, .658, f"{down:.3f} s 破裂", ha="right", va="center", fontsize=8)
    ax.set(xlim=(3.608, down+.005), ylim=(0, .69), ylabel="破裂风险分数", xlabel="原始记录时间（s）")
    ax.set_xticks([3.65, 3.70, 3.75, 3.80, 3.85, 3.90])
    label(ax, "a", "70320：在同一条报警线下，首报提早 213 ms", y=1.065)


def zoom_70320(ax, data, meta):
    t = data["t70320"]
    threshold = meta["threshold"]
    n = meta["cases"]["70320"]["scores"]["current_5"]["first_index"]
    ix = np.arange(n-3, n+10)
    x = np.rint(1000*(t[ix]-t[n])).astype(int)
    ax.axvspan(-.42, .42, color=LIGHT, lw=0)
    threshold_line(ax, threshold)
    for k in ("bias_only", "earlier_5", "current_5"):
        curve(ax, data, 70320, k, ix, x=x, markers=True)
    ax.set(xlim=(x[0]-.45, x[-1]+.45), ylim=(.176, .573), yticks=[.2, .3, .4, .5],
           xticks=[-2, 0, 2, 4, 6, 8], xlabel=f"距首次报警的时间（ms；0 = {t[n]:.3f} s）",
           ylabel="破裂风险分数")
    ax.annotate(f"{data['s70320_current_5'][n]:.4f}", (0, data["s70320_current_5"][n]),
                xytext=(-1.3, .55), textcoords="data", fontsize=8, color=LATEST, weight="bold",
                arrowprops={"arrowstyle": "-", "lw": .65, "color": LATEST})
    label(ax, "b", "首报放大：最新误差让分数刚好越线", y=1.065)


def zoom_73743(ax, ramp, data, meta):
    t = data["t73743"]
    s = meta["cases"]["73743"]["scores"]
    threshold = meta["threshold"]
    n = s["current_0"]["first_index"]
    release = int(data["release73743"][np.searchsorted(data["release73743"], n, side="right")-1])
    wait, width = (meta["trace"][k] for k in ("wait_samples", "patch_width"))
    ix = np.arange(release-4, release+22)
    x = ix-release
    threshold_line(ax, threshold)
    for k in ("bias_only", "current_0", "current_5"):
        curve(ax, data, 73743, k, ix, x=x, markers=True)
    ax.scatter(n-release, s["current_0"]["first_score"], color=NOWAIT, edgecolor="white",
               linewidth=.55, s=22, zorder=5)
    ax.axvline(n-release, color="#c3cbd4", lw=.65, ls=(0, (2, 2)), zorder=1)
    ax.text(n-release+4, .575, "不等：误报", fontsize=8, color=NOWAIT, va="center")
    ax.text(n-release+4, .444, "等 5 点：未报", fontsize=8, color=LATEST, va="center")
    ax.set(xlim=(-4.5, 21.5), ylim=(.173, .627), yticks=[.2, .3, .4, .5, .6])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    label(ax, "c", "未破裂炮 73743：等 5 点避开一次误报", y=1.065)
    for k, delay in (("current_0", 0), ("current_5", wait)):
        c, ls, _ = CURVES[k]
        series(ramp, x, np.clip((x-delay)/width, 0., 1.), color=c, ls=ls, lw=1.4)
    ramp.axvline(n-release, color="#c3cbd4", lw=.65, ls=(0, (2, 2)), zorder=1)
    ramp.set(xlim=ax.get_xlim(), ylim=(-.10, 1.13), xticks=[-4, 0, 5, 8, 13, 20],
             yticks=[0, 1], yticklabels=["旧", "新"])
    ramp.set_xlabel(f"误差可用后（ms；0 = {t[release]:.3f} s）", labelpad=2)
    ramp.tick_params(axis="y", length=0)
    ramp.spines["left"].set_visible(False)


def counts_panel(ax, meta):
    counts = meta["counts"]
    ax.set_axis_off()
    label(ax, "d", "完整 360 炮：多两炮及时，误报名单不变", y=1.025)
    cols = [.015, .62, .90]
    for x, h, ha in zip(cols, ("回放版本", "及时检出／180", "误报／180"), ("left", "center", "center")):
        ax.text(x, .735, h, transform=ax.transAxes, ha=ha, va="center", color=MUTED, fontsize=8)
    ax.add_patch(Rectangle((0, .025), 1, .204, facecolor=LIGHT, lw=0, transform=ax.transAxes))
    ax.hlines(.62, 0, 1, transform=ax.transAxes, color="#d3dce5", lw=.7)
    for k, y in zip(("bias_only", "current_0", "current_5"), (.51, .315, .128)):
        c, _, name = CURVES[k]
        ax.text(cols[0], y, name, color=c, transform=ax.transAxes, fontsize=8, va="center",
                weight="bold" if k == "current_5" else "normal")
        for x, value in zip(cols[1:], counts[k]):
            ax.text(x, y, str(value), color=INK, transform=ax.transAxes, fontsize=9,
                    weight="bold" if k == "current_5" else "normal", ha="center", va="center")


def effects(data, meta):
    fig = plt.figure(figsize=(183/25.4, 186/25.4))
    title(fig, "提前报警来自误差方向；额外等待改变交接时机",
          f"EAST 360 炮开发回放。两炮共用报警阈值 {meta['threshold']:.7f}；散点为原始采样时刻。")
    handles = [Line2D([0], [0], c=CURVES[k][0], ls=CURVES[k][1], lw=1.45, label=CURVES[k][2])
               for k in ("bias_only", "current_5", "earlier_5", "current_0")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.056, .91), ncol=2,
               borderaxespad=0, handlelength=2.1, columnspacing=1.7)
    overview(fig.add_axes([.125, .525, .817, .282]), data, meta)
    zoom_70320(fig.add_axes([.125, .277, .369, .166]), data, meta)
    zoom_73743(fig.add_axes([.572, .277, .37, .166]),
               fig.add_axes([.572, .232, .37, .039]), data, meta)
    counts_panel(fig.add_axes([.125, .019, .817, .123]), meta)
    return fig


def reader_guide(data, meta):
    fig = plt.figure(figsize=(183/25.4, 209/25.4))
    x0 = .073
    fig.text(x0, .961, "EAST 核聚变实验 · 360 次历史放电", fontsize=9, color=LATEST, va="top")
    fig.text(x0, .923, "怎样让 EAST 更早发现破裂？", fontsize=19, weight="bold", va="top")
    fig.text(x0, .853,
             "EAST 用磁场约束高温等离子体。一次放电中，等离子体突然失稳、\n"
             "磁约束迅速丧失，叫作破裂。早一点报警，就能给保护设备多留一点时间。",
             fontsize=10, va="top", linespacing=1.55)
    fig.text(x0, .749, "模型怎样用刚收到的实测信号？", fontsize=11, weight="bold", va="top")
    fig.text(x0, .715,
             "模型用另一台装置 J-TEXT 的数据训练。到了 EAST，它边读仪器信号，\n"
             "边预测接下来的 25 个读数。等实测到了，它看哪里猜错了，再用这些\n"
             "差别调整这次放电后面的风险打分。已经发出的分数保持原样。",
             fontsize=10, va="top", linespacing=1.5)

    shock = meta["cases"]["70320"]
    scores = shock["scores"]
    before, after = (scores[k]["lead_ms"] for k in ("bias_only", "current_5"))
    down = shock["down_s"]
    threshold = meta["threshold"]
    t = data["t70320"]
    ms = 1000*(t-down)
    deadline = meta["timely_strictly_over_ms"]
    fig.text(x0, .588, f"70320：第一次报警早了 {after-before:g} 毫秒",
             fontsize=11, weight="bold", va="top")

    ax = fig.add_axes([.134, .305, .808, .211])
    ax.axvspan(-deadline, 0, color=WAIT, lw=0)
    ax.axvline(0, lw=.9, color=INK)
    threshold_line(ax, threshold)
    window = np.flatnonzero(ms >= -(after+30))
    for k, color, ls in (("bias_only", BASE, (0, (4, 2))), ("current_5", LATEST, "-")):
        y = data[f"s70320_{k}"]
        series(ax, ms[window], y[window], color=color, ls=ls, lw=1.6, zorder=3)
        alarm = scores[k]
        ax.scatter(-alarm["lead_ms"], y[alarm["first_index"]], s=24, color=color,
                   edgecolor="white", linewidth=.55, zorder=4)
    for arm, position, color in (("current_5", (-209, .645), LATEST), ("bias_only", (-53, .627), BASE)):
        alarm = scores[arm]
        ax.annotate(f"提前 {alarm['lead_ms']:g} ms", (-alarm["lead_ms"], alarm["first_score"]),
                    xytext=position, textcoords="data", fontsize=9, color=color, ha="center",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": .7})
    ax.text(-166, threshold+.021, "报警线", fontsize=8, color=THRESH, va="bottom")
    ax.text(-3, .73, "破裂", color=INK, fontsize=8, ha="right", va="top")
    ax.set(xlim=(-257, 5), ylim=(0, .76), xticks=[-250, -200, -150, -100, -50, 0],
           yticks=[0, .25, .5, .75], ylabel="破裂风险分数",
           xlabel="时间（毫秒；0 是实际破裂时刻，负值表示破裂前）")
    ax.legend(handles=[
        Line2D([0], [0], color=BASE, linestyle=(0, (4, 2)), label="做了前四步修正"),
        Line2D([0], [0], color=LATEST, label="在前四步上再加第 5 步"),
    ], loc="lower left", bbox_to_anchor=(-.005, 1.08), ncol=2, borderaxespad=0, fontsize=9)

    recap = json.loads((HERE / "EAST360_评分核对.json").read_text(encoding="utf-8"))
    static, four, five = (recap["rows"][i] for i in (0, 4, 5))
    fig.text(x0, .216, "放到 360 次历史放电里看", fontsize=11, weight="bold", va="top")
    fig.text(x0, .18,
             f"静态基线及时报 {static['timely']} 炮、误报 {static['false_alarms']} 炮；它没有做炮内修正。\n"
             f"做了前四步修正及时报 {four['timely']} 炮，再加第 5 步后是 {five['timely']} 炮。\n"
             f"两者误报的是同样 {five['false_alarms']} 炮。提前超过 {deadline} 毫秒才算及时。",
             fontsize=10, va="top", linespacing=1.5)
    fig.text(x0, .093,
             "70320 的第一次越线只持续一个采样点。若要求连续两点越线，\n"
             "这次就不能算及时。第二页列出全部 360 次放电的结果。",
             fontsize=9, color=MUTED, va="top", linespacing=1.5)
    return fig


def readout_360(_data, _meta):
    a = json.loads((HERE / "EAST360_评分核对.json").read_text(encoding="utf-8"))
    n, d, s = a["n"], a["n_disruptive"], a["n_safe"]
    fig = plt.figure(figsize=(183/25.4, 215/25.4))
    title(fig, "EAST 360｜准确率怎么算，误报是哪几炮",
          "180 炮破裂、180 炮未破裂。统计单位是整炮；两条准确率都以 360 为分母。")
    ax = fig.add_axes([.062, .636, .883, .24])
    ax.set_axis_off()
    xs = [.015, .34, .53, .72, .906]
    for j, (x, name) in enumerate(zip(xs, ("怎么打分", "及时／180", "误报／180",
                                          "按时才算对", "只看报没报"), strict=True)):
        ax.text(x, .90, name, ha="left" if j == 0 else "center", va="center", fontsize=8.5, color=MUTED)
    ax.axhline(.81, color=GRID, lw=.8)
    for i, name, y in ((0, "静态基线", .68), (4, "前四步修正", .42), (5, "前四步 + 第 5 步", .16)):
        m = a["rows"][i]
        if i == 5:
            ax.add_patch(Rectangle((0, .055), 1, .216, facecolor=LIGHT, lw=0, zorder=0))
        cells = (name, f"{m['timely']} 炮\n{m['timely']/d:.2%}", f"{m['false_alarms']} 炮\n{m['false_alarms']/s:.2%}",
                 f"{m['timely_correct']}/{n}\n{m['timely_correct']/n:.2%}",
                 f"{m['binary_correct']}/{n}\n{m['binary_correct']/n:.2%}")
        for j, (x, cell) in enumerate(zip(xs, cells, strict=True)):
            ax.text(x, y, cell, ha="left" if j == 0 else "center", va="center", fontsize=8.5,
                    color=LATEST if i == 5 else INK, weight="bold" if i == 5 else "normal", linespacing=1.2)
    fig.text(.073, .593, "按时才算对：及时报警的破裂炮 + 没报警的未破裂炮。\n"
                        "只看报没报：晚报也算对。所以第 5 步提前报警，后一项仍一样。",
             fontsize=8.5, color=MUTED, va="top", linespacing=1.48)
    fig.text(.073, .492, "34 炮误报，全部在做完前四步时就已存在", fontsize=11, weight="bold")
    fig.text(.073, .452, "其中 6 炮在前 20 炮准备段；7 炮在第 5 步第一次改动前就报了。\n"
                        "剩下 21 炮在第 5 步生效后才报，但只做前四步也会报。",
             fontsize=9, va="top", linespacing=1.5)
    for rect, sid, head, expected in (
        ([.133, .104, .344, .226], 67900, "67900：第 5 步还没开始，两条线就越线", (0, 4, 5)),
        ([.591, .104, .344, .226], 73743, "73743：不额外等会误报，等 5 点后不报", (4, 5, 7)),
    ):
        q = np.load(HERE / f"_case_{sid}.npz", allow_pickle=False)
        ax = fig.add_axes(rect)
        x = 1000*(q["seconds"]-q["center"])
        threshold_line(ax, float(q["threshold"]))
        for row in expected:
            color, ls = {0: ("#9c94a2", (0, (2, 2))), 4: (BASE, (0, (4, 2))), 5: (LATEST, "-"),
                         7: (NOWAIT, (0, (5, 2, 1.2, 2)))}[row]
            label_text = {0: "静态", 4: "前四步", 5: "再加第 5 步", 7: "不额外等"}[row]
            series(ax, x, q[f"row{row}"], color=color, ls=ls, lw=1.3, label=label_text)
        ax.axvline(0, color=GRID, lw=.6)
        ax.set(xlim=(-20.7, 21.7), xticks=[-20, -10, 0, 10, 20], xlabel="距图中首次误报（毫秒）")
        if sid == 67900:
            ax.set_ylabel("破裂风险分数")
        ax.text(0, 1.135, head, transform=ax.transAxes, va="bottom", fontsize=8, weight="bold")
        ax.legend(loc="upper left", ncol=3, handlelength=1.2, columnspacing=.56, fontsize=7,
                  bbox_to_anchor=(-.03, 1.10))
        q.close()
    fig.text(.073, .027, "水平虚线是报警线。左图蓝线与深灰线重合；每次误报的时间和最高分见逐炮清单。",
             fontsize=8, color=MUTED)
    return fig


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--preview-dir", type=Path, help="write scaled previews and full-size panel crops here")
    arg.add_argument("--guide-only", action="store_true", help="render the guide and accuracy sheet")
    opt = arg.parse_args()
    style()
    meta = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    with np.load(HERE / "figure_data.npz", allow_pickle=False) as data:
        figures = (("70320_mechanism", mechanism), ("70320_alarm_and_wait", effects),
                   ("EAST360_reader_guide", reader_guide), ("EAST360_readout_360", readout_360))
        for name, draw in figures[-2:] if opt.guide_only else figures:
            fig = draw(data, meta)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for artist in fig.findobj(mpl.text.Text):
                if artist.get_text().strip() and artist.get_visible():
                    bounds = artist.get_window_extent(renderer)
                    if not (fig.bbox.contains(*bounds.min) and fig.bbox.contains(*bounds.max)):
                        raise ValueError(f"clipped figure text: {name}: {artist.get_text()}")
            for suffix in ("png", "pdf", "svg"):
                fig.savefig(HERE / f"{name}.{suffix}", facecolor="white")
            if opt.preview_dir:
                from PIL import Image
                opt.preview_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(opt.preview_dir / f"{name}.png", dpi=170, facecolor="white")
                if name in ("EAST360_reader_guide", "EAST360_readout_360"):
                    plt.close(fig)
                    continue
                with Image.open(HERE / f"{name}.png") as img:
                    w, h = img.size
                    regions = ((0, .21), (.21, .445), (.445, .738), (.738, 1)) if name.endswith("mechanism") else (
                        (0, .479), (.479, .805), (.805, 1))
                    for i, (top, bottom) in enumerate(regions, 1):
                        view = img.crop((0, int(top*h), w, int(bottom*h)))
                        view.thumbnail((1520, 880))
                        view.save(opt.preview_dir / f"{name}_region{i}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
