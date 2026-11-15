"""Produce the EAST360 per-shot atlas and an overview in Seaborn."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from plot_figures import (
    BASE, CH, CURVES, ERR, HERE, INK, LATEST, LIGHT, MUTED, NOWAIT, OLDER, WAIT,
    guides, label, mpl, plt, series, sns, style, threshold_line, title,
)


DEST = HERE / "atlas"
ARMS = ("bias_only", "earlier_5", "current_0", "current_5")
KEY_SHOTS = {70320: "新增及时", 73405: "新增及时", 73743: "等待去掉误报"}
DATA = DEST / "data"


def score(ax, x, y, arm, *, points=False):
    color, ls, _ = CURVES[arm]
    series(ax, x, y, color=color, ls=ls, lw=1.4 if arm == "current_5" else 1.,
           marker="o" if points else None, ms=2.5, zorder=4 if arm == "current_5" else 3)


def at(a, j):
    return a["times"][j]


def score_text(row, k):
    a = row["alarms"][k]
    if not a["alarm"]:
        return "未报警"
    if not row["disruptive"]:
        return "误报"
    return f"{a['lead_ms']:g} ms（{'及时' if a['timely'] else '太晚'}）"


def full_curves(ax, delta, write, a, row):
    t = a["times"]
    threshold_line(ax, row["threshold"])
    if row["disruptive"]:
        down = row["down_s"]
        ax.axvspan(down-.030, down, color=WAIT, lw=0, zorder=0)
        ax.axvline(down, color=INK, lw=.75, zorder=2)
    for k in (("bias_only",) if row["rank"] <= 20 else ("bias_only", "current_5")):
        score(ax, t, a[k], k)
        n = row["alarms"][k]["index"]
        if n is not None:
            ax.scatter(t[n], a[k][n], color=CURVES[k][0], s=18,
                       edgecolor="white", linewidth=.5, zorder=5)
    ax.axvline(t[row["focus"]], color="#a4b4c3", lw=.7, ls=(0, (2, 3)))
    guides(ax)
    ax.set(xlim=(t[0], max(t[-1], row["down_s"] or t[-1])+.005),
           ylim=(0, 1.04), yticks=[0, .25, .50, .75, 1], ylabel="破裂风险分数")
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    label(ax, "a", "整炮风险曲线与首次报警", y=1.045)
    d = a["current_5"] - a["bias_only"]
    series(delta, t, d, color=LATEST, lw=.95)
    active = a["dose"][-1] > 0.
    v = max(.007, float(abs(d).max())*1.07)
    delta.axhline(0, lw=.65, color=BASE, zorder=0)
    delta.fill_between(t, 0, d, where=d >= 0, color=LIGHT, lw=0)
    delta.fill_between(t, 0, d, where=d < 0, color="#eadde8", lw=0)
    delta.axvline(t[row["focus"]], color="#a4b4c3", lw=.7, ls=(0, (2, 3)))
    delta.set(xlim=ax.get_xlim(), ylim=(-v, v), yticks=[-v/1.07, 0, v/1.07], ylabel="分数差")
    delta.tick_params(axis="x", bottom=False, labelbottom=False)
    delta.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f" if v > .05 else "%.3f"))
    label(delta, "b", "第 5 行减第 4 行；下方蓝线是实际改动幅度", y=1.12)
    series(write, t, a["dose"][-1], color=LATEST, lw=.6)
    write.fill_between(t, 0, a["dose"][-1], color=LIGHT, lw=0)
    write.axvline(t[row["focus"]], color="#a4b4c3", lw=.7, ls=(0, (2, 3)))
    write.set(xlim=ax.get_xlim(), ylim=(-.08, 2.5), yticks=[0, 2],
              ylabel="幅度", xlabel="原始记录时间（s）")
    write.xaxis.set_major_locator(MaxNLocator(7, prune="both"))
    if not np.any(active):
        delta.text(.98, .82, "冷启动：没有炮内修正" if row["rank"] <= 20 else "本炮没有实际施加的修正",
                   ha="right", va="top", transform=delta.transAxes, fontsize=8, color=MUTED)


def local_curves(ax, a, row):
    t = a["times"]
    n = row["focus"]
    ix = np.arange(max(0, n-25), min(n+22, len(t)))
    ms = 1000 * (t[ix]-t[n])
    threshold_line(ax, row["threshold"])
    for k in ("bias_only", "earlier_5", "current_0", "current_5"):
        if row["rank"] <= 20 and k != "bias_only":
            continue
        score(ax, ms, a[k][ix], k, points=k == "current_5")
    ax.axvline(0, color="#9cabb9", lw=.75)
    vals = np.concatenate([a[k][ix] for k in ARMS])
    low, high = min(vals.min(), row["threshold"]), max(vals.max(), row["threshold"])
    pad = max(.026, .07*(high-low))
    ax.set(ylim=(low-pad, high+pad), xlim=(ms[0]-.5, ms[-1]+.5),
           xlabel=f"距标记点的时间（ms；0 = {t[n]:.4f} s）", ylabel="风险分数")
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="both"))
    name = {"alarm": "首报", "largest_change": "整炮分数差绝对值最大点",
            "parent_peak": "原分数峰值"}[row["focus_reason"]]
    label(ax, "c", f"{name}附近的逐点对照", y=1.095)


def feedback_slot(a, row):
    releases = a["release"]
    delay = 0 if row["shot"] == 73743 else row["wait_samples"]
    dose = a["dose"][0 if delay == 0 else -1, row["focus"]]
    width = row["patch_width"]
    candidate = np.searchsorted(releases + delay, row["focus"], side="right")-1
    if candidate >= 0 and a["eligible"][candidate] and dose > 0:
        # At fraction zero the transition still has only the old coefficients.
        elapsed = row["focus"] - (releases[candidate]+delay)
        if elapsed == 0 and candidate and a["eligible"][candidate-1]:
            return int(candidate-1), delay, "active"
        return int(candidate), delay, "mixed" if elapsed < width else "active"
    available = np.flatnonzero(a["eligible"])
    if not available.size:
        return None, delay, "unused"
    last = available[releases[available] <= row["focus"]]
    return (int(last[-1]), delay, "unused") if last.size else (int(available[0]), delay, "later")


def show_feedback(ax, heat, a, row):
    j, delay, status = feedback_slot(a, row)
    if j is None:
        label(ax, "d", "反馈可用性", y=1.095)
        ax.set_axis_off()
        heat.set_axis_off()
        ax.text(.02, .82, "冷启动炮：本炮只评分，不使用诊断反馈。" if row["rank"] <= 20 else
                "本炮没有符合条件的诊断反馈。", color=MUTED, transform=ax.transAxes)
        return
    x = np.arange(1, 26)
    pred, obs, e = (a[k][j] for k in ("forecast", "observation", "new_error"))
    c = (int(np.flatnonzero(a["channels"] == "dy_a")[0]) if row["shot"] == 70320
         else int(np.argmax(abs(e).mean(0))))
    channel = str(a["channels"][c])
    for k in range(5):
        if k % 2 == 0:
            ax.axvspan(.5+5*k, 5.5+5*k, color="#f3f6f9", lw=0, zorder=0)
    series(ax, x, pred[:, c], color=BASE, ls="--", lw=1.25)
    series(ax, x, obs[:, c], color=LATEST, marker="o", ms=2, lw=1.35)
    ax.legend([Line2D([0], [0], color=BASE, ls="--"), Line2D([0], [0], color=LATEST, marker="o", ms=2)],
              ["预测", "实测"], loc="upper right", ncol=2, handlelength=1.4, columnspacing=.8)
    ax.set(xlim=(.5, 25.5), xticks=[5, 10, 15, 20, 25], ylabel="模型坐标（无量纲）",
           xlabel="本次预测的采样点")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    label(ax, "d", f"反馈波形示例：{CH[channel]}", y=1.095)
    group_title = {"active": "当时用于修正", "mixed": "新旧交接（此处画新反馈）",
                   "unused": "此前最近反馈（该点未用）", "later": "后来才到的反馈（该点未用）"}[status]
    if delay == 0 and status in ("active", "mixed"):
        group_title = "第 7 行当时使用" if status == "active" else "第 7 行新旧交接（画新反馈）"
    ax.text(0, 1.017, f"{group_title} · 发出 {at(a, a['issue'][j]):.4f} s，"
            f"可用 {at(a, a['release'][j]):.4f} s", transform=ax.transAxes, fontsize=7, color=MUTED)
    norm = TwoSlopeNorm(0, -2, 2)
    sns.heatmap(e.T, ax=heat, cmap=ERR, norm=norm, cbar=False, annot=True, fmt="+.2f",
                annot_kws={"fontsize": 6.2}, yticklabels=[CH[k] for k in a["channels"]],
                xticklabels=["1–5", "6–10", "11–15", "16–20", "21–25"])
    heat.tick_params(axis="y", length=0, pad=4, labelrotation=0)
    heat.tick_params(axis="x", labelrotation=0)
    label(heat, "e", "扣除历史偏差后的新误差", y=1.078)
    heat.set_xlabel("25 点，每 5 点一段；色标 −2 至 +2")
    heat.add_patch(plt.Rectangle((.01, c+.04), 4.98, .92, fill=False, lw=.75, ec=INK))


def status_panel(ax, a, row):
    ax.set_axis_off()
    label(ax, "f", "这炮的结果与修正用量", y=1.09)
    col = ("第 4 行｜原方法", "第 5 行｜最新，等 5 点",
           "第 6 行｜较早，等 5 点", "第 7 行｜最新，不等")
    for i, (k, name) in enumerate(zip(("bias_only", "current_5", "earlier_5", "current_0"), col, strict=True)):
        y = .89-i*.145
        if k == "current_5":
            ax.add_patch(plt.Rectangle((0, y-.062), 1, .127, transform=ax.transAxes, facecolor=LIGHT, lw=0))
        ax.text(.01, y, name, fontsize=8, color=CURVES[k][0], va="center", transform=ax.transAxes)
        ax.text(.99, y, score_text(row, k), fontsize=8, ha="right", va="center", transform=ax.transAxes)
    n = row["focus"]
    ax.text(.01, .255, f"标记点：第 4 行 {a['bias_only'][n]:.4f}，第 5 行 {a['current_5'][n]:.4f}",
            fontsize=8, transform=ax.transAxes)
    ax.text(.01, .132, f"第 5 行的第一层改动长度：{a['dose'][-1, n]:.4f}",
            fontsize=8, transform=ax.transAxes)
    ax.text(.01, .013, f"全炮：{row['eligible_windows']} 笔合格反馈，{row['changed_points']} 个采样点实际改动",
            fontsize=8, transform=ax.transAxes, color=MUTED)


def shot_page(row):
    with np.load(DATA / f"{row['file']}.npz", allow_pickle=False) as pack:
        a = dict(pack)
    fig = plt.figure(figsize=(183/25.4, 229/25.4))
    klass = "破裂" if row["disruptive"] else "未破裂"
    ending = f" · {row['down_s']:.4f} s 破裂" if row["disruptive"] else ""
    cold = " · 冷启动" if row["rank"] <= 20 else ""
    note = " · " + KEY_SHOTS[row["shot"]] if row["shot"] in KEY_SHOTS else ""
    if row["disruptive"]:
        ending += "（灰区：破裂前最后 30 ms）"
    title(fig, f"EAST {row['shot']}｜第 {row['rank']:03d} 炮 · {klass}{cold}{note}",
          f"{row['n']} 个采样点{ending}　｜　报警阈值 {row['threshold']:.7f}")
    handles = [Line2D([0], [0], c=CURVES[k][0], ls=CURVES[k][1], lw=1.3, label=CURVES[k][2])
               for k in ("bias_only", "current_5", "earlier_5", "current_0")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .918), ncol=2,
               borderaxespad=0, handlelength=1.6, columnspacing=1.4)
    full_curves(fig.add_axes([.13, .695, .819, .156]), fig.add_axes([.13, .572, .819, .077]),
                fig.add_axes([.13, .526, .819, .032]), a, row)
    local_curves(fig.add_axes([.13, .285, .35, .154]), a, row)
    show_feedback(fig.add_axes([.596, .285, .353, .154]),
                  fig.add_axes([.195, .065, .269, .139]), a, row)
    status_panel(fig.add_axes([.547, .063, .402, .147]), a, row)
    return fig


def flips_overview(rows):
    """Every discharge contributes one row; outcomes are not filtered by improvement."""
    fig = plt.figure(figsize=(183/25.4, 203/25.4))
    n_changed = sum(r["max_abs_score_change"] > 0 for r in rows)
    title(fig, f"EAST 360｜{n_changed} 炮分数变化，两炮由晚报转为及时",
          "180 炮破裂、180 炮未破裂；同一阈值比较第 5 行和第 4 行，误报名单相同。")
    names = ("第 4 行\n原方法", "第 6 行\n较早，等 5 点", "第 7 行\n最新，不等", "第 5 行\n最新，等 5 点")
    arms = ("bias_only", "earlier_5", "current_0", "current_5")
    chosen = {k: [r for r in rows if bool(r["disruptive"]) == d] for k, d in (("D", True), ("S", False))}
    for ax, key, letter in ((fig.add_axes([.14, .655, .80, .21]), "D", "a"),
                            (fig.add_axes([.14, .344, .80, .21]), "S", "b")):
        ss = chosen[key]
        mat = np.zeros((4, len(rows)), np.float64)
        mat.fill(np.nan)
        for row in ss:
            i = row["rank"]-1
            for j, k in enumerate(arms):
                a = row["alarms"][k]
                mat[j, i] = (2 if a["timely"] else 1 if a["alarm"] else 0) if key == "D" else int(a["alarm"])
        palette = ["#e5e9ee", "#8d99a8", "#186a9f"] if key == "D" else ["#e5e9ee", "#b75615"]
        sns.heatmap(mat, cmap=mpl.colors.ListedColormap(palette), vmin=0, vmax=len(palette)-1,
                    ax=ax, cbar=False, xticklabels=False, yticklabels=names, linewidths=0)
        ax.tick_params(axis="y", labelrotation=0, length=0, pad=5)
        ax.axvline(20, lw=.75, color=MUTED, ls="--")
        for sid, color in ((70320, LATEST), (73405, LATEST), (73743, NOWAIT)):
            selected = next(r for r in rows if r["shot"] == sid)
            if selected not in ss:
                continue
            x = selected["rank"]-.5
            ax.axvline(x, lw=.9, color=color)
            ax.text(x, -.18, str(sid), ha="center", va="bottom", fontsize=8, color=color,
                    clip_on=False)
        label(ax, letter, "破裂炮：蓝 = 及时，深灰 = 晚报，浅灰 = 未报" if key == "D" else
              "未破裂炮：橙 = 误报，浅灰 = 未报", y=1.155, x=-.075)
        ax.set_xticks(np.array([20, 60, 120, 180, 240, 300, 360])-.5,
                      [20, 60, 120, 180, 240, 300, 360])
        if key == "S":
            ax.set_xlabel("回放顺序（空白位置是另一类炮；虚线前为冷启动）")
    ax = fig.add_axes([.14, .078, .80, .135])
    # Show the actual temporal maximum of the paired score change for all shots,
    # including zeros; this statistic does not use the outcome to choose a time.
    for d, marker, color, name in ((True, "o", LATEST, "破裂"), (False, "x", "#8a687e", "未破裂")):
        ss = [r for r in rows if r["disruptive"] == d]
        sns.scatterplot(x=[r["rank"] for r in ss], y=[r["max_abs_score_change"] for r in ss], ax=ax,
                        color=color, marker=marker, s=10, linewidth=.55, alpha=.69, label=name, legend="auto")
    ax.axvline(20.5, lw=.75, color=MUTED, ls="--")
    ax.set(xlim=(1, 361), ylim=(-.017, None), ylabel="最大分数差的绝对值",
           xlabel="回放顺序", xticks=[20, 60, 120, 180, 240, 300, 360])
    ax.yaxis.set_major_locator(MaxNLocator(5))
    guides(ax)
    label(ax, "c", "每炮第 5 行相对第 4 行，整炮最大分数变化", y=1.13, x=-.075)
    ax.legend(loc="upper right", ncol=2)
    return fig


def check_bounds(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden_ticks = {label for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)
                    for tick in axis.get_major_ticks() + axis.get_minor_ticks()
                    if not min(axis.get_view_interval()) <= tick.get_loc() <= max(axis.get_view_interval())
                    for label in (tick.label1, tick.label2)}
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text().strip() and text not in hidden_ticks:
            r = text.get_window_extent(renderer)
            if not fig.bbox.contains(*r.min) or not fig.bbox.contains(*r.max):
                raise ValueError(f"figure text exceeds page: {text.get_text()}")


def volume(task):
    j, rows = task
    style()
    with PdfPages(DEST / f"EAST360_atlas_{j:02d}.pdf", metadata={
        "Title": f"EAST360 intra-shot trace atlas: {rows[0]['rank']}–{rows[-1]['rank']}",
        "Subject": "Same-parent sealed risk scores, causal forecast feedback, and replay outcomes",
    }) as pdf:
        for row in rows:
            fig = shot_page(row)
            check_bounds(fig)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    print(f"saved atlas volume {j:02d}: {rows[0]['rank']}–{rows[-1]['rank']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--preview-dir", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    opt = ap.parse_args()
    rows = json.loads((DEST / "index.json").read_text(encoding="utf-8"))
    trace = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))["trace"]
    for row in rows:
        row.update(wait_samples=trace["wait_samples"], patch_width=trace["patch_width"])
    style()
    fig = flips_overview(rows)
    check_bounds(fig)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"EAST360_overview.{suffix}", facecolor="white")
    if opt.preview_dir:
        opt.preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(opt.preview_dir / "EAST360_overview.png", dpi=170, facecolor="white")
        for sid in (rows[0]["shot"], 70320, 73405, 73743, rows[-1]["shot"]):
            r = next(r for r in rows if r["shot"] == sid)
            one = shot_page(r)
            check_bounds(one)
            one.savefig(opt.preview_dir / f"atlas_{sid}.png", dpi=170, facecolor="white")
            if sid in KEY_SHOTS:
                one.savefig(DEST / f"shot_{sid}.pdf", facecolor="white")
            plt.close(one)
    plt.close(fig)
    if not opt.preview_only:
        batches = [(k//60+1, rows[k:k+60]) for k in range(0, len(rows), 60)]
        with ProcessPoolExecutor(max_workers=opt.workers) as ex:
            list(ex.map(volume, batches))


if __name__ == "__main__":
    main()
