"""Plot four EAST 360 decisions from sealed replay; export the numbers used."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import seaborn as sns
from scipy.special import logit

from . import commissioning_time_fourier_20261113 as probe, simple_data
from .history_gradient_metric_20261102.run import read, sha, write


CASES = (
    (79649, 'fourier', '环电压', 'v_loop', '傅里叶组提前', .20),
    (71076, 'fourier', '环电压', 'v_loop', '傅里叶组新增误报', .08),
    (90874, 'time', '软 X 射线偏度', 'sxr_skew', '时域组新增及时检出', .046),
    (73405, 'original', '软 X 射线偏度', 'sxr_skew', '时域组晚 3 ms', .035),
)
LABELS = ('原方法', '加时域', '加傅里叶')
INK, MUTED = '#172633', '#5a6671'
COLORS = ('#404954', '#bc540e', '#0066a1')
STYLES = ('-', (0, (5, 2)), '-')


def plot(folder):
    proto, fit, per, seal, result = (read(folder / f'{k}.json') for k in
                                    ('protocol', 'fit', 'per_shot', 'seal', 'result'))
    mapping = probe.lag.load_inputs()['mapping']
    main = per[probe.MAIN]
    comparisons = result['comparisons'][probe.MAIN]
    changed = {s for a in comparisons.values() for v in a['flips'].values() for s in v}
    pcase = {row[0] for row in CASES}
    scale = np.asarray(proto['channel_source_scale'])
    rows = {}
    exported = []
    for rank, sid in enumerate(proto['order'], 1):
        if sid not in changed | pcase | {67900, 70320}:
            continue
        scores = [main[arm][rank-1] for arm in probe.ARMS]
        f = folder / 'shots' / f'{rank:04d}_{sid}.npz'
        if seal['curve_sha256'][f.name] != sha(f):
            raise ValueError('unsealed plot curve')
        with np.load(f, allow_pickle=False) as a:
            z = dict(a)
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        sig = probe.lag.prepare_actual(raw, rank, mapping)
        np.testing.assert_array_equal(raw['times'], z['times'])
        # Export the first actual crossing of each arm and the correction from
        # every contributing diagnostic at that same sample.
        idx = sorted({r['first_idx'] for r in scores if r['first_idx'] is not None})
        x, observed = probe.features(sig, scale, np.asarray(idx, int))
        new, parts = probe.apply_extra(z['original'][idx], x, observed, fit)
        np.testing.assert_allclose(new, np.stack([z[k][idx] for k in probe.ARMS[1:]]), rtol=0, atol=1e-7)
        for j, i in enumerate(idx):
            contributions = {k: {c: dict(zip(probe.DESCRIPTORS[b], parts[b, j, channel].tolist(), strict=True))
                                for channel, c in enumerate(probe.CHANNELS)} for b, k in enumerate(probe.ARMS[1:])}
            tau = scores[0]['threshold']
            exported.append(dict(shot=sid, index=i, time_s=float(z['times'][i]),
                                 first=[arm for arm, row in zip(probe.ARMS, scores, strict=True)
                                        if i == row['first_idx']], disrupting=scores[0]['disruptive'],
                                 scores={k: float(z[k][i]) for k in probe.ARMS}, threshold=tau,
                                 logit_needed=float(logit(tau)-logit(float(z['original'][i]))),
                                 shared_step5_norm=float(z['dose'][i]),
                                 logit_delta=dict(zip(probe.ARMS[1:], parts[:, j].sum((1, 2)).tolist(), strict=True)),
                                 logit_contributions=contributions))
        if sid in pcase:
            rows[sid] = (z, scores, sig['values'], sig['available'])
    write(folder / 'decision_points.json', exported)

    font = 'Noto Sans CJK SC'
    try:
        mpl.font_manager.findfont(font, fallback_to_default=False)
    except ValueError:
        font = 'Noto Sans CJK JP'
    sns.set_theme(context='paper', style='ticks', font=font, palette=COLORS)
    mpl.rcParams.update({'text.color': INK, 'axes.edgecolor': '#72808c', 'axes.labelcolor': INK,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
                         'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'xtick.direction': 'out',
                         'ytick.direction': 'out', 'legend.frameon': False, 'legend.fontsize': 8,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.dpi': 400})
    fig = plt.figure(figsize=(185/25.4, 156/25.4), layout='constrained')
    outer = fig.add_gridspec(3, 2, height_ratios=(.19, 1, 1), hspace=.11, wspace=.10)
    key = fig.add_subplot(outer[0, :])
    key.set_axis_off()
    key.legend(handles=[*[Line2D([], [], color=c, ls=s, lw=1.5, label=k)
                          for c, s, k in zip(COLORS, STYLES, LABELS, strict=True)],
                        Line2D([], [], color=MUTED, ls=':', lw=1., label='报警阈值'),
                        Line2D([], [], color=MUTED, ls='--', lw=.75, label='破裂时刻')],
               ncol=5, loc='center', columnspacing=1.65)
    with (folder / 'figure_curves.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(('shot', 'time_s', *probe.ARMS, 'threshold', 'true_disruption_s', 'in_128_point_window',
                         'plotted_channel', 'observed_channel_value', 'shared_step5_norm',
                         'time_logit_delta', 'fourier_logit_delta'))
        for j, (sid, anchor, name, col, title, right) in enumerate(CASES):
            z, rows_for_case, values, valid = rows[sid]
            stamp = z['times']
            channel = simple_data.io23.RAW23.index(col)
            start_row = rows_for_case[probe.ARMS.index(anchor)]
            crossing = start_row['first_idx']
            tau, down = start_row['threshold'], start_row['down_s']
            origin = stamp[crossing]
            left_edge, right_edge = origin-.142, origin+right
            coord = (stamp >= left_edge) & (stamp <= right_edge)
            ii = np.flatnonzero(coord)
            wing = outer[1+j//2, j%2].subgridspec(2, 1, height_ratios=(1.85, 1.), hspace=.07)
            risk = fig.add_subplot(wing[0])
            obs = fig.add_subplot(wing[1], sharex=risk)
            for arm, color, style, detail in zip(probe.ARMS, COLORS, STYLES, rows_for_case, strict=True):
                risk.plot(stamp[ii], z[arm][ii], color=color, ls=style, lw=1.15)
                hit = detail['first_idx']
                if hit is not None and left_edge <= stamp[hit] <= right_edge:
                    risk.plot(stamp[hit], z[arm][hit], marker='o', markeredgecolor=color,
                              markerfacecolor='white', markeredgewidth=.8, ms=3.6, zorder=5)
            risk.axhline(tau, color=MUTED, ls=':', lw=.8)
            if down is not None and left_edge <= down <= right_edge:
                for ax in (risk, obs):
                    ax.axvline(down, color=MUTED, ls='--', lw=.85)
            risk.set(title=f'{chr(97+j)}   {sid}  ·  {title}', ylabel='报警分数')
            risk.title.set_ha('left')
            risk.title.set_x(0.)
            risk.set_ylim(0, 1.02 if max(z[k][ii].max() for k in probe.ARMS) > .70 else .69)
            risk.yaxis.set_major_locator(MultipleLocator(.2))
            risk.tick_params(labelbottom=False)
            painting = np.where(valid[ii, channel], values[ii, channel], np.nan)
            obs.axvspan(stamp[crossing-probe.W+1], origin, color='#e6edf3', lw=0., zorder=-1)
            obs.plot(stamp[ii], painting, color='#567483', lw=.9)
            obs.set(xlim=(left_edge, right_edge), ylabel=name, xlabel='炮内时间（秒）')
            obs.xaxis.set_major_locator(MaxNLocator(5))
            obs.yaxis.set_major_locator(MaxNLocator(3))
            for i in ii:
                writer.writerow([sid, stamp[i], *(z[k][i] for k in probe.ARMS), tau, down,
                                 crossing-probe.W+1 <= i <= crossing, col,
                                 values[i, channel] if valid[i, channel] else None, z['dose'][i],
                                 *z['delta_logit'][:, i]])
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(folder / f'EAST360_四炮首报对照.{ext}', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    plot(parser.parse_args().out.resolve())
