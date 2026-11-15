"""Two first-alarm timelines for the EAST 360 technical report."""

import csv
from functools import partial
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRACE = ROOT / 'outputs/east360_row5_70320'
REPLAY = ROOT / 'probe_outputs/commissioning_write_delay_east360_20261112_v2'
CLOCK = ROOT / 'probe_outputs/stratified_cold20_context360_v1/order.json'
INK, MUTED, BLUE, RULE = '#192935', '#616c77', '#006ba5', '#7d8994'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def curves():
    with (TRACE / 'score_traces.csv').open(newline='') as f:
        x = [r for r in csv.DictReader(f) if r['shot'] == '70320']
    fields = ('time_s', 'row4_no_new_update', 'row5_latest_error_wait5')
    t, parent, active = (np.array([float(r[k]) for r in x]) for k in fields)
    summary = read(TRACE / 'summary.json')
    t_disrupt = summary['cases']['70320']['down_s']
    p = (t >= 3.615) & (t <= t_disrupt)
    a = (t[p], parent[p], active[p])

    audit = read(TRACE / 'EAST360_评分核对.json')
    record = next(r for r in audit['false_alarm_cases'] if r['shot'] == 67900)
    prefix = f'{record["rank"]:04d}_67900'
    row = read(CLOCK)[record['rank']-1]
    assert row['shot'] == 67900 and record['threshold'] == summary['threshold']
    with np.load(REPLAY / 'shots' / f'{prefix}.npz', allow_pickle=False) as f:
        t = row['signal_start_time'] + np.arange(len(f['current_5'])) / row['actual_rate_hz']
        p = t <= 3.36
        b = (t[p], f['bias_only'][p], f['current_5'][p])
    return a, b, t_disrupt, record


def main():
    a, b, end, record = curves()
    alarm = record['threshold']
    font = 'Noto Sans CJK JP'
    sns.set_theme(context='paper', style='ticks', font=font)
    mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'text.color': INK,
        'axes.labelcolor': INK, 'axes.edgecolor': RULE, 'axes.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'legend.fontsize': 8,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 75/25.4), sharey=True)
    fig.subplots_adjust(left=.095, right=.977, bottom=.19, top=.76, wspace=.15)
    fig.legend(handles=[Line2D([],[],color=MUTED,ls=(0,(3,2)),lw=1.25,label='前四步'),
        Line2D([],[],color=BLUE,lw=1.35,label='再加第 5 步'),
        Line2D([],[],color=RULE,ls=':',lw=1.,label='报警线')],
        ncol=3, loc='upper center', bbox_to_anchor=(.53,.99), columnspacing=2.)
    panels = ((70320, a, '破裂炮：第 5 步让首报提前', 3.910, 3.697, end),
              (67900, b, '安全炮：误报早于第 5 步',
               record['row4_first_s'], record['row5_first_s'], record['first_step5_write_s']))
    with (HERE / 'two_shot_curves.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(('shot','time_s','before_step5','with_step5','threshold','milestone_s'))
        for ax, (shot, (t,y0,y1), title, first0, first1, line) in zip(axes, panels, strict=True):
            w.writerows((shot,tv,p0,p1,alarm,line) for tv,p0,p1 in zip(t,y0,y1,strict=True))
            ax.plot(t,y1,color=BLUE,lw=1.25,zorder=2)
            ax.plot(t,y0,color=MUTED,lw=1.05,ls=(0,(3,2)),zorder=3)
            ax.axhline(alarm,color=RULE,ls=':',lw=.8)
            ax.axvline(line,color=RULE,lw=.7,alpha=.75)
            ax.set(xlim=(t[0],t[-1]+.003), ylim=(0,.73), yticks=(0,.2,.4,.6), xlabel='记录时间（秒）')
            ax.tick_params(direction='out',length=3,width=.65)
            ax.set_title(f'{"a" if shot == 70320 else "b"}   {shot}  {title}', loc='left',pad=7.)
            annotate = partial(ax.annotate, xycoords='data', textcoords='offset points', fontsize=7.4)
            for q,c,y in ((first0,MUTED,y0),(first1,BLUE,y1)):
                i = int(np.abs(t-q).argmin())
                assert abs(t[i]-q) < 1e-8 and y[i] >= alarm and not (y[:i] >= alarm).any()
                ax.scatter(t[i],y[i],s=14,zorder=4,facecolors='white',edgecolors=c,linewidths=.95)
            if shot == 70320:
                annotate('提前 225 ms', (first1,.5077), xytext=(0,11), ha='center',color=BLUE)
                annotate('提前 12 ms', (first0,.5226), xytext=(-9,11), ha='right',color=MUTED)
                ax.set_xticks((3.65,3.75,3.85,3.922))
                ax.set_xticklabels(('3.65','3.75','3.85','3.922'))
                ax.text(end-.003,.711,'破裂',ha='right',va='top',fontsize=7.4,color=MUTED)
            else:
                annotate('3.176 s 误报', (first1,record['row5_score_at_first']), xytext=(0,12), ha='center',color=INK)
                ax.set_xticks((3.00,3.10,3.20,3.30))
                ax.text(line-.005,.711,'第 5 步开始生效\n3.287 s',ha='right',va='top',fontsize=7.4,color=MUTED)
    axes[0].set_ylabel('报警分数')
    for ext in ('png','pdf','svg'):
        fig.savefig(HERE / f'two_shot_timelines.{ext}',dpi=400,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
