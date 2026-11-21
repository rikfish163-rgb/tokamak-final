"""Plot a retained disruption and a retained false alarm on the same axes."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from . import commissioning_fourier_confirmation_20261114 as run, simple_data
from .history_gradient_metric_20261102.run import read, sha


def plot(out):
    p, s, cases = (read(out/f'{k}.json') for k in ('protocol', 'seal', 'fourier_11_cases'))
    cases = {r['shot']:r for r in cases['rows']}
    names = ('original','fourier','confirmed')
    palette = ('#454d55','#006ca8','#bd5613')
    styles = ('-', (0,(5,2)), '-')
    tau = p['calibration'][run.prior.MAIN]
    figdata = []
    for shot in (92394,74939):
        rank = p['order'].index(shot)+1
        file = out/'shots'/f'{rank:04d}_{shot}.npz'
        if sha(file) != s['curve_sha256'][file.name]:
            raise ValueError('plot requires the sealed replay')
        raw = simple_data.read_shot('EAST', shot, rate_contract='nominal')
        q = int(np.searchsorted(raw['times'], cases[shot]['fourier_first_s']))
        indices = np.arange(q-127, q+41)
        mir = np.abs(raw['values'][indices, run.COLS[2]].astype(float))
        if not raw['available'][indices, run.COLS[2]].all():
            raise ValueError('plot windows require observed sensor readings')
        scale = mir[:128].mean()
        with np.load(file, allow_pickle=False) as f:
            np.testing.assert_array_equal(raw['times'], f['times'])
            figdata.append((shot, (f['times'][indices]-f['times'][q])*1000.,
                            [f[k][indices] for k in names], mir/scale, mir, scale))
    font = 'Noto Sans CJK SC'
    try:
        mpl.font_manager.findfont(font, fallback_to_default=False)
    except ValueError:
        font = 'Noto Sans CJK JP'
    sns.set_theme(style='ticks', context='paper', font=font, palette=palette)
    mpl.rcParams.update({'font.size':8, 'axes.labelsize':8, 'axes.titlesize':9,
        'axes.spines.right':False, 'axes.spines.top':False, 'axes.linewidth':.65,
        'legend.frameon':False, 'legend.fontsize':8, 'xtick.labelsize':7, 'ytick.labelsize':7,
        'svg.fonttype':'none', 'pdf.fonttype':42})
    fig = plt.figure(figsize=(183/25.4,88/25.4), layout='constrained')
    gs = fig.add_gridspec(3,2,height_ratios=(.3,1.6,1),wspace=.06)
    key = fig.add_subplot(gs[0,:])
    key.set_axis_off()
    key.legend(handles=[*[Line2D([],[],color=c,ls=ls,lw=1.5,label=title)
                           for c,ls,title in zip(palette,styles,('原方法','傅里叶直接加分','三路确认后加分'),strict=True)],
                        Line2D([],[],color='#606b76',ls=':',lw=.85,label='报警线')], ncol=4, loc='center')
    ymax = max(y.max() for _,_,_,y,_,_ in figdata)
    with (out/'figure_curves.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('shot','ms_from_fourier_first',*names,'threshold','mir_vv_over_window_mean',
                         'mir_vv_magnitude','mir_vv_128_mean'))
        for j, (sid, x, curves, m, raw, avg) in enumerate(figdata):
            a = fig.add_subplot(gs[1,j])
            b = fig.add_subplot(gs[2,j],sharex=a)
            for y,c,ls in zip(curves,palette,styles,strict=True):
                a.plot(x,y,color=c,ls=ls,lw=1.25)
            a.axhline(tau,color='#606b76',ls=':',lw=.85)
            a.scatter(0,curves[-1][127],s=14,facecolor='white',edgecolor=palette[-1],linewidth=.7,zorder=5)
            a.set_title(('a' if j==0 else 'b')+f'   {sid}   '+('破裂炮' if j==0 else '安全炮'),loc='left')
            a.set(ylim=(0.,.64),yticks=(0.,.2,.4,.6))
            a.tick_params(labelbottom=False)
            b.fill_between(x,0,ymax*1.05,where=x<=0,color='#eef2f5',linewidth=0)
            b.plot(x,m,color='#526b78',lw=.95)
            b.set(xlim=(-127,40),ylim=(0,ymax*1.05),xticks=(-120,-80,-40,0,40),xlabel='距傅里叶首报（ms）')
            for ax in (a,b):
                ax.axvline(0,color='#80909b',alpha=.8,lw=.6,zorder=0)
            if j==0:
                a.set_ylabel('报警分数')
                b.set_ylabel('磁探针读数（倍）')
            for k,t in enumerate(x):
                writer.writerow([sid,t,*(v[k] for v in curves),tau,m[k],raw[k],avg])
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'fourier_confirmation_cases.{ext}',dpi=360,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    plot(parser.parse_args().out.resolve())
