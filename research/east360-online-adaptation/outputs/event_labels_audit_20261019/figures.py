"""Descriptive figures from the completed audit; no model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.unicode_minus':False})
s = json.loads((HERE/'summary.json').read_text())
e = pd.read_csv(HERE/'events_joined.csv')
fam = pd.read_csv(HERE/'family_summary.csv')
blue, orange, gray = '#0072B2', '#D55E00', '#777777'

fig, ax = plt.subplots(1, 3, figsize=(15,4.7), constrained_layout=True)
x = np.arange(6)
ax[0].bar(x-.18, fam.disruptive_shots, .36, label='破裂炮', color=orange)
ax[0].bar(x+.18, fam.safe_shots, .36, label='安全炮', color=blue)
for i,r in fam.iterrows():
    ax[0].text(i-.18,r.disruptive_shots+3,str(r.disruptive_shots),ha='center',fontsize=9)
    ax[0].text(i+.18,r.safe_shots+3,str(r.safe_shots),ha='center',fontsize=9)
ax[0].set(xticks=x,xticklabels=fam.family, ylabel='出现该类的炮数',ylim=(0,265),title='a  六类都出现在破裂炮和安全炮中')
ax[0].legend(frameon=False,loc='upper left')
first=e[e.hdf5_found & e.disruption_label.eq(1)].sort_values('event_index').drop_duplicates('shot_no')
lead=np.sort(first.start_lead_ms)
ax[1].plot(lead,np.arange(1,len(lead)+1)/len(lead),color=blue,lw=2)
ax[1].axvline(30,color=gray,linestyle='--',linewidth=1)
ax[1].axvline(float(np.median(lead)),color=orange,linestyle=':',linewidth=1.5)
ax[1].text(185,.40,f'中位数 {np.median(lead):.1f} ms\n{len(lead)} 炮',color=orange)
ax[1].set(xlabel='首个离线事件起点距破裂 / ms',ylabel='累计炮数比例',xlim=(0,520),ylim=(0,1.02),
          title='b  事件起点通常早于临近破裂阶段')
overlap=pd.DataFrame(s['split_overlap']).pivot(index='canonical_role',columns='cohort',values='shots').loc[['train','val','test','outside_canonical'],['training_pool','sanity']]
ax[2].imshow(overlap.to_numpy(),cmap='Blues',aspect='auto',vmin=0,vmax=1400)
for i in range(4):
    for j in range(2):
        ax[2].text(j,i,int(overlap.iloc[i,j]),ha='center',va='center',color='white' if overlap.iloc[i,j]>700 else 'black',fontsize=13)
ax[2].set(xticks=[0,1],xticklabels=['标注 training_pool','标注 sanity'],yticks=range(4),
          yticklabels=['现有 TRAIN1353','现有 VAL160','现有 TEST220','额外 83 炮'],title='c  两个项目的数据划分不一致')
fig.suptitle('事件标注的数据价值与使用边界',fontsize=15,fontweight='bold')
fig.savefig(HERE/'event_labels_overview.pdf')
fig.savefig(HERE/'event_labels_overview.svg')
plt.close(fig)

fig, axes=plt.subplots(3,1,figsize=(11,6.6),constrained_layout=True)
pred=pd.read_csv(HERE/'source_test220_errors_joined.csv').set_index('shot_no')
for ax,sid in zip(axes,[1052127,1057443,1053313]):
    g=e[e.shot_no.eq(sid)].sort_values('event_index')
    r=g.iloc[0]; down=r.down_s
    left=min(r.event_start_s,down-r.horizon_ms/1000)-.015
    right=down+.018
    ax.axvspan(down-r.horizon_ms/1000,down,color=orange,alpha=.14,label='当前破裂正样本窗口')
    for t in g.itertuples():
        ax.broken_barh([(t.event_start_s,t.event_end_s-t.event_start_s)],(.3,.25),facecolors=blue)
        ax.text((t.event_start_s+t.event_end_s)/2,.425,t.event_class,ha='center',va='center',color='white',fontweight='bold')
    ax.axvline(down,color='black',linewidth=1.1)
    pr=pred.loc[sid]
    if pr.alarm:
        at=down-pr.lead_ms/1000
        ax.axvline(at,color=orange,ls='--',lw=1.5)
        result=f'原基模晚报：提前 {pr.lead_ms:.1f} ms'
    else: result='原基模未报警'
    ax.text(left,.83,f'{sid}：{result}；首事件离线起点提前 {r.start_lead_ms:.1f} ms',fontsize=11)
    ax.set(xlim=(left,right),ylim=(0,1.1),yticks=[],xlabel='原始诊断时钟 / s')
    ax.spines['left'].set_visible(False)
fig.suptitle('已有错误炮的事件时间证据',fontsize=15,fontweight='bold')
fig.legend(handles=[Patch(facecolor=blue,label='离线候选事件'),
                    Patch(facecolor=orange,alpha=.14,label='当前破裂正样本窗口'),
                    Line2D([0],[0],color=orange,ls='--',label='原模型报警'),
                    Line2D([0],[0],color='black',label='破裂时刻')],
           loc='outside lower center',ncol=4,frameon=False)
fig.savefig(HERE/'source_error_event_timelines.pdf')
fig.savefig(HERE/'source_error_event_timelines.svg')
plt.close(fig)

profile_path=HERE/'family_signal_profile_medians.csv'
if profile_path.exists():
    p=pd.read_csv(profile_path)
    names=['dx_a','dy_a','Mir_avg_amp','CIII','sxr_mean','sxr_var','xuv_var','Z_proxy']
    matrix=p.pivot(index='event_class',columns='feature',values='relative_to_whole_shot').reindex(index=list('ABCDEF'),columns=names)
    fig,ax=plt.subplots(figsize=(13,4.4),constrained_layout=True)
    finite=matrix.to_numpy()[np.isfinite(matrix.to_numpy())]
    limit=max(1.,float(np.max(np.abs(finite))))
    im=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    for i in range(6):
        for j in range(len(names)):
            value=matrix.iloc[i,j]
            if np.isfinite(value):ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(value)>.65*limit else 'black')
    ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(6),yticklabels=list('ABCDEF'),
           title='现有特征中的事件差异：以炮为单位汇总')
    plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
    fig.colorbar(im,ax=ax,label='事件中位值相对同炮中位值的差 / 同炮 IQR')
    fig.savefig(HERE/'existing_feature_profiles.pdf')
    fig.savefig(HERE/'existing_feature_profiles.svg')
    plt.close(fig)
print('Figures saved')
