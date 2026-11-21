"""Summarize forecast skill and signed warning response on identical legal windows."""

from pathlib import Path

import numpy as np

from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, write


OUT = Path(__file__).resolve().parent


def same_windows(rows):
    values = []
    for shot in rows:
        use = np.asarray(shot['nonzero_dose'],dtype=bool)
        if not use.any():
            continue
        skill = np.asarray(shot['feedback_skill'])[use]
        signed = np.asarray(shot['signed_delta'])[use]
        gap = np.asarray(shot['gap'])[use]
        values.append(dict(id=shot['shot'], skill=skill, signed=signed,gap=gap))
    if not values:
        return dict(shots=0, windows=0)
    combined = {k:np.concatenate([v[k] for v in values]) for k in ('skill','signed','gap')}
    by_shot = {v['id']:dict(windows=len(v['signed']), forecast_skill=v['skill'].mean(0).tolist(),
               signed_mean=v['signed'].mean(0).tolist(),best_gap=float(v['gap'].min()),
               largest_signed_change=v['signed'].max(0).tolist()) for v in values}
    s,d = combined['skill'],combined['signed']
    quadrants = {f'{name}_beat_hold':dict(right=int(np.sum((s[:,1]>0.) & (d[:,i]>0.))),
        wrong_or_tied=int(np.sum((s[:,1]>0.) & (d[:,i]<=0.))))
        for i,name in enumerate(('current','earlier'))}
    return dict(shots=len(values),windows=len(d),by_shot=by_shot,
        paired_skill_shot_mean=np.mean([v['skill'].mean(0) for v in values],axis=0).tolist(),
        signed_shot_mean=np.mean([v['signed'].mean(0) for v in values],axis=0).tolist(),
        positive_skill_shape_and_current_beats_earlier=int(np.sum((s[:,1]>0.) & (d[:,0]>d[:,1]))),
        signed_risk_by_skill=quadrants,
        minimum_gap_shot_median=float(np.median([v['gap'].min() for v in values])),
        max_signed_change_shot_median=np.median([v['signed'].max(0) for v in values],axis=0).tolist())


def main():
    r = read(OUT/'result.json')
    old = r['attribution']
    groups = {label:same_windows([v for v in old if v['kind']==label]) for label in ('late','false')}
    for name,v in groups.items():
        print(name,{k:x for k,x in v.items() if k!='by_shot'},flush=True)
    write(OUT/'postseal_usable_feedback.json',dict(sets='old delayed/missed and false-alarm shots; '
        'actual common dose at full handoff only; HDF5 outcomes used in this analysis only',
        skill_columns=('hold minus model waveform MAE','hold minus model bin-increment MAE'),
        signed_columns=('current feedback','earlier feedback'),groups=groups))


if __name__=='__main__':
    main()
