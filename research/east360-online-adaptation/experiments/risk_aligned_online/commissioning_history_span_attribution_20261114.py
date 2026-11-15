"""Explain the span intervention using exactly the sealed finite encoder actions."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logit

from . import commissioning_history_span_20261114 as model
from .history_gradient_metric_20261102.run import read, sha, write
from .paper131_alarm_audit import alarm_endpoints


MENUS = {
    'fixed_spans': model.DIALS[:3],
    'all_spans': model.DIALS,
    'keep_old_or_use_spans': ('legacy_matched', *model.DIALS),
    'keep_parent_old_or_use_spans': ('bias_only', 'legacy_matched', *model.DIALS),
}


def timely(times, downtime):
    return np.round((downtime-times)*1000., 9)>30.


def margin_capacity(scores, valid, threshold):
    hits = scores[:,valid]>=np.float32(threshold)
    static = hits.any(axis=1)
    return dict(can_raise=bool(static.any()), min_static=bool(static.all()),
                forced_pointwise=bool(hits.all(axis=0).any()))


def confirmation(run, *, names=('bias_only','legacy_matched','fixed_short','span_now','span_earlier','span_shifted')):
    """Sensitivity only: charge the confirmation time, not the first point of the run."""
    p, s, per = (read(run/f'{k}.json') for k in ('protocol','seal','per_shot'))
    rows = {b:{a:{n:[0,0] for n in (1,2,3,5)} for a in names} for b in p['calibration']}
    if len(s['curves'])!=360:
        raise ValueError('score only a complete sealed stream')
    for rank,sid in enumerate(p['order'],1):
        y, down, times = model.group.hdf._raw_truth(sid)
        valid = (down-times)*1000.>=-1e-6 if y else np.ones(len(times),bool)
        clocks = times[valid]
        name = f'{rank:04d}_{sid}.npz'
        path = run/'shots'/name
        if sha(path)!=s['curves'][name]:
            raise ValueError('sealed confirmation curve changed')
        with np.load(path,allow_pickle=False) as f:
            curves = {k:f[k][valid] for k in names}
        for b,th in p['calibration'].items():
            tau = float(np.float32(p['original'] if rank<=p['startup'] else th))
            for a,curve in curves.items():
                original = per[b][a][rank-1]
                for n, counts in rows[b][a].items():
                    c = alarm_endpoints(clocks,curve,tau,points=n)
                    hit = None if c is None else c[1]
                    early = bool(y and hit is not None and timely(clocks[hit],down))
                    if n==1 and (early!=original['timely'] or (hit is not None)!=original['alarm']):
                        raise ValueError('confirmation sensitivity changed the original first alarm')
                    counts[0] += early
                    counts[1] += bool(not y and hit is not None)
    return dict(scope='POSTSEAL_CONFIRMATION_SENSITIVITY_NO_THRESHOLD_CHANGE',
                pair='timely, false alarm', tables=rows)


def bounds(run):
    protocol, seal, per = (read(run/f'{k}.json') for k in ('protocol','seal','per_shot'))
    if protocol['scope']!='EAST360_CAUSAL_ENCODER_HISTORY_SPAN_DEVELOPMENT' or len(seal['curves'])!=360:
        raise ValueError('sealed full-stream history-span replay required')
    result = {b:{mode:defaultdict(list) for mode in MENUS} for b in protocol['calibration']}
    selected = defaultdict(list)
    for rank, sid in enumerate(protocol['order'], 1):
        rows = {b:per[b]['current_5'][rank-1] for b in result}
        y, down, times = model.group.hdf._raw_truth(sid)
        ref = rows['0.2']
        if (ref['shot'],ref['disruptive'],ref['down_s'])!=(sid,y,down):
            raise ValueError('sealed outcomes disagree with HDF5')
        name = f'{rank:04d}_{sid}.npz'
        path = run/'shots'/name
        if sha(path)!=seal['curves'][name]:
            raise ValueError('sealed prediction changed')
        with np.load(path,allow_pickle=False) as f:
            arrays = {k:f[k] for k in model.ARMS}
            valid = timely(times,down) if y else np.ones(len(times), bool)
            for b, tau in protocol['calibration'].items():
                if rank<=protocol['startup']:
                    tau = protocol['original']
                for mode, names in MENUS.items():
                    scores = np.stack([arrays[k] for k in names])
                    z = margin_capacity(scores,valid,tau)
                    d = result[b][mode]
                    if y:
                        if z['can_raise']:
                            d['tp_upper'].append(sid)
                            if not rows[b]['timely']:
                                d['outside_old_tp'].append(sid)
                        elif rows[b]['timely']:
                            d['unreachable_old_tp'].append(sid)
                    else:
                        if z['min_static']:
                            d['fp_lower_whole_shot'].append(sid)
                        if z['forced_pointwise']:
                            d['fp_lower_pointwise'].append(sid)

            if rank<=protocol['startup']:
                continue
            frames, follow = f['frames'], f['follow']
            if y and not ref['timely']:
                lead = np.round((down-times[frames[:,1]])*1000.,9)
                watch = (lead>30.) & (lead<=125.) & timely(times[frames[:,3]],down)
                key = 'late_before_deadline'
            elif not y and ref['alarm']:
                first = int(np.flatnonzero(arrays['current_5']>=np.float32(protocol['calibration']['0.2']))[0])
                watch = frames[:,3]<=first
                key = 'safe_before_false_alarm'
            else:
                continue
            if not watch.any():
                continue
            witness = follow[watch]
            action = frames[watch,1]
            view = {a:witness[:,model.WITNESS.index(a)] for a in model.WITNESS}
            # Measure direction from the actual finite action, not a decoder Jacobian.
            p = arrays['bias_only'][action]
            z = logit(arrays['span_now'][action].clip(1e-6,1-1e-6))-logit(p.clip(1e-6,1-1e-6))
            earlier_win = view['span_now']<view['span_earlier']
            shift_win = view['span_now']<view['span_shifted']
            correct = z>0. if y else z<0.
            selected[key].append((sid, len(z), float(z.mean()), int(correct.sum()),
                                  int((correct & earlier_win & shift_win).sum()),
                                  float((view['span_now']-view['no_added_write']).mean())))

    fields = ('tp_upper','outside_old_tp','unreachable_old_tp','fp_lower_whole_shot','fp_lower_pointwise')
    compact = {b:{m:dict(counts={k:len(v[k]) for k in fields},shots={k:v[k] for k in fields})
                    for m,v in arms.items()} for b,arms in result.items()}
    gains = {g:dict(shots=len(v),blocks=sum(x[1] for x in v),
              right_sign_shots=sum((x[2]>0. if g=='late_before_deadline' else x[2]<0.) for x in v),
              right_sign_blocks=sum(x[3] for x in v),right_and_both_holdout_controls=sum(x[4] for x in v),
              by_shot=v) for g,v in selected.items()}
    return dict(scope='POSTSEAL_FIXED_ACTION_MENU_NOT_A_POLICY',action_menu=compact,direction=gains,
        direction_columns=('shot','eligible_held_blocks','mean_current_minus_parent_logit',
                           'correct_direction_blocks','correct_and_beats_two_controls_blocks',
                           'mean_current_minus_parent_held_motion_loss'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--confirmation',action='store_true')
    a = parser.parse_args()
    if a.confirmation:
        result = confirmation(a.run.resolve())
        write(a.run/'postseal_confirmation.json',result)
        print(result,flush=True)
    else:
        result = bounds(a.run.resolve())
        write(a.run/'postseal_span_attribution.json',result)
        print({k:{s:a['counts'] for s,a in v.items()} for k,v in result['action_menu'].items()},flush=True)
        print({g:{k:v for k,v in d.items() if k!='by_shot'} for g,d in result['direction'].items()},flush=True)
