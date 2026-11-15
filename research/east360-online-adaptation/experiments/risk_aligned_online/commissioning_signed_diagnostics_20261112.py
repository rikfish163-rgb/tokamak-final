"""Check availability and warning specificity of two directed diagnostic proxies."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np

from . import commissioning_coincident_innovation_20261111 as joint
from . import commissioning_excursion_order_20261111 as excursion
from . import commissioning_growth_signature_20261112 as growth
from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .commissioning_roughness_20261110 import coordinates
from .history_gradient_metric_20261102.run import read, sha, write


PRIOR = io23.ROOT/'probe_outputs/commissioning_growth_signature_east360_20261112_v1'
H, FIRST, STRIDE = growth.H, growth.FIRST, growth.STRIDE
NAMES = ('Mir_avg_amp','Mir_avg_fre','n=1 amplitude','xuv_skew','xuv_var',
         'v_loop','ip','xuv_ratio')
AMP,FREQ,N1,XSKEW,XVAR,VOLT,CURRENT,RATIO = range(len(NAMES))
COLS = tuple(io23.RAW23.index(k) for k in NAMES)
KNOTS = np.arange(0,25,8)
STRENGTH = .05
RAMP = .01
SCOPE = ('all','late','miss','false_alarm')


def magnetic_pair(amplitude, falling_frequency, shift):
    """Change the coincidence only: keep both amplitude series and all frequency steps."""
    null = growth.permute_steps(falling_frequency, shift)
    return tuple(np.minimum(amplitude, f[:, :, None]).mean(1) for f in (falling_frequency,null))


def radiative_pair(shape, voltage):
    """Adjacent 8-sample increments: radiation leading or following positive voltage."""
    return (shape[:, :-1]*voltage[:, 1:]).mean(1), (voltage[:, :-1]*shape[:, 1:]).mean(1)


def observe_block(z, mask, raw, release, rank):
    q = release-H-1
    idx = q[:, None]+np.arange(-4,H+1)
    k = q[:, None, None]+KNOTS[None, :, None]+np.arange(-4,1)
    valid = mask[idx] & np.isfinite(raw[idx]) & np.isfinite(z[idx])
    good = valid.all(1)
    means = np.where(mask[k] & np.isfinite(z[k]),z[k],0.).mean(2,dtype=float)
    step = np.tanh(np.diff(means,axis=1)/STRENGTH)
    step = np.where(good[:, None],step,0.)

    f = raw[idx,FREQ]
    f_obs = valid[:,:,FREQ]
    freq_zero = f_obs & (f==0.)
    freq_pos = f_obs & (f>0.)
    drop_zero = (freq_pos[:, :-1] & freq_zero[:, 1:]).any(1)
    mag_finite = good[:,[AMP,FREQ,N1]].all(1)
    mag_usable = mag_finite & freq_pos.all(1)
    amplitude = step[:,:,[AMP,N1]].clip(min=0.)
    falling = (-step[:,:,FREQ]).clip(min=0.)
    shift = 1 + ((rank+(release-FIRST)//STRIDE) % 2)
    true, wrong = magnetic_pair(amplitude,falling,shift)

    rad_usable = good[:,[XSKEW,XVAR,VOLT]].all(1)
    shape = np.sqrt(np.square(step[:,:,[XSKEW,XVAR]]).mean(2))
    voltage = step[:,:,VOLT].clip(min=0.)
    lead,trail = radiative_pair(shape,voltage)
    now,prev = (np.abs(raw[k[:,i],CURRENT]).mean(1,dtype=float) for i in (3,0))
    ip_known = good[:,CURRENT] & np.isfinite(prev) & np.isfinite(now) & (prev>0.)
    ip_change = np.zeros(len(q),float)
    np.divide(now-prev,prev,out=ip_change,where=ip_known)
    # The availability flags and all single-channel changes belong to the
    # common control. No path term gets credit for recognizing a missing value.
    common = np.c_[good[:,[AMP,FREQ,N1,XSKEW,XVAR,VOLT]],mag_usable,rad_usable,
                  freq_zero.mean(1),drop_zero,np.tanh(ip_change/RAMP),
                  np.sort(step[:,:,[AMP,FREQ,N1,XSKEW,XVAR,VOLT]],axis=1).reshape(len(q),18)]
    return dict(common=common.astype(np.float32),magnetic=np.where(mag_usable[:,None],true,0.),
                magnetic_shift=np.where(mag_usable[:,None],wrong,0.),
                naive=np.where(mag_finite[:,None],true,0.),
                radiation=np.where(rad_usable,lead,0.),voltage_first=np.where(rad_usable,trail,0.),
                mag_usable=mag_usable,rad_usable=rad_usable,zero_transition=drop_zero,
                all_zero=freq_zero.all(1),ratio_usable=good[:,RATIO],
                ramp_down=ip_known & (ip_change < -RAMP),
                steady=ip_known & (np.abs(ip_change)<=RAMP))


def observe(out):
    began = time.monotonic()
    p,s = [read(PRIOR/k) for k in ('protocol.json','seal.json')]
    if any(sha(PRIOR/k)!=h for k,h in s['assets'].items()):
        raise ValueError('prior fixed observations changed')
    with np.load(PRIOR/'frames.npz',allow_pickle=False) as f:
        frames = {k:f[k] for k in ('rank','shot','release','time','marginal','match','usable')}
    lock = lag.load_inputs()
    if lock['ids']!=p['order']:
        raise ValueError('fixed flow changed')
    coord = joint.BRIDGE/'decoders.npz'
    earlier = read(Path(p['prior'])/'protocol.json')
    if sha(coord)!=earlier['coordinate_sha256']:
        raise ValueError('fixed scaler changed')
    with np.load(coord,allow_pickle=False) as f:
        center,scale = f['center'][list(COLS)],f['scale'][list(COLS)]
    out.mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(**p, scope_detail='DIRECTED_PROXIES_ONLY_NO_ALARM_CHANGES',
        baseline_features_sha=s['assets']['frames.npz'],coordinate_sha=sha(coord), channels=NAMES,
        step='difference of five-point trailing means at q and q+8/16/24; tanh(delta/.05)',
        frequency_support='full q-4 through q+25 finite and strictly positive; zeros not assumed locked',
        magnetic='two joint positive-amplitude / falling-frequency minima; cyclic frequency null',
        radiation='XUV variance and skewness move; compare one 8-point increment leading or trailing voltage rise',
        ramp='1% decline in mean magnitude of recorded current across 24 points; not an operator command',
        fit_detail='same previously ended block pairs as growth experiment; no late/miss labels in fitting',
        qualification='path-specific unavailable rows contribute zero only to path, not removed from flow',
        code_sha256=sha(Path(__file__))))
    rows,witnesses = [],{}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome unavailable before seal')):
        for rank,sid in enumerate(lock['ids'],1):
            if rank<=p['startup']:
                continue
            path = io23.path_for('EAST',sid)
            if sha(path)!=lock['hashes'][str(sid)]:
                raise ValueError('raw signal changed')
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            x = raw['values'][:,COLS]
            mapped = lag.prepare_actual(raw,rank,lock['mapping'])
            z,m = coordinates(mapped['values'][:,COLS],mapped['available'][:,COLS],center,scale)
            at = frames['rank']==rank
            r = frames['release'][at]
            np.testing.assert_array_equal(raw['times'][r],frames['time'][at])
            result = observe_block(z,m,x,r,rank)
            if rank in (21,120,360):
                n = min(len(z),410)
                short = observe_block(z[:n],m[:n],x[:n],r[r<n],rank)
                for k,v in short.items():
                    np.testing.assert_array_equal(result[k][r<n],v)
                witnesses[str(sid)] = n
            rows.append(result)
            if rank%85==0 or rank==360:
                print(dict(stage='seal_directed_proxies',rank=rank),flush=True)
    data = excursion._stack(rows)
    if not all(np.isfinite(v).all() for v in data.values()):
        raise FloatingPointError('nonfinite diagnostic proxy')
    joint._npz(out/'frames.npz',**frames,**data)
    write(out/'seal.json',dict(prefix_witnesses=witnesses,seconds=time.monotonic()-began,
        assets={k:sha(out/k) for k in ('protocol.json','frames.npz','executed_probe.py')}))


def designs(rows):
    x = np.c_[rows['marginal'],rows['common']]
    return dict(unordered=x,**{k:np.c_[x,rows[k]] for k in
        ('magnetic','magnetic_shift','naive','radiation','voltage_first')})


def pair_scores(rows, held, learned):
    return growth.comparisons(rows,rows[held[0]],rows[held[1]],learned)


def coverage(rows, where):
    def count(use):
        return dict(windows=int(use.sum()),shots=len(np.unique(rows['shot'][use])))
    evidence = dict(magnetic_evidence=(rows['magnetic']>0.).any(1),
                    naive_evidence=(rows['naive']>0.).any(1),
                    naive_at_zero=(rows['naive']>0.).any(1)&~rows['mag_usable']&rows['zero_transition'],
                    rad_leads=rows['radiation']>rows['voltage_first'])
    flags = ('mag_usable','zero_transition','all_zero','rad_usable','ratio_usable','ramp_down','steady')
    return dict(total=count(where),**{k:count(where & v) for k,v in evidence.items()},
                **{k:count(where & rows[k]) for k in flags})


def score(out):
    p,s = [read(out/k) for k in ('protocol.json','seal.json')]
    if (any(sha(out/k)!=h for k,h in s['assets'].items())
            or sha(out/'executed_probe.py')!=p['code_sha256']):
        raise ValueError('directed proxies changed after sealing')
    ref = Path(p['parent'])
    if sha(ref/'seal.json')!=p['parent_seal'] or sha(ref/'per_shot.json')!=p['parent_result']:
        raise ValueError('reference scores changed')
    refs = read(ref/'seal.json')['curve_sha256']
    with np.load(out/'frames.npz',allow_pickle=False) as f:
        rows = dict(f)
    _,records = growth.reveal(p,ref,refs,rows,1,p['startup']+1)
    history, blocks = [],[]
    held = dict(all=('positive','safe'),late=('late','fp'),miss=('miss','fp'))
    for start in range(p['startup']+1,len(p['order'])+1,p['block_shots']):
        stop = min(start+p['block_shots'],len(p['order'])+1)
        use = (rows['rank']>=start)&(rows['rank']<stop)
        current = {k:v[use] for k,v in rows.items()}
        learned = dict(parent=current['marginal'][:,0])
        fit = None
        if history:
            prior = excursion._stack(history)
            if prior['rank'].max()>=start:
                raise ValueError('ranker saw current block')
            a,b = growth.pairs(prior,prior['positive'],prior['safe'])
            fit = dict(pairs=len(a),ranks=np.c_[prior['rank'][a],prior['rank'][b]].tolist())
            if len(a):
                dx = designs(current)
                for name,old in designs(prior).items():
                    coef = excursion.pair_fit(old[a],old[b])
                    fit[name] = coef.tolist()
                    learned[name] = dx[name]@coef
        file = out/f'block_{start:03d}_predictions.npz'
        joint._npz(file,rank=current['rank'],release=current['release'],**learned)
        digest = sha(file)
        r,rec = growth.reveal(p,ref,refs,rows,start,stop)
        history.append(r)
        records.extend(rec)
        checks = {k:pair_scores(r,v,learned) for k,v in held.items()}
        blocks.append(dict(start=start,stop=stop-1,fit=fit,file=file.name,sha256=digest,checks=checks))
        print(dict(stage='score_sealed_block',start=start,pairs={k:v['pairs'] for k,v in checks.items()}),
              flush=True)
    if records!=read(ref/'per_shot.json')['0.2'][p['parent_curve']] or len(records)!=360:
        raise ValueError('raw outcome or warning clock differs from parent')
    r = excursion._stack(history)
    masks = dict(all=np.ones(len(r['rank']),bool),positive=r['positive'],late=r['late'],
                 miss=r['miss'],false_alarm=r['fp'],safe_fall=r['safe']&r['ramp_down'],
                 safe_steady=r['safe']&r['steady'])
    physical = dict(magnetic=r['magnetic'].mean(1),shifted=r['magnetic_shift'].mean(1),
                    naive=r['naive'].mean(1),radiation=r['radiation'],voltage_first=r['voltage_first'])
    checks = {k:pair_scores(r,v,physical) for k,v in held.items()}
    result = dict(status='COMPLETED_SIGNED_DIAGNOSTIC_PROBE_NOT_ALARMS',raw_verified_shots=len(records),
                  parent=dict(timely=sum(r['timely'] for r in records),
                              false_alarm=sum(r['alarm'] and not r['disruptive'] for r in records)),
                  coverage={k:coverage(r,v) for k,v in masks.items()},direct=checks,blocks=blocks)
    write(out/'result.json',result)
    print(dict(raw_shots=len(records),windows=len(r['rank']),parent=result['parent']),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    observe(args.out.resolve())
    score(args.out.resolve())
