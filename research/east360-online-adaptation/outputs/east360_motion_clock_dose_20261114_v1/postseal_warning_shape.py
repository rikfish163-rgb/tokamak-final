"""Separate sustained alarm evidence from a one-point maximum after full sealing."""

from pathlib import Path

import numpy as np

from experiments.risk_aligned_online import commissioning_motion_clock_dose_20261114 as dose
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha, write


OUT = Path(__file__).resolve().parent


def score_curve(times, risk, down, minimum, threshold):
    eligible = times<=down+1e-9 if down is not None else np.ones(len(times),bool)
    crossed = eligible & (risk>=threshold)
    hits = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(crossed,minimum).all(1))
    if not len(hits):
        return False, None
    t = int(hits[0])+minimum-1
    return True, round((down-times[t])*1e3,9) if down is not None else None


def main():
    protocol,result,seal = (read(OUT/f'{k}.json') for k in ('protocol','per_shot','seal'))
    ref = read(dose.history.ARCHIVE/'seal.json')
    cases = (dose.CASES + (73541, 78617, 66458))
    bands = {k:{n:{a:dict(tp=0,fp=0) for a in dose.ARMS} for n in (1,2,3)}
             for k in ('0.2','0.15')}
    witnesses, difference = {}, {k:{a:0 for a in dose.DOSE} for k in bands}
    for rank,sid in enumerate(protocol['order'],1):
        key = f'{rank:04d}_{sid}.npz'
        f = OUT/'shots'/key
        if sha(f)!=seal['curves'][key]:
            raise ValueError('changed curve after the anatomy seal')
        with np.load(f,allow_pickle=False) as r:
            new = {k:r[k] for k in dose.ARMS[3:]}
        old = dose.history.archived_shot(rank,sid,ref)
        flag,down,clock = dose.motion.group.hdf._raw_truth(sid)
        if sid in cases:
            witnesses[str(sid)] = {}
        for band,matrix in bands.items():
            tau = protocol['original'] if rank<=protocol['startup'] else protocol['calibration'][band]
            if sid in cases:
                witnesses[str(sid)][band] = {a:[] for a in dose.ARMS}
            for method in dose.ARMS:
                series = (new if method in new else old)[method]
                if method in dose.DOSE:
                    difference[band][method] += int(((series>=tau)!=(new['source_representation']>=tau)).sum())
                for n,summary in matrix.items():
                    alarm,lead = score_curve(clock,series,down,n,tau)
                    summary[method]['tp'] += bool(flag and alarm and lead>30.)
                    summary[method]['fp'] += bool(not flag and alarm)
                    if n==1:
                        y = result[band][method][rank-1]
                        if (y['alarm'],y['lead_ms'])!=(alarm,lead):
                            raise ValueError('mismatch at inherited first-hit rule')
                    if sid in cases:
                        witnesses[str(sid)][band][method].append(dict(points=n,alarm=alarm,lead_ms=lead))
    write(OUT/'postseal_warning_shape.json',dict(
        confirmation=bands, score_endpoint_disagreements=difference, selected_first_alarms=witnesses))
    for band,v in bands.items():
        print(band,v)


if __name__=='__main__':
    main()
