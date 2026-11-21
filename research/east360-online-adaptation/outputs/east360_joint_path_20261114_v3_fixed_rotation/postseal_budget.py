"""Decompose fixed-threshold results using shot peaks and an equal error budget."""

from pathlib import Path

import numpy as np

from experiments.risk_aligned_online import commissioning_joint_path_20261114 as exp
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, write
from experiments.risk_aligned_online.operating_point_audit import best_at_count, peak_for_shot


HERE = Path(__file__).resolve().parent
WEAK = HERE.parent/'east360_joint_path_20261114_v2'
NAMES = (*exp.BASES, *(f'{family}_{arm}'
                      for family in ('weak','turn') for arm in exp.TRIALS))


def main():
    protocol = read(HERE/'protocol.json')
    other = read(WEAK/'protocol.json')
    lock = read(exp.history.ARCHIVE/'seal.json')
    locks = [read(p/'seal.json')['completed'] for p in (WEAK,HERE)]
    order = protocol['order']
    names = [f'{i:04d}_{s}.npz' for i,s in enumerate(order,1)]
    if any(protocol[k]!=other[k] for k in exp.history.LOCK) or any(names!=a for a in locks):
        raise ValueError('same fully sealed replay required')
    cold = {k:[0,0] for k in NAMES}
    suffix = {k:[] for k in NAMES}
    for i,(shot,key) in enumerate(zip(order,names,strict=True),1):
        d,down,t = exp.group.hdf._raw_truth(shot)
        baseline = exp.history.archived_shot(i,shot,lock)
        with (np.load(WEAK/'shots'/key,allow_pickle=False) as a,
              np.load(HERE/'shots'/key,allow_pickle=False) as b):
            curves = {k:baseline[k] for k in exp.BASES} | {
                f'{family}_{arm}':f[arm]
                for family,f in (('weak',a),('turn',b)) for arm in exp.TRIALS}
        for k in NAMES:
            peak = peak_for_shot(t,curves[k],d,down)
            if i<=protocol['startup']:
                if peak is not None and peak>=protocol['original']:
                    cold[k][int(not d)] += 1
            else:
                suffix[k].append(dict(shot=shot,disruptive=d,peak=peak))
    result = {}
    for n in (18,27,36):
        matched = {}
        for k in NAMES:
            ct,cf = cold[k]
            r = best_at_count(suffix[k],n-cf)
            matched[k] = dict(tp=ct+r['tp'],fp=cf+r['fp'],cold_tp=ct,cold_fp=cf)
        result[n] = matched
    write(HERE/'postseal_budget.json',dict(note='postseal peak ordering; thresholds not used for prediction',
        limits=list(result),arm_order=NAMES,matched=result))
    print(result)


if __name__=='__main__':
    main()
