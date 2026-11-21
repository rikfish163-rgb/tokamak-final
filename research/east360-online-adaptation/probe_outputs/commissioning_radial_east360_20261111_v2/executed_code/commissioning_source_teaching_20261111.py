"""Source-only support and label-window ablations for a fixed EAST update."""

import numpy as np
import torch

from . import commissioning_persistence_20261111 as persistence
from . import commissioning_persistence_learning_20261111 as teacher
from . import commissioning_split_learning_20261111 as direct
from . import simple_data


POLICIES = ('legacy', 'coverage', 'native')
BINS = ((30., 60., 45.), (60., 95., 75.), (95., 125., 105.))


def _pair_table(times, near):
    far = np.searchsorted(times, times[near]-.032+1e-9, side='right')-1
    return np.column_stack((far, near)).astype(np.int64)


def source_endpoints(times, valid, positive, down, horizon):
    """Keep every legacy row; fill empty lead bins from earlier causal context.

    The near endpoint must still have a mature update. The far endpoint is a
    matching covariate, not a supervision target or a state-update opportunity.
    The native arm changes only the J-TEXT near-anchor policy: three bins of
    the source's own useful positive interval, respecting its original mask.
    """
    times, valid, positive = map(np.asarray, (times, valid, positive))
    legacy = direct.pairs.endpoints(times, valid, True, down, 0.,
                                    first_effect=persistence.FIRST_EFFECT)
    if positive.shape != valid.shape or not np.isfinite(horizon) or horizon <= 10.:
        raise ValueError('an aligned source positive mask and source horizon >10 ms are required')
    near = np.flatnonzero(valid & (np.arange(len(times)) >= persistence.FIRST_EFFECT))
    candidates = _pair_table(times, near)
    keep = (candidates[:,0] >= direct.FIRST) & (candidates[:,0] < candidates[:,1])
    keep[keep] = valid[candidates[keep,0]]
    candidates = candidates[keep]
    lead = np.round((down-times)*1000., 9)
    earlier = []
    for lo, hi, center in BINS:
        fixed_lead = lead[legacy[:,1]]
        if np.any((fixed_lead > lo) & (fixed_lead <= hi)):
            continue
        ll = lead[candidates[:,1]]
        ix = np.flatnonzero((ll > lo) & (ll <= hi))
        if len(ix):
            earlier.append(candidates[ix[np.argmin(np.abs(ll[ix]-center))]])
    extra = np.asarray(earlier, dtype=np.int64).reshape(-1,2)
    native_candidates = candidates[positive[candidates[:,1]].astype(bool) &
                                    (lead[candidates[:,1]] > 10.)]
    native = []
    # The endpoint label mask already includes the HDF5 roundoff tolerance.
    location = np.minimum(1., (lead[native_candidates[:,1]]-10.)/(horizon-10.))
    for i in range(3):
        ids = np.flatnonzero((location > i/3) & (location <= (i+1)/3))
        if len(ids):
            native.append(native_candidates[ids[np.argmin(np.abs(location[ids]-(i+.5)/3))]])
    return dict(legacy=legacy, coverage=np.concatenate((legacy, extra)),
                native=np.asarray(native, dtype=np.int64).reshape(-1,2))


def _retain_endpoint(record, times):
    q = record['indices']
    record['pairs']['endpoint'] = q
    record['pairs']['context_endpoint'] = _pair_table(times, q)[:,0]
    return record


@torch.no_grad()
def collect(net, ids, advanced, params, mapper, role):
    """Read each source shot once; share all safe rows and the legacy rows."""
    result = {k: [] for k in POLICIES}
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        t = raw['times']
        if len(t) > direct.FIRST:
            legacy = _retain_endpoint(teacher.labelled(net, raw, raw, advanced, params, mapper), t)
            result['legacy'].append(legacy)
            result['coverage'].append(legacy)
            if not legacy['disruptive']:
                result['native'].append(legacy)
            else:
                truth = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
                e = source_endpoints(t, truth['valid'], truth['y'], truth['down'], truth['horizon'])
                extra = e['coverage'][len(e['legacy']):]
                points = np.union1d(extra.ravel(), e['native'].ravel())
                if len(points):
                    tr, pre, ready = teacher.observed_track(net, raw, points, params, mapper)
                    for policy, index in (('coverage', extra), ('native', e['native'])):
                        pos = np.searchsorted(points, index)
                        keep = tr['matchable'][pos].all(1)
                        index, pos = index[keep], pos[keep]
                        q = index[:,1]
                        if len(q):
                            pairs = dict(pre=pre[pos[:,1]].numpy(), key=tr['key'][pos[:,1]],
                                         context=tr['context'][pos], tail=np.zeros(len(q),bool),
                                         shot=np.full(len(q),sid), disruptive=np.ones(len(q),bool),
                                         official_positive=truth['y'][q].astype(bool), endpoint=q,
                                         context_endpoint=index[:,0])
                            result[policy].append(dict(ready=ready, pairs=pairs, indices=q,
                                                       alarm=None, disruptive=True))
        if i%450 == 0 or i == len(ids):
            print(dict(stage=role, shots=i, total=len(ids)), flush=True)
    return result


def receipt(pack):
    rows, match, _ = pack
    pos, neg = match[:2]
    return [dict(positive_shot=int(rows['shot'][a]), positive=int(rows['endpoint'][a]),
                 positive_context=int(rows['context_endpoint'][a]),
                 inside_official=bool(rows['official_positive'][a]),
                 donor_shot=int(rows['shot'][b]), donor=int(rows['endpoint'][b]),
                 donor_context=int(rows['context_endpoint'][b])) for a,b in zip(pos,neg,strict=True)]
