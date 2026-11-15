"""Estimate local outward/return response from already observed position vectors."""

import numpy as np

from . import io23
from .commissioning_cross_response_20261111 import FIRST, STRIDE, WINDOW


FIRST_RELEASE = FIRST + WINDOW + 1
LOOKBACK, NEIGHBORS = 16, 4
CONTEXT_RADIUS, SPREAD_FLOOR = 1., .05
CHANNELS = tuple(io23.RAW23.index(k) for k in ('ip','Bt','dx_a','dy_a'))
N_FEATURES = 6


def excursion(points, reference, spread):
    """A signed local radial response; not a measured instability growth rate."""
    relative = points-reference
    v = np.diff(relative, axis=0)
    distance = np.linalg.norm(relative, axis=1)
    numer = (relative[:2]*v).sum(1)
    den = distance[:2]**2+spread**2
    pair = numer/den
    common = np.sign(pair[1])*min(abs(pair[0]),abs(pair[1])) if pair[0]*pair[1] > 0. else 0.
    amplitude = np.r_[distance[2], np.linalg.norm(v[1]), np.linalg.norm(v[0])]
    values = np.r_[numer.sum()/den.sum(), common, (distance[2]-distance[1])/spread, amplitude/spread]
    return np.tanh(values).astype(np.float32)


def replay(x, mask):
    """Every release uses a historical reference and the same three snapshots."""
    first_snapshot = FIRST_RELEASE-1-3*STRIDE
    ends = np.arange(first_snapshot,len(x)-1,STRIDE,dtype=np.int64)
    release = ends[3:]+1
    windows = ends[:,None]+np.arange(-31,1)
    bins = x[windows].reshape(len(ends),4,8,4).mean(2)
    ip, bt = bins[:,-1,:2].T
    ctx = np.column_stack((ip,bt,ip-bins[:,0,0]))
    position = bins[:,-1,2:]
    valid = mask[windows][:,:,:2].all((1,2)) & mask[windows[:,-8:]][:,:,2:].all((1,2))
    compact = np.zeros((len(release),N_FEATURES),np.float32)
    eligible = np.zeros(len(release),bool)
    nearest = np.full(len(release),-1,np.int64)
    amplitude = np.zeros((len(release),3),np.float32)
    state = np.zeros((len(x),N_FEATURES),np.float32)
    for j,k in enumerate(range(3,len(ends))):
        if not valid[k-2:k+1].all() or np.any(
                np.linalg.norm(ctx[k-2:k]-ctx[k],axis=1)>CONTEXT_RADIUS):
            continue
        cand = np.arange(max(0,k-2-LOOKBACK),k-2)[::-1]
        d = np.linalg.norm(ctx[cand]-ctx[k],axis=1)
        cand = cand[valid[cand] & (d<=CONTEXT_RADIUS)]
        if not len(cand):
            continue
        order = np.argsort(np.linalg.norm(ctx[cand]-ctx[k],axis=1),kind='stable')
        ref = cand[order[:NEIGHBORS]]
        center = np.median(position[ref],axis=0)
        spread = max(SPREAD_FLOOR,np.median(np.linalg.norm(position[ref]-center,axis=1)))
        points = position[k-2:k+1]
        a,b,c = points
        compact[j] = excursion(points,center,spread)
        amplitude[j] = (np.linalg.norm(c-center),np.linalg.norm(c-b),np.linalg.norm(b-a))
        eligible[j] = True
        nearest[j] = ends[ref[0]]
        state[release[j]:release[j]+STRIDE] = compact[j]
    return dict(state=state, release=release, compact=compact, eligible=eligible,
                context=ctx[3:], amplitude=amplitude, reference_end=nearest)


def observe(signal, center, scale):
    values = signal['values'][:, CHANNELS].astype(np.float64)
    values[:,:2] = abs(values[:,:2])
    available = signal['available'][:,CHANNELS]
    z = np.where(available,np.arcsinh((values-center)/scale),0.)
    return replay(z,available)


def unsigned(state):
    """Remove the direction without changing feature dimension or magnitudes."""
    return np.abs(state)


def lagged(state):
    return np.concatenate((np.zeros_like(state[:STRIDE]),state[:-STRIDE]))
