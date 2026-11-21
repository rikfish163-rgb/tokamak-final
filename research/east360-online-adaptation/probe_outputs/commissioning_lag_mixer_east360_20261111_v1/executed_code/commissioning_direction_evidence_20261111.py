"""Causal comparison of two diagnostic forecasts at a fixed update budget."""

import numpy as np
import torch

from .commissioning_persistence_20261111 import STRIDE, WINDOW
from .commissioning_source_teaching_east_20261111 import match_magnitude
from .commissioning_weak_shape_data_20261111 import GROUPS


_CHANNELS = np.asarray(GROUPS)
_FAMILIES = tuple(_CHANNELS==k for k in range(4))
FEATURES = 1 + len(_FAMILIES)


def decision_evidence(q, old, native, target, observed, length):
    """Release relative Huber evidence at q+26, never at q or on partial blocks."""
    if (old.shape != native.shape or old.shape != target.shape or old.shape != observed.shape
            or old.shape != (len(q), WINDOW, len(_CHANNELS))
            or np.any(np.diff(q)!=STRIDE) or np.any((q<0)|(q>=length))):
        raise ValueError('aligned direct forecasts on the locked issuance grid required')
    dense = np.zeros((length,FEATURES),np.float32)
    ready = q+WINDOW+1 < length
    m = observed[ready]
    channel_support = m.sum(1)>=(WINDOW+1)//2
    family_support = np.column_stack([channel_support[:,k].any(1) for k in _FAMILIES])
    supported = family_support.sum(1)>=2

    def loss(x):
        r = np.abs(np.where(m,x[ready]-target[ready],0.))
        return .5*np.minimum(r,1.)**2 + np.maximum(r-1.,0.)

    a,b = loss(old),loss(native)
    delta = a-b
    def agree(d, mask):
        den = np.abs(d).sum((1,2))+.02*mask.sum((1,2))
        return np.divide(d.sum((1,2)),den,out=np.zeros(len(d),np.float32),where=den>0.)

    x = np.column_stack([agree(delta,m),*(agree(delta[:,:,k],m[:,:,k]) for k in _FAMILIES)])
    x[~supported] = 0.
    if len(x):
        start = int(q[0]+WINDOW+1)
        dense[start:] = np.repeat(x,STRIDE,axis=0)[:length-start]
    improvement = delta.sum((1,2))[supported]
    diff = np.any((old[ready]!=native[ready])&m,axis=(1,2)) & supported
    return dense, dict(comparisons=int(supported.sum()), distinct=int(diff.sum()),
                       native_wins=int((improvement>0.).sum()),old_wins=int((improvement<0.).sum()),
                       old_loss=float(a[supported].sum(dtype=np.float64)),
                       native_loss=float(b[supported].sum(dtype=np.float64)),
                       observations=int(m[supported].sum()))


def shift_evidence(e):
    """The causal lag control receives the previous block, not a future shuffle."""
    z = np.zeros_like(e)
    z[STRIDE:] = e[:-STRIDE]
    return z


def mix_direction(old, native, weight):
    """Change the direction, then project back to the old pointwise L2 norm."""
    w = weight.unsqueeze(-1)
    return match_magnitude((1.-w)*old+w*native,old)


def weights(evidence, gate):
    """Scalar prior, fixed blend and evidence-dependent policies at each sample."""
    x = torch.from_numpy(evidence)
    intercept, coef, constant = gate
    base = torch.full_like(x[:,0],torch.sigmoid(intercept).item())
    signal = x@coef
    previous = torch.from_numpy(shift_evidence(evidence))@coef
    def route(v):
        # With no directional evidence the policy is exactly its fixed prior.
        return torch.where(v!=0.,(intercept+v).sigmoid(),base)
    return {
        'midpoint': torch.full_like(x[:,0],.5),
        'constant': torch.full_like(x[:,0],torch.sigmoid(constant).item()),
        'neutral': base,
        'agreement': .5+.5*x[:,0],
        'learned': route(signal),
        'lagged': route(previous),
        'inverted': route(-signal),
    }
