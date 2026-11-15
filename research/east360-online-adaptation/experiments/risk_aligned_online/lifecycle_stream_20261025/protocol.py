"""Whole-flow commissioning contract, including all startup predictions."""
import numpy as np

from ..commissioning_stream import read_prefix, _record


def order_stream(ids, reader=read_prefix):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('nonempty distinct full roster required')
    rows=[]
    for sid in ids:
        r=_record(sid,reader(sid))
        # Ai2026 potential DBC inspires this ranking, not actual damage.
        # All positive, fixed unit/reference factors cancel in the order.
        r['proxy']=(r['initial_ip_A']/100000.)**1.2*r['initial_abs_Bt_raw']
        rows.append(r)
    return sorted(rows,key=lambda r:(r['proxy'],r['shot']))


class AccessLedger:
    def __init__(self, order, warmup):
        if len(set(order)) != len(order) or not 0 < warmup < len(order):
            raise ValueError('distinct complete stream and internal warmup required')
        self.order=list(order); self.warmup=warmup; self.history=[]; self.pending=None

    @property
    def ready(self):
        return len(self.history)>=self.warmup

    def seal(self,sid,digest):
        if (self.pending is not None or len(self.history)>=len(self.order)
                or sid != self.order[len(self.history)] or len(digest)!=64):
            raise ValueError('current-shot durable seal required, in stream order')
        self.pending=(sid,digest)

    def reveal(self,sid):
        if self.pending is None or self.pending[0]!=sid:
            raise ValueError('cannot reveal before matching seal or reveal twice')
        self.history.append(sid); self.pending=None


def sampled_endpoints(times,y,valid,scores,*,down):
    times,y,valid,scores=map(np.asarray,(times,y,valid,scores))
    if times.ndim!=1 or any(a.shape!=times.shape for a in (y,valid,scores)):
        raise ValueError('matching per-endpoint training arrays required')
    allowed=valid & (np.arange(len(times))>=127)
    if down is not None:
        allowed &= np.round((down-times)*1000,9)>30
    ids=np.flatnonzero(allowed)
    if not len(ids):
        raise ValueError('no observed valid timely training window')
    uniform=ids[np.linspace(0,len(ids)-1,min(32,len(ids)),dtype=int)]
    hard=ids[np.argsort(scores[ids],kind='stable')[-16:]]
    positive=ids[y[ids]>0]
    if len(positive):positive=positive[np.linspace(0,len(positive)-1,min(16,len(positive)),dtype=int)]
    return np.unique(np.r_[uniform,hard,positive])
