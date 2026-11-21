"""Materialize reached shadow observations and separately masked future labels."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np


@dataclass
class Window:
    reached: np.ndarray
    tail: np.ndarray
    length: np.ndarray
    allowed: np.ndarray
    delay: np.ndarray


def sample_late(ids, owner, clock):
    ids, owner, clock = (np.asarray(v) for v in (ids,owner,clock))
    if (ids.ndim!=1 or owner.ndim!=1 or owner.shape!=clock.shape
            or any(x.dtype.kind not in "iu" for x in (ids,owner,clock))
            or (owner<0).any() or (owner>=len(ids)).any()):
        raise ValueError("invalid original shot/release keys")
    return np.array([9+int.from_bytes(
        sha256(f"patch-margin-late-v1/{int(ids[j])}/{int(t)}".encode("ascii")).digest()[:8],"big"
    )%23 for j,t in zip(owner,clock,strict=True)],dtype=np.int64)


def select(tape,delays,tau,bound):
    clock,curves,menu = (np.asarray(v) for v in (tape.data["clock"],tape.data["curves"],tape.allowed))
    n = len(clock)
    delay = np.asarray(delays)
    if (clock.shape!=(n,) or clock.dtype.kind not in "iu" or (clock<0).any()
            or curves.shape!=(n,13,32) or menu.shape!=(n,13) or menu.dtype!=bool
            or delay.shape not in ((),(n,)) or delay.dtype.kind not in "iu"
            or (delay<8).any() or (delay>31).any()
            or not np.isfinite(tau) or not np.isfinite(bound) or bound<0):
        raise ValueError("invalid tape, clock or paired bound")
    delay = np.broadcast_to(delay,(n,)).astype(np.int64,copy=True)
    length,seen,no_alarm = np.zeros(n,np.int64),np.zeros(n,bool),np.zeros(n,bool)
    covered = 0
    for q in tape.shots:
        sl = q.blocks
        native = np.asarray(q.native)
        if (sl.start!=covered or sl.stop<sl.start or sl.stop>n or native.ndim!=1
                or (clock[sl]>=len(native)).any()):
            raise ValueError("misaligned source shot blocks")
        covered = sl.stop
        t = clock[sl]+delay[sl]
        seen[sl] = t<len(native)
        length[sl] = np.clip(len(native)-1-t,0,32-delay[sl])
        ix = np.flatnonzero(seen[sl])
        if len(ix):
            native_prefix = np.maximum.accumulate(native[:int(t[ix].max())+1].astype(np.float64))
            no_alarm[sl.start+ix] = native_prefix[t[ix]]<tau
    if covered!=n:
        raise ValueError("source shot blocks do not cover release rows")
    reached = np.zeros((n,13,8),curves.dtype)
    tail = np.zeros((n,13,24),curves.dtype)
    for d in np.unique(delay):
        rows = (delay==d)&seen
        reached[rows] = curves[rows,:,d-8:d]
        real = np.arange(32-d)[None,:]<length[rows,None]
        tail[rows,:,:32-d] = np.where(real[:,None,:],curves[rows,:,d:],0)
    shift = reached[:,1:,-1].astype(np.float64)-reached[:,0:1,-1].astype(np.float64)
    pair = (np.abs(shift).reshape(n,6,2)<=bound).all(2)&menu[:,1:].reshape(n,6,2).all(2)
    allowed = np.ones((n,13),bool)
    allowed[:,1:] = pair.repeat(2,axis=1)&no_alarm[:,None]&(length>0)[:,None]
    return Window(reached,tail,length,allowed,delay)
