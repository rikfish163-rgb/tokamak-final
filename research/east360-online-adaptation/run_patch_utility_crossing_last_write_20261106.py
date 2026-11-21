"""Find the last still-protected commit before each real helpful crossing."""

import argparse
from pathlib import Path
import time

import numpy as np

import run_patch_utility_pivotal_support_20261106 as p
from verify_confirmation_raw_replay_20261105 import Hashes


b,m = p.b,p.m
PLAN = b.r.ROOT / "PATCH_UTILITY_CROSSING_LAST_WRITE_PROTOCOL_20261106.md"


def legal_delays(curves,native,clock,crossing_delay,action,tau,bound):
    z = np.asarray(curves,dtype=np.float64)
    if (z.ndim!=2 or z.shape[1]!=32 or z.shape[0]%2!=1 or not 0<action<len(z)
            or not 8<crossing_delay<=32 or not 0<=clock<clock+crossing_delay<len(native)):
        raise ValueError("invalid actual crossing or original action")
    i = 1+2*((action-1)//2)
    delays = np.arange(8,crossing_delay)
    d = np.abs(z[i:i+2,delays-1]-z[0,delays-1])
    already = np.maximum.accumulate(np.asarray(native,dtype=np.float64)>=tau)[clock+delays]
    return delays[(d.max(axis=0)<=bound)&~already]


def witness(q,d,case):
    loc = np.flatnonzero(d["clock"][q.blocks]==case["clock"])
    if len(loc)!=1:
        raise ValueError("original release was not found exactly once")
    j = q.blocks.start+int(loc[0])
    c = d["curves"][j].astype(np.float64)
    clock,a = int(d["clock"][j]),case["action"]
    g,valid,*_ = p.payoff(q.native,q.times,q.disruptive,q.down,q.horizon,
                          np.array([clock]),b.TAU)
    orig,f = p.effect(c[None,:,8:],g,valid,b.TAU)
    k = int(f[0,a]+9)
    if clock+k!=case["sample"] or orig[0,a]!=1:
        raise ValueError("old crossing witness does not reproduce")
    bound = b.pair.LIMIT-b.pair.GUARD_HEADROOM
    delays = legal_delays(c,q.native,clock,k,a,b.TAU,bound)
    if not len(delays) or delays[0]!=8:
        raise ValueError("old protected clock did not reproduce")
    held = []
    for d0 in delays:
        delta = c[a,d0-1]-c[0,d0-1]
        hold = c[0,8:].copy()
        hold[d0-8:] += delta
        z = np.stack((c[0,8:],hold))[None]
        r,offset = p.effect(z,g,valid,b.TAU)
        if r[0,1]>0:
            held.append(dict(delay=int(d0),local_first_offset=int(offset[0,1]+1)))
    return dict(shot=int(q.sid),action=a,release=clock,crossing_delay=k,
                latest=int(delays[-1]),necessary_frames=int(k-delays[-1]),
                delays=delays.tolist(),lead_ms=case["lead_ms"],held_success=held,
                last_delta=float(c[a,delays[-1]-1]-c[0,delays[-1]-1]),
                delta_at_hit=float(c[a,k-1]-c[0,k-1]),
                static_deficit_at_hit=float(b.TAU-c[0,k-1]),bound=bound)


def run(native,masked,baseline,proof,out):
    if out.exists():
        raise FileExistsError(out)
    start,h = time.monotonic(),Hashes()
    for path in (Path(__file__),Path(p.__file__),PLAN,proof):
        h(path)
    rec = b.r.read(proof)
    if rec["status"]!="EXACT_OFFLINE_FIRST_ALARM_FACTOR_AND_SOURCE_SUPPORT":
        raise ValueError("missing independently derived rescue cases")
    tapes,_,_ = m.load(native,masked,baseline,h)
    views = {}
    for v in m.VIEWS:
        t = tapes[v]
        qs = {q.sid:q for q in t.shots}
        views[v] = [witness(qs[x["shot"]],t.data,x) for x in rec["views"][v]["rescue_actions"]]
    all_cases = [x for rows in views.values() for x in rows]
    grouped = {i:min(q["necessary_frames"] for q in all_cases if q["shot"]==i)
               for i in sorted({x["shot"] for x in all_cases})}
    held = sorted({x["shot"] for x in all_cases if x["held_success"]})
    h.check()
    b.r.write_new(out,dict(status="LATEST_PROTECTED_COMMIT_DIAGNOSIS_NOT_ONLINE",views=views,
                            necessary_frames_per_shot=grouped,hold_preserves=held,new_fits=0,
                            new_val_test_east_reads=0,first_alarm_policy_replays=0,
                            input_sha256={str(a):r[1] for a,r in h.cache.items()},
                            elapsed_seconds=time.monotonic()-start))
    print(grouped,"hold_preserves",held,flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("native","masked","baseline","proof","output"):
        cli.add_argument(key,type=Path)
    a = cli.parse_args()
    run(a.native,a.masked,a.baseline,a.proof,a.output)
