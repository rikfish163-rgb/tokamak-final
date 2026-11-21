"""Recount the decisive timely-rescue comparison with whole native curves."""

import argparse
from pathlib import Path

import numpy as np

import run_patch_utility_frame_margin_20261106 as a
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import deterministic_folds, first_alarm, menu_analysis


b,m,s = a.b,a.m,a.s
ROOT = b.r.ROOT/"probe_outputs"


def reconstruction(head,x):
    use = head["cols"]
    return np.einsum("d,dh->h",(x[use]-head["center"])/head["scale"],
                     np.asarray(head["coef"]).reshape(len(use),24))+head["intercept"]


def check(tapes,ids,folds,records,models):
    arms = ("hold",*s.ARMS)
    factors = ("estimated","exact_static","exact_effect")
    evidence = {}
    for v,t in tapes.items():
        data = t.data
        owner = t.owner
        sampled = a.windows.sample_late(ids,owner,data["clock"])
        evidence[v] = {}
        for name,delay in (("early",np.full(len(owner),8)),("late",sampled)):
            candidates,events = 0,0
            actual = set()
            rescue = {arm:{k:set() for k in factors} for arm in arms}
            for qi,q in enumerate(t.shots):
                sl = q.blocks
                c = data["curves"][sl]
                clock = data["clock"][sl]
                ok = np.zeros(c.shape[:2],bool)
                ok[:,0] = True
                transformed = c.copy()
                ds = delay[sl]
                for j,(release,d) in enumerate(zip(clock,ds,strict=True)):
                    n = len(q.native)-1-release-d
                    if n<=0 or any(x>=b.TAU for x in q.native[:release+d+1].astype(float)):
                        continue
                    row = c[j,:,d-1].astype(np.float64)
                    for k in range(1,13,2):
                        ok[j,k:k+2] = bool(
                            t.allowed[sl.start+j,k:k+2].all()
                            and max(abs(row[k]-row[0]),abs(row[k+1]-row[0]))<=a.BOUND
                        )
                    transformed[j,:,:d] = c[j,:1,:d]
                candidates += int(ok[:,1:].sum())
                if not q.disruptive or not ok[:,1:].any():
                    continue
                y = menu_analysis(q,clock,transformed,ok,b.TAU)["hard"]
                fold = int(folds[qi])
                for j,k in np.argwhere(y>0):
                    actual.add(int(q.sid))
                    events += 1
                    d = int(ds[j])
                    off = int(clock[j]+d+1)
                    n = min(32-d,len(q.native)-off)
                    c0 = c[j,0,d:d+n].astype(float)
                    c1 = c[j,k,d:d+n].astype(float)
                    z0,delta = c0,c1-c0
                    prefix = c[j:j+1,:,d-8:d]
                    g = a.common_features(data["risk"][sl][j:j+1],prefix,np.array([d]))
                    s0 = reconstruction(models["static"][fold],g[0])[:n]+float(prefix[0,0,-1])
                    for arm in arms:
                        d0 = float(prefix[0,k,-1])-float(prefix[0,0,-1])
                        if arm=="hold":
                            estimated = np.full(n,d0)
                        else:
                            x = a.action_features(g,prefix,data["effect"][sl][j:j+1],arm)[0,k]
                            estimated = d0+reconstruction(models["action"][arm][fold],x)[:n]
                        for factor,z,dv in zip(factors,(s0,z0,s0),(estimated,estimated,delta),strict=True):
                            t0,t1 = q.native.astype(float).copy(),q.native.astype(float).copy()
                            t0[off:off+n],t1[off:off+n] = z,z+dv
                            base = first_alarm(q.times,t0,True,q.down,b.TAU)
                            action = first_alarm(q.times,t1,True,q.down,b.TAU)
                            if action["timely"] and not base["timely"]:
                                rescue[arm][factor].add(int(q.sid))
            assert candidates==records["writable"][name][v]
            for arm in arms:
                for factor in factors:
                    cell = records["report"][arm][name]["first"][factor][v]
                    assert sorted(actual)==cell["true_benefit"]["late"],(v,name,actual)
                    assert sorted(rescue[arm][factor])==cell["kept"]["late"],(v,name,arm,factor)
            evidence[v][name] = dict(actions=candidates,late_true=sorted(actual),late_witnesses=events,
                                    late_recount={arm:{k:sorted(x) for k,x in y.items()} for arm,y in rescue.items()})
    return evidence


def run(out):
    h = Hashes()
    root = ROOT/"patch_utility_frame_margin_20261106_v1"
    result = b.r.read(root/"result.json")
    assert h(root/"fits.json")==result["fit_sha256"]
    tapes,ids,_ = m.load(ROOT/"patch_utility_bank_20261106_v1",ROOT/"patch_utility_target_views_20261106_v1",
                         ROOT/"patch_utility_selection_20261106_v1/result.json",h)
    summary = check(tapes,ids,deterministic_folds(ids),result,b.r.read(root/"fits.json"))
    if out is not None:
        b.r.write_new(out,dict(status="INDEPENDENT_WHOLE_CURVE_TIMELY_RECOUNT",
                                source_result_sha256=h(root/"result.json"),views=summary,
                                new_val_test_east_reads=0))
    print({v:{k:x["late_witnesses"] for k,x in y.items()} for v,y in summary.items()},flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--write-receipt",type=Path)
    run(cli.parse_args().write_receipt)
