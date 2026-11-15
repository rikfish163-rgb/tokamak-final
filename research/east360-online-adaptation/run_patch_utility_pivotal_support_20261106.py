"""Check exact first-crossing payoff factorization and its TRAIN support."""

import argparse
from pathlib import Path
import time

import numpy as np

import run_patch_utility_shadow_response_20261106 as s
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import deterministic_folds, menu_analysis


b, m = s.b, s.m
WIDTH, DELAY, TAIL = 32, 8, 24
PLAN = b.r.ROOT / "PATCH_UTILITY_PIVOTAL_SUPPORT_PROTOCOL_20261106.md"
STRATA = ("all_complete", "no_prior_complete", "writable_complete")
SUPPORT = ("any", "original_positive_window", "positive_pivotal",
           "positive_pivotal_in_original_window", "positive_pivotal_static_not_timely",
           "negative_pivotal", "negative_pivotal_static_no_alarm")


def payoff(native, times, disruptive, down, horizon, clock, tau):
    """Offline value of a local first crossing, independent of action identity."""
    native, times = np.asarray(native, np.float64), np.asarray(times, np.float64)
    clock = np.asarray(clock)
    if (native.ndim != 1 or not len(native) or times.shape != native.shape
            or clock.ndim != 1 or clock.dtype.kind not in "iu"
            or (clock < 0).any() or (clock >= len(native)).any()):
        raise ValueError("inconsistent original shot clock")
    n = len(times)
    lead = (down-times)*1000 if disruptive else np.zeros(n)
    eligible = lead >= -1e-6 if disruptive else np.ones(n, dtype=bool)
    rounded = np.round(lead, 9)
    timely = rounded > 10
    label = timely & (rounded <= horizon + 1e-6)
    hits = (native >= tau) & eligible
    prefix = np.maximum.accumulate(hits)[np.minimum(clock+DELAY,n-1)]
    decisive_suffix = np.r_[np.maximum.accumulate((hits & timely if disruptive else hits)[::-1])[::-1], False]
    pivotal = ~prefix & ~decisive_suffix[np.minimum(clock+WIDTH+1,n)]
    grid = clock[:,None]+np.arange(DELAY+1,WIDTH+1)
    actual = grid < n
    at = np.minimum(grid,n-1)
    valid = actual & eligible[at]
    band = valid & label[at] if disruptive else np.zeros_like(valid)
    signed = pivotal[:,None] & valid & (timely[at] if disruptive else True)
    reward = signed.astype(np.int8)*(1 if disruptive else -1)
    return reward, valid, band, prefix, actual.all(axis=1)


def effect(curves, reward, valid, tau):
    """Difference of payoff at the two local first crossings; -1 is no hit."""
    if curves.ndim != 3 or curves.shape[::2] != reward.shape or reward.shape != valid.shape:
        raise ValueError("misaligned local payoff and curves")
    crossing = (curves.astype(np.float64) >= tau) & valid[:,None,:]
    first = np.where(crossing.any(axis=2),crossing.argmax(axis=2),-1)
    taken = np.where(first>=0,np.take_along_axis(reward,np.maximum(first,0),axis=1),0)
    return (taken-taken[:,:1]).astype(np.int8), first


def score_view(tape, ids, fold):
    d = tape.data
    support = {key:{metric:{} for metric in SUPPORT} for key in STRATA}
    witness, seen, exact_actions = [], 0, 0
    for i,q in enumerate(tape.shots):
        sl = q.blocks
        clock, c, old = d["clock"][sl], d["curves"][sl], tape.allowed[sl]
        reward, valid, band, prefix, full = payoff(
            q.native,q.times,q.disruptive,q.down,q.horizon,clock,b.TAU,
        )
        allowed = s.commit_mask(old,c[:,:,:DELAY],clock+DELAY<len(q.native)-1,
                                d["risk"][sl,8],b.TAU)
        elapsed = ~prefix
        writable = allowed[:,1:].any(axis=1)
        if (writable & ~elapsed).any():
            raise ValueError("write opened after an eligible static crossing")
        for name,mask in zip(STRATA,(full,full & elapsed,full & writable),strict=True):
            block = {
                "any": mask,
                "original_positive_window": mask & band.any(axis=1),
                "positive_pivotal": mask & (reward>0).any(axis=1),
                "positive_pivotal_in_original_window": mask & ((reward>0)&band).any(axis=1),
                "positive_pivotal_static_not_timely": (
                    mask & (reward>0).any(axis=1) & (not q.teacher["baseline"]["timely"])
                ),
                "negative_pivotal": mask & (reward<0).any(axis=1),
                "negative_pivotal_static_no_alarm": (
                    mask & (reward<0).any(axis=1) & (not q.teacher["baseline"]["false_alarm"])
                ),
            }
            for key,y in block.items():
                if y.any():
                    support[name][key][q.sid] = int(y.sum())
        predicted, first = effect(c[:,:,DELAY:],reward,valid,b.TAU)
        if len(clock):
            tail = c.copy()
            tail[:,:,:DELAY] = c[:,:1,:DELAY]
            old_score = menu_analysis(q,clock,tail,allowed,b.TAU)
            if not np.array_equal(np.where(allowed,predicted,0),old_score["hard"]):
                raise ValueError(f"payoff factorization differs from original first alarm for {q.sid}")
        seen += len(clock)
        exact_actions += int(allowed[:,1:].sum())
        if q.disruptive:
            for j,a in np.argwhere(allowed & (predicted>0)):
                at = first[j,a]
                sample = int(clock[j]+DELAY+1+at)
                lead = round(float((q.down-q.times[sample])*1000),9)
                if at < 0 or lead <= 10:
                    raise ValueError("an actual rescue must have a timely local first crossing")
                witness.append(dict(shot=int(q.sid),fold=int(fold[i]),clock=int(clock[j]),
                                    action=int(a),sample=sample,lead_ms=lead,
                                    source_horizon_ms=float(q.horizon),in_original_window=bool(band[j,at]),
                                    block_has_original_window=bool(band[j].any()),complete=bool(full[j])))
    index = dict(zip(ids.tolist(),fold.tolist(),strict=True))
    def counts(shots):
        return dict(shots=sorted(shots),blocks=sum(shots.values()),
                    folds=[sum(index[sid]==k for sid in shots) for k in range(5)])
    return dict(
        support={k:{n:counts(q) for n,q in x.items()} for k,x in support.items()},
        rescue_actions=witness, checked_release_blocks=seen,checked_nonidentity_actions=exact_actions,
    )


def run(native,masked,baseline,proof,out):
    if out.exists():
        raise FileExistsError(out)
    start,h = time.monotonic(),Hashes()
    for p in (Path(__file__),Path(s.__file__),PLAN,s.PLAN,proof):
        h(p)
    guard = b.r.read(proof)
    if not guard["commit_guard_gate"] or guard["commit_plan_sha256"]!=h(s.PLAN):
        raise ValueError("cannot change the audited write boundary")
    tapes,ids,_ = m.load(native,masked,baseline,h)
    folds = deterministic_folds(ids)
    views = {}
    for v in m.VIEWS:
        views[v] = score_view(tapes[v],ids,folds)
        checked = views[v]["checked_nonidentity_actions"]
        if checked != guard["views"][v]["action_counts"]["protected"]:
            raise ValueError("not the same action menu as the eight-frame proof")
        print(v,checked,views[v]["support"]["writable_complete"]["positive_pivotal"]["folds"],
              len(views[v]["rescue_actions"]),flush=True)
    h.check()
    result = dict(status="EXACT_OFFLINE_FIRST_ALARM_FACTOR_AND_SOURCE_SUPPORT",
                  views=views, source_shots=len(ids),new_val_test_east_reads=0,
                  input_sha256={str(p): q[1] for p,q in h.cache.items()},
                  policy_fitted=False,first_alarm_policy_replays=0,elapsed_seconds=time.monotonic()-start)
    b.r.write_new(out,result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("native","masked","native_result","guard_proof","output"):
        p.add_argument(k,type=Path)
    q = p.parse_args()
    run(q.native,q.masked,q.native_result,q.guard_proof,q.output)
