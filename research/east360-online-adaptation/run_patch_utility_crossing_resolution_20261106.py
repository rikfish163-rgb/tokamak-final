"""Isolate temporal resolution using the actual future static curve."""

import argparse
from pathlib import Path
import time

import numpy as np

import run_patch_utility_shadow_effect_20261106 as effect
import run_patch_utility_pivotal_support_20261106 as p
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import deterministic_folds


b,m,s = p.b,p.m,p.s
PLAN = b.r.ROOT / "PATCH_UTILITY_CROSSING_RESOLUTION_PROTOCOL_20261106.md"
NAMES = ("exact_frame","exact_bin_mean","learned_real","learned_reverse","hold")


def exact_mean(z):
    a = z.astype(np.float64)
    d = a-a[:,:1]
    return d.reshape(*d.shape[:2],3,8).mean(axis=-1)


def proxy(static,shift):
    if static.ndim!=2 or static.shape[1]!=24 or shift.shape!=(len(static),shift.shape[1],3):
        raise ValueError("static 24 frames and per-action three segment means required")
    z = static[:,None,:]+np.repeat(shift,8,axis=-1)
    z[:,0] = static
    return z


def one_view(name,tape,ids,flags,folds,models,known,dense,receipt):
    d = tape.data
    c = d["curves"].astype(np.float64)
    clock = d["clock"]
    full = np.concatenate([clock[q.blocks]+32<len(q.native) for q in tape.shots])
    allowed = s.commit_mask(tape.allowed,c[:,:,:8],full,d["risk"][:,8],b.TAU)
    allowed[:,0] = False
    if int(allowed.sum())!=receipt["available_actions"][name]:
        raise ValueError("dense forecast and resolution audit have different windows")
    real = c[:,:,8:]
    last = c[:,:,7]-c[:,[0],7]
    mean = exact_mean(real)
    base = s.sel.features(d,"risk")
    proxies = dict(exact_frame=real,exact_bin_mean=proxy(real[:,0],mean),
                   hold=proxy(real[:,0],np.repeat(last[...,None],3,axis=-1)))
    for arm in ("real","reverse"):
        x = s.design(base,c[:,:,:8],arm,b.TAU)
        y = np.empty((*last.shape,3))
        for k,head in enumerate(models[arm]):
            rows = folds[tape.owner]==k
            y[rows] = effect.predict(head,x[rows])
        if arm=="real" and not np.allclose(y[allowed],dense[f"real_prediction__{name}"],rtol=0,atol=1e-10):
            raise ValueError("stored shadow effect no longer reproduces")
        proxies[f"learned_{arm}"] = proxy(real[:,0],last[...,None]+y)
    truth = np.zeros(allowed.shape,dtype=np.int8)
    scores = {k:np.zeros_like(truth) for k in NAMES}
    witnesses = []
    for q in tape.shots:
        sl = q.blocks
        g,valid,*_ = p.payoff(q.native,q.times,q.disruptive,q.down,q.horizon,clock[sl],b.TAU)
        f = {}
        for k in NAMES:
            scores[k][sl],f[k] = p.effect(proxies[k][sl],g,valid,b.TAU)
        truth[sl] = scores["exact_frame"][sl]
        if q.disruptive:
            for j,a in np.argwhere(allowed[sl]&(truth[sl]>0)):
                idx = int(sl.start+j)
                frame = int(f["exact_frame"][j,a])
                c0 = int(clock[idx])
                rec = dict(
                    shot=int(q.sid),block=int(idx),commit=c0+8,action=int(a),offset=frame+1,
                    observed_cross_logit=float(real[idx,a,frame]),
                    static_at_cross_logit=float(real[idx,0,frame]),threshold=float(b.TAU),
                    exact_delta_at_cross=float(real[idx,a,frame]-real[idx,0,frame]),
                    exact_bin_delta=float(mean[idx,a,frame//8]),
                    alternatives={k:dict(kept=bool(scores[k][idx,a]>0),
                                         first_offset=int(f[k][j,a]+1),
                                         at_true_cross_logit=float(proxies[k][idx,a,frame]))
                                  for k in NAMES},
                )
                witnesses.append(rec)
    if {q["shot"] for q in witnesses}!={q["shot"] for q in known["rescue_actions"] if q["complete"]}:
        raise ValueError("true curve full-support recoveries do not match sealed earlier result")
    shot = ids[tape.owner]
    disruptive = flags[tape.owner]
    def compact(mask):
        which = mask.any(axis=1)
        return dict(actions=int(mask.sum()),shots=np.unique(shot[which]).tolist(),
                    late=np.unique(shot[which&disruptive]).tolist())
    report = {}
    for k,y in scores.items():
        plus,minus = allowed & (y>0),allowed & (y<0)
        report[k] = dict(benefit=compact(plus),harm=compact(minus),
                         benefit_same_action=compact(plus&(truth>0)),harm_same_action=compact(minus&(truth<0)),
                         invented_benefit=compact(plus&(truth<=0)),invented_harm=compact(minus&(truth>=0)))
    return dict(witnesses=witnesses,report=report,writable_actions=int(allowed.sum()))


def run(native,masked,baseline,support,dense,out):
    if out.exists():
        raise FileExistsError(out)
    t0,h = time.monotonic(),Hashes()
    rec = b.r.read(dense/"result.json")
    check = b.r.read(dense/"verification.json")
    for path in (Path(__file__),Path(effect.__file__),PLAN,support,dense/"verification.json"):
        h(path)
    if (rec["fits_sha256"]!=h(dense/"fits.json") or rec["errors_sha256"]!=h(dense/"held_errors.npz")
            or check["source_result_sha256"]!=h(dense/"result.json")
            or rec["first_alarm_replays"] or rec["new_val_test_east_reads"]):
        raise ValueError("missing independently verified dense forecast")
    proof = b.r.read(support)
    if proof["status"]!="EXACT_OFFLINE_FIRST_ALARM_FACTOR_AND_SOURCE_SUPPORT":
        raise ValueError("missing first alarm payoff support")
    tapes,ids,flags = m.load(native,masked,baseline,h)
    folds = deterministic_folds(ids)
    models = b.r.read(dense/"fits.json")["fits"]
    with np.load(dense/"held_errors.npz",allow_pickle=False) as old:
        if not np.array_equal(old["ids"],ids) or not np.array_equal(old["folds"],folds):
            raise ValueError("dense predictor fold mismatch")
        result = {}
        for name in m.VIEWS:
            result[name] = one_view(name,tapes[name],ids,flags,folds,models,proof["views"][name],old,rec)
            print(name,{k:x["benefit_same_action"]["late"] for k,x in result[name]["report"].items()},flush=True)
    h.check()
    b.r.write_new(out,dict(
        status="ORACLE_FUTURE_STATIC_RESOLUTION_DIAGNOSIS_NOT_ONLINE",
        views=result,new_val_test_east_reads=0,new_fits=0,first_alarm_policy_replays=0,
        input_sha256={str(path): v[1] for path,v in h.cache.items()},
        elapsed_seconds=time.monotonic()-t0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("native","masked","baseline","support","dense","output"):
        cli.add_argument(key,type=Path)
    args = cli.parse_args()
    run(args.native,args.masked,args.baseline,args.support,args.dense,args.output)
