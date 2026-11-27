"""Two clocks: within-shot measured forecast innovations, between-shot outcome credit. EAST360 only."""
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp, softmax

from intershot360 import ORDER_HASH, SOURCE_HASH, paired, scored, summary
from shot_memory import contract, digest, pack
from zero_cold_start import backbone, data, engine, experiments


OUT = Path(__file__).with_suffix('.json')
BRANCHES = ('parent', 'multi', 'single', 'frozen_fast', 'bias', 'sham')


def divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def innovations(times, values, mask, patch=8, max_gap_ms=2.):
    """For each contiguous valid run, replay the issued one-block-ahead slope forecasts.

    For target block k, v is delta[k-1], u is delta[k]. The cumsum[:-1]
    gives the fit at issue (end of k-1). The residual is released in block k+1.
    """
    n, d = values.shape
    out, persist = np.zeros((n, d)), np.zeros((n, d))
    ready, block = np.zeros((n, d), bool), np.zeros(n, int)
    acc = np.zeros(4)  # matured forecasts, nonzero issued slopes, two comparable absolute losses
    for seg in np.split(np.arange(n), np.flatnonzero(np.diff(times)*1000 > max_gap_ms + 1e-7) + 1):
        m = len(seg); full = m // patch
        block[seg] = seg[0] + np.arange(m)//patch
        if full < 3:
            continue
        ids = seg[:full*patch].reshape(-1, patch)
        valid = mask[ids].all(axis=1) & np.isfinite(values[ids]).all(axis=1)
        means = np.where(mask[ids], values[ids], 0.).mean(axis=1, dtype=float)
        amp, frozen, active = np.zeros((full+1,d)), np.zeros((full+1,d)), np.zeros((full+1,d), bool)
        for j in range(d):
            edges = np.flatnonzero(np.diff(np.r_[False, valid[:,j], False]))
            for lo, hi in edges.reshape(-1,2):
                if hi-lo < 3:
                    continue
                delta = np.diff(means[lo:hi,j]); v, u = delta[:-1], delta[1:]
                vv = np.cumsum(v*v); vu = np.cumsum(v*u)
                coef = np.r_[0., np.clip(divide(vu[:-1], vv[:-1]), 0., 1.)]
                rms = np.sqrt(vv / np.arange(1, len(v)+1))  # known with v, before u
                err = u - coef*v; dest = slice(lo+3, hi+1)
                amp[dest,j] = divide(err, abs(err)+rms)
                frozen[dest,j] = divide(u, abs(u)+rms)
                active[dest,j] = True
                denom = abs(u)+abs(v)+rms
                acc += (len(v), np.count_nonzero(coef), divide(abs(err),denom).sum(), divide(abs(u),denom).sum())
        b = np.arange(m)//patch
        # The current block's future missing points must not suppress its earlier evidence.
        alive = np.zeros(((full+1)*patch, d), bool)
        alive[:m] = mask[seg] & np.isfinite(values[seg])
        alive = np.logical_and.accumulate(alive.reshape(-1,patch,d),axis=1).reshape(-1,d)[:m]
        ready[seg] = active[b] & alive
        out[seg] = amp[b] * ready[seg]; persist[seg] = frozen[b] * ready[seg]
    for x in (out, persist):
        x /= np.maximum(1., np.linalg.norm(x,axis=1,keepdims=True))
    return out, persist, ready.any(axis=1)[:,None].astype(float), block, acc


def candidates(t, block, margin, eligible, k, separation_ms):
    idx = np.flatnonzero(eligible); choices = []
    while len(idx) and len(choices)<k:
        i = idx[np.argmax(margin[idx])]; choices.append(int(i))
        idx = idx[(block[idx] != block[i]) & (abs(t[idx]-t[i])*1000. >= separation_ms-1e-7)]
    return choices


class OutcomeMemory:
    """Every observed discharge has one class vote; w stays unchanged during its prediction."""
    def __init__(self, dim):
        self.w = np.zeros(dim); self.pos = []; self.neg = []; self.tp_anchors = []
        self.stats = Counter(); self.release = 0

    def observe(self, x, margin, weak, parent, own, down, first, parent_first):
        if down is not None:
            if parent == 'TP_timely':
                take = [parent_first]; self.tp_anchors.append((margin[parent_first],x[parent_first].copy()))
            elif own == 'TP_timely':
                take = [first]  # preserve a real timely success, not manufactured labels on its weak bag
            else:
                take = weak; self.stats['failure_bags'] += 1; self.stats['witnesses'] += len(take)
                if not take: self.stats['empty'] += 1
            self.pos.append((margin[take], x[take]))
        else:
            # ||x||, ||w|| <= 1. No omitted point can cross for ANY w in the family.
            keep = np.flatnonzero(margin >= -1.-1e-12)
            issued = None if first is None else int(np.searchsorted(keep, first))
            if issued is not None: assert keep[issued] == first
            self.neg.append((margin[keep], x[keep], issued, parent=='TN'))
        if not self.pos or not self.neg:
            return
        self.release += 1
        dgrad, ngrad = np.zeros_like(self.w), np.zeros_like(self.w)
        for m, f in self.pos:
            if not len(m): continue
            s = m + f@self.w
            if s.max() < 0.:
                bag = logsumexp(s)-np.log(len(s))
                dgrad += expit(-bag) * (softmax(s) @ f)
        for m, f, issued, _ in self.neg:
            s = m + f@self.w; hit = np.flatnonzero(s>=0.)
            if not len(hit): continue
            firsts = [hit[0]]
            if issued is not None and issued != hit[0] and s[issued]>=0.: firsts.append(issued)
            top = np.argmax(s)
            ngrad += .5*((expit(s[firsts])[:,None]*f[firsts]).mean(0) + expit(s[top])*f[top])
        grad = .5*(dgrad/len(self.pos) - ngrad/len(self.neg))
        w = self.w + grad/np.sqrt(self.release)
        w /= max(1., np.linalg.norm(w))
        if not self.safe(w):
            self.stats['guard_shrank'] += 1
            for _ in range(48):
                w *= .5
                if self.safe(w): break
            else: w[:] = 0.
        assert self.safe(w) and np.linalg.norm(w)<=1.+1e-12
        self.stats['changed'] += int(not np.array_equal(self.w,w))
        self.w = w

    def safe(self, w):
        if any(m + f@w < 0 for m, f in self.tp_anchors): return False
        return not any(np.any(m + f@w >= 0.) for m,f,_,parent_tn in self.neg if parent_tn)

    def historical_only(self):
        return dict(normal_fp=sum(int(np.any(m+f@self.w>=0.)) for m,f,_,_ in self.neg),
                    positive_bags_or_anchors_crossed=sum(int(np.any(m+f@self.w>=0.)) for m,f in self.pos),
                    past_parent_guard_ok=self.safe(self.w), normal=len(self.neg), disruptive=len(self.pos))


def issue(pred, z, x, w):
    delta = x@w
    assert np.isfinite(delta).all() and np.max(abs(delta),initial=0.)<=1.+1e-12
    p = np.where(delta==0., pred.risk_curve, engine._sigmoid(z + delta))
    hits = np.flatnonzero(p >= pred.threshold_curve)  # pre-registered hold=1 sample
    i = int(hits[0]) if len(hits) else None
    return p, i, delta


def example(sid, t, down, selected, pred, margin, x, w):
    a = margin[selected] + x[selected]@w
    return dict(shot=sid, weight_before=w.tolist(), witnesses=[dict(time=float(t[i]),
        lead_ms=round((down-t[i])*1000.,6), parent_score=float(pred.risk_curve[i]),
        shared_threshold=float(pred.threshold_curve[i]), margin_logit=float(margin[i]),
        candidate_margin_logit=float(v), soft_credit=float(s)) for i,v,s in zip(selected,a,softmax(a))])


def run():
    ids = data._east360(); assert sha256(','.join(map(str,ids)).encode()).hexdigest()==ORDER_HASH
    assert digest(data.ARTIFACTS/'jtext_source.pt') == SOURCE_HASH
    model, ckpt = backbone.load_source_model(); cfg = experiments.locked_configs(float(ckpt['val_threshold_fpr5']))['COH6_Reflective_M100']
    assert cfg.hold_ms == 1 and model.cfg['patch']==8 and cfg.min_lead_ms == 30.
    sc, ss = experiments.source_coords(ckpt, cfg.align_mode); state=engine.InterShotState.from_source(sc,ss,cfg)
    cols = list(cfg.align_channels); k = len(cols); assert k==6
    models = {n:OutcomeMemory(1 if n=='bias' else k) for n in ('multi','single','frozen_fast','bias')}
    direction = np.random.default_rng(20261126).normal(size=k); direction /= np.linalg.norm(direction)
    rows, dose, ledger = defaultdict(list), defaultdict(list), []
    frozen, traces, reach, examples = {}, [], Counter(), {}
    pred_error, counts = [], np.zeros(2)
    for rank,sid in enumerate(ids,1):
        sig = data.read_signals('EAST',sid); t,v,a = (sig[j] for j in ('times','values','available'))
        pred = engine.run_shot_causal(sid,t,v,a,lambda x,m:backbone.predict_shot_proba(model,x,m),state,cfg,sc,ss)
        fast, static, ready, block, forecast = innovations(t,v[:,cols],a[:,cols],model.cfg['patch'],cfg.max_gap_ms)
        counts += forecast[:2]
        if forecast[0]: pred_error.append(forecast[2:]/forecast[0])
        z = engine._logit(pred.risk_curve,cfg); m = z-engine._logit(pred.threshold_curve,cfg)
        W = dict(parent=np.zeros(k), **{n:c.w.copy() for n,c in models.items()}, sham=direction*np.linalg.norm(models['multi'].w))
        if rank>270: W.update(frozen)
        X = {n:(static if n=='frozen_fast' else ready if n=='bias' else fast) for n in W}
        signals, ix, firsts = {}, {}, {}; seal=sha256(pred.threshold_curve.tobytes())
        for n,w in W.items():
            p, i, delta = issue(pred,z,X[n],w); signals[n], ix[n] = p, i
            firsts[n] = None if i is None else float(t[i])
            seal.update(n.encode()); seal.update(w.tobytes()); seal.update(p.tobytes())
            dose[n].append([float(delta.mean()),float(np.mean(delta**2)),float(np.max(abs(delta)))])
        assert np.array_equal(signals['parent'], pred.risk_curve) and firsts['parent']==pred.alarm_time
        assert np.array_equal(issue(pred,z,static,W['parent'])[0], pred.risk_curve)  # other zero cell
        ledger.append([sid,seal.hexdigest(),firsts]); traces.append(W['multi'].tolist())
        y, down = data.read_outcome('EAST',sid)  # all branches and first alarms already issued
        ledger[-1] += [y, down]
        for n,p in signals.items(): rows[n].append(scored(replace(pred,risk_curve=p),pred.threshold_curve,firsts[n],y,down,cfg))
        outcome = rows['parent'][-1]['outcome']; wk = []
        if y:
            lead = np.round((down-t)*1000.,6); timely = lead>cfg.min_lead_ms; weak = timely & (lead<=125.)
            wk = candidates(t,block,m,weak,3,2*model.cfg['patch']*cfg.sample_ms)
            if outcome!='TP_timely':
                bnd = m+np.linalg.norm(fast,axis=1)
                can = bool(np.any(bnd[timely]>=0.)); inside = bool(np.any(bnd[weak]>=0.))
                reach['parent_not_timely'] += 1; reach['reachable_any_timely'] += int(can); reach['reachable_in_weak_window'] += int(inside)
                reach['reachable_selected'] += int(np.any(bnd[wk]>=0.))
                if wk:
                    for tag, ok in (('first_eligible_parent_error',True),('first_reachable_weak_parent_error',inside)):
                        if ok and tag not in examples: examples[tag]=example(sid,t,down,wk,pred,m,fast,W['multi'])
            if rows['multi'][-1]['outcome'] != 'TP_timely': reach['multi_not_timely'] += 1
            if np.any(signals['multi'][weak]>=pred.threshold_curve[weak]): reach['multi_online_crosses_in_weak_window'] += 1
        elif outcome=='FP':
            reach['parent_fp'] += 1; reach['parent_fp_forced_by_a_single_point'] += int(np.any(m-np.linalg.norm(fast,axis=1)>=0.))
        for n,c in models.items():
            c.observe(X[n],m,wk[:1] if n=='single' else wk,outcome,rows[n][-1]['outcome'],down,ix[n],ix['parent'])
        engine.reflect_and_update(pred,v,a,y,down,state,cfg,sc,ss)
        if rank in (90,180,270):
            f = models['multi'].w.copy(); f.flags.writeable=False; frozen[f'frozen_{rank}']=f
        if rank%90==0: print(f'{rank}/360 predictions sealed',flush=True)

    archive = json.loads(Path(__file__).with_name('intershot360.json').read_text())  # only after last reveal
    pidx=archive['arms'].index('old_joint')
    assert all((a[0],a[2]['parent'],a[3],a[4])==(b[0],b[2][pidx],b[4],b[5]) for a,b in zip(ledger,archive['ledger'],strict=True))
    assert pack(state)==archive['checkpoint_states']['360']['old_joint']
    actual={n:summary(rows[n]) for n in BRANCHES}; assert (actual['parent']['tp_timely'],actual['parent']['fp'])==(135,31)
    freport=np.asarray(pred_error)
    return dict(contract=contract(cfg),source_code_sha256=digest(__file__),order_sha256=ORDER_HASH,features=[data.RAW23[j] for j in cols],
        online=actual,blocks90=[{n:summary(rows[n][i:i+90]) for n in BRANCHES} for i in range(0,360,90)],
        last90={n:summary(r[-90:]) for n,r in rows.items()},paired={n+'_to_multi':paired(rows[n],rows['multi']) for n in BRANCHES if n!='multi'},
        last90_paired={n+'_to_multi':paired(rows[n][-90:],rows['multi'][-90:]) for n in ('parent',*frozen)},
        within_shot_forecast=dict(matured_channel_forecasts=int(counts[0]),issued_nonzero_slopes=int(counts[1]),
            equal_shot_mean_normalized_mae={'fitted':float(freport[:,0].mean()),'persistence':float(freport[:,1].mean())},
            fitted_lower_shots=int((freport[:,0]<freport[:,1]).sum()),fitted_higher_shots=int((freport[:,0]>freport[:,1]).sum()),shots=len(freport)),
        reachability_diagnostic=dict(reach),weak_examples=examples,sham_direction=direction.tolist(),
        correction_dose={n:dict(mean=float(np.mean(d,axis=0)[0]),rms=float(np.sqrt(np.mean(d,axis=0)[1])),max=float(np.max(d,axis=0)[2])) for n,d in dose.items()},
        learning={n:dict(stats=dict(c.stats),release_opportunities=c.release,final_w=c.w.tolist(),final_in_sample_only=c.historical_only()) for n,c in models.items()},
        frozen_w={n:w.tolist() for n,w in frozen.items()},issued_multi_weights=traces,
        ledger_columns=['shot','pre_reveal_digest','issued_first_seconds','IsDisrupt','DownTime_if_disruptive'],ledger=ledger)


if __name__=='__main__':
    result = run(); tmp=OUT.with_suffix('.pending'); tmp.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'); tmp.replace(OUT)
    for arm,s in result['online'].items(): print(arm,s['tp_timely'],s['fp'])
