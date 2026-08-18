"""Per-shot binary alarm: each shot is judged 报警 / 不报警.

Uses the 110-shot adapted model (rebuilt17 base + ea125 + adapt_high) and
combines single-point and sustained-over-threshold evidence into ONE per-shot
score, then one threshold => binary alarm. No window/candidate restrictions:
sliding 128ms risk over the full shot feeds the aggregator.

Aggregators:
  max     : max risk
  p90     : 90th percentile of risk
  top5    : mean of top-5 risk peaks
  area    : mean of max(risk - tau, 0)  (sustained exceedance)
  combo   : 0.6*max + 0.4*mean_top5
"""
import json, sys
import numpy as np, torch
import grid_search as GS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = 128

def train(seed=131):
    src, val, tr, te = GS.load_all()
    src = GS.transform(src, "base"); tr = GS.transform(tr, "base"); te = GS.transform(te, "base")
    med, sc = GS.fit_scale(src)
    for g in (src, tr, te): GS.apply_scale(g, med, sc)
    pos, gray = GS.POS["ea125"], GS.GRAY["gray0"]
    ss = GS.build_sampler(
        src, None, pos["jt"], gray["jt"], seed,
        minimum_pos_ms=10.0, allow_left_padding=True,
        context_lengths=(32, 64, 96, 128),
    )
    ts = GS.build_sampler(
        tr, None, pos["ea"], gray["ea"], seed,
        minimum_pos_ms=30.0, allow_left_padding=True,
        context_lengths=(32, 64, 96, 128),
    )
    C = src[0]["v"].shape[1]
    enc, head = GS.train_adapt(
        ss, ts, C, 12, 128, seed, source_epochs=96,
    )
    return enc, head, tr, te


def train_support_only(seed=131):
    """Train the identical model without loading any EAST test shot or label."""
    src, val, tr, te = GS.load_all(include_test=False)
    if te:
        raise RuntimeError("support-only training unexpectedly loaded test shots")
    src = GS.transform(src, "base"); tr = GS.transform(tr, "base")
    med, sc = GS.fit_scale(src)
    for group in (src, tr):
        GS.apply_scale(group, med, sc)
    pos, gray = GS.POS["ea125"], GS.GRAY["gray0"]
    ss = GS.build_sampler(
        src, None, pos["jt"], gray["jt"], seed,
        minimum_pos_ms=10.0, allow_left_padding=True,
        context_lengths=(32, 64, 96, 128),
    )
    ts = GS.build_sampler(
        tr, None, pos["ea"], gray["ea"], seed,
        minimum_pos_ms=30.0, allow_left_padding=True,
        context_lengths=(32, 64, 96, 128),
    )
    channels = src[0]["v"].shape[1]
    enc, head = GS.train_adapt(
        ss, ts, channels, 12, 128, seed, source_epochs=96,
    )
    return enc, head, tr, med, sc

def full_risk(enc, head, s, bs=512):
    v=s["v"]; n=len(v)
    if n<W: return np.array([0.0], np.float32)
    idx=np.arange(W-1,n); p=np.zeros(len(idx), np.float32)
    enc.eval(); head.eval()
    for b0 in range(0,len(idx),bs):
        batch=idx[b0:b0+bs]
        xs=np.stack([v[j-W+1:j+1] for j in batch])
        with torch.no_grad():
            p[b0:b0+len(batch)]=torch.sigmoid(head(enc(torch.from_numpy(xs).to(DEV)))).cpu().numpy()
    return p

def agg(p, kind, tau=0.5):
    if p.size==0: return 0.0
    if kind=="max": return float(p.max())
    if kind=="p90": return float(np.percentile(p,90))
    if kind=="top5": return float(np.sort(p)[-min(5,len(p)):].mean())
    if kind=="area": return float(np.clip(p-tau,0,None).mean())
    if kind=="combo": return 0.6*float(p.max()) + 0.4*float(np.sort(p)[-min(5,len(p)):].mean())
    raise ValueError(kind)

def main():
    enc, head, tr, te = train(int(sys.argv[1]) if len(sys.argv)>1 else 131)
    R_tr=[full_risk(enc,head,s) for s in tr]
    R_te=[full_risk(enc,head,s) for s in te]
    kinds=["max","p90","top5","area","combo"]
    out={}
    for k in kinds:
        s_tr=np.array([agg(p,k) for p in R_tr]); s_te=np.array([agg(p,k) for p in R_te])
        y_tr=np.array([1 if s["is_disrupt"] else 0 for s in tr])
        y_te=np.array([1 if s["is_disrupt"] else 0 for s in te])
        def eval_at(x, th, y):
            tp=( (x[y==1]>=th).sum() ); fp=((x[y==0]>=th).sum())
            return tp, fp
        # paper: sweep on test, TPR>=0.8 min FPR
        best=None
        for th in sorted(set(np.round(s_te,5))) + [0.0]:
            tp,fp=eval_at(s_te,th,y_te); tpr=tp/(y_te.sum()); fpr=fp/(len(y_te)-y_te.sum())
            if tpr>=0.8 and (best is None or fpr<best[1]): best=(th,tpr,fpr)
        rec={"tpr0.8_test": dict(tpr=round(best[1],4), fpr=round(best[2],4), th=round(best[0],5)) if best else None}
        # pre-registered: sweep on tr, apply to te
        bt=None
        for th in sorted(set(np.round(s_tr,5))) + [0.0]:
            tp,fp=eval_at(s_tr,th,y_tr); tpr=tp/(y_tr.sum()); fpr=fp/(len(y_tr)-y_tr.sum())
            if tpr>=0.8 and (bt is None or fpr<bt[1]): bt=(th,tpr,fpr)
        if bt:
            tp,fp=eval_at(s_te,bt[0],y_te)
            rec["prereg"]=dict(train_tpr=round(bt[1],4), train_fpr=round(bt[2],4),
                               test_tpr=round(tp/(y_te.sum()),4), test_fpr=round(fp/(len(y_te)-y_te.sum()),4), th=round(bt[0],5))
        out[k]=rec
    print(json.dumps(out, indent=1, ensure_ascii=False))
    json.dump(out, open("shot_alarm.json","w"), indent=1)

if __name__=="__main__":
    main()
