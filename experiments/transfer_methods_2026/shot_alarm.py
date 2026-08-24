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
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import grid_search as GS
import advanced_time_labels as ATL

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = 128
EAST_ADVANCED_TIME = GS.ROOT / "shot_list/EAST/AdvancedTime_EAST.json"

def source_checkpoint_path(seed=131):
    return GS.default_source_checkpoint(seed)


def _source_contract(seed, source_checkpoint=None):
    path = Path(source_checkpoint or source_checkpoint_path(seed)).expanduser().resolve()
    _enc, _head, checkpoint = GS.load_source_checkpoint(
        path, channels=len(GS.RH.REBUILD), expected_seed=seed, device="cpu"
    )
    center = checkpoint["scaler_center"].cpu().numpy()
    scale = checkpoint["scaler_scale"].cpu().numpy()
    return path, center, scale, checkpoint


def train(seed=131, source_checkpoint=None):
    source_path, med, sc, _checkpoint = _source_contract(seed, source_checkpoint)
    tr, te = GS.load_east(include_test=True)
    tr = GS.transform(tr, "base"); te = GS.transform(te, "base")
    for g in (tr, te): GS.apply_scale(g, med, sc)
    pos, gray = GS.POS["ea125"], GS.GRAY["gray0"]
    ts = GS.build_sampler(tr, None, pos["ea"], gray["ea"], seed)
    channels = len(GS.RH.REBUILD)
    enc, head = GS.train_adapt(
        None, ts, channels, 12, 128, seed, source_checkpoint=source_path
    )
    return enc, head, tr, te


def prepare_east_training_labels(
    shots, *, policy="fixed125", advanced_time_path=EAST_ADVANCED_TIME,
):
    """Attach per-shot horizons while retaining HDF5 as outcome authority."""

    advanced_path = Path(advanced_time_path).expanduser().resolve()
    advanced = (
        ATL.load_advanced_time_ms(advanced_path)
        if policy == "advanced_floor" else None
    )
    prepared, audit = ATL.attach_positive_horizons(
        shots, device="EAST", policy=policy, advanced_time_ms=advanced,
    )
    audit = dict(audit)
    audit.update({
        "outcome_authority": "EAST HDF5 meta/IsDisrupt",
        "down_time_authority": "EAST HDF5 meta/DownTime",
        "advanced_time_role": "positive-window upper boundary only",
        "advanced_time_path": (
            str(advanced_path.relative_to(GS.ROOT))
            if advanced is not None and advanced_path.is_relative_to(GS.ROOT)
            else str(advanced_path) if advanced is not None else None
        ),
        "advanced_time_sha256": (
            ATL.sha256_file(advanced_path) if advanced is not None else None
        ),
        "formula": (
            "upper_ms=max(125, AdvancedTime_ms); missing->125"
            if policy == "advanced_floor" else "upper_ms=125"
        ),
    })
    return prepared, audit


def historical_shot_first_sampler(shots, *, pos_ms, gray_ms, seed):
    """Reproduce the sealed support110 migration sampler exactly.

    The registered EAST migration first samples a shot uniformly and only then
    chooses a positive or negative endpoint inside that shot.  In particular,
    it must not rebalance the 15D/95N support cohort by choosing the class
    before the shot.
    """

    pools = []
    for shot in shots:
        shot_pos_ms = float(shot.get("positive_horizon_ms", pos_ms))
        values = np.asarray(shot["v"])
        times = float(shot["start"]) + np.arange(len(values)) / float(shot["rate"])
        pos_endpoints = []
        neg_endpoints = []
        if bool(shot["is_disrupt"]) and shot.get("down") is not None:
            for endpoint in range(W - 1, len(values)):
                ttd_ms = (float(shot["down"]) - times[endpoint]) * 1000.0
                if 0.0 <= ttd_ms <= shot_pos_ms:
                    pos_endpoints.append(endpoint)
                elif shot_pos_ms + float(gray_ms) < ttd_ms <= 60_000.0:
                    neg_endpoints.append(endpoint)
        else:
            neg_endpoints = list(range(W - 1, len(values)))
        pools.append({"positive": pos_endpoints, "negative": neg_endpoints})
    positive = [pool["positive"] for pool in pools]
    negative = [pool["negative"] for pool in pools]
    generator = np.random.default_rng(int(seed))

    def sample(positive_probability=0.5):
        while True:
            shot_index = int(generator.integers(len(shots)))
            pos_endpoints = positive[shot_index]
            neg_endpoints = negative[shot_index]
            if not pos_endpoints and not neg_endpoints:
                continue
            if pos_endpoints and generator.random() < float(positive_probability):
                endpoints = pos_endpoints
                label = np.float32(1.0)
            else:
                if not neg_endpoints:
                    continue
                endpoints = neg_endpoints
                label = np.float32(0.0)
            endpoint = endpoints[int(generator.integers(len(endpoints)))]
            window = np.asarray(shots[shot_index]["v"])[
                endpoint - W + 1:endpoint + 1
            ]
            return np.ascontiguousarray(window), label

    return sample


def train_support_only(
    seed=131, source_checkpoint=None, *, target_label_policy="fixed125",
    advanced_time_path=EAST_ADVANCED_TIME, adapt_epochs=12,
    adapt_batch_size=128, return_label_audit=False,
):
    """Train the identical model without loading any EAST test shot or label."""
    source_path, med, sc, checkpoint = _source_contract(seed, source_checkpoint)
    selected_source = (
        isinstance(checkpoint.get("input_contract"), dict)
        and checkpoint["input_contract"].get("schema")
        == "jtext-rebuilt17-prefix-causal-v2"
    )
    tr, te = (
        GS.load_east_selected_support()
        if selected_source else GS.load_east(include_test=False)
    )
    if te:
        raise RuntimeError("support-only training unexpectedly loaded test shots")
    tr = GS.transform(tr, "base")
    tr, label_audit = prepare_east_training_labels(
        tr, policy=target_label_policy, advanced_time_path=advanced_time_path,
    )
    for group in (tr,):
        GS.apply_scale(group, med, sc)
    pos, gray = GS.POS["ea125"], GS.GRAY["gray0"]
    ts = historical_shot_first_sampler(
        tr, pos_ms=pos["ea"], gray_ms=gray["ea"], seed=seed,
    )
    channels = len(GS.RH.REBUILD)
    enc, head = GS.train_adapt(
        None, ts, channels, int(adapt_epochs), int(adapt_batch_size), seed,
        source_checkpoint=source_path,
        allow_fallback=not selected_source,
    )
    result = (enc, head, tr, med, sc)
    return result + (label_audit,) if return_label_audit else result

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


def select_tpr_first_threshold(scores, labels, *, target_tpr=0.8):
    """Existing EAST110 rule: meet TPR first, then minimize FPR."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    if not labels.any() or labels.all():
        raise ValueError("threshold selection requires both outcomes")
    best = None
    for threshold in sorted(set(scores.tolist()), reverse=True) + [-1e9]:
        tpr = float(np.mean(scores[labels] >= threshold))
        fpr = float(np.mean(scores[~labels] >= threshold))
        if tpr < float(target_tpr):
            continue
        candidate = (fpr, -float(threshold), -tpr)
        if best is None or candidate < best[0]:
            best = (candidate, float(threshold), tpr, fpr)
    if best is None:
        raise RuntimeError(f"support cannot reach target TPR={target_tpr}")
    return {"threshold": best[1], "tpr": best[2], "fpr": best[3]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, nargs="?", default=131)
    parser.add_argument("--source-checkpoint", type=Path)
    args = parser.parse_args()
    enc, head, tr, te = train(args.seed, args.source_checkpoint)
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
        bt = select_tpr_first_threshold(s_tr, y_tr, target_tpr=0.8)
        tp,fp=eval_at(s_te,bt["threshold"],y_te)
        rec["prereg"]=dict(train_tpr=round(bt["tpr"],4), train_fpr=round(bt["fpr"],4),
                           test_tpr=round(tp/(y_te.sum()),4), test_fpr=round(fp/(len(y_te)-y_te.sum()),4), th=round(bt["threshold"],5))
        out[k]=rec
    print(json.dumps(out, indent=1, ensure_ascii=False))
    json.dump(out, open("shot_alarm.json","w"), indent=1)

if __name__=="__main__":
    main()
