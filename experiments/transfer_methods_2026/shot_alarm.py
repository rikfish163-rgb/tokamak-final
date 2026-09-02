"""Per-shot binary alarm: each shot is judged 报警 / 不报警.

The default CLI is leakage-safe: it trains on EAST support110, selects the
threshold on support only, predicts the complete balanced360 signal stream
without opening outcome metadata, and reveals test outcomes only after all
predictions are sealed. ``--diagnostic-test-tuned`` reproduces the historical
test-tuned sweep, but its output is explicitly marked leakage-prone.

The model combines single-point and sustained-over-threshold evidence into ONE
per-shot score, then one threshold => binary alarm. No window/candidate
restrictions: sliding 128ms risk over the full shot feeds the aggregator.

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
import h5py
import grid_search as GS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = 128

def source_checkpoint_path(seed=131):
    return GS.default_source_checkpoint(seed)


def _source_contract(seed, source_checkpoint=None):
    path = Path(source_checkpoint or source_checkpoint_path(seed)).expanduser().resolve()
    _enc, _head, checkpoint = GS.load_source_checkpoint(
        path, channels=len(GS.RH.REBUILD), expected_seed=seed, device="cpu"
    )
    center = checkpoint["scaler_center"].cpu().numpy()
    scale = checkpoint["scaler_scale"].cpu().numpy()
    return path, center, scale


def train(seed=131, source_checkpoint=None):
    source_path, med, sc = _source_contract(seed, source_checkpoint)
    tr, te = GS.load_east(include_test=True)
    tr = GS.transform(tr, "base"); te = GS.transform(te, "base")
    for g in (tr, te): GS.apply_scale(g, med, sc)
    pos, gray = GS.POS["ea125"], GS.GRAY["gray0"]
    ts = GS.build_sampler(
        tr, None, pos["ea"], gray["ea"], seed, sampling_mode="shot_first"
    )
    channels = len(GS.RH.REBUILD)
    enc, head = GS.train_adapt(
        None, ts, channels, 12, 128, seed, source_checkpoint=source_path
    )
    return enc, head, tr, te


def train_support_only(seed=131, source_checkpoint=None):
    """Train the identical model without loading any EAST test shot or label."""
    source_path, med, sc = _source_contract(seed, source_checkpoint)
    tr, te = GS.load_east(include_test=False)
    if te:
        raise RuntimeError("support-only training unexpectedly loaded test shots")
    tr = GS.transform(tr, "base")
    for group in (tr,):
        GS.apply_scale(group, med, sc)
    pos, gray = GS.POS["ea125"], GS.GRAY["gray0"]
    ts = GS.build_sampler(
        tr, None, pos["ea"], gray["ea"], seed, sampling_mode="shot_first"
    )
    channels = len(GS.RH.REBUILD)
    enc, head = GS.train_adapt(
        None, ts, channels, 12, 128, seed, source_checkpoint=source_path
    )
    return enc, head, tr, med, sc


def load_east_signal_only(shot_id, center, scale):
    """Read one EAST test signal without opening HDF5 outcome metadata."""
    path = GS.RH.DATA / "EAST" / "unified_hdf5" / str(shot_id // 100 * 100) / f"{shot_id}.hdf5"
    with h5py.File(path, "r") as handle:
        arrays = {
            name: np.asarray(handle[f"data/{name}"][:], dtype=np.float32)
            for name in GS.RH.RAW23
        }
        signal = handle["data/ip"]
        rate = float(np.asarray(signal.attrs.get("SampleRate", 1000.0)).reshape(-1)[0])
        start = float(np.asarray(signal.attrs.get("StartTime", 0.0)).reshape(-1)[0])
    values, available = GS.RH.rebuild_features(arrays, "EAST")
    values = np.clip((values - center) / scale, -10.0, 10.0)
    values = np.where(available, values, 0.0).astype(np.float32)
    return {
        "shot": int(shot_id), "v": values, "a": available,
        "start": start, "rate": rate,
    }


def read_east_outcome(shot_id):
    """Reveal one outcome after its complete test prediction has been sealed."""
    path = GS.RH.DATA / "EAST" / "unified_hdf5" / str(shot_id // 100 * 100) / f"{shot_id}.hdf5"
    with h5py.File(path, "r") as handle:
        is_disrupt = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
        down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
    return {
        "is_disrupt": is_disrupt,
        "down_time_s": float(down[0]) if is_disrupt and down.size else None,
    }

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
    parser.add_argument(
        "--diagnostic-test-tuned", action="store_true",
        help=(
            "reproduce the historical test-tuned threshold sweep; this is "
            "explicitly leakage-prone diagnostic output and is never a "
            "registered evaluation"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("shot_alarm.json"),
        help="JSON result path; parent directories are created if needed",
    )
    args = parser.parse_args()
    if args.diagnostic_test_tuned:
        enc, head, tr, te = train(args.seed, args.source_checkpoint)
        evaluation_mode = "diagnostic_test_tuned"
    else:
        enc, head, tr, med, sc = train_support_only(
            args.seed, args.source_checkpoint
        )
        test_ids = sorted(int(s) for s in GS.C.DataSpec().east["test_locked"])
        if len(test_ids) != 360 or len(set(test_ids)) != 360:
            raise RuntimeError("authoritative EAST test roster is not balanced360")
        # Only data/* and timebase attributes are read in this phase.  Outcome
        # metadata is deliberately opened below, after every test prediction
        # has been computed and held in memory.
        te = [load_east_signal_only(shot, med, sc) for shot in test_ids]
        evaluation_mode = "support_threshold_then_sealed_test"
    R_tr=[full_risk(enc,head,s) for s in tr]
    R_te=[full_risk(enc,head,s) for s in te]
    if args.diagnostic_test_tuned:
        test_outcomes = [
            {
                "is_disrupt": bool(s["is_disrupt"]),
                "down_time_s": s.get("down_time_s"),
            }
            for s in te
        ]
    else:
        # This is the sole test-label reveal point in the safe path.
        test_outcomes = [read_east_outcome(s["shot"]) for s in te]
    kinds=["max","p90","top5","area","combo"]
    out={}
    for k in kinds:
        s_tr=np.array([agg(p,k) for p in R_tr]); s_te=np.array([agg(p,k) for p in R_te])
        y_tr=np.array([1 if s["is_disrupt"] else 0 for s in tr])
        y_te=np.array([1 if s["is_disrupt"] else 0 for s in test_outcomes])
        def eval_at(x, th, y):
            tp=( (x[y==1]>=th).sum() ); fp=((x[y==0]>=th).sum())
            return tp, fp
        # The historical test-tuned sweep is retained only behind an explicit
        # flag so it cannot silently become a reported result.
        best=None
        if args.diagnostic_test_tuned:
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
    result_rows = out
    out = {
        "evaluation_mode": evaluation_mode,
        "test_threshold_selection": (
            "test_tuned_diagnostic" if args.diagnostic_test_tuned
            else "support_only"
        ),
        "outcomes_revealed_after_all_test_predictions": not args.diagnostic_test_tuned,
        "results": result_rows,
        # Keep the historical method-key access pattern for downstream
        # diagnostics while making the safe/unsafe evaluation mode explicit.
        **result_rows,
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

if __name__=="__main__":
    main()
