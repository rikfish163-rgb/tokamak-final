"""Grid search over feature transforms x labeling rules (few-shot J-TEXT->EAST).

Base pipeline: rebuilt17 features (paper131-aligned, derived cols only).
Transforms (applied per-shot before robust scaling):
  base      : raw rebuilt17 (median/IQR robust scale)
  diff      : temporal first-difference of rebuilt17
  base_diff : concat(base, diff) aligned in time  -> 2K channels
  roll      : base + rolling mean/std (window 8)  -> 3K channels
Labeling (positive horizon per device and FLS gray band):
  sym50_gray0    : pos<=50ms both devices, no gray band
  paper_gray0    : pos<=25ms(J-TEXT)/125ms(EAST), no gray
  paper_grayFLS  : same + gray band 50ms(J-TEXT)/200ms(EAST) excluded
  ea125_gray0    : pos<=125ms both, no gray
Backbone: patch+Transformer; method: adapt_high (freeze low, train top block+head).
Metric: 360 balanced test, per-shot max-pool over candidate windows;
        AUC + paper131-style operating point (TPR>=0.8 -> min FPR).
"""
import argparse, json, math, os, time, sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/"experiments"/"transfer_methods_2026"
sys.path.insert(0, str(HERE))
import rebuilt_harness as RH
import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = 128


def load_all(*, include_test=True):
    def jt_split():
        return {k: {int(l.strip()) for l in open(ROOT/f"splits/{k}.txt") if l.strip()} for k in ("train","val","test")}
    jt = jt_split()
    east = C.DataSpec().east
    tr_ids = sorted(east["transfer"])
    te_ids = sorted(east["test_locked"]) if include_test else []
    def load(device, ids, lm=None):
        out, miss = [], []
        for sid in ids:
            try:
                arr, rate, start, down, isd = RH.load_raw(RH.DATA/device/"unified_hdf5"/str(sid//100*100)/f"{sid}.hdf5", device)
            except Exception as e:
                miss.append(int(sid)); continue
            if lm is not None and int(sid) in lm:
                isd = bool(lm[int(sid)])
                if not isd: down = None
            v, a = RH.rebuild_features(arr, device)
            n = len(v)
            out.append(dict(shot=int(sid), v=v.astype(np.float32), a=a, is_disrupt=isd,
                            down=down if isd else None, start=start, rate=rate))
        return out, miss
    src, _ = load("J-TEXT", jt["train"])
    val, _ = load("J-TEXT", jt["val"])
    # HDF5 meta/IsDisrupt and meta/DownTime are the sole label authority.
    tr, _ = load("EAST", tr_ids)
    te, _ = load("EAST", te_ids) if include_test else ([], [])
    return src, val, tr, te


def transform(shots, kind):
    out = []
    for s in shots:
        v, a = s["v"], s["a"]
        if kind == "base":
            d=dict(s); d['v']=v; d['a']=a; out.append(d)
        elif kind == "diff":
            d = np.zeros_like(v); d[1:] = np.diff(v, axis=0)
            ad = np.zeros_like(a); ad[1:] = a[1:] & a[:-1]
            dd=dict(s); dd['v']=d; dd['a']=ad; out.append(dd)
        elif kind == "base_diff":
            d = np.zeros_like(v); d[1:] = np.diff(v, axis=0)
            vv = np.concatenate([v, d], axis=1).astype(np.float32)
            aa = np.concatenate([a, a], axis=1)
            d=dict(s); d['v']=vv; d['a']=aa; out.append(d)
        elif kind == "roll":
            r_mean = np.zeros_like(v); r_std = np.zeros_like(v)
            for i in range(len(v)):
                lo = max(0, i-7); seg = v[lo:i+1]; r_mean[i] = seg.mean(0); r_std[i] = seg.std(0)
            vv = np.concatenate([v, r_mean, r_std], axis=1).astype(np.float32)
            aa = np.concatenate([a, a, a], axis=1)
            d=dict(s); d['v']=vv; d['a']=aa; out.append(d)
        else:
            raise ValueError(kind)
    return out


def fit_scale(src):
    X = np.vstack([s["v"] for s in src])
    med = np.median(X, 0); iqr = np.percentile(X, 75, 0) - np.percentile(X, 25, 0)
    return med, np.maximum(iqr, 1e-6)


def apply_scale(shots, med, sc):
    for s in shots:
        s["v"] = np.clip((s["v"]-med)/sc, -10, 10).astype(np.float32)
        s["v"] = np.where(s["a"], s["v"], 0.0)


POS = dict(sym50=dict(jt=50.0, ea=50.0), paper=dict(jt=25.0, ea=125.0),
           ea125=dict(jt=125.0, ea=125.0))
GRAY = dict(gray0=dict(jt=0.0, ea=0.0), grayFLS=dict(jt=50.0, ea=200.0))


def build_sampler(shots, device, pos_ms, gray_ms, seed=0):
    pos, neg = [], []
    for s in shots:
        n = s["v"].shape[0]; t = s["start"] + np.arange(n)/s["rate"]
        p, q = [], []
        if s["is_disrupt"] and s["down"] is not None:
            for end in range(W-1, n):
                ttd = (s["down"] - t[end]) * 1000.0
                if 0 <= ttd <= pos_ms: p.append(end)
                elif pos_ms + gray_ms < ttd <= 60000.0: q.append(end)
        else:
            q = list(range(W-1, n))
        pos.append(p); neg.append(q)
    rng = np.random.default_rng(seed)
    def sample(pp=0.5):
        while True:
            i = int(rng.integers(len(shots)))
            p, q = pos[i], neg[i]
            if not p and not q: continue
            if p and rng.random() < pp:
                end = p[int(rng.integers(len(p)))]; y = 1.0
            else:
                if not q: continue
                end = q[int(rng.integers(len(q)))]; y = 0.0
            x = shots[i]["v"][end-W+1:end+1]
            return np.ascontiguousarray(x), np.float32(y)
    return sample


class Enc(nn.Module):
    def __init__(self, C):
        super().__init__()
        d = RH.D_MODEL
        self.embed = nn.Conv1d(C, d, RH.PATCH, RH.STRIDE)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        lay = nn.TransformerEncoderLayer(d, RH.N_HEADS, d*4, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(lay, RH.N_LAYERS)
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.cls, std=0.02)
    def forward(self, x):
        h = self.embed(x.transpose(1, 2)).transpose(1, 2)
        h = h + C.sinusoidal_position_encoding(h)
        cl = self.cls.expand(x.shape[0], -1, -1)
        h = self.encoder(torch.cat([cl, h], 1))
        return self.norm(h[:, 0])


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(RH.D_MODEL, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, z): return self.net(z).squeeze(-1)


def train_adapt(src_sampler, tr_sampler, C, epochs=12, bs=128, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    enc = Enc(C).to(DEV); head = Head().to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), 3e-4, weight_decay=1e-4)
    nb = max(1, 1000 // bs)  # ~1000 windows per source epoch? use fixed budget
    for ep in range(epochs[0] if isinstance(epochs, tuple) else 6):
        enc.train(); head.train()
        for _ in range(8):
            xs, ys = [], []
            for _ in range(bs):
                x, y = src_sampler(0.5); xs.append(x); ys.append(y)
            loss = F.binary_cross_entropy_with_logits(head(enc(torch.from_numpy(np.stack(xs)).to(DEV))), torch.from_numpy(np.stack(ys)).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    # freeze low, adapt top on transfer
    for p in enc.parameters(): p.requires_grad_(False)
    for p in enc.encoder.layers[-1].parameters(): p.requires_grad_(True)
    for p in enc.norm.parameters(): p.requires_grad_(True)
    head = Head().to(DEV)
    opt = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, enc.parameters()))+list(head.parameters()), 1e-4, weight_decay=1e-4)
    nb2 = 32
    for ep in range(epochs):
        enc.train(); head.train()
        for _ in range(nb2):
            xs, ys = [], []
            for _ in range(bs):
                x, y = tr_sampler(0.5); xs.append(x); ys.append(y)
            loss = F.binary_cross_entropy_with_logits(head(enc(torch.from_numpy(np.stack(xs)).to(DEV))), torch.from_numpy(np.stack(ys)).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return enc, head


TTDS = [10,20,30,40,50,60,80,100,150,200,300,400,500,700,1000]
def score_shot(enc, head, s):
    value, _ = C.full_trace_max_score(enc, head, s["v"], window=W, device=DEV)
    return value

def score_all(enc, head, shots):
    ys, ps = [], []
    for s in shots:
        ys.append(1 if s["is_disrupt"] else 0); ps.append(score_shot(enc, head, s))
    return np.asarray(ys), np.asarray(ps)

def auc(y, p):
    positive, negative = p[y == 1], p[y == 0]
    if not len(positive) or not len(negative): return float("nan")
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean(comparison > 0) + 0.5*np.mean(comparison == 0))

def posthoc(y, p, target):
    best = None
    for th in sorted(set(p.tolist()), reverse=True) + [-1e9]:
        tpr = (p[y==1]>=th).mean(); fpr = (p[y==0]>=th).mean()
        if tpr >= target and (best is None or fpr < best[1] or (abs(fpr-best[1])<1e-9 and th>best[0])):
            best = (th, fpr, tpr)
    return dict(tpr=round(best[2],4), fpr=round(best[1],4), threshold=round(best[0],6)) if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default="base")
    ap.add_argument("--label", default="sym50_gray0")
    ap.add_argument("--seed", type=int, default=131)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    src, val, tr, te = load_all()
    src = transform(src, args.transform); val = transform(val, args.transform)
    tr = transform(tr, args.transform); te = transform(te, args.transform)
    med, sc = fit_scale(src)
    for g in (src, val, tr, te): apply_scale(g, med, sc)
    pos_kind, gray_kind = args.label.split("_")
    pos = POS[pos_kind]; gray = GRAY[gray_kind]
    sr_s = build_sampler(src, None, pos["jt"], gray["jt"], args.seed)
    tr_s = build_sampler(tr, None, pos["ea"], gray["ea"], args.seed)
    C = src[0]["v"].shape[1]
    enc, head = train_adapt(sr_s, tr_s, C, args.epochs, args.batch, args.seed)
    enc.eval(); head.eval()
    yte, pte = score_all(enc, head, te)
    res = {"transform": args.transform, "label": args.label, "seed": args.seed, "K": int(C),
           "test_auc": round(float(auc(yte, pte)), 4),
           "tpr0.8_op": posthoc(yte, pte, 0.8), "tpr0.9_op": posthoc(yte, pte, 0.9)}
    print("RESULT", json.dumps(res))
    p = HERE/"grid_runs"/args.transform/args.label
    p.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p/f"seed{args.seed}.json","w"), indent=1)


if __name__ == "__main__":
    main()
