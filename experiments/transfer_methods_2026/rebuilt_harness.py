"""Rebuilt-feature harness: paper131-aligned transfer with only derivable columns.

Avoids previously-hit pitfalls explicitly:
  * ip harmonized exactly once (J-TEXT kA -> A); no double conversion.
  * Z_proxy, Bt/Ip, Ip_diff recomputed from harmonized ip.
  * Radiation "bad columns" (CIII, xuv_ratio, sxr_mean) dropped; MNM dropped on
    both devices (paper drops it on EAST); qa_proxy/ne_nG replaced by Bt/Ip.
  * Mirnov amplitude (Mir_VV / Mir_avg_amp) dropped (probe-dependent unit, and
    ~74% zeros on EAST); only Mir_avg_fre kept with per-shot availability mask.
  * Threshold selected ONLY on the 10D+100N transfer set; the 360 test is scored
    once and never used for tuning.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")).resolve()
OUT = ROOT / "experiments" / "transfer_methods_2026" / "rebuilt_runs"
sys.path.insert(0, str(ROOT / "experiments/transfer_methods_2026"))
from common import (  # noqa: E402
    DataSpec,
    ARCHITECTURE_VERSION,
    TEST_CONTRACT_VERSION,
    full_trace_max_score,
    require_checkpoint_contract,
    sinusoidal_position_encoding,
)

# canonical raw 23 columns (order fixed by repo contract)
RAW23 = ("Bt","CIII","MNM","Mir_VV","Mir_avg_amp","Mir_avg_fre","Z_proxy","dx_a","dy_a",
         "ip","mode_number_n","n=1 amplitude","ne_nG","qa_proxy","sxr_kurt","sxr_mean",
         "sxr_skew","sxr_var","v_loop","xuv_kurt","xuv_ratio","xuv_skew","xuv_var")
RIDX = {n: i for i, n in enumerate(RAW23)}

# rebuilt output columns (K channels)
REBUILD = ("Bt", "ip_A", "Bt_over_Ip", "Ip_diff", "v_loop", "Z_proxy",
           "sxr_kurt", "sxr_skew", "sxr_var", "xuv_kurt", "xuv_skew", "xuv_var",
           "Mir_avg_fre", "n=1 amplitude", "mode_number_n", "dx_a", "dy_a")
K = len(REBUILD)

WINDOW = 128
HORIZON_MS = 50.0
NEG_MAX_MS = 1200.0
PATCH, STRIDE, D_MODEL = 8, 8, 64
N_LAYERS, N_HEADS = 2, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------- data
def load_raw(path: Path, device: str):
    with h5py.File(path, "r") as f:
        arr = {n: np.asarray(f[f"data/{n}"][:], dtype=np.float32) for n in RAW23}
        rate = float(np.asarray(f["data/ip"].attrs.get("SampleRate", 1000.0)).reshape(-1)[0])
        start = float(np.asarray(f["data/ip"].attrs.get("StartTime", 0.0)).reshape(-1)[0])
        down = np.asarray(f["meta/DownTime"][()]).reshape(-1)
        isd = bool(np.asarray(f["meta/IsDisrupt"][()]).reshape(-1)[0])
    return arr, rate, start, (float(down[0]) if down.size else None), isd


def rebuild_features(arr, device):
    T = len(arr["Bt"])
    out = np.zeros((T, K), dtype=np.float32)
    avail = np.zeros((T, K), dtype=bool)
    ip = arr["ip"].copy()
    if device == "J-TEXT":
        ip = ip * 1000.0  # exactly once: kA -> A
    ip_fin = np.isfinite(ip)
    Bt = arr["Bt"]; vl = arr["v_loop"]

    def put(name, vals, finite):
        c = REBUILD.index(name)
        v = np.where(np.isfinite(vals) & finite, vals, 0.0).astype(np.float32)
        out[:, c] = v
        avail[:, c] = np.isfinite(vals) & finite
        return c

    put("Bt", Bt, np.isfinite(Bt))
    put("ip_A", ip, ip_fin)
    denom = np.where(np.abs(ip) > 1e-9, ip, np.nan)
    put("Bt_over_Ip", Bt / denom, ip_fin & np.isfinite(Bt) & (np.abs(ip) > 1e-9))
    ipdiff = np.zeros_like(ip); ipdiff[1:] = np.diff(ip)
    put("Ip_diff", ipdiff, ip_fin)
    put("v_loop", vl, np.isfinite(vl))
    put("Z_proxy", vl / denom, np.isfinite(vl) & (np.abs(ip) > 1e-9))
    for n in ("sxr_kurt","sxr_skew","sxr_var","xuv_kurt","xuv_skew","xuv_var",
              "dx_a","dy_a","n=1 amplitude"):
        put(n, arr[n], np.isfinite(arr[n]))
    # Mir_avg_fre / mode_number_n: all-zero shot => unavailable (MHD baseline, not physics 0)
    for n in ("Mir_avg_fre","mode_number_n"):
        v = arr[n].copy(); c = put(n, v, np.isfinite(v))
        obs = v[np.isfinite(v)]
        if obs.size and np.allclose(obs, 0.0):
            avail[:, c] = False
    return out, avail


class Shots:
    def __init__(self, device, ids, label_map=None, scaler=None):
        self.device = device
        self.shots = []
        self.missing = []
        for sid in ids:
            try:
                arr, rate, start, down, isd = load_raw(DATA / device / "unified_hdf5"
                                                       / str(sid // 100 * 100) / f"{sid}.hdf5", device)
            except Exception as e:  # noqa: BLE001
                self.missing.append((int(sid), repr(e)[:120])); continue
            if label_map is not None and int(sid) in label_map:
                isd = bool(label_map[int(sid)])
                if not isd:
                    down = None
            vals, avail = rebuild_features(arr, device)
            t = start + np.arange(len(vals), dtype=np.float64) / rate
            self.shots.append(dict(shot=int(sid), values=vals, avail=avail, times=t,
                                   is_disrupt=isd, down_time_s=(down if isd else None)))
        if scaler is None:
            self.center, self.scale = self._fit()
        else:
            self.center, self.scale = scaler
        for s in self.shots:
            s["values"] = np.clip((s["values"] - self.center) / self.scale, -10, 10)
            s["values"] = np.where(s["avail"], s["values"], 0.0).astype(np.float32)

    def _fit(self):
        center = np.zeros(K, np.float32); scale = np.ones(K, np.float32)
        for c in range(K):
            acc = []
            for s in self.shots:
                v = s["values"][s["avail"][:, c], c]
                if v.size:
                    idx = np.linspace(0, len(v)-1, min(64, len(v)), dtype=int)
                    acc.extend(v[idx].tolist())
            if acc:
                a = np.asarray(acc); q = np.quantile(a, [0.25, 0.5, 0.75])
                center[c] = float(q[1]); scale[c] = max(float(q[2]-q[0]), 1e-6)
        return center, scale

    def __len__(self):
        return len(self.shots)


class Sampler:
    def __init__(self, shots, seed=0):
        self.shots = [s for s in shots if len(s["values"]) >= WINDOW]
        self.rng = np.random.default_rng(seed)
        self.pos = []; self.neg = []
        for s in self.shots:
            n = len(s["values"]); t = s["times"]
            p, q = [], []
            if s["is_disrupt"] and s["down_time_s"] is not None:
                d = s["down_time_s"]
                for end in range(WINDOW-1, n):
                    ttd = (d - t[end]) * 1000.0
                    if 0 <= ttd <= HORIZON_MS: p.append(end)
                    elif HORIZON_MS < ttd <= NEG_MAX_MS: q.append(end)
            else:
                q = list(range(WINDOW-1, n))
            self.pos.append(p); self.neg.append(q)

    def sample(self, pp=0.5):
        while True:
            i = int(self.rng.integers(len(self.shots)))
            p, q = self.pos[i], self.neg[i]
            if not p and not q: continue
            if p and self.rng.random() < pp:
                end = p[int(self.rng.integers(len(p)))]; y = 1.0
            else:
                if not q: continue
                end = q[int(self.rng.integers(len(q)))]; y = 0.0
            x = self.shots[i]["values"][end-WINDOW+1:end+1]
            return np.ascontiguousarray(x), np.float32(y)


# ----------------------------------------------------------------------- model
class Enc(nn.Module):
    def __init__(self, C=K, d=D_MODEL, layers=N_LAYERS, heads=N_HEADS):
        super().__init__()
        self.embed = nn.Conv1d(C, d, PATCH, STRIDE)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        lay = nn.TransformerEncoderLayer(d, heads, d*4, 0.1, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(lay, layers)
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x):
        h = self.embed(x.transpose(1, 2)).transpose(1, 2)
        h = h + sinusoidal_position_encoding(h)
        cl = self.cls.expand(x.shape[0], -1, -1)
        h = self.encoder(torch.cat([cl, h], 1))
        return self.norm(h[:, 0])


class Head(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, z): return self.net(z).squeeze(-1)


# --------------------------------------------------------------------- scoring
TTDS = [10,20,30,40,50,60,80,100,150,200,300,400,500,700,1000]

def shot_score(enc, head, s):
    value, _ = full_trace_max_score(
        enc, head, s["values"], window=WINDOW, device=DEV
    )
    return value

def score(enc, head, shots):
    ys, ps = [], []
    for s in shots:
        ys.append(1 if s["is_disrupt"] else 0); ps.append(shot_score(enc, head, s))
    return np.asarray(ys), np.asarray(ps)

def auc(y, p):
    positive, negative = p[y == 1], p[y == 0]
    if not len(positive) or not len(negative): return float("nan")
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean(comparison > 0) + 0.5*np.mean(comparison == 0))

def eval_model(enc, head, tr, te):
    if hasattr(enc, 'eval'): enc.eval()
    head.eval()
    yt, pt = score(enc, head, tr.shots); yv, pv = score(enc, head, te.shots)
    res = dict(n_train=len(yt), n_test=len(yv), test_auc=float(auc(yv, pv)), train_auc=float(auc(yt, pt)))
    # threshold on transfer set only: TPR>=0.9 -> lowest FPR
    best = None
    for th in sorted(set(pt.tolist()), reverse=True) + [-1e9]:
        tpr = (pt[yt == 1] >= th).mean(); fpr = (pt[yt == 0] >= th).mean()
        if tpr >= 0.9 and (best is None or fpr < best[1] or (abs(fpr-best[1]) < 1e-9 and th > best[0])):
            best = (th, fpr, tpr)
    if best:
        res.update(threshold=best[0], train_tpr=best[2], train_fpr=best[1],
                   test_tpr=float((pv[yv == 1] >= best[0]).mean()),
                   test_fpr=float((pv[yv == 0] >= best[0]).mean()))
    # post-hoc test operating point (paper131-style): TPR first, then min FPR
    for target in (0.8, 0.85, 0.9):
        bc = None
        for th in sorted(set(pv.tolist()), reverse=True) + [-1e9]:
            tpr = (pv[yv == 1] >= th).mean(); fpr = (pv[yv == 0] >= th).mean()
            if tpr >= target and (bc is None or fpr < bc[1] or (abs(fpr-bc[1]) < 1e-9 and th > bc[0])):
                bc = (th, fpr, tpr)
        if bc:
            res[f"posthoc_tpr{target}"] = dict(tpr=round(bc[2],4), fpr=round(bc[1],4), threshold=round(bc[0],6))
    return res


# --------------------------------------------------------------------- methods
def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def source_pretrain(cfg, src, val):
    seed_all(cfg["seed"])
    enc = Enc().to(DEV); head = Head().to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), 3e-4, weight_decay=1e-4)
    samp = Sampler(src.shots, cfg["seed"])
    nb = max(1, len(samp.shots)//cfg["batch"])
    best = -1
    for ep in range(cfg["source_epochs"]):
        enc.train(); head.train()
        for _ in range(nb):
            xs, ys = [], []
            for _ in range(cfg["batch"]):
                x, y = samp.sample(0.5); xs.append(x); ys.append(y)
            xb = torch.from_numpy(np.stack(xs)).to(DEV); yb = torch.from_numpy(np.stack(ys)).to(DEV)
            loss = F.binary_cross_entropy_with_logits(head(enc(xb)), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        a = eval_model(enc, head, val, val)["test_auc"]
        if not math.isnan(a) and a > best:
            best = a
            torch.save(dict(
                enc=enc.state_dict(),
                head=head.state_dict(),
                architecture_version=ARCHITECTURE_VERSION,
                test_contract_version=TEST_CONTRACT_VERSION,
            ), OUT / f"src_posv2_seed{cfg['seed']}.pt")
    ck = torch.load(OUT / f"src_posv2_seed{cfg['seed']}.pt", map_location=DEV)
    require_checkpoint_contract(ck)
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
    return enc, head

def target_only(cfg, tr, te):
    seed_all(cfg["seed"])
    enc = Enc().to(DEV); head = Head().to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), cfg["lr"], weight_decay=1e-4)
    samp = Sampler(tr.shots, cfg["seed"]); nb = max(64, len(samp.shots)//4)
    for _ in range(cfg["epochs"]):
        enc.train(); head.train()
        for _ in range(nb):
            xs, ys = [], []
            for _ in range(cfg["batch"]):
                x, y = samp.sample(0.5); xs.append(x); ys.append(y)
            loss = F.binary_cross_entropy_with_logits(head(enc(torch.from_numpy(np.stack(xs)).to(DEV))),
                                                      torch.from_numpy(np.stack(ys)).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return eval_model(enc, head, tr, te)

def fine_tune(cfg, enc, head, tr, te):
    seed_all(cfg["seed"])
    head = Head().to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), 1e-4, weight_decay=1e-4)
    samp = Sampler(tr.shots, cfg["seed"]); nb = max(64, len(samp.shots)//4)
    for _ in range(cfg["epochs"]):
        enc.train(); head.train()
        for _ in range(nb):
            xs, ys = [], []
            for _ in range(cfg["batch"]):
                x, y = samp.sample(0.5); xs.append(x); ys.append(y)
            loss = F.binary_cross_entropy_with_logits(head(enc(torch.from_numpy(np.stack(xs)).to(DEV))),
                                                      torch.from_numpy(np.stack(ys)).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return eval_model(enc, head, tr, te)

def adapt_high(cfg, enc, head, tr, te):
    seed_all(cfg["seed"])
    head = Head().to(DEV)
    for p in enc.parameters(): p.requires_grad_(False)
    for p in enc.encoder.layers[-1].parameters(): p.requires_grad_(True)
    for p in enc.norm.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, enc.parameters()))+list(head.parameters()), 1e-4, weight_decay=1e-4)
    samp = Sampler(tr.shots, cfg["seed"]); nb = max(64, len(samp.shots)//4)
    for _ in range(cfg["epochs"]):
        enc.train(); head.train()
        for _ in range(nb):
            xs, ys = [], []
            for _ in range(cfg["batch"]):
                x, y = samp.sample(0.5); xs.append(x); ys.append(y)
            loss = F.binary_cross_entropy_with_logits(head(enc(torch.from_numpy(np.stack(xs)).to(DEV))),
                                                      torch.from_numpy(np.stack(ys)).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return eval_model(enc, head, tr, te)

def coral_cond(cfg, enc, head, src, tr, te):
    seed_all(cfg["seed"])
    for p in enc.parameters(): p.requires_grad_(False)
    proj = nn.Linear(D_MODEL, D_MODEL).to(DEV); h2 = Head().to(DEV)
    def coral(a, b):
        ca = torch.cov(a.T)+1e-6*torch.eye(D_MODEL, device=DEV)
        cb = torch.cov(b.T)+1e-6*torch.eye(D_MODEL, device=DEV)
        return ((ca-cb)**2).sum()/ (4*D_MODEL*D_MODEL)
    ssamp = Sampler(src.shots, cfg["seed"]); tsamp = Sampler(tr.shots, cfg["seed"])
    opt = torch.optim.AdamW(list(proj.parameters())+list(h2.parameters()), cfg["lr"], weight_decay=1e-4)
    nb = max(32, min(len(ssamp.shots), len(tsamp.shots))//3)
    def draw_cls(samp, pos):
        for _ in range(200):
            x, y = samp.sample(0.5)
            if (y > 0.5) == pos: return x
        return None
    for _ in range(cfg["epochs"]):
        proj.train(); h2.train()
        for _ in range(nb):
            sp = [draw_cls(ssamp, True) for _ in range(cfg["batch"]//4)]
            sn = [draw_cls(ssamp, False) for _ in range(cfg["batch"]//4)]
            tp = [draw_cls(tsamp, True) for _ in range(cfg["batch"]//4)]
            tn = [draw_cls(tsamp, False) for _ in range(cfg["batch"]//4)]
            sp=[x for x in sp if x is not None]; sn=[x for x in sn if x is not None]; tp=[x for x in tp if x is not None]; tn=[x for x in tn if x is not None]
            n = min(len(sp), len(sn), len(tp), len(tn))
            if n == 0: continue
            def T(L): return torch.from_numpy(np.stack(L[:n])).to(DEV)
            with torch.no_grad():
                zsp, zsn = enc(T(sp)), enc(T(sn)); ztp, ztn = enc(T(tp)), enc(T(tn))
            zsp, zsn, ztp, ztn = proj(zsp), proj(zsn), proj(ztp), proj(ztn)
            yp = torch.ones(n, device=DEV); yn = torch.zeros(n, device=DEV)
            cl = F.binary_cross_entropy_with_logits(h2(torch.cat([ztp, ztn])), torch.cat([yp, yn]))
            loss = cl + 0.5*(coral(zsp, ztp)+coral(zsn, ztn))
            opt.zero_grad(); loss.backward(); opt.step()
    def encw(x): return proj(enc(x))
    return eval_model(encw, h2, tr, te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["target_only","fine_tune","adapt_high","coral_cond"])
    ap.add_argument("--seed", type=int, default=131)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--source-epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")

    # splits (same as earlier sweep)
    def jt_split():
        d = {}
        for k in ("train","val","test"):
            d[k] = {int(l.strip()) for l in open(ROOT/f"splits/{k}.txt") if l.strip()}
        return d
    jt = jt_split()
    east = DataSpec().east
    tr_ids = sorted(east["transfer"])
    te_ids = sorted(east["test_locked"])
    tr_map = {int(s): True for s in east["transfer_disrupt"]} | {int(s): False for s in east["transfer_safe"]}
    te_map = {int(s): True for s in east["test_disrupt"]} | {int(s): False for s in east["test_safe"]}

    src = Shots("J-TEXT", jt["train"])
    val = Shots("J-TEXT", jt["val"], scaler=(src.center, src.scale))
    tr = Shots("EAST", tr_ids, label_map=tr_map, scaler=(src.center, src.scale))
    te = Shots("EAST", te_ids, label_map=te_map, scaler=(src.center, src.scale))
    print(f"rebuilt K={K} | J-TEXT train={len(src)} val={len(val)} | EAST transfer={len(tr)} test={len(te)} "
          f"(transfer pos={sum(s['is_disrupt'] for s in tr.shots)}, test pos={sum(s['is_disrupt'] for s in te.shots)})")

    cfg = dict(seed=args.seed, batch=args.batch, epochs=args.epochs,
               source_epochs=args.source_epochs, lr=args.lr)
    t0 = time.time()
    if args.method == "target_only":
        res = target_only(cfg, tr, te)
    else:
        ck = OUT / f"src_posv2_seed{args.seed}.pt"
        if ck.exists():
            enc = Enc().to(DEV); head = Head().to(DEV)
            c = torch.load(ck, map_location=DEV)
            require_checkpoint_contract(c)
            enc.load_state_dict(c["enc"]); head.load_state_dict(c["head"])
        else:
            enc, head = source_pretrain(cfg, src, val)
        if args.method == "fine_tune":
            res = fine_tune(cfg, enc, head, tr, te)
        elif args.method == "adapt_high":
            res = adapt_high(cfg, enc, head, tr, te)
        else:
            res = coral_cond(cfg, enc, head, src, tr, te)
    res["method"] = args.method; res["seed"] = args.seed
    res["features"] = "rebuilt17"; res["elapsed_s"] = round(time.time()-t0, 1)
    p = OUT / args.method / f"seed{args.seed}"
    p.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p / "result.json", "w"), indent=1)
    print("RESULT", json.dumps(res))
    print("WROTE", p / "result.json")

if __name__ == "__main__":
    main()
