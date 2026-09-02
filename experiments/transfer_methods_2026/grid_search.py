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

ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT/"experiments"/"transfer_methods_2026"
sys.path.insert(0, str(HERE))
import rebuilt_harness as RH
import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = 128
SOURCE_CHECKPOINT_SCHEMA = "jtext_rebuilt17_transformer_source_v1"
SOURCE_ARCHITECTURE_VERSION = "rebuilt17-patch-transformer-128x8-d64-l2-h4-v1"


def source_model_config(channels):
    return {
        "channels": int(channels),
        "window": W,
        "patch": int(RH.PATCH),
        "stride": int(RH.STRIDE),
        "d_model": int(RH.D_MODEL),
        "layers": int(RH.N_LAYERS),
        "heads": int(RH.N_HEADS),
    }


def default_source_checkpoint(seed):
    trained = HERE / "source_transformer_runs" / f"seed{int(seed)}" / "source_checkpoint.pt"
    if trained.is_file():
        return trained
    bundled = (
        HERE / "paper_grade_package/evidence/source_lineage"
        / f"seed{int(seed)}/source_checkpoint.pt"
    )
    return bundled


def _load_rebuilt_shots(device, ids):
    out, miss = [], []
    for sid in ids:
        try:
            arr, rate, start, down, isd = RH.load_raw(
                RH.DATA/device/"unified_hdf5"/str(sid//100*100)/f"{sid}.hdf5",
                device,
            )
        except Exception:
            miss.append(int(sid))
            continue
        v, a = RH.rebuild_features(arr, device)
        out.append(dict(
            shot=int(sid), v=v.astype(np.float32), a=a,
            is_disrupt=isd, down=down if isd else None,
            start=start, rate=rate,
        ))
    return out, miss


def load_east(*, include_test=True):
    east = C.DataSpec().east
    transfer, missing_transfer = _load_rebuilt_shots(
        "EAST", sorted(east["transfer"])
    )
    test, missing_test = (
        _load_rebuilt_shots("EAST", sorted(east["test_locked"]))
        if include_test else ([], [])
    )
    if missing_transfer or missing_test:
        raise RuntimeError(
            "missing EAST HDF5 shots: "
            f"transfer={missing_transfer}, test={missing_test}"
        )
    return transfer, test


def load_all(*, include_test=True):
    def jt_split():
        return {k: {int(l.strip()) for l in open(ROOT/f"splits/{k}.txt") if l.strip()} for k in ("train","val","test")}
    jt = jt_split()
    src, missing_src = _load_rebuilt_shots("J-TEXT", jt["train"])
    val, missing_val = _load_rebuilt_shots("J-TEXT", jt["val"])
    if missing_src or missing_val:
        raise RuntimeError(
            f"missing J-TEXT HDF5 shots: train={missing_src}, val={missing_val}"
        )
    # HDF5 meta/IsDisrupt and meta/DownTime are the sole label authority.
    tr, te = load_east(include_test=include_test)
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


def causal_window(values, *, endpoint, width=W, context_points=None):
    """Return a fixed-width causal view, repeating only its first past value."""

    endpoint = int(endpoint)
    width = int(width)
    context = width if context_points is None else int(context_points)
    if values.ndim != 2 or not 0 <= endpoint < len(values):
        raise ValueError("causal endpoint must index a two-dimensional trace")
    if width <= 0 or not 0 < context <= width:
        raise ValueError("require 0 < context_points <= width")
    start = max(0, endpoint - context + 1)
    observed = values[start:endpoint + 1]
    padding = width - len(observed)
    if padding:
        observed = np.concatenate([
            np.repeat(observed[:1], padding, axis=0), observed,
        ], axis=0)
    return np.ascontiguousarray(observed)


def build_sampler(
    shots, device, pos_ms, gray_ms, seed=0, *, minimum_pos_ms=0.0,
    allow_left_padding=False, context_lengths=(W,),
    sampling_mode="class_first",
):
    """Build a causal window sampler.

    ``class_first`` is retained for the historical feature/label sweep.  The
    registered V3C EAST support adaptation used ``shot_first``: draw a shot
    uniformly, then draw its positive/negative branch.  Those two operations
    are not interchangeable when the support cohort is 15 disruptive versus
    95 safe shots, so callers that belong to the V3C lineage must select the
    mode explicitly.
    """
    sampling_mode = str(sampling_mode)
    if sampling_mode not in {"class_first", "shot_first"}:
        raise ValueError(f"unknown sampling mode: {sampling_mode}")
    context_lengths = tuple(int(value) for value in context_lengths)
    if not context_lengths or any(not 0 < value <= W for value in context_lengths):
        raise ValueError("context lengths must lie in [1, W]")
    pos, neg = [], []
    for s in shots:
        n = s["v"].shape[0]; t = s["start"] + np.arange(n)/s["rate"]
        p, q = [], []
        first_endpoint = 0 if allow_left_padding else W - 1
        if s["is_disrupt"] and s["down"] is not None:
            for end in range(first_endpoint, n):
                ttd = (s["down"] - t[end]) * 1000.0
                if minimum_pos_ms + 1e-9 < ttd <= pos_ms: p.append(end)
                elif pos_ms + gray_ms < ttd <= 60000.0: q.append(end)
        else:
            q = list(range(first_endpoint, n))
        pos.append(p); neg.append(q)
    rng = np.random.default_rng(seed)
    positive_shots = np.flatnonzero([bool(values) for values in pos])
    negative_shots = np.flatnonzero([bool(values) for values in neg])

    if sampling_mode == "shot_first":
        def sample(pp=0.5):
            # This is the exact V3C order: shot -> branch -> endpoint.  A
            # safe shot cannot produce a positive example, and a disruptive
            # shot with no eligible negative endpoint is retried, matching the
            # sealed historical support-parent recipe.
            while True:
                i = int(rng.integers(len(shots)))
                positive_endpoints = pos[i]
                negative_endpoints = neg[i]
                if not positive_endpoints and not negative_endpoints:
                    continue
                if positive_endpoints and rng.random() < float(pp):
                    endpoints = positive_endpoints
                    label = np.float32(1.0)
                else:
                    if not negative_endpoints:
                        continue
                    endpoints = negative_endpoints
                    label = np.float32(0.0)
                endpoint = endpoints[int(rng.integers(len(endpoints)))]
                context = context_lengths[int(rng.integers(len(context_lengths)))]
                window = causal_window(
                    shots[i]["v"], endpoint=endpoint, width=W,
                    context_points=context,
                )
                return np.ascontiguousarray(window), label
        return sample

    def sample(pp=0.5):
        if not len(positive_shots) and not len(negative_shots):
            raise RuntimeError("sampler has no eligible causal window")
        choose_positive = bool(
            len(positive_shots)
            and (not len(negative_shots) or rng.random() < float(pp))
        )
        eligible = positive_shots if choose_positive else negative_shots
        i = int(eligible[int(rng.integers(len(eligible)))])
        endpoints = pos[i] if choose_positive else neg[i]
        end = endpoints[int(rng.integers(len(endpoints)))]
        context = context_lengths[int(rng.integers(len(context_lengths)))]
        x = causal_window(
            shots[i]["v"], endpoint=end, width=W, context_points=context,
        )
        return np.ascontiguousarray(x), np.float32(choose_positive)
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
    def forward(self, x, valid_lengths=None):
        h = self.embed(x.transpose(1, 2)).transpose(1, 2)
        h = h + C.sinusoidal_position_encoding(h)
        cl = self.cls.expand(x.shape[0], -1, -1)
        h = torch.cat([cl, h], 1)
        padding_mask = None
        if valid_lengths is not None:
            lengths = torch.as_tensor(valid_lengths, device=x.device, dtype=torch.long)
            if lengths.ndim != 1 or lengths.shape[0] != x.shape[0]:
                raise ValueError("valid_lengths must contain one value per batch row")
            if torch.any(lengths <= 0) or torch.any(lengths > x.shape[1]):
                raise ValueError("valid_lengths must be within the input window")
            left_padding = x.shape[1] - lengths
            patch_ends = (
                torch.arange(h.shape[1] - 1, device=x.device) * RH.STRIDE
                + RH.PATCH
            )
            invalid_patch = patch_ends.unsqueeze(0) <= left_padding.unsqueeze(1)
            padding_mask = torch.cat(
                [torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device),
                 invalid_patch],
                dim=1,
            )
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.norm(h[:, 0])


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(RH.D_MODEL, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, z): return self.net(z).squeeze(-1)


def load_source_checkpoint(path, *, channels, expected_seed=None, device=None):
    run_device = device or DEV
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"J-TEXT source checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=run_device, weights_only=True)
    if checkpoint.get("schema_version") != SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError("source checkpoint schema contract mismatch")
    if checkpoint.get("architecture_version") != SOURCE_ARCHITECTURE_VERSION:
        raise ValueError("source checkpoint architecture contract mismatch")
    if checkpoint.get("feature_names") != list(RH.REBUILD):
        raise ValueError("source checkpoint feature contract mismatch")
    if checkpoint.get("model_config") != source_model_config(channels):
        raise ValueError("source checkpoint model configuration mismatch")
    if expected_seed is not None and int(checkpoint.get("seed", -1)) != int(expected_seed):
        raise ValueError("source checkpoint seed contract mismatch")
    if checkpoint.get("split_counts") != {"train": 1353, "val": 160, "test": 220}:
        raise ValueError("source checkpoint split contract mismatch")
    if int(checkpoint.get("test_shards_read", -1)) != 0:
        raise ValueError("source checkpoint was selected after reading J-TEXT TEST220")
    center = checkpoint.get("scaler_center")
    scale = checkpoint.get("scaler_scale")
    if not isinstance(center, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError("source checkpoint scaler contract is missing")
    if tuple(center.shape) != (channels,) or tuple(scale.shape) != (channels,):
        raise ValueError("source checkpoint scaler shape mismatch")
    if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("source checkpoint scaler is not finite and positive")
    enc = Enc(channels).to(run_device)
    head = Head().to(run_device)
    enc.load_state_dict(checkpoint["encoder"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    return enc, head, checkpoint


def train_adapt(
    src_sampler, tr_sampler, C, epochs=12, bs=128, seed=0,
    *, source_checkpoint=None, device=None,
):
    run_device = device or DEV
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if source_checkpoint is not None:
        enc, head, _checkpoint = load_source_checkpoint(
            source_checkpoint, channels=C, expected_seed=seed, device=run_device
        )
    else:
        if src_sampler is None:
            raise ValueError("legacy source training requires a source sampler")
        enc = Enc(C).to(run_device); head = Head().to(run_device)
        opt = torch.optim.AdamW(
            list(enc.parameters())+list(head.parameters()), 3e-4, weight_decay=1e-4
        )
        for _ep in range(epochs[0] if isinstance(epochs, tuple) else 6):
            enc.train(); head.train()
            for _ in range(8):
                xs, ys = [], []
                for _ in range(bs):
                    x, y = src_sampler(0.5); xs.append(x); ys.append(y)
                loss = F.binary_cross_entropy_with_logits(
                    head(enc(torch.from_numpy(np.stack(xs)).to(run_device))),
                    torch.from_numpy(np.stack(ys)).to(run_device),
                )
                opt.zero_grad(); loss.backward(); opt.step()
    # freeze low, adapt top on transfer
    for p in enc.parameters(): p.requires_grad_(False)
    for p in enc.encoder.layers[-1].parameters(): p.requires_grad_(True)
    for p in enc.norm.parameters(): p.requires_grad_(True)
    head = Head().to(run_device)
    opt = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, enc.parameters()))+list(head.parameters()), 1e-4, weight_decay=1e-4)
    nb2 = 32
    for ep in range(epochs):
        enc.train(); head.train()
        for _ in range(nb2):
            xs, ys = [], []
            for _ in range(bs):
                x, y = tr_sampler(0.5); xs.append(x); ys.append(y)
            loss = F.binary_cross_entropy_with_logits(
                head(enc(torch.from_numpy(np.stack(xs)).to(run_device))),
                torch.from_numpy(np.stack(ys)).to(run_device),
            )
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
