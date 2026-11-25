"""J-TEXT-only Simple23 training and exact batched causal inference."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import ARTIFACTS, load_jtext_splits, read_advanced_time, read_outcome, read_signals
from .metrics import score_shot, summarize_stream

ARCH_CONFIG = dict(window=128, patch=8, dim=64, heads=4, layers=2, ffn=128)


class Simple23(nn.Module):
    """46-input patched Transformer; masked diagnostics never enter normalization."""
    def __init__(self, center=None, scale=None, arch_config=None):
        super().__init__()
        self.cfg = dict(ARCH_CONFIG if arch_config is None else arch_config)
        c = self.cfg
        if c["window"] % c["patch"] or c["dim"] % c["heads"]:
            raise ValueError("window/patch and dim/heads must divide")
        center = torch.zeros(23) if center is None else torch.as_tensor(center, dtype=torch.float32)
        scale = torch.ones(23) if scale is None else torch.as_tensor(scale, dtype=torch.float32)
        if (center.shape != (23,) or scale.shape != (23,) or not torch.isfinite(center).all()
                or not torch.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError("finite 23-channel normalization and positive scale required")
        self.register_buffer("center", center.clone())
        self.register_buffer("scale", scale.clone())
        self.embedding = nn.Linear(c["patch"] * 46, c["dim"])
        self.position = nn.Parameter(torch.randn(1, c["window"] // c["patch"] + 1, c["dim"]) * .02)
        self.cls = nn.Parameter(torch.zeros(1, 1, c["dim"]))
        layer = nn.TransformerEncoderLayer(c["dim"], c["heads"], c["ffn"], dropout=0.,
                                            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c["layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c["dim"])
        self.risk_head = nn.Linear(c["dim"], 1)

    def forward(self, values, available):
        if values.ndim != 3 or values.shape[1:] != (self.cfg["window"], 23):
            raise ValueError("values must be [batch, window, 23]")
        if available.shape != values.shape or available.dtype != torch.bool:
            raise ValueError("matching boolean availability required")
        available = available & torch.isfinite(values)
        clean = torch.where(available, values, self.center)
        normalized = torch.asinh((clean - self.center) / self.scale)
        x = torch.cat((normalized, available.to(normalized.dtype)), dim=-1)
        batch, patches = len(values), self.cfg["window"] // self.cfg["patch"]
        x = self.embedding(x.reshape(batch, patches, -1))
        valid = available.reshape(batch, patches, -1).any(-1)
        x = torch.cat((x, self.cls.expand(batch, -1, -1)), dim=1) + self.position
        valid = torch.cat((valid, torch.ones(batch, 1, dtype=torch.bool, device=x.device)), dim=1)
        x = self.encoder(x, src_key_padding_mask=~valid)
        return self.risk_head(self.norm(x[:, -1])).squeeze(-1)


def load_source_model(path=ARTIFACTS / "jtext_source.pt", device="cuda"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Simple23(checkpoint["norm_center"], checkpoint["norm_scale"], checkpoint["arch_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def _windows(values, available, window, device):
    values = torch.as_tensor(values, dtype=torch.float32, device=device)
    available = torch.as_tensor(available, dtype=torch.bool, device=device)
    if values.ndim != 2 or values.shape[1] != 23 or available.shape != values.shape:
        raise ValueError("matching [T,23] arrays required")
    x = F.pad(values, (0, 0, window - 1, 0)).unfold(0, window, 1).permute(0, 2, 1)
    a = F.pad(available, (0, 0, window - 1, 0)).unfold(0, window, 1).permute(0, 2, 1)
    return x, a


@torch.inference_mode()
def predict_shot_proba(model, values, available, device="cuda", batch_size=2048):
    """Endpoint t sees only values[max(0,t-W+1):t+1], left-padded unavailable.

    No Python window copies: unfold views are batched on the accelerator.
    Float32 inference preserves single-endpoint equivalence without AMP drift.
    """
    if batch_size < 1:
        raise ValueError("positive batch_size required")
    model.eval()
    target = next(model.parameters()).device if device is None else torch.device(device)
    if next(model.parameters()).device != target:
        model.to(target)
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    x, a = _windows(values, available, model.cfg["window"], target)
    result = torch.cat([model(x[i:i + batch_size], a[i:i + batch_size]).sigmoid()
                        for i in range(0, len(x), batch_size)])
    return result.cpu().numpy()


def _load_supervised(ids, advanced):
    shots = []
    for shot in ids:
        s = read_signals("J-TEXT", shot)
        disruptive, down = read_outcome("J-TEXT", shot)
        ttd = 1000. * (down - s["times"]) if disruptive else np.full(len(s["times"]), np.inf)
        valid = ttd >= -1e-6
        horizon = max(25., advanced.get(shot, 25.))
        ip = s["ip_A"][:100][s["available"][:100, 9]]
        dbc = (float(np.median(np.abs(ip))) / 5e5) ** 2 if len(ip) else None
        shots.append(dict(s, disruptive=disruptive, down_time=down,
                          adv_time_ms=advanced.get(shot), dbc_ip2=dbc,
                          valid=valid, positive=valid & (ttd <= horizon + 1e-6)))
    return shots


def _source_statistics(shots):
    """TRAIN-only observations strided to ~128 per shot/channel, as in the source scaler."""
    result = {k: [] for k in ("norm_center", "norm_scale", "robust_median", "robust_iqr",
                              "source_mean", "source_std")}
    for channel in range(23):
        samples = []
        for s in shots:
            v = s["values"][s["valid"] & s["available"][:, channel], channel]
            if len(v):
                samples.append(v[::max(1, len(v) // 128)])
        if not samples:
            raise ValueError(f"no valid training values: channel {channel}")
        x = np.concatenate(samples).astype(np.float64)
        q25, median, q75 = np.quantile(x, [.25, .5, .75])
        values = dict(norm_center=median, norm_scale=max(q75 - q25, abs(median) * .01, 1e-6),
                      robust_median=median, robust_iqr=q75 - q25,
                      source_mean=float(x.mean()), source_std=float(x.std()))
        for key, value in values.items():
            result[key].append(value)
    return {k: np.asarray(v, dtype=np.float32) for k, v in result.items()}


def _training_pool(shots, window, device):
    xs, masks, groups, offset = [], [], {0: [], 1: []}, 0
    for s in shots:
        xs.append(np.pad(s["values"], ((window - 1, 0), (0, 0))))
        masks.append(np.pad(s["available"], ((window - 1, 0), (0, 0))))
        for label in (0, 1):
            ends = np.flatnonzero(s["valid"] & (s["positive"] == bool(label)))
            if len(ends):
                groups[label].append(ends + offset + window - 1)
        offset += len(s["values"]) + window - 1
    return (torch.as_tensor(np.concatenate(xs), device=device),
            torch.as_tensor(np.concatenate(masks), device=device), groups)


def _score(model, shots, device):
    return [dict(shot=s["shot"], times=s["times"], disruptive=s["disruptive"], down_time=s["down_time"],
                 adv_time_ms=s["adv_time_ms"], dbc_ip2=s["dbc_ip2"],
                 scores=predict_shot_proba(model, s["values"], s["available"], device=device)) for s in shots]


def _calibrate(rows, fpr_cap=.05, deadline_ms=10.):
    maxima, labels = [], []
    for r in rows:
        eligible = (np.round(1000. * (r["down_time"] - r["times"]), 6) > deadline_ms
                    if r["disruptive"] else np.ones(len(r["times"]), bool))
        maxima.append(float(r["scores"][eligible].max()) if eligible.any() else -np.inf)
        labels.append(r["disruptive"])
    maxima, labels = np.asarray(maxima), np.asarray(labels, dtype=bool)
    candidates = np.r_[np.nextafter(1., 2.), np.unique(maxima[np.isfinite(maxima)])]
    best = None
    for threshold in candidates:
        tpr, fpr = float(np.mean(maxima[labels] >= threshold)), float(np.mean(maxima[~labels] >= threshold))
        if fpr <= fpr_cap + 1e-12:
            rank = (tpr - fpr, tpr, -fpr, float(threshold))
            if best is None or rank > best:
                best = rank
    return best[-1], best[0]


def _evaluate(rows, threshold, deadline_ms=10.):
    scored = [dict(shot=r["shot"], dbc_ip2=r["dbc_ip2"],
                   **score_shot(r["times"], r["scores"], threshold,
                                r["disruptive"], r["down_time"], deadline_ms=deadline_ms,
                                adv_time_ms=r["adv_time_ms"])) for r in rows]
    return summarize_stream(scored, deadline_ms=deadline_ms, checkpoints=())


def train_source(*, epochs=40, steps=256, batch_size=64, patience=20, seed=133, device="cuda"):
    """Select epoch and threshold ONLY on VAL160; expose TEST220 once locked."""
    if min(epochs, steps, batch_size, patience) < 1:
        raise ValueError("positive training dimensions required")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for source training")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    splits, advanced = load_jtext_splits(), read_advanced_time("J-TEXT")
    train = _load_supervised(splits["train"], advanced)
    val = _load_supervised(splits["val"], advanced)
    statistics = _source_statistics(train)
    model = Simple23(statistics["norm_center"], statistics["norm_scale"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps)
    x, a, groups = _training_pool(train, model.cfg["window"], device)
    offsets = torch.arange(-model.cfg["window"] + 1, 1, device=device)
    best_rank, selected, stale, history = None, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses, started = [], time.perf_counter()
        for _ in range(steps):
            labels = rng.integers(0, 2, batch_size)
            # Uniform class → uniform eligible shot → uniform endpoint.
            ends = [int(rng.choice(groups[int(y)][int(rng.integers(len(groups[int(y)])))])) for y in labels]
            idx = torch.as_tensor(ends, device=device)[:, None] + offsets
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x[idx], a[idx]),
                                                       torch.as_tensor(labels, dtype=torch.float32, device=device))
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        val_rows = _score(model, val, device)
        threshold, j = _calibrate(val_rows)
        measured = _evaluate(val_rows, threshold)
        rank = (j, measured["roc_auc"], measured["timely_tpr"], -measured["fpr"])
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), val=measured,
                            threshold=threshold, seconds=time.perf_counter() - started))
        print(json.dumps(history[-1]), flush=True)
        if best_rank is None or rank > best_rank:
            best_rank, stale = rank, 0
            selected = dict(state_dict=copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                            arch_config=model.cfg, epoch=epoch, val_threshold=threshold,
                            val_threshold_fpr5=threshold,
                            val_threshold_fpr2p5=_calibrate(val_rows, fpr_cap=.025)[0],
                            val_metrics=measured, **statistics)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(selected["state_dict"])
    test = _load_supervised(splits["test"], advanced)
    selected["test220_metrics"] = _evaluate(_score(model, test, device), selected["val_threshold"])
    selected["training"] = dict(seed=seed, epochs=epochs, steps=steps, batch_size=batch_size,
                                selected_epoch=selected["epoch"], history=history,
                                positive_window="max(25 ms, AdvancedTime_J-TEXT); missing floor=25 ms",
                                selection="VAL160 max Youden J at FPR<=5%, AUC tie-break; TEST220>10ms",
                                statistics="TRAIN-only per-channel samples strided to ~128 per shot; source_std ddof=0",
                                split_sha256=hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest())
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(selected, ARTIFACTS / "jtext_source.pt")
    print(json.dumps(dict(completed=True, selected_epoch=selected["epoch"],
                          threshold=selected["val_threshold"], test220=selected["test220_metrics"])), flush=True)
    return model.eval(), selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=133)
    args = parser.parse_args()
    train_source(epochs=args.epochs, steps=args.steps, batch_size=args.batch_size,
                 patience=args.patience, seed=args.seed)
