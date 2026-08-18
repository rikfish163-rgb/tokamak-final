"""Shared primitives for J-TEXT -> EAST transfer-method sweep (Transformer backbone)."""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")).resolve()
OUT = ROOT / "experiments" / "transfer_methods_2026"

CHANNELS = (
    "Bt", "CIII", "MNM", "Mir_VV", "Mir_avg_amp", "Mir_avg_fre",
    "Z_proxy", "dx_a", "dy_a", "ip", "mode_number_n", "n=1 amplitude",
    "ne_nG", "qa_proxy", "sxr_kurt", "sxr_mean", "sxr_skew", "sxr_var",
    "v_loop", "xuv_kurt", "xuv_ratio", "xuv_skew", "xuv_var",
)
N_CH = len(CHANNELS)
CIDX = {name: i for i, name in enumerate(CHANNELS)}

# feature-subset presets (zero-out the rest so the same backbone can ablate channels)
FEATURE_PRESETS = {
    "full": list(CHANNELS),
    "top8": ["Bt", "ip", "v_loop", "Mir_avg_fre", "sxr_kurt", "sxr_skew", "xuv_kurt", "xuv_skew"],
    "control12": ["Bt", "ip", "v_loop", "Z_proxy", "qa_proxy", "ne_nG", "dx_a", "dy_a",
                  "sxr_mean", "xuv_ratio", "CIII", "Mir_VV"],
    "radiation13": ["CIII", "sxr_mean", "sxr_var", "sxr_skew", "sxr_kurt",
                    "xuv_ratio", "xuv_var", "xuv_skew", "xuv_kurt", "Mir_VV",
                    "Mir_avg_amp", "Mir_avg_fre", "MNM"],
    "mhd6": ["MNM", "Mir_VV", "Mir_avg_amp", "Mir_avg_fre", "mode_number_n", "n=1 amplitude"],
    # leave-one-group-out presets (full 23 minus one radiation/MHD subgroup)
    "no_sxr": [c for c in CHANNELS if not c.startswith("sxr_")],
    "no_xuv": [c for c in CHANNELS if not c.startswith("xuv_")],
    "no_mirnov": [c for c in CHANNELS if c not in ("Mir_VV", "Mir_avg_amp", "Mir_avg_fre")],
    "no_ciii": [c for c in CHANNELS if c != "CIII"],
    "no_mnm": [c for c in CHANNELS if c != "MNM"],
    "sxr4": ["sxr_mean", "sxr_var", "sxr_skew", "sxr_kurt"],
    "xuv4": ["xuv_ratio", "xuv_var", "xuv_skew", "xuv_kurt"],
    "mir3": ["Mir_VV", "Mir_avg_amp", "Mir_avg_fre"],
}
ACTIVE_CHANNELS: list[str] | None = None  # None => full; else subset kept active


def set_active_channels(names: Sequence[str] | None, preset: str | None = None):
    global ACTIVE_CHANNELS
    if preset is not None:
        ACTIVE_CHANNELS = list(FEATURE_PRESETS[preset])
    elif names is not None:
        ACTIVE_CHANNELS = [str(n) for n in names]
    else:
        ACTIVE_CHANNELS = None


def channel_mask() -> np.ndarray:
    if ACTIVE_CHANNELS is None:
        return np.ones(N_CH, dtype=np.float32)
    m = np.zeros(N_CH, dtype=np.float32)
    for n in ACTIVE_CHANNELS:
        m[CIDX[n]] = 1.0
    return m

# ---- label / window contract -------------------------------------------------
WINDOW = 128          # input history points (1000 Hz => 128 ms)
NEG_MAX_MS = 1200.0   # early negatives: window end within (HORIZON, NEG_MAX] ms before
POSITIVE_BANDS_MS = {"J-TEXT": (10.0, 25.0), "EAST": (30.0, 125.0)}
PATCH = 8
STRIDE = 8
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
ARCHITECTURE_VERSION = "patch-transformer-sinusoidal-position-v2"
TEST_CONTRACT_VERSION = "east-hdf5-authoritative-balanced360-v1"
EVALUATION_VERSION = "full-causal-trace-sustained10-lead30-v1"
TRAINING_LABEL_VERSION = "device-bands-jtext10-25-east30-125-v1"
PREPROCESSING_VERSION = "domain-local-robust-scaler-east110-v1"


def require_checkpoint_contract(checkpoint: dict) -> None:
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("checkpoint architecture contract is missing or obsolete")
    if checkpoint.get("test_contract_version") != TEST_CONTRACT_VERSION:
        raise ValueError("checkpoint TEST contract is missing or obsolete")
    if checkpoint.get("evaluation_version") != EVALUATION_VERSION:
        raise ValueError("checkpoint evaluation contract is missing or obsolete")
    if checkpoint.get("training_label_version") != TRAINING_LABEL_VERSION:
        raise ValueError("checkpoint training-label contract is missing or obsolete")
    if checkpoint.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("checkpoint preprocessing contract is missing or obsolete")


def load_shot(path: Path, device: str) -> dict:
    """Load one HDF5 shot into standardized values/times/meta."""
    with h5py.File(path, "r") as f:
        vals = []
        starts = []
        rates = []
        for name in CHANNELS:
            d = f[f"data/{name}"]
            v = np.asarray(d[:], dtype=np.float32)
            if v.ndim == 0:
                v = v.reshape(1)
            starts.append(float(np.asarray(d.attrs.get("StartTime", 0.0)).reshape(-1)[0]))
            rates.append(float(np.asarray(d.attrs.get("SampleRate", 1000.0)).reshape(-1)[0]))
            vals.append(v)
        down = np.asarray(f["meta/DownTime"][()]).reshape(-1)[0]
        isd = bool(np.asarray(f["meta/IsDisrupt"][()]).reshape(-1)[0])
    lengths = {v.shape[0] for v in vals}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent channel lengths in {path}")
    x = np.stack(vals, axis=1).astype(np.float32)          # [T, C]
    finite = np.isfinite(x)
    # sentinel / all-zero MHD handling (mirrors atlas23 availability, simplified)
    avail = finite.copy()
    ci = CIDX["CIII"]
    avail[:, ci] &= ~np.isclose(x[:, ci], -10.0, atol=1e-2)
    # harmonize units: J-TEXT ip kA -> A; Z_proxy = v_loop / ip
    if device == "J-TEXT":
        x[:, CIDX["ip"]] *= 1000.0
    ip = x[:, CIDX["ip"]]
    vl = x[:, CIDX["v_loop"]]
    valid = avail[:, CIDX["ip"]] & avail[:, CIDX["v_loop"]] & (np.abs(ip) > 1e-12)
    x[:, CIDX["Z_proxy"]] = 0.0
    x[np.where(valid), CIDX["Z_proxy"]] = vl[np.where(valid)] / ip[np.where(valid)]
    avail[:, CIDX["Z_proxy"]] = valid
    x = np.where(avail, x, 0.0)
    t0 = float(starts[0])
    rate = float(rates[0])
    t = t0 + np.arange(x.shape[0], dtype=np.float64) / rate
    down_raw = float(down) if isd else None
    return {
        "values": x, "times": t, "down_time_s": down_raw,
        "is_disrupt": isd, "rate": rate, "path": str(path),
    }


def _load_splits() -> dict:
    root = ROOT / "splits"
    out = {}
    for name in ("train", "val", "test"):
        with open(root / f"{name}.txt") as fh:
            out[name] = {int(line.strip()) for line in fh if line.strip()}
    return out


def _load_east_splits(root: Path = ROOT) -> dict:
    """Load the transfer support and authoritative TEST cohort fail-closed."""

    with open(root / "config/splits/east_paper131_split_all_in.json") as fh:
        allin = json.load(fh)
    transfer_disrupt = [int(s) for s in allin["train"]["disruption_shots"]]
    transfer_safe = [int(s) for s in allin["train"]["non_disruption_shots"]]
    test_manifest = json.loads(
        (
            root
            / "config/splits/east_hdf5_authoritative_balanced360_v1.json"
        ).read_text()
    )
    test_disrupt = [int(s) for s in test_manifest["test"]["disruption_shots"]]
    test_safe = [int(s) for s in test_manifest["test"]["non_disruption_shots"]]

    roles = {
        "transfer disruption": transfer_disrupt,
        "transfer safe": transfer_safe,
        "test disruption": test_disrupt,
        "test safe": test_safe,
    }
    for role, values in roles.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate shot IDs in {role}")
    transfer_disrupt_set = set(transfer_disrupt)
    transfer_safe_set = set(transfer_safe)
    test_disrupt_set = set(test_disrupt)
    test_safe_set = set(test_safe)
    if transfer_disrupt_set & transfer_safe_set:
        raise ValueError("label overlap in EAST transfer support")
    if test_disrupt_set & test_safe_set:
        raise ValueError("label overlap in authoritative EAST TEST")
    transfer = transfer_disrupt_set | transfer_safe_set
    test_locked = test_disrupt_set | test_safe_set
    overlap = transfer & test_locked
    if overlap:
        raise ValueError(
            f"train/test overlap in EAST split contract: {sorted(overlap)}"
        )
    return {
        "transfer": transfer,
        "transfer_disrupt": transfer_disrupt_set,
        "transfer_safe": transfer_safe_set,
        "test_locked": test_locked,
        "test_disrupt": test_disrupt_set,
        "test_safe": test_safe_set,
        "test_label_authority": str(test_manifest["label_authority"]),
    }


@dataclass
class DataSpec:
    jtext_split: dict = field(default_factory=_load_splits)
    east: dict = field(default_factory=_load_east_splits)

    def shot_path(self, device: str, shot: int) -> Path:
        bucket = str(shot // 100 * 100)
        return DATA / device / "unified_hdf5" / bucket / f"{shot}.hdf5"

    def east_native_ids(self, exclude_360: bool = True) -> list[int]:
        """All EAST HDF5 shots, optionally excluding the 360 balanced test set."""
        ids = []
        base = DATA / "EAST" / "unified_hdf5"
        for dp, _, fns in os.walk(base):
            for fn in fns:
                if fn.endswith(".hdf5"):
                    ids.append(int(fn[:-5]))
        if exclude_360:
            excl = self.east["test_locked"]
            ids = [s for s in ids if s not in excl]
        return sorted(ids)


class ModeCatShots:
    """Loads only the shots needed per role into memory."""

    def __init__(self, device: str, shot_ids: Sequence[int], spec: DataSpec, label_role: str,
                 scaler: tuple[np.ndarray, np.ndarray] | None = None,
                 label_map: dict[int, bool] | None = None):
        self.device = device
        self.shots = []
        missing = []
        for sid in shot_ids:
            p = spec.shot_path(device, int(sid))
            if not p.exists():
                missing.append(int(sid))
                continue
            try:
                sh = load_shot(p, device)
            except Exception as e:  # noqa: BLE001
                missing.append((int(sid), repr(e)))
                continue
            sh["shot_id"] = int(sid)
            sh["device"] = device
            if label_map is not None and int(sid) in label_map:
                lab = bool(label_map[int(sid)])
                raw_down = sh["down_time_s"]
                sh["is_disrupt"] = lab
                # drop curated safe shots whose HDF5 disagrees
                sh["down_time_s"] = raw_down if lab else None
            self.shots.append(sh)
        self.label_role = label_role
        self.missing = missing
        if scaler is not None:
            self.center = np.asarray(scaler[0], dtype=np.float32).copy()
            self.scale = np.asarray(scaler[1], dtype=np.float32).copy()
        else:
            self.center = np.zeros(N_CH, dtype=np.float32)
            self.scale = np.ones(N_CH, dtype=np.float32)
            self._fit_scaler()
        for sh in self.shots:
            self._transform(sh)

    def _fit_scaler(self):
        for c in range(N_CH):
            acc = []
            for sh in self.shots:
                v = sh["values"][:, c]
                v = v[np.isfinite(v)]
                if v.size:
                    idx = np.linspace(0, len(v) - 1, min(64, len(v)), dtype=int)
                    acc.extend(v[idx].tolist())
            if acc:
                a = np.asarray(acc, dtype=np.float64)
                q = np.quantile(a, [0.25, 0.5, 0.75])
                self.center[c] = float(q[1])
                self.scale[c] = max(float(q[2] - q[0]), 1e-6)

    def _transform(self, sh: dict):
        sh["values"] = np.clip(
            (sh["values"] - self.center) / self.scale, -10.0, 10.0
        ).astype(np.float32)
        sh["values"] = sh["values"] * channel_mask()

    def __len__(self):
        return len(self.shots)


class WindowSampler:
    """Samples [WINDOW, N_CH] windows with causal time-to-disruption labels."""

    def __init__(self, shots: Sequence[dict], seed: int = 0):
        self.shots = [s for s in shots if len(s["values"]) >= WINDOW]
        self.rng = np.random.default_rng(seed)
        self.valid_pos = []
        self.valid_neg = []
        for sh in self.shots:
            n = len(sh["values"])
            t = sh["times"]
            rate = sh["rate"]
            if sh["is_disrupt"] and sh["down_time_s"] is not None:
                d = sh["down_time_s"]
                lead_min, lead_max = POSITIVE_BANDS_MS[sh["device"]]
                pos = []
                neg = []
                for end in range(WINDOW - 1, n):
                    e = t[end]
                    ttd_ms = (d - e) * 1000.0
                    # Millisecond grids represented as float seconds otherwise
                    # lose exact boundary samples (for example 30 ms -> 29.999...).
                    eps_ms = 1e-6
                    if lead_min - eps_ms <= ttd_ms <= lead_max + eps_ms:
                        pos.append(end)
                    elif lead_max + eps_ms < ttd_ms <= NEG_MAX_MS + eps_ms:
                        neg.append(end)
                self.valid_pos.append(pos)
                self.valid_neg.append(neg)
            else:
                self.valid_pos.append([])
                self.valid_neg.append(list(range(WINDOW - 1, n)))

    def sample(self, pos_prob: float = 0.5):
        while True:
            i = int(self.rng.integers(len(self.shots)))
            pos = self.valid_pos[i]
            neg = self.valid_neg[i]
            if not pos and not neg:
                continue
            use_pos = bool(self.rng.random() < pos_prob) and bool(pos)
            if use_pos:
                end = pos[int(self.rng.integers(len(pos)))]
                label = 1.0
            else:
                if not neg:
                    continue
                end = neg[int(self.rng.integers(len(neg)))]
                label = 0.0
            sh = self.shots[i]
            x = sh["values"][end - WINDOW + 1: end + 1]
            return np.ascontiguousarray(x, dtype=np.float32), np.float32(label)

    def n_disrupt(self):
        return sum(1 for s in self.shots if s["is_disrupt"])

    def n_safe(self):
        return sum(1 for s in self.shots if not s["is_disrupt"])


def sinusoidal_position_encoding(h: torch.Tensor) -> torch.Tensor:
    """Return deterministic positions for a [batch, time, width] token tensor."""

    length, width = h.shape[1], h.shape[2]
    position = torch.arange(length, device=h.device, dtype=h.dtype).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, device=h.device, dtype=h.dtype)
        * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros((length, width), device=h.device, dtype=h.dtype)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


def full_trace_scores(
    enc: nn.Module,
    head: nn.Module,
    values: np.ndarray,
    *,
    window: int,
    device: str,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every causal endpoint without consulting outcome or event time."""

    endpoints = np.arange(int(window) - 1, len(values), dtype=np.int64)
    if endpoints.size == 0:
        return endpoints, np.empty(0, dtype=np.float32)
    scores = np.empty(len(endpoints), dtype=np.float32)
    enc.eval()
    head.eval()
    for start in range(0, len(endpoints), int(batch_size)):
        batch = endpoints[start : start + int(batch_size)]
        x = torch.from_numpy(
            np.stack([values[idx - int(window) + 1 : idx + 1] for idx in batch])
        ).to(device)
        with torch.no_grad():
            scores[start : start + len(batch)] = torch.sigmoid(head(enc(x))).cpu().numpy()
    return endpoints, scores


def full_trace_max_score(
    enc: nn.Module,
    head: nn.Module,
    values: np.ndarray,
    *,
    window: int,
    device: str,
    batch_size: int = 512,
) -> tuple[float, int | None]:
    """Return the maximum score from the complete outcome-blind causal trace."""

    endpoints, scores = full_trace_scores(
        enc, head, values, window=window, device=device, batch_size=batch_size
    )
    if not len(scores):
        return 0.0, None
    winner = int(np.argmax(scores))
    return float(scores[winner]), int(endpoints[winner])


def sustained_trace_score(scores: np.ndarray, sustained_samples: int) -> float:
    """Highest threshold supported by at least ``sustained_samples`` points."""

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    width = int(sustained_samples)
    if width <= 0:
        raise ValueError("sustained_samples must be positive")
    if len(scores) < width:
        return 0.0
    rolling_min = np.lib.stride_tricks.sliding_window_view(scores, width).min(axis=1)
    return float(rolling_min.max())


def first_sustained_confirmation(
    scores: np.ndarray, *, threshold: float, sustained_samples: int
) -> int | None:
    """Index of the first sample confirming a sustained threshold crossing."""

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    width = int(sustained_samples)
    if width <= 0:
        raise ValueError("sustained_samples must be positive")
    if len(scores) < width:
        return None
    passing = np.lib.stride_tricks.sliding_window_view(scores, width).min(axis=1) >= threshold
    starts = np.flatnonzero(passing)
    return None if not len(starts) else int(starts[0] + width - 1)


class TransformerEnc(nn.Module):
    """Channel-mixing patch embedding + causal-free transformer encoder."""

    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                 dropout=0.1, patch=PATCH, stride=STRIDE):
        super().__init__()
        self.embed = nn.Conv1d(N_CH, d_model, kernel_size=patch, stride=stride)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x):
        # x: [B, W, C] -> [B, C, W] -> tokens [B, n_tokens, d]
        h = self.embed(x.transpose(1, 2)).transpose(1, 2)
        h = h + sinusoidal_position_encoding(h)
        cls = self.cls.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(h)
        z = self.norm(h[:, 0])
        return z

    def forward_tokens(self, x):
        # token sequence WITHOUT cls, for masked-autoencoding reconstruction
        h = self.embed(x.transpose(1, 2)).transpose(1, 2)
        h = h + sinusoidal_position_encoding(h)
        h = self.encoder(h)
        return h


class Head(nn.Module):
    def __init__(self, d_model=D_MODEL, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


class DomainDisc(nn.Module):
    def __init__(self, d_model=D_MODEL, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


@dataclass
class RunConfig:
    method: str
    epochs: int = 4
    batch: int = 64
    lr: float = 3e-4
    seed: int = 131
    device: str = "cuda"
    out_dir: str = str(OUT / "runs")
    features: str = "full"

    @property
    def run_out(self):
        p = Path(self.out_dir) / self.method / f"seed{self.seed}"
        p.mkdir(parents=True, exist_ok=True)
        return p
