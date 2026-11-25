"""Causal zero-cold-start cross-device disruption warning engine.

Intra-shot (real-time, causal, zero labels):
  1. Causal active-channel statistical alignment shrunk toward inter-shot target memory.
  2. Per-shot initial reference risk (median & MAD in logit space over [0, ref_ms)).
  3. Relative dynamic threshold tau = sigmoid(q_anchor + max(log(M), k_spread * s0))
     with static persistence hold_ms and conservative J-TEXT startup fallback during [0, ref_ms).

Inter-shot (post-shot, bounded finite-influence, after mature outcome feedback):
  1. Safe-segment bounded EMA update of target alignment statistics (center, scale, n_eff).
  2. Root-cause reflection before updating relative multiplier M:
     - ALIGNMENT_SHIFT: residual prefix shift -> boost alignment EMA, preserve M.
     - REF_CONTAMINATED: FN on shot starting already hot -> preserve M.
     - CAUSE_UNKNOWN: no counterfactual M* in box resolves shot -> preserve M.
     - THRESHOLD_MISMATCH: bounded asymmetric damped step toward counterfactual M*,
       committed only if same-shot counterfactual replay improves outcome/margin.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable
import numpy as np

CHANNELS = 23
COHERENT6 = (3, 7, 8, 9, 12, 13)  # Mir_VV, dx_a, dy_a, ip, ne_nG, qa_proxy
EFD2 = (9, 12)                    # ip, ne_nG


@dataclass(frozen=True)
class EngineConfig:
    # Alignment (intra-shot causal + inter-shot safe-segment EMA)
    align_mode: str = "shrinkage_robust"          # "none" | "shrinkage_robust" | "efd"
    align_channels: tuple[int, ...] | str = COHERENT6
    align_prefix_ms: int = 200
    shrinkage_k: float = 5.0
    z_clip: float = 10.0
    scale_floor: float = 1e-12
    scale_ratio_bounds: tuple[float, float] = (1e-5, 1e5)
    # Per-shot relative dynamic threshold (intra-shot)
    threshold_mode: str = "logit_margin"          # "logit_margin" | "fixed"
    ref_ms: int = 100
    init_multiplier: float = 100.0
    fixed_threshold: float = 0.842946
    source_threshold: float = 0.842946
    r_floor: float = 1e-4
    ref_cap: float = 0.45
    tau_min: float = 1e-4
    tau_max: float = 0.995
    k_spread: float = 2.0
    spread_floor: float = 0.05
    eps: float = 1e-6
    hold_ms: int = 1
    sample_ms: float = 1.0
    max_gap_ms: float = 2.0
    min_lead_ms: float = 30.0
    # Inter-shot reflective memory
    enable_reflection: bool = True
    reflect_align: bool = True
    reflect_threshold: bool = True
    diagnose_reflection: bool = True              # False = blind-update ablation control
    align_ema: float = 0.02
    align_shift_thresh: float = 1.5
    align_boost: float = 2.0
    max_center_step: float = 0.25
    max_log_scale_step: float = 0.10
    max_n_eff_step: float = 1.0
    n_eff_cap: float = 200.0
    max_log_mult_step: float = 0.15
    fp_weight: float = 2.5
    fn_weight: float = 1.0
    mult_bounds: tuple[float, float] = (1.2, 5000.0)

    def __post_init__(self):
        if self.align_mode not in {"none", "efd", "shrinkage_robust"}:
            raise ValueError("unknown align_mode")
        if self.threshold_mode not in {"fixed", "logit_margin"}:
            raise ValueError("unknown threshold_mode")
        if isinstance(self.align_channels, str) and self.align_channels not in {"all", "coherent6", "efd2"}:
            raise ValueError("unknown align_channels")
        if not isinstance(self.align_channels, str) and any(j < 0 or j >= CHANNELS for j in self.align_channels):
            raise ValueError("channel indices must be in [0, 23)")
        if not (0 < self.r_floor < self.ref_cap < 1 and 0 < self.tau_min <= self.tau_max < 1):
            raise ValueError("invalid probability bounds")
        if any(not 0 < v <= 1 for v in (self.source_threshold, self.fixed_threshold)):
            raise ValueError("invalid threshold value")
        if self.ref_ms <= 0 or self.align_prefix_ms <= 0 or self.shrinkage_k <= 0 or self.scale_floor <= 0 or self.hold_ms < 1:
            raise ValueError("timing, shrinkage and scale floor must be positive")
        if not (self.mult_bounds[0] <= self.init_multiplier <= self.mult_bounds[1] and self.mult_bounds[0] > 1.0):
            raise ValueError("init_multiplier outside mult_bounds")
        if not (0 < self.sample_ms <= self.max_gap_ms and 0 <= self.align_ema <= 1):
            raise ValueError("invalid timing or EMA")
        if self.z_clip <= 0 or not (0 < self.eps < 0.5 and 0 < self.scale_ratio_bounds[0] <= self.scale_ratio_bounds[1]):
            raise ValueError("invalid alignment regularization")
        if any(not np.isfinite(v) or v < 0 for v in vars(self).values() if isinstance(v, (int, float))):
            raise ValueError("non-finite or negative numeric parameter")


@dataclass
class InterShotState:
    target_center: np.ndarray
    target_scale: np.ndarray
    target_n_eff: np.ndarray
    multiplier: float
    shots_seen: int = 0
    consecutive_fp: int = 0
    consecutive_fn: int = 0

    @classmethod
    def from_source(cls, source_center, source_scale, config: EngineConfig):
        c, s = _source(source_center, source_scale, config)
        return cls(c.copy(), s.copy(), np.zeros(CHANNELS), float(config.init_multiplier))


@dataclass(frozen=True)
class IntraShotResult:
    shot: int
    times: np.ndarray
    risk_curve: np.ndarray
    threshold_curve: np.ndarray
    baseline_curve: np.ndarray
    alarm: bool
    alarm_idx: int | None
    alarm_time: float | None
    ref_risk: float
    ref_spread: float
    ref_contaminated: bool
    max_risk: float
    safe_prefix_center: np.ndarray
    safe_prefix_scale: np.ndarray
    safe_prefix_valid: np.ndarray
    channel_z_shift: np.ndarray
    aligned_features: np.ndarray


@dataclass(frozen=True)
class ReflectionRecord:
    shot: int
    outcome: str
    root_cause: str
    reason: str
    lead_ms: float | None
    safe_samples: int
    counterfactual_multiplier: float | None
    before: dict
    after: dict
    applied: bool
    max_center_delta_z: float
    max_log_scale_delta: float


def _source(center, scale, cfg: EngineConfig):
    c, s = np.asarray(center, float), np.asarray(scale, float)
    if c.shape != (CHANNELS,) or s.shape != (CHANNELS,) or not np.isfinite(c).all() or not np.isfinite(s).all():
        raise ValueError("source statistics must be finite (23,) vectors")
    return c, np.maximum(np.abs(s), cfg.scale_floor)


def _active(cfg: EngineConfig) -> np.ndarray:
    ch = cfg.align_channels
    if ch == "all":
        return np.arange(CHANNELS, dtype=int)
    if ch == "coherent6":
        return np.array(COHERENT6, dtype=int)
    if ch == "efd2":
        return np.array(EFD2, dtype=int)
    return np.array(ch, dtype=int)


def _stats(x, fallback_c, fallback_s, cfg: EngineConfig, mean: bool = False, cols=None):
    c, s, count = fallback_c.copy(), fallback_s.copy(), np.sum(np.isfinite(x), axis=0)
    indices = np.flatnonzero(count) if cols is None else [int(j) for j in cols if count[j] > 0]
    for j in indices:
        v = x[np.isfinite(x[:, j]), j]
        c[j] = np.mean(v) if mean else np.median(v)
        s[j] = np.std(v) if mean else np.subtract(*np.percentile(v, [75, 25]))
    valid = (count >= 2) & (s >= cfg.scale_floor)
    return np.where(valid, c, fallback_c), np.where(valid, s, fallback_s), count, valid


def _clip_for_stats(x, ref_c, ref_s, has_target, cfg: EngineConfig):
    out = x.copy()
    if np.any(has_target):
        lo = ref_c[has_target] - cfg.z_clip * ref_s[has_target]
        hi = ref_c[has_target] + cfg.z_clip * ref_s[has_target]
        out[:, has_target] = np.clip(out[:, has_target], lo, hi)
    return out


def _align(times, x, state: InterShotState, cfg: EngineConfig, sc, ss):
    y = x.copy()
    end = int(np.searchsorted((times - times[0]) * 1000.0, cfg.align_prefix_ms, side="left"))
    active = _active(cfg)
    has_target = state.target_n_eff > 0
    if cfg.align_mode != "none" and len(active):
        if cfg.align_mode == "efd" and np.all(has_target[active]):
            tc, ts = state.target_center, state.target_scale
            eff = ss / np.clip(ss / ts, *cfg.scale_ratio_bounds)
            z = np.clip((x[:, active] - tc[active]) / eff[active], -cfg.z_clip, cfg.z_clip)
            y[:, active] = sc[active] + ss[active] * z
        else:
            tc, ts = state.target_center.copy(), state.target_scale.copy()
            steps = [i for i in range(end) if i < 8 or i % 5 == 0]
            if end < len(x):
                steps.append(end)
            for pos, i in enumerate(steps):
                n = min(i + 1, end)
                clipped = _clip_for_stats(x[:n], state.target_center, state.target_scale, has_target, cfg)
                pc, ps, pn, valid = _stats(clipped, state.target_center, state.target_scale, cfg, cfg.align_mode == "efd", cols=active)
                if cfg.align_mode == "efd":
                    tc = np.where(has_target, state.target_center, pc)
                    ts = np.where(has_target, state.target_scale, ps)
                else:
                    wp = np.where(valid, pn / cfg.shrinkage_k, 0.0)
                    wh = state.target_n_eff
                    total = wp + wh
                    has_any = total > 0
                    safe_total = np.where(has_any, total, 1.0)
                    tc = np.where(has_any, (wp * pc + wh * state.target_center) / safe_total, sc)
                    var = (wp * (ps**2 + (pc - tc)**2) + wh * (state.target_scale**2 + (state.target_center - tc)**2)) / safe_total
                    ts = np.where(has_any, np.maximum(np.sqrt(np.maximum(var, 0.0)), cfg.scale_floor), ss)
                nxt = len(x) if i == end else (steps[pos + 1] if pos + 1 < len(steps) else end)
                rows = slice(i, nxt)
                eff = ss / np.clip(ss / ts, *cfg.scale_ratio_bounds)
                z = np.clip((x[rows][:, active] - tc[active]) / eff[active], -cfg.z_clip, cfg.z_clip)
                y[rows, active] = sc[active] + ss[active] * z
    pc, ps, _, valid = _stats(x[:end], state.target_center, state.target_scale, cfg, cfg.align_mode == "efd")
    apc, _, _, avalid = _stats(y[:end], sc, ss, cfg, cfg.align_mode == "efd")
    return y, pc, ps, valid, apc, avalid


def _logit(r, cfg: EngineConfig):
    p = np.clip(r, cfg.eps, 1.0 - cfg.eps)
    return np.log(p) - np.log1p(-p)


def _sigmoid(q):
    return np.exp(-np.logaddexp(0.0, -np.asarray(q)))


def _reference(r, elapsed, cfg: EngineConfig):
    prefix = r[elapsed < cfg.ref_ms]
    if len(prefix) == 0:
        prefix = r[:1]
    q = _logit(prefix, cfg)
    med_q = float(np.median(q))
    s0 = max(float(1.4826 * np.median(np.abs(q - med_q))), cfg.spread_floor)
    return float(_sigmoid(med_q)), s0


def _threshold(anchor_prob: float, elapsed, s0: float, cfg: EngineConfig, multiplier: float):
    if cfg.threshold_mode == "fixed":
        return np.full(len(elapsed), cfg.fixed_threshold)
    q_anchor = float(_logit(np.clip(anchor_prob, cfg.r_floor, cfg.ref_cap), cfg))
    margin = max(float(np.log(multiplier)), cfg.k_spread * s0)
    tau_val = float(np.clip(_sigmoid(min(q_anchor + margin, float(_logit(cfg.tau_max, cfg)))), cfg.tau_min, cfg.tau_max))
    tau = np.full(len(elapsed), tau_val)
    tau[elapsed < cfg.ref_ms] = cfg.source_threshold
    return tau


def _persistence(r, tau, times, cfg: EngineConfig):
    start = None
    for i, (risk, threshold) in enumerate(zip(r, tau)):
        if i and (times[i] - times[i - 1]) * 1000.0 > cfg.max_gap_ms + 1e-7:
            start = None
        if risk >= threshold:
            start = times[i] if start is None else start
            if (times[i] - start) * 1000.0 + cfg.sample_ms >= cfg.hold_ms - 1e-7:
                return i
        else:
            start = None
    return None


def _detect(r, times, cfg: EngineConfig, multiplier: float):
    elapsed = (times - times[0]) * 1000.0
    r0, s0 = _reference(r, elapsed, cfg)
    anchor = float(np.clip(r0, cfg.r_floor, cfg.ref_cap))
    b = np.where(elapsed < cfg.ref_ms, cfg.r_floor, anchor)
    tau = _threshold(anchor, elapsed, s0, cfg, multiplier)
    idx = _persistence(r, tau, times, cfg)
    return b, tau, idx, r0, s0


def run_shot_causal(shot: int, times, values, available, risk_fn: Callable, state: InterShotState,
                    config: EngineConfig, source_center, source_scale) -> IntraShotResult:
    """Strictly causal single-shot execution. No labels accepted, no state mutation."""
    t, x, mask = np.asarray(times, float), np.asarray(values, float), np.asarray(available, bool)
    if t.ndim != 1 or len(t) == 0 or x.shape != (len(t), CHANNELS) or mask.shape != x.shape or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("need increasing finite times and matching (N,23) values/availability")
    mask = mask & np.isfinite(x)
    x = np.where(mask, x, np.nan)
    sc, ss = _source(source_center, source_scale, config)
    aligned, pc, ps, valid, apc, avalid = _align(t, x, state, config, sc, ss)
    aligned = np.where(mask, aligned, 0.0).astype(np.float32)
    r = np.asarray(risk_fn(aligned, mask), float).reshape(-1)
    if r.shape != t.shape or not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError("risk_fn must return N finite probabilities in [0,1]")
    b, tau, idx, r0, s0 = _detect(r, t, config, state.multiplier)
    res_shift = np.where(valid & avalid, (apc - sc) / ss, 0.0)
    arrays = [a.copy() for a in (t, r, tau, b, pc, ps, valid, res_shift, aligned)]
    for a in arrays:
        a.flags.writeable = False
    t, r, tau, b, pc, ps, valid, shift, aligned = arrays
    return IntraShotResult(
        int(shot), t, r, tau, b, idx is not None, idx, None if idx is None else float(t[idx]),
        r0, s0, r0 > config.ref_cap, float(r.max()),
        pc, ps, valid, shift, aligned,
    )


def classify_outcome(result: IntraShotResult, disruptive: bool, down_time: float | None,
                     config: EngineConfig) -> tuple[str, float | None]:
    if not disruptive:
        return ("FP" if result.alarm else "TN"), None
    if down_time is None or not np.isfinite(down_time):
        raise ValueError("disruptive feedback requires finite down_time in seconds")
    if result.alarm_time is None or result.alarm_time > down_time + 1e-9:
        return "MISS", None
    lead = (down_time - result.alarm_time) * 1000.0
    return ("TARDY" if lead <= config.min_lead_ms else "TP_timely"), lead


def _counterfactual_multiplier(result: IntraShotResult, cfg: EngineConfig, down_time: float | None, fp: bool):
    elapsed = (result.times - result.times[0]) * 1000.0
    anchor = float(np.clip(result.ref_risk, cfg.r_floor, cfg.ref_cap))

    def alarms(m: float) -> bool:
        tau = _threshold(anchor, elapsed, result.ref_spread, cfg, m)
        i = _persistence(result.risk_curve, tau, result.times, cfg)
        return i is not None and (fp or result.times[i] <= down_time - cfg.min_lead_ms / 1000.0 + 1e-12)

    lo, hi = float(np.log(cfg.mult_bounds[0])), float(np.log(cfg.mult_bounds[1]))
    if not alarms(float(np.exp(lo))) or alarms(float(np.exp(hi))):
        return None
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if alarms(float(np.exp(mid))):
            lo = mid
        else:
            hi = mid
    return float(np.exp(hi if fp else lo))


def _diagnose(result: IntraShotResult, state: InterShotState, cfg: EngineConfig, outcome: str, down_time: float | None):
    if outcome not in {"FP", "MISS", "TARDY"}:
        return "NONE", "successful outcome; multiplier unchanged", None
    fp = outcome == "FP"
    if not cfg.diagnose_reflection:
        return "BLIND_UPDATE", "blind-update ablation mode", None
    active = _active(cfg)
    valid = active[result.safe_prefix_valid[active]]
    shift = float(np.median(np.abs(result.channel_z_shift[valid]))) if len(valid) else 0.0
    if shift > cfg.align_shift_thresh:
        return "ALIGNMENT_SHIFT", f"median active |z_shift|={shift:.4g} > {cfg.align_shift_thresh:g}", None
    if (not fp) and (result.ref_contaminated or result.ref_risk >= 0.9 * cfg.ref_cap) and result.max_risk >= 0.5:
        return "REF_CONTAMINATED", f"hot initial reference={result.ref_risk:.4g} near cap={cfg.ref_cap:g}", None
    critical = _counterfactual_multiplier(result, cfg, down_time, fp)
    if critical is None:
        return "CAUSE_UNKNOWN", "no multiplier-identifiable boundary in box; preserve M", None
    return "THRESHOLD_MISMATCH", f"counterfactual M*={critical:.6g}; log ratio={np.log(critical / state.multiplier):.4g}", critical


def _quality(result: IntraShotResult, cfg: EngineConfig, multiplier: float, down_time: float | None, fp: bool):
    _, tau, idx, _, _ = _detect(result.risk_curve, result.times, cfg, multiplier)
    grade = int(idx is None) if fp else (0 if idx is None else 2 if result.times[idx] <= down_time - cfg.min_lead_ms / 1000.0 else 1)
    eligible = np.ones(len(tau), bool) if fp else result.times <= down_time - cfg.min_lead_ms / 1000.0
    margin = result.risk_curve - tau if cfg.threshold_mode == "fixed" else _logit(result.risk_curve, cfg) - _logit(tau, cfg)
    maximum = float(np.max(margin[eligible])) if eligible.any() else -np.inf
    return grade, -maximum if fp else maximum


def _scalars(state: InterShotState) -> dict:
    return {
        "multiplier": float(state.multiplier),
        "shots_seen": int(state.shots_seen),
        "consecutive_fp": int(state.consecutive_fp),
        "consecutive_fn": int(state.consecutive_fn),
    }


def reflect_and_update(result: IntraShotResult, values, available, disruptive: bool, down_time: float | None,
                       state: InterShotState, config: EngineConfig, source_center, source_scale) -> ReflectionRecord:
    """Reveal outcome here only; diagnose under old state, update alignment EMA and gated M."""
    cfg, x, mask = config, np.asarray(values, float), np.asarray(available, bool)
    if x.shape != (len(result.times), CHANNELS) or mask.shape != x.shape:
        raise ValueError("reflection values/availability must match completed shot")
    x = np.where(mask & np.isfinite(x), x, np.nan)
    _source(source_center, source_scale, cfg)
    outcome, lead = classify_outcome(result, disruptive, down_time, cfg)
    cause, reason, critical = _diagnose(result, state, cfg, outcome, down_time)
    before, old_c, old_s = _scalars(state), state.target_center.copy(), state.target_scale.copy()
    elapsed = (result.times - result.times[0]) * 1000.0
    safe = (elapsed <= cfg.ref_ms) & (result.times < down_time - 0.250) if disruptive else result.times < result.times[-1] - 0.050

    if cfg.enable_reflection and cfg.reflect_align and safe.any():
        has_target = state.target_n_eff > 0
        clipped = _clip_for_stats(x[safe], old_c, old_s, has_target, cfg)
        c, s, _, valid = _stats(clipped, old_c, old_s, cfg, cfg.align_mode == "efd")
        active = np.zeros(CHANNELS, bool)
        active[_active(cfg)] = True
        valid &= active
        boot = valid & ~has_target
        upd = valid & has_target
        eta = np.full(CHANNELS, cfg.align_ema)
        if cause == "ALIGNMENT_SHIFT":
            eta[np.abs(result.channel_z_shift) > cfg.align_shift_thresh] *= cfg.align_boost
        eta = np.clip(eta, 0.0, 1.0) * upd
        delta = np.clip(eta * (c - old_c), -cfg.max_center_step * old_s, cfg.max_center_step * old_s)
        variance = (1.0 - eta) * old_s**2 + eta * s**2 + eta * (1.0 - eta) * (c - old_c)**2
        proposal = np.maximum(np.sqrt(variance), cfg.scale_floor)
        new_c = old_c + delta
        new_s = old_s * np.exp(np.clip(np.log(proposal / old_s), -cfg.max_log_scale_step, cfg.max_log_scale_step))
        if np.any(boot):
            new_c = np.where(boot, np.clip(c, -1e5, 1e5), new_c)
            new_s = np.where(boot, np.clip(s, cfg.scale_floor, 1e5), new_s)
        state.target_center = new_c
        state.target_scale = new_s
        state.target_n_eff = np.minimum(cfg.n_eff_cap, state.target_n_eff + cfg.max_n_eff_step * valid)

    applied = False
    if cfg.enable_reflection and cfg.reflect_threshold and cfg.threshold_mode == "logit_margin":
        fp = outcome == "FP"
        w = cfg.fp_weight if fp else cfg.fn_weight
        if cause == "BLIND_UPDATE":
            step = cfg.max_log_mult_step * (cfg.fp_weight if fp else -cfg.fn_weight)
            new_m = float(np.clip(state.multiplier * np.exp(step), *cfg.mult_bounds))
            applied = abs(new_m - state.multiplier) > 1e-12
            state.multiplier = new_m
        elif cause == "THRESHOLD_MISMATCH" and critical is not None:
            rep, opp = (state.consecutive_fp, state.consecutive_fn) if fp else (state.consecutive_fn, state.consecutive_fp)
            step = 0.5 * float(np.log(critical / state.multiplier)) * w / ((1 + opp) * np.sqrt(1 + rep))
            cap = cfg.max_log_mult_step * w
            step = float(np.clip(step, 0.0, cap) if fp else np.clip(step, -cap, 0.0))
            cand_m = float(np.clip(state.multiplier * np.exp(step), *cfg.mult_bounds))
            old_q = _quality(result, cfg, state.multiplier, down_time, fp)
            new_q = _quality(result, cfg, cand_m, down_time, fp)
            applied = new_q[0] > old_q[0] or (new_q[0] == old_q[0] and new_q[1] > old_q[1] + 1e-12)
            reason += f"; replay {old_q} -> {new_q}, applied={applied}"
            if applied:
                state.multiplier = cand_m
            else:
                cause, reason = "CAUSE_UNKNOWN", f"THRESHOLD_MISMATCH rejected by replay gate; {reason}"

    if outcome == "FP":
        state.consecutive_fp, state.consecutive_fn = state.consecutive_fp + 1, 0
    elif outcome in {"MISS", "TARDY"}:
        state.consecutive_fn, state.consecutive_fp = state.consecutive_fn + 1, 0
    else:
        state.consecutive_fp = state.consecutive_fn = 0
    state.shots_seen += 1
    return ReflectionRecord(
        result.shot, outcome, cause, reason, lead, int(safe.sum()), critical,
        before, _scalars(state), applied,
        float(np.max(np.abs(state.target_center - old_c) / old_s)),
        float(np.max(np.abs(np.log(state.target_scale / old_s)))),
    )


def verify_invariants():
    """Deterministic checks: causality, tail isolation, masking, boundedness, reflection."""
    sc, ss, t = np.zeros(CHANNELS), np.ones(CHANNELS), np.arange(700) / 1000.0
    x = np.random.default_rng(7).normal(sc, ss, (len(t), CHANNELS))
    available = np.random.default_rng(8).random(x.shape) > 0.05
    available[:, 1:3] = False
    x[:, 1:3] = -10.0

    def model(a, mask):
        assert a.dtype == np.float32 and mask.dtype == bool and np.all(a[~mask] == 0)
        return _sigmoid(a[:, 0])

    def infer(cfg, times=t, values=x, r=None):
        st = InterShotState.from_source(sc, ss, cfg)
        before = asdict(st)
        res = run_shot_causal(1, times, values, available[:len(times)], model if r is None else (lambda a, m: r), st, cfg, sc, ss)
        assert all(np.array_equal(v, getattr(st, k)) for k, v in before.items())
        return st, res

    curves = ("aligned_features", "risk_curve", "baseline_curve", "threshold_curve")
    for mode in ("none", "efd", "shrinkage_robust"):
        for threshold in ("fixed", "logit_margin"):
            cfg = EngineConfig(align_mode=mode, threshold_mode=threshold)
            st, a = infer(cfg)
            for cut in (1, 30, 100, 350):
                _, b = infer(cfg, t[:cut], x[:cut])
                assert all(np.array_equal(getattr(a, k)[:cut], getattr(b, k)) for k in curves), (mode, threshold, cut)
                assert (a.alarm_idx if a.alarm_idx is not None and a.alarm_idx < cut else None) == b.alarm_idx
            altered = x.copy()
            altered[500:] = 1e6
            other, b = infer(cfg, values=altered)
            assert a.ref_risk == b.ref_risk and np.array_equal(a.safe_prefix_center, b.safe_prefix_center)
            assert all(np.array_equal(getattr(a, k)[:500], getattr(b, k)[:500]) for k in curves)
            assert not a.safe_prefix_valid[1:3].any() and not a.channel_z_shift[1:3].any()
            reflect_and_update(a, x, available, True, 0.7, st, cfg, sc, ss)
            reflect_and_update(b, altered, available, True, 0.7, other, cfg, sc, ss)
            assert np.array_equal(st.target_center, other.target_center) and np.array_equal(st.target_scale, other.target_scale)
    print("PASS 1 prefix invariance (3 align x 2 threshold modes, masked inputs)")
    print("PASS 2 disruptive-tail isolation; PASS 3 persistent-state isolation")

    cfg = EngineConfig(align_mode="none", threshold_mode="logit_margin", init_multiplier=10.0, hold_ms=1, k_spread=0.0)
    assert infer(cfg, r=np.where(t == 0, 1.0, 0.01))[1].alarm_idx == 0
    extreme = np.full_like(x, 1e6)
    st, res = infer(cfg, values=extreme, r=np.ones(len(t)))
    st.target_n_eff[:] = 1.0
    before = asdict(st)
    rec = reflect_and_update(res, extreme, available, False, None, st, cfg, sc, ss)
    assert rec.max_center_delta_z <= cfg.max_center_step + 1e-12
    assert rec.max_log_scale_delta <= cfg.max_log_scale_step + 1e-12
    assert abs(np.log(st.multiplier / before["multiplier"])) <= cfg.max_log_mult_step * cfg.fp_weight + 1e-12
    assert np.max(st.target_n_eff - before["target_n_eff"]) <= cfg.max_n_eff_step
    print("PASS 4 startup fallback alarms at sample 0; PASS 5 finite per-shot influence under extreme inputs")

    # Reflective attribution checks: REF_CONTAMINATED (hot FN), clean FP (M up), clean FN (M down), CAUSE_UNKNOWN
    r_hot = np.where(t >= 0.2, 0.65, 0.50)
    st, res = infer(replace(cfg, init_multiplier=5.0), r=r_hot)
    old_m = st.multiplier
    rec = reflect_and_update(res, x, available, True, 0.7, st, replace(cfg, init_multiplier=5.0), sc, ss)
    assert rec.root_cause == "REF_CONTAMINATED" and not rec.applied and st.multiplier == old_m

    r_fp = np.where((t >= 0.2) & (t < 0.4), 0.30, 0.005)
    st, res = infer(cfg, r=r_fp)
    old_m = st.multiplier
    rec = reflect_and_update(res, x, available, False, None, st, cfg, sc, ss)
    assert rec.root_cause == "THRESHOLD_MISMATCH" and rec.applied and st.multiplier > old_m

    r_fn = np.where(t >= 0.2, 0.03, 0.005)
    st, res = infer(replace(cfg, init_multiplier=20.0), r=r_fn)
    old_m = st.multiplier
    rec = reflect_and_update(res, x, available, True, 0.7, st, replace(cfg, init_multiplier=20.0), sc, ss)
    assert rec.root_cause == "THRESHOLD_MISMATCH" and rec.applied and st.multiplier < old_m

    for r_flat in (np.full(len(t), 0.005), np.full(len(t), 0.80)):
        st, res = infer(cfg, r=r_flat)
        old_m = st.multiplier
        rec = reflect_and_update(res, x, available, True, 0.7, st, cfg, sc, ss)
        assert rec.root_cause in {"CAUSE_UNKNOWN", "REF_CONTAMINATED"} and not rec.applied and st.multiplier == old_m
    print("PASS 6 reflective attribution (REF_CONTAMINATED, FP->M+, FN->M-, CAUSE_UNKNOWN no-update)")
    print("All 6 algorithmic invariants passed.")


if __name__ == "__main__":
    verify_invariants()
