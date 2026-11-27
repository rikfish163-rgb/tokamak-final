"""Flat23 signals and separately revealed outcomes; all times are seconds."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(__file__).with_name("artifacts")
RAW23 = ("Bt", "CIII", "MNM", "Mir_VV", "Mir_avg_amp", "Mir_avg_fre",
         "Z_proxy", "dx_a", "dy_a", "ip", "mode_number_n", "n=1 amplitude",
         "ne_nG", "qa_proxy", "sxr_kurt", "sxr_mean", "sxr_skew", "sxr_var",
         "v_loop", "xuv_kurt", "xuv_ratio", "xuv_skew", "xuv_var")


def _check_device(device):
    if device not in ("J-TEXT", "EAST"):
        raise ValueError("device must be J-TEXT or EAST")


def _path(device, shot):
    _check_device(device)
    shot = int(shot)
    return ROOT / "data" / device / "unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"


def _scalar(value):
    a = np.asarray(value)
    if a.size != 1:
        raise ValueError("expected a scalar HDF5 attribute or outcome")
    return float(a.reshape(-1)[0])


def read_signals(device, shot):
    """Read /data only. J-TEXT ip is converted kA→A; other units are unchanged.

    CIII=-10 is an unavailable diagnostic, not a physical negative observation.
    No full-shot statistics, metadata, interpolation, or timestamp rounding.
    """
    with h5py.File(_path(device, shot), "r") as f:
        ip = f["data/ip"]
        rate, start = (_scalar(ip.attrs[k]) for k in ("SampleRate", "StartTime"))
        tolerance = .5 if device == "EAST" else 1e-9
        if not np.isfinite(start) or not np.isfinite(rate) or abs(rate - 1000.) > tolerance:
            raise ValueError(f"invalid 1 kHz contract: {device} {shot}, {rate}")
        channels = []
        for key in RAW23:
            ds = f[f"data/{key}"]
            if ds.ndim != 1 or len(ds) != len(ip):
                raise ValueError(f"unaligned channel length: {device} {shot} {key}")
            channel_rate, channel_start = (_scalar(ds.attrs[k]) for k in ("SampleRate", "StartTime"))
            if (not np.isfinite(channel_rate) or not np.isfinite(channel_start)
                    or abs(channel_rate - rate) > 1e-9 or abs(channel_start - start) > 1e-9):
                raise ValueError(f"unaligned channel timestamps: {device} {shot} {key}")
            channels.append(np.asarray(ds[:], dtype=np.float32))
    values = np.stack(channels, axis=1)
    if device == "J-TEXT":
        values[:, 9] *= 1000.
    available = np.isfinite(values)
    available[:, 0] &= (np.abs(values[:, 0]) > .5) & (np.abs(values[:, 0]) < 5.)
    available[:, 1] &= ~np.isclose(values[:, 1], -10., atol=1e-2, rtol=0.)
    available[:, 7] &= np.abs(values[:, 7]) <= 1000.
    available[:, 8] &= np.abs(values[:, 8]) <= 1000.
    if device == "EAST":
        available[:, 2] = False  # MNM is not transferable, matching simple_data.py.
    values[~available] = 0.
    return dict(shot=int(shot), values=values, available=available,
                times=start + np.arange(len(values), dtype=np.float64) / rate,
                rate=rate, start=start, ip_A=values[:, 9].copy())


def read_outcome(device, shot):
    """Reveal raw HDF5 ground truth, never a shot-list inferred label."""
    with h5py.File(_path(device, shot), "r") as f:
        flag = _scalar(f["meta/IsDisrupt"][()])
        if flag not in (0., 1.):
            raise ValueError(f"nonbinary IsDisrupt: {device} {shot}")
        if not flag:
            return False, None
        down = _scalar(f["meta/DownTime"][()])
        if not np.isfinite(down):
            raise ValueError(f"nonfinite DownTime: {device} {shot}")
    return True, down


def read_advanced_time(device):
    _check_device(device)
    payload = json.loads((ROOT / "shot_list" / device / f"AdvancedTime_{device}.json").read_text())
    result = {int(shot): float(value) for shot, value in payload.items()}
    if any(not np.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError(f"invalid AdvancedTime: {device}")
    return result


def _manifests():
    return json.loads((ARTIFACTS / "manifests.json").read_text())


def load_jtext_splits():
    payload = _manifests()["jtext_splits"]
    splits = {name: list(map(int, payload[name])) for name in ("train", "val", "test")}
    if {k: len(v) for k, v in splits.items()} != {"train": 1353, "val": 160, "test": 220}:
        raise ValueError("canonical J-TEXT split sizes changed")
    if len(set(sum(splits.values(), []))) != 1733:
        raise ValueError("duplicate or overlapping J-TEXT splits")
    return splits


def _east360():
    ids = list(map(int, _manifests()["east360"]))
    if len(ids) != 360 or len(set(ids)) != 360:
        raise ValueError("EAST360 membership changed")
    return sorted(ids)


def _archived_order():
    return list(map(int, _manifests()["east360_archived_order"]))


def compute_shot_meta(device, shot, advanced=None):
    """First 100 samples only define planning fields; outcomes are audit-only.

    DBC is a relative impact proxy, NOT an Ai-2026 disruption-probability model:
    magnetic/thermal impact ∝ initial Ip². Raw ne_nG is retained without
    assuming a shared Greenwald-fraction scale across devices.
    dbc_init=dbc_ip2=(Ip0_MA/0.5)². Raw qa_proxy likewise has no certified
    dimensionless operational-limit scale. Neither DownTime, AdvancedTime nor
    full-shot duration enters these proxies.
    """
    s = read_signals(device, shot)
    def median(channel, absolute=False):
        v = s["values"][:100, channel][s["available"][:100, channel]]
        return float(np.median(np.abs(v) if absolute else v)) if len(v) else None
    ip, bt, ne, qa = median(9, True), median(0, True), median(12), median(13)
    ip_ma = ip / 1e6 if ip is not None else None
    ip2 = (ip_ma / .5) ** 2 if ip_ma is not None else None
    disruptive, down = read_outcome(device, shot)
    if advanced is None:
        advanced = read_advanced_time(device)
    return dict(shot=int(shot), ip_init_MA=ip_ma, bt_init=bt, ne_ng_init=ne, qa_init=qa,
                dbc_init=ip2, dbc_ip2=ip2,
                disruptive=disruptive, down_time=down, adv_time_ms=advanced.get(int(shot)),
                n_samples=len(s["times"]), start_time=s["start"])


def build_east_meta(force=False):
    """Persist an audit cache; NEVER pass this outcome-containing map to a predictor."""
    path = ARTIFACTS / "east_meta.json"
    if path.exists() and not force:
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    advanced = read_advanced_time("EAST")
    ids = sorted(int(p.stem) for p in (ROOT / "data/EAST/unified_hdf5").rglob("*.hdf5"))
    meta = {shot: compute_shot_meta("EAST", shot, advanced) for shot in ids}
    ARTIFACTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(meta, allow_nan=False, separators=(",", ":")))
    return meta


def load_east_cohorts(meta_map=None):
    meta = build_east_meta() if meta_map is None else meta_map
    all_ids, cohort, archived = sorted(map(int, meta)), _east360(), _archived_order()
    if len(all_ids) != 2953 or set(archived) != set(cohort) or len(archived) != 360:
        raise ValueError("local EAST coverage or archived360 order changed")
    if sum(meta[s]["disruptive"] for s in cohort) != 180:
        raise ValueError("raw HDF5 EAST360 labels are not 180D/180N")
    return dict(dev360=archived, east_indep2593=sorted(set(all_ids) - set(cohort)))


def sort_shots(shot_ids, meta_map, mode="chrono"):
    """Planning orders use prefix-only Ip/DBC; chronology uses shot ID, not outcome."""
    ids = list(map(int, shot_ids))
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate shots")
    if mode == "chrono":
        return sorted(ids)
    if mode == "archived360":
        archived = _archived_order()
        if set(ids) != set(archived):
            raise ValueError("archived360 requires the exact locked cohort")
        return archived
    if mode not in ("ip_asc", "ip_desc", "ip_stratified", "dbc_asc", "dbc_desc"):
        raise ValueError(f"unknown shot order: {mode}")
    key = "ip_init_MA" if mode.startswith("ip_") else "dbc_init"
    observed = [s for s in ids if meta_map[s][key] is not None]
    observed.sort(key=lambda s: (meta_map[s][key], s), reverse=mode.endswith("desc"))
    if mode == "ip_stratified":
        # Ten deterministic rank strata interleaved, with no label balancing.
        strata = np.array_split(observed, min(10, len(observed))) if observed else []
        observed = [int(a[i]) for i in range(max(map(len, strata), default=0)) for a in strata if i < len(a)]
    return observed + sorted(set(ids) - set(observed))
