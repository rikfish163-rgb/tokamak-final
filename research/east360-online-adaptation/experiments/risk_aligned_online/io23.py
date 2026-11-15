"""Standalone, label-separated HDF5 access for the flat 23-channel stream.

No legacy model modules are imported.  Values (including J-TEXT ``ip``) are
returned in HDF5 units; the device harmonization belongs to ``data.py``.
``read_signals`` never opens ``/meta``.  Call ``read_outcome`` only after an
online prediction has been sealed, or in an explicitly supervised context.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / 'data'
JTEXT_INDEX = ROOT / 'artifacts/canonical/jtext_source_bundle_p131_v1/index.json'
JTEXT_ADVANCED_TIME_PATH = ROOT / 'shot_list/J-TEXT/AdvancedTime_J-TEXT.json'
EAST_COHORT = ROOT / 'config/splits/east_hdf5_authoritative_balanced360_v1.json'

# All-disk EAST audit found 234 rates differing from 1000 by floating-point
# serialization neighbors (<1e-9 Hz), not a distinct sampling regime. Accept
# only that absolute numerical tolerance; retain the ACTUAL rate/timestamps.
# Other rates need an explicit model contract, never silent rounding.
RATE_ATOL_HZ = 1e-9
# Opt-in EAST contract, grounded in SAMPLE_INTERVAL_AUDIT.md (all 138 rejected
# records). W128 span error <=0.064ms and H25 error <=0.01251ms. Source/archived
# readers retain the strict default; declared rates/times are NEVER rounded.
NOMINAL_EAST_RATE_ATOL_HZ = .5

# Canonical HDF5 channel order, not the older derived 17+8 representation.
RAW23 = ('Bt', 'CIII', 'MNM', 'Mir_VV', 'Mir_avg_amp', 'Mir_avg_fre',
         'Z_proxy', 'dx_a', 'dy_a', 'ip', 'mode_number_n', 'n=1 amplitude',
         'ne_nG', 'qa_proxy', 'sxr_kurt', 'sxr_mean', 'sxr_skew', 'sxr_var',
         'v_loop', 'xuv_kurt', 'xuv_ratio', 'xuv_skew', 'xuv_var')


def _device(device):
    if device not in ('J-TEXT', 'EAST'):
        raise ValueError('device must be J-TEXT or EAST')


def rate_tolerance_hz(device, rate_contract):
    _device(device)
    if rate_contract == 'strict':
        return RATE_ATOL_HZ
    if rate_contract == 'nominal' and device == 'EAST':
        return NOMINAL_EAST_RATE_ATOL_HZ
    raise ValueError('rate contract must be strict, or explicit EAST-only nominal')


def path_for(device, shot):
    _device(device)
    shot = int(shot)
    return DATA_ROOT / device / 'unified_hdf5' / str(shot // 100 * 100) / f'{shot}.hdf5'


def read_signals(device, shot, features='flat23', *, rate_contract='strict'):
    """Read raw flat23 without metadata, interpolation, or unit conversion.

    The planning fields are *raw* HDF5 ``ip`` and ``Bt`` (not derived input
    slots).  In particular their first 100 samples remain untouched, so an
    upper layer can apply its J-TEXT kA-to-A conversion exactly once.
    """
    _device(device)
    rate_atol = rate_tolerance_hz(device, rate_contract)
    if features != 'flat23':
        raise ValueError(f'unsupported feature mode: {features}')
    with h5py.File(path_for(device, shot), 'r') as f:
        raw = {key: np.asarray(f['data'][key][:], dtype=np.float32) for key in RAW23}
        ip = f['data/ip']
        rate = float(np.asarray(ip.attrs['SampleRate']).reshape(-1)[0])
        start = float(np.asarray(ip.attrs['StartTime']).reshape(-1)[0])
        if not math.isfinite(rate) or not math.isclose(rate,1000.,rel_tol=0.,abs_tol=rate_atol):
            raise ValueError(f'1 kHz {rate_contract} contract requires +/-{rate_atol} Hz; no resampling')
        for key in RAW23:
            ds = f['data'][key]
            if len(raw[key]) != len(raw['ip']):
                raise ValueError(f'unaligned channel length: {device} {shot} {key}')
            for attr, expected in (('SampleRate', rate), ('StartTime', start)):
                if attr not in ds.attrs or not np.isclose(
                    float(np.asarray(ds.attrs[attr]).reshape(-1)[0]), expected
                ):
                    raise ValueError(f'unaligned {attr}: {device} {shot} {key}')
    values = np.stack([raw[key] for key in RAW23], axis=1)
    available = np.isfinite(values)
    clean = np.where(available, values, 0.).astype(np.float32)
    return dict(shot=int(shot), v=clean, a=available, rate=rate, start=start,
                times=start + np.arange(len(clean)) / rate,
                feature_names=list(RAW23), feature_mode=features,
                planning_ip=np.asarray(raw['ip'], dtype=np.float32),
                planning_Bt=np.asarray(raw['Bt'], dtype=np.float32))


def read_outcome(device, shot):
    """Use HDF5 alone; malformed metadata is an error, never a shot-list fallback.

    A safe shot can have a recorded terminal DownTime, but it is not a
    disruption timestamp. Neither shot-list classes nor event-label outcomes
    may override IsDisrupt. This reader remains separate from signal access.
    """
    with h5py.File(path_for(device, shot), 'r') as f:
        flag = np.asarray(f['meta/IsDisrupt'][()]).reshape(-1)
        raw = np.asarray(f['meta/DownTime'][()]).reshape(-1)
    if flag.size != 1 or flag.dtype.kind not in 'biuf' or flag[0] not in (0, 1):
        raise ValueError(f'invalid HDF5 IsDisrupt (one binary value required): {shot}')
    disruptive = bool(flag[0])
    if disruptive and (raw.size != 1 or raw.dtype.kind not in 'biuf'
                       or not np.isfinite(raw[0])):
        raise ValueError(f'invalid disruption timestamp: {shot}')
    down = float(raw[0]) if disruptive else None
    return disruptive, down


def labels_at(times, disruptive, down, horizon):
    """Return (positive, preterminal-valid); keep the legacy 1e-6 ms tolerance."""
    if not disruptive:
        return np.zeros(len(times), np.float32), np.ones(len(times), bool)
    ttd = (down - times) * 1000
    valid = ttd >= -1e-6
    return ((ttd <= horizon + 1e-6) & valid).astype(np.float32), valid


def source_roles():
    """Read the canonical J-TEXT TRAIN1353/VAL160/TEST220 role lists."""
    if JTEXT_INDEX.is_file():
        raw = json.loads(JTEXT_INDEX.read_text(encoding='utf-8')).get('split_shot_ids')
        if not isinstance(raw, dict):
            raise ValueError('canonical J-TEXT bundle has no split_shot_ids')
        roles = {name: [int(shot) for shot in raw[name]]
                 for name in ('train', 'val', 'test')}
    else:
        roles = {name: [int(line.strip()) for line in (ROOT / f'splits/{name}.txt').read_text().splitlines()
                        if line.strip()]
                 for name in ('train', 'val', 'test')}
    counts = {name: len(ids) for name, ids in roles.items()}
    if counts != {'train': 1353, 'val': 160, 'test': 220}:
        raise ValueError(f'canonical J-TEXT role counts changed: {counts}')
    for name, ids in roles.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f'duplicate shot IDs in J-TEXT {name}')
    if any(set(roles[a]) & set(roles[b]) for a, b in
           (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('canonical J-TEXT roles overlap')
    return roles


def advanced_times():
    """Read and validate J-TEXT shot-list AdvancedTime values in milliseconds."""
    path = JTEXT_ADVANCED_TIME_PATH.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'J-TEXT AdvancedTime file does not exist: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('J-TEXT AdvancedTime must be a JSON object shot_id -> milliseconds')
    result = {}
    for raw_shot, raw_value in payload.items():
        shot = int(raw_shot)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'invalid J-TEXT AdvancedTime for shot {shot}: {raw_value!r}') from exc
        if not math.isfinite(value) or value < 0.:
            raise ValueError(f'invalid J-TEXT AdvancedTime for shot {shot}: {raw_value!r}')
        if shot in result:
            raise ValueError(f'duplicate J-TEXT AdvancedTime shot {shot}')
        result[shot] = value
    return result


def effective_horizon_ms(shot_id: int, advanced_times_ms: Mapping[int, float], *,
                         floor_ms: float = 25.) -> tuple[float, float | None, bool]:
    """Return (max(floor, AdvancedTime), raw or None, fallback-used)."""
    floor = float(floor_ms)
    if not math.isfinite(floor) or floor <= 0.:
        raise ValueError('J-TEXT AdvancedTime floor must be finite and positive')
    raw = advanced_times_ms.get(int(shot_id))
    if raw is None:
        return floor, None, True
    value = float(raw)
    if not math.isfinite(value) or value < 0.:
        raise ValueError(f'invalid AdvancedTime for shot {shot_id}: {raw!r}')
    return max(floor, value), value, False


def cohort_ids(device):
    """Return locked online test IDs; labels never enter the signal reader."""
    _device(device)
    if device == 'J-TEXT':
        return source_roles()['test']
    manifest = json.loads(EAST_COHORT.read_text(encoding='utf-8'))
    ids = manifest['test']['disruption_shots'] + manifest['test']['non_disruption_shots']
    if len(ids) != 360 or len(set(ids)) != 360:
        raise ValueError('expected distinct locked 360-shot cohort')
    return sorted(ids)
