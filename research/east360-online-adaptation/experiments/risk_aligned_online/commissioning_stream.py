"""Prepare an offline low-current -> high-current EAST commissioning replay.

Future prefixes are used ONLY by the offline schedule author. They are not
normalization/training inputs to the deployed model before its first shot.
Ip squared is a relative current-energy proxy, not measured disruption damage.
Bt has no certified unit attribute; its load proxy is only a secondary key.
"""
import argparse
import math
from pathlib import Path

import h5py
import numpy as np

from . import io23
from .redesign_20261004.run import DEFAULT_ROSTER, load_roster
from .simple_online import _durable_json, _hash


PREFIX_SECONDS = .100
CURRENT_REFERENCE_A = 100_000.


def read_prefix(sid):
    """Read only the signal prefix and clock attributes, never /meta."""
    with h5py.File(io23.path_for('EAST', sid), 'r') as f:
        ip, bt = f['data/ip'], f['data/Bt']
        rate = float(np.asarray(ip.attrs['SampleRate']).reshape(-1)[0])
        start = float(np.asarray(ip.attrs['StartTime']).reshape(-1)[0])
        if not np.isfinite(rate) or abs(rate - 1000.) > io23.NOMINAL_EAST_RATE_ATOL_HZ:
            raise ValueError('prefix needs the declared nominal 1kHz clock')
        if not np.isfinite(start):
            raise ValueError('nonfinite prefix clock origin')
        if (float(np.asarray(bt.attrs['SampleRate']).reshape(-1)[0]) != rate
                or float(np.asarray(bt.attrs['StartTime']).reshape(-1)[0]) != start):
            raise ValueError('Ip and Bt prefix clocks differ')
        n = math.ceil(PREFIX_SECONDS * rate - 1e-10)
        return dict(planning_ip=np.asarray(ip[:n], np.float64),
                    planning_Bt=np.asarray(bt[:n], np.float64), rate=rate)


def _record(sid, raw):
    rate = float(raw['rate'])
    if not np.isfinite(rate) or abs(rate - 1000.) > io23.NOMINAL_EAST_RATE_ATOL_HZ:
        raise ValueError('prefix needs the declared nominal 1kHz clock')
    n = math.ceil(PREFIX_SECONDS * rate - 1e-10)
    ip, bt = np.asarray(raw['planning_ip']), np.asarray(raw['planning_Bt'])
    if ip.ndim != 1 or bt.ndim != 1 or min(len(ip), len(bt)) < n:
        raise ValueError('insufficient initial 100ms signal coverage')
    if not np.isfinite(ip[:n]).all() or not np.isfinite(bt[:n]).all():
        raise ValueError('nonfinite initial Ip/Bt observation; no imputation')
    current, field = float(np.median(abs(ip[:n]))), float(np.median(abs(bt[:n])))
    if current <= 0 or field <= 0:
        raise ValueError('no positive initial Ip/Bt median; operating level unidentifiable')
    energy = (current / CURRENT_REFERENCE_A) ** 2
    force = current / CURRENT_REFERENCE_A * field
    if not np.isfinite(energy) or not np.isfinite(force):
        raise ValueError('nonfinite relative current/load proxy')
    return dict(shot=sid, initial_ip_A=current, initial_abs_Bt_raw=field,
                proxy=energy, current_energy_proxy=energy, force_proxy_raw=force,
                prefix_samples=n, prefix_duration_s=n / rate, actual_rate_hz=rate)


def build_stream(ids, reader=read_prefix):
    if (not ids or any(type(sid) is not int or sid <= 0 for sid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('distinct positive integer shot IDs required')
    records, rejected = [], []
    for sid in ids:
        try:
            records.append(_record(sid, reader(sid)))
        except ValueError as error:
            # Unusable observed planning prefixes cannot establish this order.
            # Keep explicit exclusions; do not replace them with future signals.
            rejected.append(dict(shot=sid, reason=str(error)))
    return sorted(records, key=lambda r: (r['proxy'], r['force_proxy_raw'], r['shot'])), rejected


def quantile_pilot(order, count):
    if type(count) is not int or not 2 <= count <= len(order):
        raise ValueError('pilot size must be an integer in [2, full stream size]')
    positions = np.rint(np.linspace(0, len(order) - 1, count)).astype(int)
    if len(set(positions)) != count:
        raise ValueError('pilot positions are not distinct')
    return [dict(order[p], full_stream_rank=int(p) + 1) for p in positions]


def publish(directory, order, original_roster, source_path, rejected, *, pilot):
    """Lock identity/order first, then access outcomes for a composition sidecar."""
    directory.mkdir(parents=True, exist_ok=False)
    _durable_json(directory / 'order.json', order)
    selection = dict(original_roster['selection_protocol'])
    selection.update(order='offline_initial100ms_current_energy',
                     interpretation='offline commissioning simulation, not historical chronology',
                     ordering_uses_outcomes=False, ordering_uses_predictions=False,
                     all_future_prefixes_used_by_offline_scheduler=True,
                     model_receives_future_prefixes=False, pilot=pilot,
                     cohort_sampling=('equal rank positions across full operating range' if pilot
                                      else 'all original roster members with usable initial Ip/Bt'),
                     proxy='(median(abs(initial100ms Ip_A))/100000)^2',
                     secondary_key='(initial Ip_A/100000)*initial abs(Bt_raw)',
                     measured_damage=False, thermal_energy_unavailable=True)
    roster = dict(schema_version='east-commissioning-roster-v1',
                  shot_ids=[r['shot'] for r in order], selection_protocol=selection,
                  selection='prefix signal quality and operating-rank only; no class balancing or prediction filtering',
                  source_roster_sha256=_hash(source_path),
                  original_selection=original_roster['selection'], rejected=rejected,
                  order_sha256=_hash(directory / 'order.json'))
    _durable_json(directory / 'roster.json', roster)
    rows = []
    for r in order:
        disruptive, _ = io23.read_outcome('EAST', r['shot'])
        rows.append(dict(shot=r['shot'], is_disrupt=bool(disruptive)))
    def counts(items):
        nd = sum(r['is_disrupt'] for r in items)
        return dict(n=len(items), n_disrupt=nd, n_safe=len(items) - nd)
    q = math.ceil(len(rows) / 4)
    _durable_json(directory / 'composition.json', dict(
        scope='OFFLINE EVALUATION SIDECAR; not supplied to predictor or used for ordering',
        outcome_reads_after_order_and_roster_locked=True,
        roster_sha256=_hash(directory / 'roster.json'),
        full=counts(rows), first30=counts(rows[:30]), after30=counts(rows[30:]),
        quarters=[counts(rows[i:i+q]) for i in range(0, len(rows), q)]))
    return roster


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-roster', type=Path, default=DEFAULT_ROSTER)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pilot-count', type=int, default=240)
    args = p.parse_args()
    if args.output.exists():
        p.error('new output directory required; preserve earlier orders')
    ids, original = load_roster(args.source_roster)
    order, rejected = build_stream(ids)
    pilot = quantile_pilot(order, args.pilot_count)
    publish(args.output / 'full', order, original, args.source_roster, rejected, pilot=False)
    publish(args.output / 'pilot', pilot, original, args.source_roster, rejected, pilot=True)
    print(dict(full=len(order), pilot=len(pilot), rejected=len(rejected),
               output=str(args.output)), flush=True)


if __name__ == '__main__':
    main()
