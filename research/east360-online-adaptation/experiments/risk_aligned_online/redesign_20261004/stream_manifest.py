"""Read-only EAST inventory and outcome-blind, SHA-locked listdict ordering.

No runner/model imports, GPU use, unit conversion, resampling or file writes.
CLI JSON goes to stdout; build audit goes to stderr. Full-root builds fail
closed; --compatible-only explicitly rejects signal/timestamp QC errors,
never outcome classes. Audit retains every discovered file and rejection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np

from ..io23 import RAW23, labels_at, rate_tolerance_hz
from ..causal_input import CIII_MISSING_VALUE, CIII_MISSING_ATOL


class ManifestError(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__('stream rejected; inspect audit.errors')


def manifest_bytes(rows):
    """Exact SHA input and CLI output: canonical UTF-8 JSON plus one newline."""
    return (json.dumps(rows, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value, reason):
    a = np.asarray(value)
    if a.size != 1:
        raise ValueError(reason)
    try:
        v = float(a.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(reason) from exc
    if not np.isfinite(v):
        raise ValueError(reason)
    return v


def _shot(value):
    if isinstance(value, bool) or not str(value).isdigit() or int(value) <= 0:
        raise ValueError('shot must be a positive integer')
    return int(value)


def _dataset(f, name):
    # Never follow external/soft HDF5 links outside explicitly selected files.
    for part in (name.split('/')[0], name):
        if not isinstance(f.get(part, getlink=True), h5py.HardLink):
            raise ValueError('missing_or_linked_field:' + name)
    ds = f[name]
    if not isinstance(ds, h5py.Dataset) or ds.is_virtual or ds.external:
        raise ValueError('unsupported_dataset:' + name)
    return ds


def _signals(f, shot, path, *, rate_contract='strict'):
    try:
        ip = _dataset(f, 'data/ip')
    except ValueError as exc:
        raise ValueError('missing_channel:ip') from exc
    if ip.ndim != 1:
        raise ValueError('channel_shape:ip')
    n = len(ip)
    if not n:
        raise ValueError('empty_signal')
    rate = _scalar(ip.attrs.get('SampleRate', []), 'invalid_SampleRate')
    start = _scalar(ip.attrs.get('StartTime', []), 'invalid_StartTime')
    rate_atol = rate_tolerance_hz('EAST',rate_contract)
    if not math.isclose(rate, 1000., rel_tol=0., abs_tol=rate_atol):
        raise ValueError(f'sample_rate:{rate}; required 1000 +/- {rate_atol} Hz; no resampling')
    available, prefixes = [], {}
    for name in RAW23:
        try:
            ds = _dataset(f, 'data/' + name)
        except ValueError as exc:
            raise ValueError('missing_channel:' + name) from exc
        if ds.ndim != 1 or len(ds) != n or ds.dtype.kind not in 'iuf':
            raise ValueError('channel_shape_or_dtype:' + name)
        for attr, expected in [('SampleRate', rate), ('StartTime', start)]:
            actual = _scalar(ds.attrs.get(attr, []), 'invalid_' + attr + ':' + name)
            # Same alignment tolerance as io23.read_signals, not a new time map.
            if not np.isclose(actual, expected):
                raise ValueError('unaligned_' + attr + ':' + name)
        raw = np.asarray(ds[:], dtype=np.float32)
        mask = np.isfinite(raw)
        if name == 'MNM':
            mask[:] = False
        if name == 'CIII':
            mask &= ~np.isclose(raw, CIII_MISSING_VALUE, rtol=0., atol=CIII_MISSING_ATOL)
        available.append(int(mask.sum()))
        if name in ('ip', 'Bt'):
            prefix = raw[:100].astype(np.float64)
            prefixes[name] = (float(np.median(np.abs(prefix)))
                              if len(prefix) == 100 and np.isfinite(prefix).all() else None)
    return dict(shot=shot, provenance=dict(path=str(path), sample_count=n,
        sample_rate=rate, diagnostic_start=start, feature_names=list(RAW23),
        available_points=available, mask_contract='finite; EAST MNM unavailable; CIII sentinel',
        ciii_missing_value=CIII_MISSING_VALUE, ciii_missing_atol=CIII_MISSING_ATOL,
        units='raw HDF5; no inferred units/conversion'),
        initial_abs_ip_raw=prefixes['ip'], initial_abs_Bt_raw=prefixes['Bt'])


def _outcome(f):
    try:
        flag = _scalar(_dataset(f, 'meta/IsDisrupt')[()], 'invalid_IsDisrupt')
        # io23 opens this field for BOTH classes, but only uses it for disruptions.
        raw_down = np.asarray(_dataset(f, 'meta/DownTime')[()]).reshape(-1)
    except ValueError as exc:
        if 'missing_or_linked_field' in str(exc):
            raise ValueError('missing_outcome:' + str(exc)) from exc
        raise
    if flag not in (0., 1.):
        raise ValueError('invalid_IsDisrupt')
    down = _scalar(raw_down[:1], 'invalid_DownTime') if flag else None
    return dict(is_disrupt=bool(flag), down=down)


def _disruption_time_audit(f, down):
    """Offline sampled coverage, not a last-sample/exclusive-end rejection rule."""
    ip = _dataset(f, 'data/ip')
    if ip.ndim != 1 or not len(ip):
        raise ValueError('empty_or_invalid_diagnostic_grid')
    rate = _scalar(ip.attrs.get('SampleRate', []), 'invalid_SampleRate')
    start = _scalar(ip.attrs.get('StartTime', []), 'invalid_StartTime')
    if rate <= 0:
        raise ValueError('invalid_SampleRate')
    times = start + np.arange(len(ip))/rate  # actual rate, NEVER rounded/resampled
    last, exclusive = float(times[-1]), start + len(ip)/rate
    preterminal = times <= down  # same strict sample mask as run_arm
    labels, _ = labels_at(times, True, down, 125.)
    timely = preterminal & (np.round((down-times)*1000., 9) > 30.)
    gap = (down-last)*1000.
    exclusive_gap = (down-exclusive)*1000.
    position = ('exclusive_sampling_end' if abs(exclusive_gap) <= 1e-9 else
                'after_exclusive_sampling_end' if exclusive_gap > 0 else
                'before_first_sample' if down < start else 'within_sampled_interval')
    return dict(last_sample=last, exclusive_sampling_end=exclusive, position=position,
        unsampled_disrupt_gap_ms=max(0., gap), gap_from_exclusive_end_ms=exclusive_gap,
        preterminal_points=int(preterminal.sum()), timely_evaluation_points=int(timely.sum()),
        label_points_125ms=int(((labels > .5) & preterminal).sum()),
        timely_label_points_30_125ms=int(((labels > .5) & timely).sum()))


def inventory(*, roots=(), paths=(), shots=None, offline_outcomes=False, terminal_slack_ms=None,
              rate_contract='strict'):
    """Enumerate explicit roots/files. Never dedupe silently or use TV labels.

    Optional shots is an explicit JSON-style ID roster, not a label filter.
    Outcomes are validated separately and never added to signal records.
    readable_count counts HDF5 opens, complete_count counts input+outcome validity.
    """
    rate_atol = rate_tolerance_hz('EAST',rate_contract)
    roots, paths = tuple(roots), tuple(paths)
    if terminal_slack_ms is not None:
        terminal_slack_ms = _scalar(terminal_slack_ms, 'invalid_terminal_slack_ms')
        if terminal_slack_ms < 0:
            raise ValueError('invalid_terminal_slack_ms')
    errors, candidates = [], [Path(p).resolve() for p in paths]
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            errors.append(dict(path=str(root), kind='fatal', reason='missing_root'))
            continue
        candidates.extend(sorted(p.resolve() for p in root.rglob('*')
                                 if p.is_file() and p.suffix.lower() in ('.h5', '.hdf5')))
    requested = None
    if shots is not None:
        requested = [_shot(s) for s in shots]
        if len(requested) != len(set(requested)):
            errors.append(dict(kind='fatal', reason='duplicate_requested_shot'))
        available_ids = {int(p.stem) for p in candidates if p.stem.isdigit()}
        for sid in sorted(set(requested) - available_ids):
            errors.append(dict(shot=sid, kind='fatal', reason='missing_requested_shot'))
        candidates = [p for p in candidates if p.stem.isdigit() and int(p.stem) in requested]
    if not candidates:
        errors.append(dict(kind='fatal', reason='empty_roster'))
    ids, rows, seen_paths, seen_ids = [], [], set(), set()
    counts, warnings = Counter(), []
    for path in sorted(candidates):
        row = dict(path=str(path), shot=None, readable=False, signals=None,
                   outcome_valid=False, evaluation_valid=False)
        rows.append(row)
        try:
            sid = _shot(path.stem)
            row['shot'] = sid
            ids.append(sid)
            if str(path) in seen_paths or sid in seen_ids:
                raise ValueError('duplicate_file_or_shot')
            seen_paths.add(str(path))
            seen_ids.add(sid)
            if not path.is_file():
                raise ValueError('missing_file')
            before = path.stat()
            with h5py.File(path, 'r') as f:
                row['readable'] = True
                try:
                    row['signals'] = _signals(f, sid, path,rate_contract=rate_contract)
                except (ValueError, KeyError, OSError, TypeError) as exc:
                    errors.append(dict(path=str(path), shot=sid, kind='signal', reason=str(exc)))
                try:
                    outcome = _outcome(f)
                    row['outcome_valid'] = True
                    if offline_outcomes:
                        counts['disruptive' if outcome['is_disrupt'] else 'safe'] += 1
                    row['evaluation_valid'] = True
                    if outcome['is_disrupt']:
                        timing = _disruption_time_audit(f, outcome['down'])
                        row['disruption_time_audit'] = timing
                        row['evaluation_valid'] = timing['preterminal_points'] > 0
                        if not row['evaluation_valid']:
                            errors.append(dict(path=str(path), shot=sid, kind='qc',
                                reason='no_preterminal_diagnostic_samples'))
                        if timing['position'] == 'after_exclusive_sampling_end':
                            warnings.append(dict(shot=sid, reason='unsampled_disrupt_gap',
                                unsampled_disrupt_gap_ms=timing['unsampled_disrupt_gap_ms'],
                                gap_from_exclusive_end_ms=timing['gap_from_exclusive_end_ms']))
                        if (terminal_slack_ms is not None
                                and timing['unsampled_disrupt_gap_ms'] > terminal_slack_ms+1e-9):
                            warnings.append(dict(shot=sid, reason='disrupt_gap_exceeds_review_allowance',
                                unsampled_disrupt_gap_ms=timing['unsampled_disrupt_gap_ms'],
                                terminal_slack_ms=terminal_slack_ms))
                except (ValueError, KeyError, OSError, TypeError) as exc:
                    errors.append(dict(path=str(path), shot=sid, kind='outcome', reason=str(exc)))
            digest = _sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('file_changed_during_inventory')
            row['sha256'] = digest
            if row['signals'] is not None:
                row['signals']['provenance']['sha256'] = digest
        except (ValueError, OSError, TypeError) as exc:
            errors.append(dict(path=str(path), shot=row['shot'], kind='fatal', reason=str(exc)))
    timing_rows = [r for r in rows if 'disruption_time_audit' in r]
    result = dict(schema_version='east-stream-inventory-v2', file_count=len(candidates),
        unique_shot_count=len(set(ids)), readable_count=sum(r['readable'] for r in rows),
        signal_compatible_count=sum(r['signals'] is not None for r in rows),
        complete_count=sum(r['signals'] is not None and r['outcome_valid'] for r in rows),
        qc_eligible_count=sum(r['signals'] is not None and r['outcome_valid']
                              and r['evaluation_valid'] for r in rows),
        rate_contract=rate_contract,rate_atol_hz=rate_atol, rate_rtol=0.,
        terminal_slack_ms=terminal_slack_ms, terminal_slack_excludes_shots=False,
        qualification='io23-readable outcomes; safe down ignored; late disruptive down audited, not excluded',
        duplicate_shots=sorted(s for s, n in Counter(ids).items() if n > 1),
        roots=[str(Path(p).resolve()) for p in roots], requested_shots=requested,
        rows=rows, errors=errors, warnings=warnings,
        disruption_time_summary=dict(shot_count=len(timing_rows),
            signal_compatible_count=sum(r['signals'] is not None for r in timing_rows),
            positions=dict(Counter(r['disruption_time_audit']['position'] for r in timing_rows)),
            no_timely_evaluation_shots=[r['shot'] for r in timing_rows
                if not r['disruption_time_audit']['timely_evaluation_points']],
            no_timely_label_window_shots=[r['shot'] for r in timing_rows
                if not r['disruption_time_audit']['timely_label_points_30_125ms']]))
    if offline_outcomes:
        result['offline_outcome_counts'] = dict(counts)
    return result


def lock_order(signals, order):
    """Only diagnostic records enter this function, never outcome metadata."""
    if order not in ('shot_number', 'prefix_ip_bt'):
        raise ValueError('unknown order')
    rows = []
    for signal in signals:
        row = dict(shot=signal['shot'], provenance=signal['provenance'], order_method=order)
        if order == 'prefix_ip_bt':
            ip, bt = signal['initial_abs_ip_raw'], signal['initial_abs_Bt_raw']
            if ip is None or bt is None:
                raise ValueError(f"{row['shot']}: invalid initial 100ms Ip/Bt prefix")
            proxy = (ip / 100000.) ** 1.2 * bt
            if not np.isfinite(proxy):
                raise ValueError(f"{row['shot']}: nonfinite proxy")
            row.update(initial_abs_ip_raw=ip, initial_abs_Bt_raw=bt, proxy=proxy)
        rows.append(row)
    return sorted(rows, key=(lambda r: (r['proxy'], r['shot']))
                  if order == 'prefix_ip_bt' else (lambda r: r['shot']))


def build_manifest(*, roots=(), paths=(), shots=None, order='shot_number',
                   compatible_only=False, offline_outcomes=False, terminal_slack_ms=None,
                   rate_contract='strict'):
    if order not in ('shot_number', 'prefix_ip_bt'):
        raise ValueError('unknown order')
    roots, paths = tuple(roots), tuple(paths)
    audit = inventory(roots=roots, paths=paths, shots=shots, offline_outcomes=offline_outcomes,
                      terminal_slack_ms=terminal_slack_ms,rate_contract=rate_contract)
    audit['compatible_only'] = bool(compatible_only)
    audit['rejected_signal_shots'] = sorted({e['shot'] for e in audit['errors'] if e['kind'] == 'signal'})
    if any(e['kind'] not in ('signal', 'qc') or not compatible_only for e in audit['errors']):
        raise ManifestError(audit)
    signals = [r['signals'] for r in audit['rows']
               if r['signals'] is not None and r['evaluation_valid']]
    try:
        rows = lock_order(signals, order)
        if not rows:
            raise ValueError('empty_compatible_roster')
    except ValueError as exc:
        audit['errors'].append(dict(kind='ordering', reason=str(exc)))
        raise ManifestError(audit) from exc
    audit.update(order_method=order, manifest_count=len(rows),
        manifest_sha256=hashlib.sha256(manifest_bytes(rows)).hexdigest(),
        shot_roster_sha256=hashlib.sha256(manifest_bytes([r['shot'] for r in rows])).hexdigest(),
        order_interpretation='shot number chronological proxy ONLY; not acquisition time'
            if order == 'shot_number' else 'offline commissioning simulation ONLY; not damage/chronology')
    roots = audit['roots']
    audit['roster'] = dict(schema_version='east-stream-roster-v1', shot_ids=[r['shot'] for r in rows],
        selection=('explicit EAST roots/list; outcome-class/prediction blind; ' +
                   audit['order_interpretation'] +
                   '; io23-compatible rate/masks/outcome QC; safe down ignored; late disrupt gap audit only'),
        selection_protocol=dict(label_blind=True, order=order, interpretation=audit['order_interpretation'],
            explicit_shots=audit['requested_shots'], signal_qc_exclusions=compatible_only,
            rate_contract=rate_contract,rate_atol_hz=audit['rate_atol_hz'], rate_rtol=0.,
            actual_rate_and_times_preserved=True,
            safe_down_used=False, late_disrupt_gap_exclusion=False,
            terminal_slack_ms=audit['terminal_slack_ms'], terminal_slack_role='offline review allowance only',
            no_preterminal_points_qc=True, outcome_class_filter=False, prediction_filter=False),
        data_root=roots[0] if len(roots) == 1 else None, data_roots=roots,
        explicit_paths=[str(Path(v).resolve()) for v in paths],
        order_sha256=audit['manifest_sha256'], shot_roster_sha256=audit['shot_roster_sha256'],
        helper_sha256=_sha(Path(__file__)),
        rejected=[e for e in audit['errors'] if e['kind'] in ('signal', 'qc')])
    return rows, audit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', action='append', type=Path, default=[])
    p.add_argument('--file-list', type=Path, help='JSON list of paths; relative to list file parent')
    p.add_argument('--shots', type=Path, help='JSON list of integer shot IDs within explicit roots')
    p.add_argument('--order', choices=('shot_number', 'prefix_ip_bt'), default='shot_number')
    p.add_argument('--inventory', action='store_true')
    p.add_argument('--compatible-only', action='store_true')
    p.add_argument('--offline-outcomes', action='store_true')
    p.add_argument('--rate-contract',choices=('strict','nominal'),default='strict')
    p.add_argument('--terminal-slack-ms', type=float, default=None,
                   help='explicit disruption-gap audit allowance, e.g. 2; default None; never a late-tail filter')
    p.add_argument('--emit', choices=('order', 'roster', 'audit'), default='order',
                   help='stdout artifact; full build audit always goes to stderr')
    args = p.parse_args(argv)
    try:
        paths = []
        if args.file_list:
            raw = json.loads(args.file_list.read_text())
            if not isinstance(raw, list) or any(not isinstance(v, str) for v in raw):
                raise ValueError('file-list must be JSON list of paths')
            paths = [args.file_list.resolve().parent / v for v in raw]
        shots = json.loads(args.shots.read_text()) if args.shots else None
        if shots is not None and not isinstance(shots, list):
            raise ValueError('shots must be JSON list of integer IDs')
        kwargs = dict(roots=args.root, paths=paths, shots=shots, offline_outcomes=args.offline_outcomes,
                      terminal_slack_ms=args.terminal_slack_ms,rate_contract=args.rate_contract)
        if args.inventory:
            sys.stdout.write(manifest_bytes(inventory(**kwargs)).decode())
        else:
            rows, audit = build_manifest(**kwargs, order=args.order, compatible_only=args.compatible_only)
            artifact = rows if args.emit == 'order' else audit['roster'] if args.emit == 'roster' else audit
            sys.stdout.write(manifest_bytes(artifact).decode())
            sys.stderr.write(manifest_bytes(audit).decode())
    except ManifestError as exc:
        sys.stderr.write(manifest_bytes(exc.audit).decode())
        return 2
    except (ValueError, OSError) as exc:
        sys.stderr.write(manifest_bytes({'error': str(exc)}).decode())
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
