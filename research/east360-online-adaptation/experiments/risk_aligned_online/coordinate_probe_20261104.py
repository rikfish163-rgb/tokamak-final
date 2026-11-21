"""CPU-only, input-only coordinate audit of fixed source64 and ended EAST20."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import io23, simple_data
from .causal_input import apply_input_contract
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .warmstart_alignment_20261022.alignment import transform

ROOT = io23.ROOT
RLS = ROOT / 'probe_outputs/direct_fast_memory_ridge360_20261104_v1'
BASELINE = ROOT / 'probe_outputs/qualified_source_static360_20261104_v1'
MAPPING = ROOT / 'probe_outputs/stratified_cold20_context360_v1/input_map.json'
ORDER = ROOT / 'probe_outputs/east_decoder_complete360_v1/protocol.json'
OUTPUT = ROOT / 'probe_outputs/input_coordinate_diagnostic_20261104_v1'
RLS_DONE_SHA = 'faeeaa386e3d160b4da4a2582adda79d3a3b226029c0767fe187bcdd90e732f6'
BASELINE_DONE_SHA = 'ef41bc2db11c19d12f7fd2f6b2fea24a5208d0daa4cf5b818c996bf80d66d78f'
DEPENDENCIES = (
    'experiments/risk_aligned_online/io23.py',
    'experiments/risk_aligned_online/simple_data.py',
    'experiments/risk_aligned_online/causal_input.py',
    'experiments/risk_aligned_online/simple23.py',
    'experiments/risk_aligned_online/redesign_20261004/prepare_prior.py',
    'experiments/risk_aligned_online/warmstart_alignment_20261022/alignment.py',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def uniform_indices(n, cap=128):
    if type(n) is not int or type(cap) is not int or n < 1 or cap < 1:
        raise ValueError('positive integer length and cap required')
    return np.linspace(0, n - 1, min(n, cap), dtype=np.int64)


def select_cohorts(roles, rls_order, baseline_order, map_ids):
    train, val, test = (list(roles[k]) for k in ('train', 'val', 'test'))
    if (len(train) < 64 or any(len(set(x)) != len(x) for x in (train, val, test))
            or set(train) & set(val) or set(train) & set(test) or set(val) & set(test)):
        raise ValueError('distinct nonoverlapping source roles with >=64 TRAIN required')
    if (len(rls_order) != 360 or len(set(rls_order)) != 360
            or list(rls_order) != list(baseline_order) or list(map_ids) != list(rls_order[:20])):
        raise ValueError('exact RLS/baseline order and first20 map IDs required')
    return sorted(train)[:64], list(rls_order[:20])


def prepare_views(shot, mapping):
    if shot['device'] not in ('J-TEXT', 'EAST'):
        raise ValueError('explicit source or EAST required')
    if shot['device'] == 'EAST':
        if shot['available'][:, 2].any():
            raise ValueError('EAST simple_data contract must disable MNM')
        prepared = apply_input_contract(shot)
    else:
        prepared = shot  # source does not use the EAST CIII sentinel
    signal = {'values': prepared['values'], 'available': prepared['available']}
    views = {'unmapped': signal}
    if shot['device'] == 'EAST':
        views['mapped'] = transform(signal, mapping)
        if not np.array_equal(signal['available'], views['mapped']['available']):
            raise AssertionError('map must not change availability')
    return views


def summarize(records, center, scale, *, prefix):
    center, scale = np.asarray(center, np.float32), np.asarray(scale, np.float32)
    if (center.shape != (23,) or scale.shape != (23,) or not np.isfinite(center).all()
            or not np.isfinite(scale).all() or (scale <= 0).any() or not records):
        raise ValueError('nonempty records and finite source coordinates required')
    points, samples, finite, present = 0, [], [], np.zeros(23, dtype=int)
    center_t, scale_t = torch.from_numpy(center), torch.from_numpy(scale)
    for rec in records:
        x, mask, raw = rec['values'], rec['available'], rec['raw_finite']
        if (x.ndim != 2 or x.shape[1] != 23 or x.dtype != np.float32 or not len(x)
                or mask.shape != x.shape or mask.dtype != bool
                or raw.shape != x.shape or raw.dtype != bool or not np.isfinite(x[mask]).all()):
            raise ValueError('matching nonempty float32/boolean input arrays required')
        idx = np.arange(min(len(x), 128)) if prefix else uniform_indices(len(x))
        x, mask, raw = x[idx], mask[idx], raw[idx]
        with torch.no_grad():
            x_t, m_t = torch.from_numpy(x), torch.from_numpy(mask)
            z = torch.asinh((torch.where(m_t, x_t, center_t) - center_t) / scale_t).numpy()
        if not np.isfinite(z[mask]).all():
            raise FloatingPointError('nonfinite observed source coordinate')
        samples.append((x, mask, z))
        finite.append(raw)
        present += mask.any(axis=0)
        points += len(idx)
    finite = np.concatenate(finite, axis=0)
    result = []
    for channel in range(23):
        values = np.concatenate([x[m[:, channel], channel] for x, m, _ in samples]).astype(np.float64)
        z = np.concatenate([v[m[:, channel], channel] for _, m, v in samples]).astype(np.float64)
        n = len(values)
        item = dict(total_points=points, sample_points=points, available_points=n,
                    available_fraction=n / points, finite_hdf_points=int(finite[:, channel].sum()),
                    finite_hdf_fraction=float(finite[:, channel].mean()),
                    records_with_available=int(present[channel]))
        if n:
            q25, median, q75 = np.quantile(values, [.25, .5, .75])
            q05, zmedian, q95 = np.quantile(z, [.05, .5, .95])
            item.update(raw_median=float(median), raw_iqr=float(q75 - q25),
                        asinh_q05=float(q05), asinh_median=float(zmedian), asinh_q95=float(q95),
                        frac_abs_gt5=float(np.mean(np.abs(z) > 5)),
                        frac_abs_gt10=float(np.mean(np.abs(z) > 10)))
        else:
            item.update({key: None for key in ('raw_median', 'raw_iqr', 'asinh_q05',
                        'asinh_median', 'asinh_q95', 'frac_abs_gt5', 'frac_abs_gt10')})
        result.append(item)
    return result


def compare_raw(source, east):
    def ratio(n, d):
        return float(n / d) if n is not None and d is not None and d != 0 else None
    return {'median_east_over_source': ratio(east['raw_median'], source['raw_median']),
            'iqr_east_over_source': ratio(east['raw_iqr'], source['raw_iqr'])}


def sample_sha(arrays):
    digest = hashlib.sha256()
    for name, value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(name.encode() + b'\0')
        digest.update(str((array.dtype.str, array.shape)).encode() + b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def _format(value):
    return '—' if value is None else f'{value:.5g}'


def write_tables(path, rows):
    def quant(s):
        return ' / '.join(_format(s[k]) for k in ('asinh_q05', 'asinh_median', 'asinh_q95'))
    def tails(s):
        return ' / '.join(_format(s[k]) for k in ('frac_abs_gt5', 'frac_abs_gt10'))
    lines = ['# All 23 channels: sampled coordinate summaries', '',
             'Coordinate tuples are q05 / median / q95; counts are model-available / sampled.',
             'The first20 mapped column is retrospective; only ne_nG is actually mapped.', '',
             '| Channel | Source center / scale | Source n/N; q05 / med / q95 | EAST full raw n/N; q05 / med / q95 | EAST full mapped q05 / med / q95 | Persistent raw / mapped |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        s, e, m = (row[k] for k in ('source', 'east_full_raw', 'east_full_mapped'))
        lines.append(f"| {row['channel']} | {_format(row['center'])} / {_format(row['scale'])} | "
                     f"{s['available_points']}/{s['total_points']}; {quant(s)} | "
                     f"{e['available_points']}/{e['total_points']}; {quant(e)} | {quant(m)} | "
                     f"{row['persistent_raw']} / {row['persistent_mapped']} |")
    lines += ['', 'Fractions are |asinh| >5 / >10. Ratios are EAST / source raw numerical median / IQR; no common physical-unit claim.', '',
              '| Channel | Source fractions | EAST full raw fractions | EAST full mapped fractions | Raw median / IQR ratios | Mapped median / IQR ratios |',
              '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        r, m = row['ratio_full_raw'], row['ratio_full_mapped']
        lines.append(f"| {row['channel']} | {tails(row['source'])} | {tails(row['east_full_raw'])} | "
                     f"{tails(row['east_full_mapped'])} | {_format(r['median_east_over_source'])} / {_format(r['iqr_east_over_source'])} | "
                     f"{_format(m['median_east_over_source'])} / {_format(m['iqr_east_over_source'])} |")
    lines += ['', 'EAST first 128 actual samples per shot; no future frames enter this prefix summary.', '',
              '| Channel | Prefix raw n/N; q05 / med / q95 | Prefix mapped q05 / med / q95 | Raw >5 / >10 | Mapped >5 / >10 |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        e, m = row['east_prefix_raw'], row['east_prefix_mapped']
        lines.append(f"| {row['channel']} | {e['available_points']}/{e['total_points']}; {quant(e)} | "
                     f"{quant(m)} | {tails(e)} | {tails(m)} |")
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    started = time.monotonic()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(1)
    rls_done, baseline_done = RLS / 'completed.json', BASELINE / 'completed.json'
    if sha(rls_done) != RLS_DONE_SHA or sha(baseline_done) != BASELINE_DONE_SHA:
        raise ValueError('locked RLS and qualified-baseline completion hashes required')
    rd, bd = read_json(rls_done), read_json(baseline_done)
    if (rd.get('status') != 'COMPLETED_DIRECT_MEMORY_INTRA_DIAGNOSTIC'
            or rd['artifact_sha256']['protocol.json'] != sha(RLS / 'protocol.json')
            or bd['artifact_sha256']['protocol.json'] != sha(BASELINE / 'protocol.json')):
        raise ValueError('completed source protocols and artifact hashes required')
    rls, baseline, order, mapping = (read_json(p) for p in
                                     (RLS / 'protocol.json', BASELINE / 'protocol.json', ORDER, MAPPING))
    if (rls['source_sha256'] != PARENT_SHA256 or baseline['source_sha256'] != PARENT_SHA256
            or sha(DEFAULT_PARENT) != PARENT_SHA256
            or rls['input_sha256'][str(MAPPING)] != sha(MAPPING)
            or baseline['input_sha256'][str(MAPPING)] != sha(MAPPING)
            or order['input_map_sha256'] != sha(MAPPING)):
        raise ValueError('qualified checkpoint and exact fixed numerical map required')
    dependencies = {str(ROOT / p): sha(ROOT / p) for p in DEPENDENCIES}
    if any(rls['code_sha256'].get(p) != digest for p, digest in dependencies.items()):
        raise ValueError('input/model dependency differs from RLS code snapshot')
    rls_order = [row['shot'] for row in rls['order']]
    base_order = [row['shot'] for row in baseline['order']]
    if [row['shot'] for row in order['order']] != rls_order:
        raise ValueError('old RLS and locked order disagree')
    train_ids, east_ids = select_cohorts(simple_data.source_roles(), rls_order,
                                         base_order, mapping['calibration_ids'])
    base, _ = load_parent(DEFAULT_PARENT)
    state = hashlib.sha256()
    for name, value in base.state_dict().items():
        state.update(name.encode())
        state.update(str((value.dtype, tuple(value.shape))).encode())
        state.update(value.detach().cpu().contiguous().numpy().tobytes())
    if state.hexdigest() != baseline['source_state_sha256'] or state.hexdigest() != rls['source_state_sha256']:
        raise ValueError('qualified source state differs from baseline or RLS')
    center, scale = base.center.cpu().numpy().copy(), base.scale.cpu().numpy().copy()
    expected = [('J-TEXT', sid) for sid in train_ids] + [('EAST', sid) for sid in east_ids]
    source, east_raw, east_map, receipts = [], [], [], []
    original_reader, calls, captured = io23.read_signals, [], {}

    def guarded_reader(device, sid, *args, **kwargs):
        if len(calls) >= len(expected) or (device, sid) != expected[len(calls)]:
            raise ValueError('only predeclared ordered source64/EAST20 reads allowed')
        calls.append((device, sid))
        raw = original_reader(device, sid, *args, **kwargs)
        captured['finite'], captured['values'] = raw['a'], raw['v']
        return raw

    def forbidden(*args, **kwargs):
        raise RuntimeError('HDF outcome access is forbidden in this signal-only audit')

    with patch.object(io23, 'read_signals', side_effect=guarded_reader), \
            patch.object(io23, 'read_outcome', side_effect=forbidden), \
            patch.object(simple_data, 'attach_outcome', side_effect=forbidden):
        for device, sid in expected:
            input_path = io23.path_for(device, sid)
            st = input_path.stat()
            shot = simple_data.read_shot(device, sid, rate_contract='nominal' if device == 'EAST' else 'strict')
            views = prepare_views(shot, mapping)
            raw_finite = captured['finite']
            if raw_finite.shape != shot['available'].shape:
                raise ValueError('reader and model input masks must match shape')
            groups = [('source', views['unmapped'])] if device == 'J-TEXT' else [
                ('east_full_raw', views['unmapped']), ('east_full_mapped', views['mapped'])]
            for name, view in groups:
                {'source': source, 'east_full_raw': east_raw, 'east_full_mapped': east_map}[name].append(
                    dict(values=view['values'], available=view['available'], raw_finite=raw_finite))
            indices = uniform_indices(len(shot['values']))
            prefix = np.arange(min(128, len(shot['values'])))
            digests = {}
            for label, idx in [('spread', indices)] + ([('prefix', prefix)] if device == 'EAST' else []):
                pairs = [('indices', idx), ('hdf_reader_float32', captured['values'][idx]),
                         ('hdf_finite_mask', raw_finite[idx]), ('contract_values', views['unmapped']['values'][idx]),
                         ('contract_mask', views['unmapped']['available'][idx])]
                if device == 'EAST':
                    pairs.append(('mapped_values', views['mapped']['values'][idx]))
                digests[label] = sample_sha(pairs)
            after = input_path.stat()
            if (st.st_size, st.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'input file changed during read: {device} {sid}')
            receipts.append(dict(device=device, shot=sid, path=str(input_path), length=len(shot['values']),
                                 size=st.st_size, mtime_ns=st.st_mtime_ns,
                                 spread_indices=indices.tolist(), prefix_count=len(prefix) if device == 'EAST' else None,
                                 sample_sha256=digests))
    if calls != expected:
        raise AssertionError('actual reader calls differed from authorized cohort')
    groups = {'source': summarize(source, center, scale, prefix=False),
              'east_full_raw': summarize(east_raw, center, scale, prefix=False),
              'east_full_mapped': summarize(east_map, center, scale, prefix=False),
              'east_prefix_raw': summarize(east_raw, center, scale, prefix=True),
              'east_prefix_mapped': summarize(east_map, center, scale, prefix=True)}
    rows = []
    for c, channel in enumerate(io23.RAW23):
        row = dict(channel=channel, center=float(center[c]), scale=float(scale[c]),
                   **{name: values[c] for name, values in groups.items()})
        for suffix in ('full_raw', 'full_mapped', 'prefix_raw', 'prefix_mapped'):
            row[f'ratio_{suffix}'] = compare_raw(row['source'], row[f'east_{suffix}'])
        for suffix in ('raw', 'mapped'):
            a, b = row[f'east_full_{suffix}']['frac_abs_gt5'], row[f'east_prefix_{suffix}']['frac_abs_gt5']
            row[f'persistent_{suffix}'] = None if a is None or b is None else bool(a >= .5 and b >= .5)
        rows.append(row)
    bind_paths = [DEFAULT_PARENT, io23.JTEXT_INDEX, RLS / 'protocol.json', rls_done,
                  BASELINE / 'protocol.json', baseline_done, ORDER, MAPPING, Path(__file__),
                  ROOT / 'tests/test_coordinate_probe_20261104.py']
    hashes = {str(path): sha(path) for path in bind_paths}
    hashes.update(dependencies)
    elapsed = time.monotonic() - started
    protocol = dict(scope='SOURCE64_EAST20_INPUT_ONLY_OFFLINE_COORDINATE_DIAGNOSTIC',
                    source_ids=train_ids, east_ids=east_ids, source_sha256=PARENT_SHA256,
                    source_state_sha256=state.hexdigest(), input_sha256=hashes,
                    code_dependencies_match_rls=True, rate_contract={'J-TEXT': 'strict', 'EAST': 'nominal'},
                    sampling='per shot deterministic linspace whole record <=128; EAST first128 separate',
                    prefix_mapping='retrospective only, fit was based on the same ended20',
                    summary_weighting='pooled observed sampled points; <=128 per shot per view',
                    persistent_flag='both full and prefix sampled frac |asinh|>5 >=0.5; descriptive',
                    availability='model mask and HDF finite mask reported separately; EAST MNM disabled by simple_data',
                    raw_numeric_ratio='EAST/source medians or IQRs; source zero denominator => null; no units implied',
                    reads='io23.read_signals data only and signal rate/start attrs; no HDF outcomes',
                    sample_hash='sha256 typed sampled arrays and masks; NOT a whole-HDF file hash',
                    full_record_io=True, no_gpu=True, no_fit=True, physical_units_certified=False,
                    outcomes_read=False, independent_validation=False, elapsed_seconds=elapsed,
                    python_torch=torch.__version__, numpy=np.__version__, receipts=receipts)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, payload in [('protocol.json', protocol), ('summary.json', {'channels': rows})]:
        (OUTPUT / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_tables(OUTPUT / 'tables.md', rows)
    with (OUTPUT / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['channel', 'center', 'scale', 'group', 'available_points', 'total_points',
                         'finite_hdf_points', 'records_with_available', 'raw_median', 'raw_iqr',
                         'asinh_q05', 'asinh_median', 'asinh_q95', 'frac_abs_gt5', 'frac_abs_gt10'])
        for row in rows:
            for group in groups:
                data = row[group]
                writer.writerow([row['channel'], row['center'], row['scale'], group] +
                                [data[k] for k in ('available_points', 'total_points', 'finite_hdf_points',
                                 'records_with_available', 'raw_median', 'raw_iqr', 'asinh_q05',
                                 'asinh_median', 'asinh_q95', 'frac_abs_gt5', 'frac_abs_gt10')])
    output_hashes = {name: sha(OUTPUT / name) for name in ('protocol.json', 'summary.json', 'summary.csv', 'tables.md')}
    (OUTPUT / 'completed.json').write_text(json.dumps(dict(status='COMPLETED_SIGNAL_ONLY_CPU',
        artifact_sha256=output_hashes, elapsed_seconds=time.monotonic() - started), indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='COMPLETED_SIGNAL_ONLY_CPU', source_shots=len(source),
                          east_shots=len(east_raw), runtime_seconds=time.monotonic()-started,
                          output=str(OUTPUT))))


if __name__ == '__main__':
    main()
