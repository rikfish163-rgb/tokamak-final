"""Read-only exposure/early-alarm audit; never shortens operational scoring.

Prefix fractions are descriptive counts on already sealed curves. They are
neither replacement FPRs nor a causal estimate of the effect of shot length.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.risk_aligned_online import io23


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_crossing(scores, threshold):
    scores = np.asarray(scores)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or not np.isfinite(threshold)):
        raise ValueError('nonempty finite one-dimensional curve required')
    hits = np.flatnonzero(scores >= threshold)
    return int(hits[0]) if len(hits) else None


def prefix_summary(rows, ms):
    def crossed(r):
        return r['first'] is not None and r['first'] / r['rate'] * 1000 < ms

    observed = [r for r in rows if (r['points'] - 1) / r['rate'] * 1000 >= ms]
    count = sum(crossed(r) for r in rows)
    return dict(before_elapsed_ms=ms, false_alarms_before_ms=count,
                all_safe_shots=len(rows), fraction_of_all_safe=count/len(rows),
                observed_through_ms=len(observed),
                false_alarms_among_observed_through_ms=sum(crossed(r) for r in observed))


def raw_record(device, sid):
    with h5py.File(io23.path_for(device, sid), 'r') as h:
        ds = h['data/ip']
        count = len(ds)
        rate = float(np.asarray(ds.attrs['SampleRate']).reshape(-1)[0])
        start = float(np.asarray(ds.attrs['StartTime']).reshape(-1)[0])
        flag = bool(np.asarray(h['meta/IsDisrupt'][()]).reshape(-1)[0])
    if not count or not np.isfinite([rate, start]).all() or rate <= 0:
        raise ValueError('invalid raw observation clock')
    return dict(shot=int(sid), points=count, rate=rate, start=start, disruptive=flag,
                observed_span_ms=(count-1)/rate*1000)


def describe(rows):
    spans = np.array([r['observed_span_ms'] for r in rows])
    points = np.array([r['points'] for r in rows])
    return dict(shots=len(rows), disruptive=sum(r['disruptive'] for r in rows),
                safe=sum(not r['disruptive'] for r in rows),
                quantile_probabilities=[0, .25, .5, .75, 1],
                span_ms_quantiles=np.quantile(spans, [0, .25, .5, .75, 1]).tolist(),
                point_count_quantiles=np.quantile(points, [0, .25, .5, .75, 1]).tolist(),
                records_shorter_than128_points=int((points < 128).sum()))


def main():
    output = Path(__file__).with_suffix('.json')
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).parent / 'probe_outputs/current_order_context360_v2'
    protocol = json.loads((root/'protocol.json').read_text())
    completed = json.loads((root/'completed.json').read_text())
    assert completed['status'] == 'COMPLETED'
    order = [r['shot'] for r in json.loads((root/'order.json').read_text())]
    saved = json.loads((root/'per_shot.json').read_text())
    hashes = json.loads((root/'input_hashes.json').read_text())
    assert len(order) == len(set(order)) == 360
    source = {role: [raw_record('J-TEXT', sid) for sid in ids]
              for role, ids in io23.source_roles().items()}
    rows, safe_curves = [], {arm: [] for arm in protocol['arms']}
    for index, sid in enumerate(order, 1):
        r = raw_record('EAST', sid)
        assert digest(io23.path_for('EAST', sid)) == hashes[str(sid)]
        prefix = root / 'shots' / f'{index:04d}_{sid}'
        sealed = json.loads(Path(str(prefix)+'_seal.json').read_text())
        path = prefix.with_suffix('.npz')
        assert digest(path) == sealed['sha256'] and not sealed['outcome_available']
        r['rank'] = index
        r['first_crossings'] = {}
        with np.load(path, allow_pickle=False) as a:
            assert np.array_equal(a['times'], r['start'] + np.arange(r['points'])/r['rate'])
            for arm in protocol['arms']:
                assert a[arm].shape == (r['points'],)
                first = first_crossing(a[arm], protocol['threshold'])
                old = saved[arm][index-1]
                assert old['shot'] == sid and old['is_disrupt'] == r['disruptive']
                r['first_crossings'][arm] = first
                if not r['disruptive']:
                    assert old['alarm'] == (first is not None)
                    safe_curves[arm].append(dict(shot=sid, points=r['points'],
                                                 rate=r['rate'], first=first))
        rows.append(r)
    summaries = {}
    for arm, records in safe_curves.items():
        alarms = [r for r in records if r['first'] is not None]
        assert len(records) == 180
        assert len(alarms) == completed['metrics'][arm]['full360']['fp']
        summaries[arm] = dict(
            full_stream_fp=len(alarms), full_stream_safe=len(records),
            full_stream_fpr=len(alarms)/len(records),
            prefix_counts=[prefix_summary(records, ms) for ms in (16, 127, 128, 512, 1000, 3000)],
            first_alarm_elapsed_ms_quantiles=np.quantile(
                [r['first']/r['rate']*1000 for r in alarms], [0, .25, .5, .75, 1]).tolist())
    declared_d = set(json.loads((io23.ROOT/'shot_list/TEST/Disruption_EAST_Test.json').read_text()))
    declared_s = set(json.loads((io23.ROOT/'shot_list/TEST/Non-disruption_EAST_Test.json').read_text()))
    assert not declared_d & declared_s and set(order) <= declared_d | declared_s
    cold = [dict(shot=r['shot'], hdf5_disrupt=r['disruptive'], shotlist_disrupt=r['shot'] in declared_d)
            for r in rows[:20]]
    result = dict(
        scope='READ_ONLY_EXPOSURE_DIAGNOSTIC_NOT_A_REPLACEMENT_METRIC',
        trained=False, gpu_used=False, threshold_changed=False, alarm_clock_changed=False,
        source_checkpoint_sha256=protocol['source_sha256'],
        parent_completed_sha256=digest(root/'completed.json'),
        script_sha256=digest(__file__), threshold=protocol['threshold'],
        raw_and_sealed_sha_verified=360, whole_safe_decisions_recomputed=180*len(summaries),
        jtext={role: describe(records) for role, records in source.items()},
        east=dict(all=describe(rows), safe=describe([r for r in rows if not r['disruptive']]),
                  disruptive=describe([r for r in rows if r['disruptive']])),
        false_alarm_timing=summaries, cold20_labels=cold,
        cold20_disputes=[r for r in cold if r['hdf5_disrupt'] != r['shotlist_disrupt']],
        jtext_records=source, east_records=rows,
        interpretation='Elapsed time starts at saved data/ip StartTime, not physical ignition. '
                       'Long/short records and prefixes have different populations; '
                       'no independent per-time-point trial or causal duration effect is assumed.')
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in (
        'jtext', 'east', 'false_alarm_timing', 'cold20_disputes')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
