"""Independent recheck of stored clocks and first-crossing metrics; no inference."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.risk_aligned_online import io23


def main():
    run = Path(__file__).parent/'probe_outputs/anchored_full360_diagnostic_v1'
    protocol = json.loads((run/'protocol.json').read_text())
    saved = json.loads((run/'per_shot.json').read_text())
    declared = set(json.loads((io23.ROOT/'shot_list/TEST/Disruption_EAST_Test.json').read_text()))
    safe = set(json.loads((io23.ROOT/'shot_list/TEST/Non-disruption_EAST_Test.json').read_text()))
    assert not declared & safe
    flags, start_diffs, disputes, stats = [], [], [], {}
    for index, sid in enumerate(protocol['order']):
        with h5py.File(io23.path_for('EAST', sid), 'r') as h:
            ds = h['data/ip']
            start, rate = float(ds.attrs['StartTime']), float(ds.attrs['SampleRate'])
            count = len(ds)
            assert rate == 1000.
            for c in io23.RAW23:
                other = h['data'][c]
                assert len(other) == count
                assert float(other.attrs['SampleRate']) == rate
                assert float(other.attrs['StartTime']) == start
            flag = bool(h['meta/IsDisrupt'][()])
            down = float(h['meta/DownTime'][()]) if flag else None
            meta_start = float(h['meta/StartTime'][()])
        flags.append(flag)
        if start != meta_start:
            start_diffs.append(dict(shot=sid, signal_start=start, meta_start=meta_start))
        assert sid in declared | safe
        if flag != (sid in declared):
            disputes.append(dict(shot=sid, HDF5_disrupt=flag, shotlist_disrupt=sid in declared))
        with np.load(run/'shots'/f'{index+1:04d}_{sid}.npz', allow_pickle=False) as a:
            times = start + np.arange(count)/rate
            assert np.array_equal(times, a['times'])
            for arm, predictions in saved.items():
                hits = np.flatnonzero(a[arm] >= protocol['threshold'])
                first = float(times[hits[0]]) if len(hits) else None
                lead = round((down-first)*1000, 9) if flag and first is not None else None
                timely = lead is not None and lead > 30.
                expected = dict(shot=sid, is_disrupt=flag, alarm=first is not None,
                                lead_ms=lead, timely=timely, first_alarm_time_seconds=first)
                assert predictions[index] == expected, (sid, arm)
                counts = stats.setdefault(arm, dict(tp=0, fp=0))
                counts['tp'] += int(timely)
                counts['fp'] += int(first is not None and not flag)
    assert sum(flags) == 180 and len(flags) == 360
    result = dict(scope='STORED_DATA_CONTRACT_NOT_INSTRUMENT_CALIBRATION',
                  trained=False, gpu_used=False, time_arrays_exact=360,
                  per_channel_clock_shape_checks=360*23, records_recomputed=360*len(saved),
                  class_disagreements=disputes, metadata_start_differences=start_diffs,
                  counts=stats, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = Path(__file__).with_suffix('.json')
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(dict(clocks=360, channels=360*23, predictions=360*len(saved),
               class_disputes=len(disputes), meta_start_differences=len(start_diffs), counts=stats))


if __name__ == '__main__':
    main()
