"""Independent endpoint arithmetic and match-set check; no fitting or GPU.

Reuses the audited raw reader and source checkpoint, not the diagnostic helper,
patch_signal, mapping transform, or matching code. Float32 preprocessing is
retained to check the actual model coordinates; differences use float64.
"""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from experiments.risk_aligned_online.simple_online import _durable_json
from experiments.risk_aligned_online.simple_train import sha

ROOT = Path(__file__).parent
AUDIT = ROOT/'probe_outputs/level_change_audit_v1'
PARENT = ROOT/'probe_outputs/current_order_context360_v2'


def read(path):
    return json.loads(path.read_text())


def main():
    output = ROOT/'LEVEL_CHANGE_VERIFY_20261031.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    base, _ = load_parent(DEFAULT_PARENT)
    bounds = read(AUDIT/'source_bounds.json')['bounds']
    lower = np.array([b['lower'] for b in bounds])
    upper = np.array([b['upper'] for b in bounds])
    order = [r['shot'] for r in read(PARENT/'order.json')]
    mapping = read(PARENT/'input_map.json')
    fixed = read(AUDIT/'snapshots128.json')
    pairs = read(AUDIT/'matched_first_alarm.json')
    done = read(AUDIT/'completed.json')
    for name, key in [('source_bounds.json', 'source_bounds_sha256'),
                      ('snapshots128.json', 'snapshots_sha256'),
                      ('matched_first_alarm.json', 'matched_sha256')]:
        assert sha(AUDIT/name) == done[key]
    cached = {}
    for rank, sid in enumerate(order):
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        x, m = raw['values'].copy(), raw['available'].copy()
        m[:, 1] &= np.abs(x[:, 1] + 10.) > .01
        bt = raw['values'][:, 0]
        if rank >= 20:
            for c in range(23):
                if mapping['gain'][c] != 1. or mapping['offset'][c] != 0.:
                    x[:, c] = (x[:, c].astype(np.float64)*mapping['gain'][c]
                               + mapping['offset'][c]).astype(np.float32)
        tx, tm = torch.from_numpy(x), torch.from_numpy(m)
        tx = torch.asinh((torch.where(tm, tx, base.center)-base.center)/base.scale)
        n = len(x)//8
        q = tx[:n*8].reshape(n, 8, 23).mean(1).numpy()
        a = m[:n*8].reshape(n, 8, 23).all(1)
        q = np.where(a, q, 0.).astype(np.float64)
        cached[sid] = (q, a, bt, len(x))

    def direct(sid, end):
        q, a, bt, length = cached[sid]
        assert 0 <= end < length
        j = end//8-1
        mask = a[j] if j >= 0 else np.zeros(23, bool)
        outside = int((((q[j] < lower) | (q[j] > upper)) & mask).sum()) if j >= 0 else 0
        observed = int(mask.sum())
        squares, count, cumulative_squares, cumulative_count = 0., 0, 0., 0
        for t in range(1, j+1):
            for lag in range(1, min(4, t)+1):
                common = a[t] & a[t-lag]
                d = q[t, common]-q[t-lag, common]
                total = float(d@d)
                cumulative_squares += total
                cumulative_count += len(d)
                if t == j:
                    squares += total
                    count += len(d)
        magnitude = abs(float(bt[end]))
        band = ('small_storage_value' if magnitude < .1 else
                'middle_storage_value' if magnitude < 1 else 'regular_storage_value')
        return dict(observed_count=observed, outside_count=outside,
                    outside_fraction=outside/observed if observed else None,
                    increment_rms=float(np.sqrt(squares/count)) if count else None,
                    feedback_pairs=count, cumulative_squared_change=cumulative_squares,
                    cumulative_feedback_pairs=cumulative_count,
                    prefix_exactly_constant=cumulative_count > 0 and cumulative_squares == 0.,
                    available_mask=''.join(str(int(v)) for v in mask), bt_band=band)

    checked = 0
    max_difference = 0.

    def check(sid, row):
        nonlocal checked, max_difference
        expected = direct(sid, row['endpoint'])
        for key, value in expected.items():
            if type(value) is float:
                assert np.isclose(value, row[key], rtol=2e-12, atol=1e-13), (sid, key, value, row[key])
                max_difference = max(max_difference, abs(value-row[key]))
            else:
                assert value == row[key], (sid, key, value, row[key])
        checked += 1
        return expected

    assert len(fixed) == 360 and [r['shot'] for r in fixed] == order
    for row in fixed:
        check(row['shot'], row)
    tn = [r['shot'] for r in fixed if r['rank'] > 20 and r['category'] == 'TN']
    assert len(tn) == 136 and len(pairs) == 42
    for row in pairs:
        case = check(row['shot'], row['snapshot'])
        end = row['snapshot']['endpoint']
        eligible = []
        for sid in tn:
            if cached[sid][3] <= end:
                continue
            q, a, bt, _ = cached[sid]
            j = end//8-1
            mask = a[j] if j >= 0 else np.zeros(23, bool)
            magnitude = abs(float(bt[end]))
            band = ('small_storage_value' if magnitude < .1 else
                    'middle_storage_value' if magnitude < 1 else 'regular_storage_value')
            if band == case['bt_band'] and ''.join(str(int(v)) for v in mask) == case['available_mask']:
                eligible.append(sid)
        assert eligible == [r['shot'] for r in row['controls']]
        for control in row['controls']:
            check(control['shot'], control)
    result = dict(status='PASS', checked_snapshots=checked, full_stream_shots=360,
                  matched_cases=42, matched_TN_pool=136, max_absolute_float_difference=max_difference,
                  source_quantiles_recomputed=False, learned_context_norm_recomputed=False,
                  source_bounds_hash_verified=True, source_checkpoint_sha256=sha(DEFAULT_PARENT),
                  audit_completed_sha256=sha(AUDIT/'completed.json'), script_sha256=sha(Path(__file__)),
                  no_training=True, no_gpu=True)
    _durable_json(output, result)
    print(result)


if __name__ == '__main__':
    main()
