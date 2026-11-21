"""Independent full-stream comparison, endpoint reconstruction and fixed replays."""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.context_memory_20261028.checkpoint import load_context_memory
from experiments.risk_aligned_online.lifecycle_stream_20261025.audit import audit, flips
from experiments.risk_aligned_online.lifecycle_stream_20261025.protocol import sampled_endpoints
from experiments.risk_aligned_online.lifecycle_stream_20261025.run import build_adapter, predict
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import transform


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(records):
    return [dict(r, disruptive=r['is_disrupt']) for r in records]


def check_training(record, times, scores, flag, down):
    y, valid = io23.labels_at(times, flag, down, 125.)
    old = sampled_endpoints(times, y, valid, scores, down=down)
    # Reconstruct the extra prefix endpoints without calling the new sampler.
    eligible = [i for i in range(min(127, len(times)))
                if valid[i] and (not flag or round((down-times[i])*1000, 9) > 30.)]
    selected = ([eligible[i] for i in np.linspace(0, len(eligible)-1,
                                                 min(16, len(eligible)), dtype=int)]
                if eligible else [])
    expected = sorted(set(old.tolist()+selected))
    assert record['ends'] == expected and record['flag'] == flag
    assert np.array_equal(record['labels'], y[expected])
    added = sorted(set(expected)-set(old.tolist()))
    used = [i for i in added if not flag or y[i] > 0]
    return dict(shot=record['shot'], added=added, added_used=used, disruptive=flag)


def main():
    output = Path(__file__).with_suffix('.json')
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).parent
    old = root/'probe_outputs/current_order_context360_v2'
    new = root/'probe_outputs/short_history_context360_v1'
    assert sha(old/'completed.json') == 'cb9cf19197ab25fb6e825687ec002bcad6135834dca1cb7dc3371234eb7883a9'
    assert read(new/'completed.json')['status'] == 'COMPLETED'
    p, q = read(old/'protocol.json'), read(new/'protocol.json')
    for key in ('source_sha256', 'threshold', 'warmup', 'source_statistics_sha256',
                'target_loss', 'readout', 'feedback', 'source_memory', 'seed',
                'arms', 'torch', 'gpu', 'order_policy', 'order_sha256'):
        assert p[key] == q[key], key
    assert q['endpoint_policy'] == 'prefix16'
    assert read(old/'input_map.json') == read(new/'input_map.json')
    assert read(old/'source_replay.json') == read(new/'source_replay.json')
    with redirect_stdout(io.StringIO()):
        audit(new)
    order = [r['shot'] for r in read(new/'order.json')]
    old_pred, new_pred = read(old/'per_shot.json'), read(new/'per_shot.json')
    first20, coverage, endpoints_verified = {}, [], 0
    early = {arm: {str(t): 0 for t in (16, 127, 128, 512)} for arm in q['arms']}
    for index, sid in enumerate(order, 1):
        flag, down = io23.read_outcome('EAST', sid)
        receipt = read(new/'shots'/f'{index:04d}_{sid}_audit.json')
        with np.load(new/'shots'/f'{index:04d}_{sid}.npz', allow_pickle=False) as a, \
                np.load(old/'shots'/f'{index:04d}_{sid}.npz', allow_pickle=False) as b:
            for arm in ('source', 'aligned'):
                assert np.array_equal(a[arm], b[arm]), (sid, arm)
            assert np.array_equal(a['times'], b['times'])
            if index <= 20:
                first20[sid] = (a['times'].copy(), a['source'].copy(), flag, down)
            else:
                for arm in ('inter', 'both'):
                    record = receipt['training_endpoints'][arm]
                    assert record['shot'] == sid
                    check_training(record, a['times'], a[arm], flag, down)
                    endpoints_verified += 1
            if not flag:
                for arm in q['arms']:
                    hits = np.flatnonzero(a[arm] >= q['threshold'])
                    if len(hits):
                        for t in (16, 127, 128, 512):
                            early[arm][str(t)] += int(hits[0] < t)
        if index == 20:
            for e in receipt['cold_training_endpoints']:
                coverage.append(check_training(e, *first20[e['shot']]))
                endpoints_verified += 1
    across = {arm: flips(rows(old_pred[arm]), rows(new_pred[arm])) for arm in q['arms']}
    within = {b+'__minus__'+a: flips(rows(new_pred[a]), rows(new_pred[b])) for a, b in (
        ('cold', 'intra'), ('cold', 'inter'), ('inter', 'both'),
        ('both_shadow', 'both'), ('intra_reverse', 'intra'))}
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    load_context_memory(model, Path(q['source_memory']['path']))
    model.restore(torch.load(new/'models/cold.pt', map_location='cuda', weights_only=True))
    mapping = read(new/'input_map.json')
    replayed = []
    for i in (20, 113, 359):
        sid = order[i]
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        curves, _, _ = predict(dict(cold=model, inter=model, both=model), signal)
        with np.load(new/'shots'/f'{i+1:04d}_{sid}.npz', allow_pickle=False) as a:
            for arm in ('aligned', 'cold', 'intra', 'intra_reverse'):
                assert np.array_equal(curves[arm], a[arm]), (sid, arm)
        replayed.append(sid)
    result = dict(complete=True, same_order_input_map_source_replay=True,
                  source_and_aligned_curves_exact=720, endpoint_records_verified=endpoints_verified,
                  cold_coverage=coverage, early_false_alarm_counts=early,
                  across_sampling_flips=across, within_new_run_flips=within,
                  outcome_authority='HDF5 meta/IsDisrupt and meta/DownTime',
                  cuda_replayed_fixed_shots=replayed, exact_cuda_curves=12,
                  persistent_online_training_not_reexecuted=True,
                  new_completed_sha256=sha(new/'completed.json'), script_sha256=sha(__file__),
                  new_metrics=read(new/'completed.json')['metrics'],
                  prior_metrics=read(old/'completed.json')['metrics'],
                  independent_confirmation=False)
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(dict(complete=True, unchanged_curves=720, endpoint_records=endpoints_verified,
               cuda_replayed=replayed), flush=True)


if __name__ == '__main__':
    main()
