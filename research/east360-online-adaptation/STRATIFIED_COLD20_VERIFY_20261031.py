"""Verify the complete coverage-order stream against fixed current-only parent."""
import ast
import hashlib
import inspect
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import zipfile

import h5py
import numpy as np
import torch

from CURRENT_ORDER_VERIFY_20261031 import read, sha, keyed, guard_summary
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.context_memory_20261028.checkpoint import load_context_memory
from experiments.risk_aligned_online.east_context_operator_20261031.run import configure
from experiments.risk_aligned_online.lifecycle_stream_20261025.audit import audit, flips
from experiments.risk_aligned_online.lifecycle_stream_20261025.run import build_adapter, predict
from experiments.risk_aligned_online.lifecycle_stream_20261025.protocol import sampled_endpoints
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import transform, fit_prefix_map


def main():
    root = Path(__file__).parent
    old = root/'probe_outputs/current_order_context360_v2'
    new = root/'probe_outputs/stratified_cold20_context360_v1'
    output = Path(__file__).with_suffix('.json')
    if output.exists() or (new/'independent_audit.json').exists():
        raise FileExistsError('independent verification output already exists')
    completed = read(new/'completed.json')
    assert completed['status'] == 'COMPLETED'
    with redirect_stdout(io.StringIO()):
        audit(new)
    first_audit = new/'independent_audit_first_pass.json'
    assert sha(first_audit) == '6f627c8a2090f3ceac4059acb1f8470535319e770c1f8dce2d8d7e95476b6c3e'
    assert first_audit.read_bytes() == (new/'independent_audit.json').read_bytes()
    p, q = read(old/'protocol.json'), read(new/'protocol.json')
    for key in ('source_sha256', 'threshold', 'warmup', 'source_statistics_sha256',
                'target_loss', 'readout', 'feedback', 'source_memory', 'seed',
                'arms', 'torch', 'gpu'):
        assert p[key] == q[key], key
    # This exact parent predates the optional endpoint_policy field. Verify
    # its archived executable path instead of assigning a guessed default.
    assert 'endpoint_policy' not in p and q['endpoint_policy'] == 'full_window'
    with zipfile.ZipFile(old/'code_snapshot.zip') as archive:
        prefix = 'experiments/risk_aligned_online/lifecycle_stream_20261025/'
        archived = {}
        for name in ('learning.py', 'protocol.py'):
            content = archive.read(prefix+name)
            assert hashlib.sha256(content).hexdigest() == p['code_sha256'][prefix+name]
            archived[name] = content.decode()
    tree = ast.parse(archived['learning.py'])
    target = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'target_example')
    assert len(target.body) == 2 and isinstance(target.body[0], ast.Assign)
    assert target.body[0].value.func.id == 'sampled_endpoints'
    assert target.body[1].value.func.id == 'cache_example'
    tree = ast.parse(archived['protocol.py'])
    sampler = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'sampled_endpoints')
    assert ast.get_source_segment(archived['protocol.py'], sampler).strip() == inspect.getsource(sampled_endpoints).strip()
    assert q['order_policy'] == 'current_stratified'
    assert sha(old/'completed.json') == 'cb9cf19197ab25fb6e825687ec002bcad6135834dca1cb7dc3371234eb7883a9'
    old_rows, new_rows = read(old/'order.json'), read(new/'order.json')
    assert read(old/'source_replay.json') == read(new/'source_replay.json')
    old_pos = {r['shot']:i for i, r in enumerate(old_rows)}
    assert set(old_pos) == {r['shot'] for r in new_rows}
    initial, source_exact = [], 0
    raw_prefix = []
    for i, row in enumerate(new_rows):
        sid = row['shot']
        with h5py.File(io23.path_for('EAST', sid), 'r') as f:
            ds = f['data/ip']
            rate, start = float(ds.attrs['SampleRate']), float(ds.attrs['StartTime'])
            n = int(np.ceil(.1*rate-1e-10))
            current = float(np.median(abs(np.asarray(ds[:n], np.float64))))
            times = start+np.arange(len(ds))/rate
        assert current == row['initial_ip_A']
        initial.append(((current/100000.)**2, sid))
        assert row['commissioning_phase'] == ('cold_coverage20' if i < 20 else 'remainder340')
        with np.load(new/'shots'/f'{i+1:04d}_{sid}.npz', allow_pickle=False) as a, \
                np.load(old/'shots'/f'{old_pos[sid]+1:04d}_{sid}.npz', allow_pickle=False) as b:
            assert np.array_equal(a['times'], times) and np.array_equal(a['times'], b['times'])
            assert np.array_equal(a['source'], b['source']), sid
            source_exact += 1
        if i < 20:
            raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            raw_prefix.append({k:raw[k] for k in ('shot', 'device', 'values', 'available')})
    sorted_initial = sorted(initial)
    selected = [k*119//19 for k in range(20)]
    expected = selected + [i for i in range(360) if i not in selected]
    assert initial == [sorted_initial[i] for i in expected]
    assert [r['initial_current_rank1based'] for r in new_rows] == [i+1 for i in expected]
    mapping = read(new/'input_map.json')
    rebuilt = fit_prefix_map(mapping['source_statistics'], raw_prefix,
                             order=[r['shot'] for r in new_rows], count=20, policy='density_gain')
    assert mapping == rebuilt
    old_pred, new_pred = read(old/'per_shot.json'), read(new/'per_shot.json')
    across, within = {}, {}
    for arm in q['arms']:
        a, b = keyed(old_pred[arm]), keyed(new_pred[arm])
        ids = sorted(a)
        across[arm] = flips([a[s] for s in ids], [b[s] for s in ids])
    for first, second in [('cold', 'intra'), ('cold', 'inter'), ('inter', 'both'),
                          ('intra', 'both'), ('both_shadow', 'both'), ('intra_reverse', 'intra')]:
        a, b = keyed(new_pred[first]), keyed(new_pred[second])
        ids = sorted(a)
        within[second+'__minus__'+first] = flips([a[s] for s in ids], [b[s] for s in ids])
    configure()
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    load_context_memory(model, Path(q['source_memory']['path']))
    model.restore(torch.load(new/'models/cold.pt', map_location='cuda', weights_only=True))
    replayed = []
    for index in (20, 113, 359):
        sid = new_rows[index]['shot']
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        curves, _, _ = predict(dict(cold=model, inter=model, both=model), signal)
        with np.load(new/'shots'/f'{index+1:04d}_{sid}.npz', allow_pickle=False) as a:
            for arm in ('aligned', 'cold', 'intra', 'intra_reverse'):
                assert np.array_equal(curves[arm], a[arm]), (sid, arm)
        replayed.append(sid)
    result = dict(complete=True, source_curves_exact_by_shot=source_exact,
        prefix_only_order_independently_recomputed=True, density_map_exactly_reconstructed=True,
        unchanged_source_event_replay=True, cuda_replayed_shots=replayed,
        cuda_replayed_fixed_parent_curves_exact=12, persistent_training_not_reexecuted=True,
        first20_overlap=len({r['shot'] for r in old_rows[:20]} & {r['shot'] for r in new_rows[:20]}),
        across_order_flips_by_identity=across, within_new_order_flips=within,
        guard_rejections=dict(old=guard_summary(old), new=guard_summary(new)),
        old_metrics={k:v['full360'] for k, v in read(old/'completed.json')['metrics'].items()},
        new_metrics={k:v['full360'] for k, v in completed['metrics'].items()},
        new_completed_sha256=sha(new/'completed.json'), script_sha256=sha(Path(__file__)),
        primary_HDF5_360_unchanged=True,
        legacy_full_window_sampler_verified_from_archived_code=True,
        outcome_authority='HDF5 meta/IsDisrupt and meta/DownTime',
        source_checkpoint_sha256=sha(DEFAULT_PARENT), independent_confirmation=False)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
    print(dict(complete=True, source_exact=source_exact, cuda_replays=replayed,
               metrics=result['new_metrics'], within=within))


if __name__ == '__main__':
    main()
