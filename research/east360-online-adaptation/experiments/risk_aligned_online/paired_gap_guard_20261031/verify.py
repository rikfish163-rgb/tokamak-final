"""Independent scoring, rule arithmetic, rollback checks and selected CUDA replay."""
import argparse
import hashlib
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..delayed_recurrent_memory_20261025.learning import batch
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .forward import POLICIES, ARMS, MATCHED, REFERENCE, REFERENCE_SHA


def main(root):
    output = root/'independent_audit.json'
    if output.exists():
        raise FileExistsError(output)
    p, done = read(root/'protocol.json'), read(root/'completed.json')
    assert done['status'] == 'COMPLETED' and sha(REFERENCE/'completed.json') == REFERENCE_SHA
    assert sha(DEFAULT_PARENT) == p['source_sha256']
    assert read(root/'source_replay.json') == read(REFERENCE/'source_replay.json')
    with zipfile.ZipFile(root/'code_snapshot.zip') as archive:
        for path, expected in p['code_sha256'].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected
    order = p['order']
    assert order == read(REFERENCE/'order.json')
    ids = [r['shot'] for r in order]
    assert len(ids) == len(set(ids)) == 360
    rows = {arm:[] for arm in ARMS}
    saved = read(root/'per_shot.json')
    receipts = read(root/'receipts.json')
    original_receipts = read(REFERENCE/'receipts.json')
    hashes = read(REFERENCE/'input_hashes.json')
    cold = torch.load(REFERENCE/'models/cold.pt', map_location='cpu', weights_only=True)
    eligible = {name:0 for name in POLICIES}
    accepted = {name:0 for name in POLICIES}
    accepted_effective = {name:0 for name in POLICIES}
    matched = 0
    flags = {}
    for index, sid in enumerate(ids):
        prefix = f'{index+1:04d}_{sid}'
        path = root/'shots'/f'{prefix}.npz'
        seal = read(root/'shots'/f'{prefix}_seal.json')
        receipt = receipts[index]
        assert seal['sha256'] == sha(path) == receipt['prediction_sha256']
        assert not seal['outcome_available'] and seal['shot'] == receipt['shot'] == sid
        assert receipt == read(root/'shots'/f'{prefix}_audit.json')
        assert sha(io23.path_for('EAST', sid)) == hashes[str(sid)]
        flag, down = io23.read_outcome('EAST', sid)
        flags[sid] = flag
        with np.load(path, allow_pickle=False) as a, \
                np.load(REFERENCE/'shots'/f'{prefix}.npz', allow_pickle=False) as b:
            assert np.array_equal(a['times'], b['times'])
            for arm in ARMS:
                assert a[arm].shape == a['times'].shape and np.isfinite(a[arm]).all()
                result = classify(a['times'], a[arm], flag, down, p['threshold'])
                assert dict(shot=sid, is_disrupt=flag, **result) == saved[arm][index]
                rows[arm].append(dict(shot=sid, disruptive=flag, **result))
                if index < 20:
                    assert np.array_equal(a[arm], b['source'])
            for arm, old in MATCHED.items():
                assert np.array_equal(a[arm], b[old])
                matched += 1
            if index >= 20:
                y, valid = io23.labels_at(a['times'], flag, down, 125.)
                allowed = valid & (np.arange(len(y)) >= 127)
                if flag:
                    allowed &= np.round((down-a['times'])*1000,9) > 30
                pool = np.flatnonzero(allowed)
                for name in POLICIES:
                    positives = pool[y[pool] > 0]
                    positive_sample = positives[np.linspace(0,len(positives)-1,min(16,len(positives)),dtype=int)] if len(positives) else positives
                    expected = np.unique(np.r_[pool[np.linspace(0,len(pool)-1,min(32,len(pool)),dtype=int)],
                                               pool[np.argsort(a[name+'_on'][pool],kind='stable')[-16:]], positive_sample])
                    assert expected.tolist() == receipt['endpoints'][name]
                    assert np.array_equal(a[name+'_on'][:16], a[name+'_off'][:16])
                assert np.array_equal(a['paired_reverse'][:16], a['paired_off'][:16])
        if index < 20:
            assert not receipt['updates'] and seal['state_sha256'] is None
            continue
        state_path = root/'states'/f'{prefix}.pt'
        assert sha(state_path) == seal['state_sha256'] == receipt['state_sha256']
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        assert set(state) == {'cold', *POLICIES}
        assert all(torch.equal(state['cold'][k], v) for k,v in cold.items())
        if index < 359:
            next_state = torch.load(root/'states'/f'{index+2:04d}_{ids[index+1]}.pt', map_location='cpu', weights_only=True)
        else:
            next_state = {name:torch.load(root/'models'/f'{name}.pt',map_location='cpu',weights_only=True) for name in POLICIES}
        for name in POLICIES:
            update = receipt['updates'][name]
            extra = update['paired_guard']
            assert update['history_ids'] == ids[index-8:index]
            assert extra['rule'] == name and extra['eligible_count_before'] == eligible[name]
            before_on, after_on = update['guard_before'], update['guard_after']
            before_off, after_off = extra['before_off'], extra['after_off']
            keys = set(before_on)
            required = {'source_event'} | {'historical_D' if flags[s] else 'historical_S' for s in ids[index-8:index]}
            assert keys == required == set(before_off) == set(after_on) == set(after_off)
            assert all(math.isfinite(value) and value >= 0 for d in (before_on,after_on,before_off,after_off) for value in d.values())
            on_ok = all(after_on[k] <= before_on[k]+1e-7 for k in keys)
            off_ok = all(after_off[k] <= before_off[k]+1e-7 for k in keys)
            gap_ok = all(after_on[k]-after_off[k] <= before_on[k]-before_off[k]+1e-7 for k in keys)
            assert extra['checks']['original_pass'] == on_ok
            assert extra['checks']['dual_absolute_pass'] == (on_ok and off_ok)
            assert extra['checks']['paired_pass'] == (on_ok and off_ok and gap_ok)
            eligible[name] += int(on_ok)
            assert extra['eligible_count_after'] == eligible[name]
            verdict = {'original':on_ok, 'dual':on_ok and off_ok,
                       'paired':on_ok and off_ok and gap_ok, 'half':on_ok and eligible[name]%2==1}[name]
            assert update['accepted'] == verdict
            accepted[name] += int(verdict)
            accepted_effective[name] += int(verdict and index < 359)
            if name == 'original':
                assert {k:v for k,v in update.items() if k!='paired_guard'} == original_receipts[index]['updates']['both']
            for key in ('slow','fast_output'):
                if not verdict:
                    assert torch.equal(state[name][key], next_state[name][key])
                    assert update['retained_delta'][key] == 0
                else:
                    norm = float((next_state[name][key]-state[name][key]).norm())
                    assert math.isclose(norm, update['retained_delta'][key], rel_tol=2e-6, abs_tol=1e-8)
    assert accepted == done['accepted'] and eligible == done['original_eligible']
    metrics = {}
    for arm, values in rows.items():
        metrics[arm] = counts(values)
        for group, subset in [('full360',values), ('warmup20',values[:20]), ('suffix340',values[20:])]:
            actual = counts(subset)
            for key in ('tp','fp','n_disrupt','n_safe','tpr','fpr','J','early_over3000'):
                assert actual[key] == done['metrics'][arm][group][key]
    comparisons = {}
    for name in POLICIES:
        comparisons[name+'_on__minus__own_off'] = flips(rows[name+'_off'], rows[name+'_on'])
    for other in ('original_on','dual_on','half_on','cold_on','paired_reverse'):
        comparisons['paired_on__minus__'+other] = flips(rows[other], rows['paired_on'])
    changed = set()
    for name in ('dual_on','paired_on','half_on'):
        for group in flips(rows['original_on'], rows[name]).values():
            changed.update(group)
    fixed = [ids[20],ids[113],ids[-1]]
    all_selected = list(dict.fromkeys(fixed+sorted(changed)))
    selected = all_selected[:12]
    _durable_json(root/'count_audit.json', dict(status='PASS', metrics=metrics, swaps=comparisons,
                  full360=True, matched_reference_curves=matched, proposals_checked=1360,
                  accepted=accepted, accepted_with_future_shot=accepted_effective,
                  original_eligible=eligible, all_four_losses_verified_arithmetically=True,
                  all_rollbacks_exact=True, own_endpoint_selection_recomputed=True,
                  changed_shots=sorted(changed), selected_neural_replay=selected,
                  all_changed_shots_neurally_replayed=set(changed)<=set(selected),
                  selected_replay_cap=12, completed_sha256=sha(root/'completed.json')))
    replay_selected(root, p, selected)
    result = read(root/'count_audit.json')
    result.update(neural_replay=read(root/'neural_replay.json'), verifier_sha256=sha(Path(__file__)),
                  independent_confirmation=False, own_forward_histories=True)
    _durable_json(output, result)
    print(result)


def replay_selected(root, protocol, selected):
    configure()
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    load_context_memory(model, Path(protocol['source_memory']['path']))
    ids = [r['shot'] for r in protocol['order']]
    mapping = read(root/'input_map.json')
    replayed = []
    with torch.no_grad():
        for sid in selected:
            index = ids.index(sid)
            prefix = f'{index+1:04d}_{sid}'
            states = torch.load(root/'states'/f'{prefix}.pt', map_location='cuda', weights_only=True)
            raw = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal = transform({k:raw[k] for k in ('values','available')},mapping)
            delta, _ = model.memory_delta(signal)
            reverse, _ = model.memory_delta(signal,'reverse')
            with np.load(root/'shots'/f'{prefix}.npz',allow_pickle=False) as saved:
                for lo in range(0,len(signal['values']),32):
                    hi = min(lo+32,len(signal['values']))
                    x, available = batch(signal,range(lo,hi),'cuda')
                    h, valid = model.tokens(x,available)
                    for name, state in states.items():
                        model.restore(state)
                        for mode in ('on','off'):
                            action = delta[lo:hi] if mode=='on' else torch.zeros_like(delta[lo:hi])
                            predicted = model.risk(h,valid,action).sigmoid().cpu().numpy()
                            assert np.array_equal(predicted,saved[name+'_'+mode][lo:hi]),(sid,name,mode,lo)
                    model.restore(states['paired'])
                    predicted = model.risk(h,valid,reverse[lo:hi]).sigmoid().cpu().numpy()
                    assert np.array_equal(predicted,saved['paired_reverse'][lo:hi]),(sid,'reverse',lo)
            replayed.append(sid)
    _durable_json(root/'neural_replay.json',dict(status='PASS',shots=replayed,
                  exact_curves=len(replayed)*11, forward_runner_not_used=True,
                  persistent_training_not_reexecuted=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
