"""Recount every curve and independently replay swaps through the original model."""
import argparse
import hashlib
import json
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
from .run import REFERENCE, REFERENCE_SHA, ARMS, MATCHED


def main(root):
    output = root/'independent_audit.json'
    if output.exists():
        raise FileExistsError(output)
    done, p = read(root/'completed.json'), read(root/'protocol.json')
    assert done['status'] == 'COMPLETED' and sha(REFERENCE/'completed.json') == REFERENCE_SHA
    assert p['counterfactual_curves_used_for_training'] is False
    assert sha(DEFAULT_PARENT) == p['source_sha256']
    with zipfile.ZipFile(root/'code_snapshot.zip') as archive:
        for path, expected in p['code_sha256'].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected
    order = p['order']
    assert order == read(REFERENCE/'order.json') and len(order) == 360
    ids = [row['shot'] for row in order]
    assert len(set(ids)) == 360
    saved = read(root/'per_shot.json')
    raw_hash = read(REFERENCE/'input_hashes.json')
    receipts = read(REFERENCE/'receipts.json')
    state_rows = read(root/'state_evolution.json')
    assert len(state_rows) == 340
    source_p = read(REFERENCE/'protocol.json')
    assert read(root/'source_replay.json') == read(REFERENCE/'source_replay.json')
    assert read(root/'cold_fit.json') == receipts[19]['cold_fit']
    assert read(root/'input_map.json') == read(REFERENCE/'input_map.json')
    cold = torch.load(REFERENCE/'models/cold.pt', map_location='cpu', weights_only=True)
    rows = {arm:[] for arm in ARMS}
    original_matches, exact_updates = 0, 0
    previous_states = None
    for index, sid in enumerate(ids):
        prefix = f'{index+1:04d}_{sid}'
        path = root/'shots'/f'{prefix}.npz'
        seal = read(root/'shots'/f'{prefix}_seal.json')
        assert seal == dict(shot=sid, index=index+1, sha256=sha(path), outcome_available=False)
        assert sha(io23.path_for('EAST', sid)) == raw_hash[str(sid)]
        flag, down = io23.read_outcome('EAST', sid)
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
            for arm, original in MATCHED.items():
                assert np.array_equal(a[arm], b[original])
                original_matches += 1
        if index >= 20:
            state_path = root/'states'/f'{prefix}.pt'
            entry = state_rows[index-20]
            assert entry['state_sha256'] == sha(state_path) and entry['shot'] == sid
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            assert set(state) == {'cold', 'inter', 'both'}
            assert all(torch.equal(state['cold'][key], value) for key, value in cold.items())
            assert torch.equal(state['inter']['fast_output'], cold['fast_output'])
            if previous_states is not None:
                for arm in ('inter', 'both'):
                    if not receipts[index-1]['updates'][arm]['accepted']:
                        assert all(torch.equal(state[arm][key], value)
                                   for key, value in previous_states[arm].items())
            previous_states = state
            updates = read(root/'updates'/f'{prefix}.json')
            assert updates == receipts[index]['updates']
            exact_updates += 2
    metrics = {arm:counts(values) for arm, values in rows.items()}
    for arm, measured in metrics.items():
        expected = done['metrics'][arm]['full360']
        for key in ('tp', 'fp', 'n_disrupt', 'n_safe', 'tpr', 'fpr', 'J', 'early_over3000'):
            assert measured[key] == expected[key]
    comparisons = {}
    for reference, candidate in (
            ('both_on', 'both_slow_cold_readout'), ('cold_on', 'cold_slow_both_readout'),
            ('both_off', 'both_slow_cold_readout'), ('inter_off', 'inter_slow_cold_readout'),
            ('both_off', 'both_on'), ('cold_on', 'both_on')):
        comparisons[candidate+'__minus__'+reference] = flips(rows[reference], rows[candidate])
    configure()
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    load_context_memory(model, Path(source_p['source_memory']['path']))
    mapping = read(root/'input_map.json')
    selected = list(dict.fromkeys([ids[20], ids[113], ids[-1], *p['case_ids']]))
    replayed = []
    for sid in selected:
        index = ids.index(sid)
        prefix = f'{index+1:04d}_{sid}'
        states = torch.load(root/'states'/f'{prefix}.pt', map_location='cuda', weights_only=True)
        c, b, i = (states[name] for name in ('cold', 'both', 'inter'))
        specs = dict(cold_off=(c,c,False), cold_on=(c,c,True), both_off=(b,b,False),
                     both_on=(b,b,True), both_slow_cold_readout=(b,c,True),
                     cold_slow_both_readout=(c,b,True), inter_off=(i,i,False),
                     inter_slow_cold_readout=(i,c,True))
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        with torch.no_grad(), np.load(root/'shots'/f'{prefix}.npz', allow_pickle=False) as saved_curves:
            delta, _ = model.memory_delta(signal)
            for lo in range(0, len(signal['values']), 32):
                hi = min(lo+32, len(signal['values']))
                x, available = batch(signal, range(lo, hi), 'cuda')
                h, valid = model.tokens(x, available)
                for arm, (slow, out, enabled) in specs.items():
                    model.restore(dict(slow=slow['slow'], fast_output=out['fast_output']))
                    action = delta[lo:hi] if enabled else torch.zeros_like(delta[lo:hi])
                    z = model.risk(h, valid, action)
                    assert np.array_equal(z.cpu().numpy(), saved_curves['logit_'+arm][lo:hi]), (sid, arm, lo)
                    assert np.array_equal(z.sigmoid().cpu().numpy(), saved_curves[arm][lo:hi]), (sid, arm, lo)
        replayed.append(sid)
    result = dict(status='PASS', full_stream_shots=360, scored_curves=2880,
                  matched_original_curves=original_matches, exact_update_receipts=exact_updates,
                  source_checkpoint_unchanged=True, cold20_training_receipt_exact=True,
                  unchanged_after_rejected_updates=True, metrics=metrics, swaps=comparisons,
                  independent_cuda_replayed_shots=replayed,
                  independent_cuda_replayed_curves=8*len(replayed),
                  helper_not_used_for_cuda_verification=True, training_not_reexecuted_by_this_verifier=True,
                  completed_sha256=sha(root/'completed.json'), verifier_sha256=sha(Path(__file__)),
                  independent_future_confirmation=False, no_deployable_policy_gain_claim=True)
    _durable_json(output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
