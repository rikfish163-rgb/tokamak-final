"""Resume an interrupted fixed-head experiment at a verified shot boundary.

This is valid only because the declared experiment has no inter-shot learning:
every shot starts with the same immutable head and fresh encoder state. Existing
outputs are checked, hashed now, and the final saved shot is actually replayed.
No claim is made that the original interrupted process sealed their hashes.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..simple_online import _durable_json
from ..simple_train import sha, metrics
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..deepcoral_prefix_20261022.run import predict_pairs
from ..head_operator_20261022.audit import validate_update
from ..warmstart_alignment_20261022.run import public
from ..warmstart_alignment_20261022.alignment import transform
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from .run import ARMS, REFERENCE, MAP_PATH, load_models


def completed_prefix(directory, order):
    predictions = {int(p.stem) for p in directory.glob('*.npz')}
    updates = {int(p.name.removesuffix('_updates.json'))
               for p in directory.glob('*_updates.json')}
    if predictions != updates:
        raise ValueError('prediction and update files must be paired; no partial file is ignored')
    prefix = order[:len(predictions)]
    if set(prefix) != predictions:
        raise ValueError('saved shots must be exactly a contiguous prefix of the fixed order')
    for sid in prefix:
        with np.load(directory / f'{sid}.npz', allow_pickle=False) as trace:
            if set(trace.files) != {'times', *ARMS}:
                raise ValueError('trace keys differ from declared experiment')
            times = trace['times']
            if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError('invalid trace time axis')
            for arm in ARMS:
                v = trace[arm]
                if v.shape != times.shape or not np.isfinite(v).all() or np.any((v < 0) | (v > 1)):
                    raise ValueError('invalid trace probabilities')
        if not isinstance(json.loads((directory / f'{sid}_updates.json').read_text()), dict):
            raise ValueError('updates must be the declared dictionary')
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory
    if (root / 'completed.json').exists():
        raise FileExistsError('completed experiment cannot be resumed')
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')
    protocol = json.loads((root / 'protocol.json').read_text())
    assert protocol['no_target_outcomes_in_updates'] and protocol['first20_common_static']
    assert protocol['source_sha256'] == PARENT_SHA256
    for path, h in protocol['code_sha256'].items():
        assert sha(io23.ROOT / path) == h
    prepared = Path(protocol['prepared'])
    assert sha(prepared / 'protocol.json') == protocol['preparation_protocol_sha256']
    assert sha(prepared / 'prepare_completed.json') == protocol['preparation_completion_sha256']
    assert sha(REFERENCE / 'protocol.json') == protocol['reference_protocol_sha256']
    assert sha(MAP_PATH) == protocol['mapping_sha256']
    contract = load_test_contract('EAST')
    assert contract == protocol['contract']
    order = contract['shot_ids']
    prefix = completed_prefix(root / 'shots', order)
    if len(prefix) < 20:
        raise ValueError('this recovery requires the original prefix20 activation receipt')
    receipt_path = root / f'resume_{len(prefix)}.json'
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    torch.set_num_threads(2)
    torch.manual_seed(131)
    models, parent = load_models(prepared)
    threshold = float(parent['threshold'])
    assert threshold == protocol['threshold']
    initial = {k: {n: v.detach().cpu().clone() for n, v in m.state_dict().items()}
               for k, m in models.items()}
    mapping = json.loads(MAP_PATH.read_text())
    assert mapping['calibration_ids'] == order[:20]
    references = {r['shot']: r for r in json.loads((REFERENCE / 'prediction_receipts.json').read_text())}
    seals = []
    start = time.monotonic()
    for i, sid in enumerate(prefix):
        shot = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        path = root / 'shots' / f'{sid}.npz'
        upath = root / 'shots' / f'{sid}_updates.json'
        refpath = REFERENCE / 'shots' / f'{sid}.npz'
        assert sha(refpath) == references[sid]['prediction_sha256']
        updates = json.loads(upath.read_text())
        with np.load(path, allow_pickle=False) as trace, np.load(refpath, allow_pickle=False) as ref:
            np.testing.assert_array_equal(trace['times'], shot['times'])
            np.testing.assert_array_equal(trace['static'], ref['static'])
            np.testing.assert_array_equal(trace['original'], ref['encoder_original'])
            for arm in models:
                np.testing.assert_array_equal(trace[arm][:256], trace['static'][:256])
                if i < 20:
                    np.testing.assert_array_equal(trace[arm], trace['static'])
            if i == len(prefix) - 1:
                signal = transform({k: shot[k] for k in ('values', 'available')}, mapping)
                fresh, replay_updates = predict_pairs(models, signal)
                for arm in models:
                    np.testing.assert_array_equal(fresh[f'{arm}_static'], trace['static'])
                    np.testing.assert_array_equal(fresh[f'{arm}_intra'], trace[arm])
                assert replay_updates == updates
        if i < 20:
            assert not updates
        else:
            assert set(updates) == set(models)
            for arm, us in updates.items():
                assert [u['first_affected_prediction'] for u in us] == list(range(256, len(shot['values']), 128))
                for u in us:
                    validate_update(arm, u, len(shot['values']))
        seals.append(dict(shot=sid, prediction_sha256=sha(path), updates_sha256=sha(upath)))
    activation = json.loads((root / 'activation_receipt.json').read_text())
    assert activation['after_seals'] == seals[:20]
    assert activation['mapping_sha256'] == sha(MAP_PATH)
    _durable_json(receipt_path, dict(status='VERIFIED_PREFIX', recovered_shots=len(prefix),
        reason='original process absent after interrupted turn; no inter-shot learned state',
        recovered_file_hashes=seals, original_protocol_sha256=sha(root / 'protocol.json'),
        resume_code_sha256=sha(__file__), replayed_last_shot=prefix[-1], replay_difference=0.,
        target_outcomes_accessed=False, hashes_captured_at_recovery=True))
    print(json.dumps(dict(resuming_after=len(prefix), replayed=prefix[-1])), flush=True)
    for i in range(len(prefix), len(order)):
        sid = order[i]
        shot = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        refpath = REFERENCE / 'shots' / f'{sid}.npz'
        assert sha(refpath) == references[sid]['prediction_sha256']
        signal = transform({k: shot[k] for k in ('values', 'available')}, mapping)
        fresh, updates = predict_pairs(models, signal)
        with np.load(refpath, allow_pickle=False) as ref:
            predictions = dict(static=ref['static'].copy(), original=ref['encoder_original'].copy())
            for arm in models:
                np.testing.assert_array_equal(fresh[f'{arm}_static'], predictions['static'])
                predictions[arm] = fresh[f'{arm}_intra']
                np.testing.assert_array_equal(predictions[arm][:256], predictions['static'][:256])
        path = root / 'shots' / f'{sid}.npz'
        upath = root / 'shots' / f'{sid}_updates.json'
        if path.exists() or upath.exists():
            raise FileExistsError(f'will not overwrite shot {sid}')
        np.savez_compressed(path, times=shot['times'], **predictions)
        _durable_json(upath, updates)
        seals.append(dict(shot=sid, prediction_sha256=sha(path), updates_sha256=sha(upath)))
        if (i + 1) % 40 == 0:
            progress = dict(shots=i + 1, resume_seconds=time.monotonic() - start)
            _durable_json(root / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    _durable_json(root / 'prediction_receipts.json', seals)
    rows = {k: [] for k in ARMS}
    for sid in order:
        shot = simple_data.attach_outcome(simple_data.read_shot('EAST', sid, rate_contract='nominal'), 'EAST', {})
        with np.load(root / 'shots' / f'{sid}.npz', allow_pickle=False) as trace:
            for arm in ARMS:
                v = shot['valid']
                rows[arm].append(dict(shot=sid, times=shot['times'][v], scores=trace[arm][v],
                                     is_disrupt=shot['is_disrupt'], down=shot['down']))
    full = {k: score_fixed_test('EAST', r, threshold) for k, r in rows.items()}
    result = {k: dict(full360=public(full[k]), common_suffix340=public(metrics(r[20:], threshold, lead=30.)))
              for k, r in rows.items()}
    for k, model in models.items():
        for name, v in model.state_dict().items():
            torch.testing.assert_close(v.cpu(), initial[k][name], rtol=0, atol=0)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    _durable_json(root / 'per_shot.json', full)
    _durable_json(root / 'completed.json', dict(status='COMPLETED', metrics=result,
        persistent_models_bitwise_unchanged=True, resumed_from=len(prefix),
        resume_receipt_sha256=sha(receipt_path), resume_seconds=time.monotonic() - start,
        uninterrupted_total_runtime_available=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
