"""Recount all sealed outcomes; rerun the full operator fit and fixed shot curves."""
import argparse
import copy
import json
from pathlib import Path
import zipfile
import hashlib

import h5py
import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..lifecycle_stream_20261025.protocol import sampled_endpoints
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .prepare import TRAINABLE, fit_operator, prepare_examples
from .run import configure, install_operator, read
from .stream import ARMS, predict


def main(root):
    output = root/'verification.json'
    if output.exists():
        raise FileExistsError(output)
    completed, p = read(root/'completed.json'), read(root/'protocol.json')
    assert completed['status'] == 'COMPLETED' and p['arms'] == list(ARMS)
    reference = Path(p['reference'])
    assert sha(reference/'completed.json') == p['reference_completed_sha256']
    assert sha(reference/'models/cold.pt') == p['reference_cold_sha256']
    assert sha(DEFAULT_PARENT) == p['source_sha256']
    with zipfile.ZipFile(root/'code_snapshot.zip') as z:
        for name, digest in p['code_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
            # The producer snapshots this auditor with the other package files,
            # but does not execute it. Its archived bytes are verified above;
            # the current auditor's hash is reported separately below.
            if (io23.ROOT/name).resolve() != Path(__file__).resolve():
                assert sha(io23.ROOT/name) == digest, ('changed executed code', name)
    order = [r['shot'] for r in read(root/'order.json')]
    assert len(order) == len(set(order)) == 360
    assert sha(root/'order.json') == p['order_sha256']
    mapping = read(root/'input_map.json')
    assert mapping == read(reference/'input_map.json') and mapping['calibration_ids'] == order[:20]
    input_hashes = read(root/'input_hashes.json')
    saved = read(root/'per_shot.json')
    rows = {arm: [] for arm in ARMS}
    probability_changes = []
    frozen_matches = zero_matches = 0
    times_and_source = {}
    for rank, sid in enumerate(order, 1):
        raw_path = io23.path_for('EAST', sid)
        assert sha(raw_path) == input_hashes[str(sid)]
        with h5py.File(raw_path, 'r') as h:
            ds = h['data/ip']
            rate, start = float(ds.attrs['SampleRate']), float(ds.attrs['StartTime'])
            times = start+np.arange(len(ds))/rate
        flag, down = io23.read_outcome('EAST', sid)
        prefix = root/'shots'/f'{rank:04d}_{sid}'
        seal = read(Path(str(prefix)+'_seal.json'))
        assert not seal['outcome_available'] and seal['index'] == rank and seal['shot'] == sid
        assert seal['model_version'] == (0 if rank <= 20 else 1)
        assert sha(prefix.with_suffix('.npz')) == seal['sha256']
        receipt = read(Path(str(prefix)+'_audit.json'))
        assert receipt['revealed_after_prediction_sha256'] == seal['sha256']
        assert receipt['suffix_optimizer_steps'] == 0
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as a, \
                np.load(reference/'shots'/f'{rank:04d}_{sid}.npz', allow_pickle=False) as b:
            assert np.array_equal(times, a['times'])
            assert np.array_equal(a['cold'], b['cold'])
            assert np.array_equal(a['source_rule'], b['intra'])
            frozen_matches += 2
            assert np.array_equal(a['zero_feedback'], a['cold'])
            zero_matches += 1
            assert all(np.array_equal(a[arm][:16], a['cold'][:16]) for arm in ARMS)
            if rank <= 20:
                assert all(np.array_equal(a[arm], b['source']) for arm in ARMS)
                times_and_source[sid] = times, b['source'].copy(), flag, down
            else:
                info = receipt['fast']
                assert all(info[arm]['reset_each_shot'] for arm in ARMS)
                assert info['zero_feedback']['updates'] == 0 and info['negative_shift']['delta_sign'] == -1
                assert info['negative_shift']['norm'] == info['learned_rule']['norm']
                real_norm = np.array(info['learned_rule']['norm'])
                reverse_norm = np.array(info['reverse_feedback']['norm'])
                zero = np.array(info['reverse_feedback']['zero_direction'], bool)
                assert np.allclose(real_norm[~zero], reverse_norm[~zero], atol=1e-6, rtol=1e-6)
            for arm in ARMS:
                assert a[arm].shape == times.shape and np.isfinite(a[arm]).all()
                prediction = classify(times, a[arm], flag, down, p['threshold'])
                expected = dict(shot=sid, is_disrupt=flag, **prediction)
                assert expected == saved[arm][rank-1], (sid, arm)
                rows[arm].append(dict(shot=sid, disruptive=flag, **prediction))
            difference = a['learned_rule']-a['source_rule']
            probability_changes.append(dict(shot=sid, mean=float(difference.mean()),
                mean_abs=float(abs(difference).mean()), max_abs=float(abs(difference).max())))
    measured = {arm:counts(r) for arm, r in rows.items()}
    for arm, metric in measured.items():
        original = completed['metrics'][arm]['full360']
        assert all(metric[key] == original[key] for key in ('tp', 'fp', 'tpr', 'fpr', 'J', 'early_over3000'))
    manifest = read(root/'training_manifest.json')
    assert [r['shot'] for r in manifest] == order[:20]
    for e in manifest:
        times, scores, flag, down = times_and_source[e['shot']]
        y, valid = io23.labels_at(times, flag, down, 125.)
        ends = sampled_endpoints(times, y, valid, scores, down=down)
        assert e['ends'] == ends.tolist() and e['labels'] == y[ends].tolist()
        assert e['used_by_loss'] == (ends[y[ends] > 0] if flag else ends).tolist()
    fitting = read(root/'training.json')
    ds, ss = ([e['shot'] for e in manifest if e['flag'] == flag] for flag in (True, False))
    assert len(ds) == 18 and len(ss) == 2
    assert len(fitting['training']) == 36 and fitting['parameters'] == 7001
    for step, row in enumerate(fitting['training']):
        assert row['step'] == step+1 and row['epoch'] == step//18+1
        assert [s['shot'] for s in row['shots']] == [ds[step % 18], ss[(step % 18) % 2]]
        assert row == read(root/'training_steps'/f'{step+1:03d}.json')
        assert set(row['gradients']) == TRAINABLE and np.isfinite(list(row['gradients'].values())).all()
    activation = read(root/'activation.json')
    assert activation['after_revealed_shots'] == 20 and activation['first_affected_stream_rank'] == 21
    assert activation['fit_ids'] == order[:20] and sha(root/'models/operator.pt') == activation['operator_sha256']
    checkpoint = torch.load(root/'models/operator.pt', map_location='cpu', weights_only=True)
    assert set(checkpoint['operator']) == TRAINABLE and checkpoint['fit_ids'] == order[:20]
    configure()
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    load_context_memory(model, Path(p['source_memory']['path']))
    model.restore(torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True))
    learner = copy.deepcopy(model)
    pending = []
    for sid in order[:20]:
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        pending.append((simple_data.attach_outcome(raw, 'EAST', {}), times_and_source[sid][1]))
    examples, repeated_manifest = prepare_examples(learner, pending, mapping)
    assert repeated_manifest == manifest
    reproduced = fit_operator(learner, examples, progress=lambda r: None)
    assert reproduced == fitting, 'full36-step training receipt mismatch'
    assert all(torch.equal(parameter.cpu(), checkpoint['operator'][name])
               for name, parameter in learner.named_parameters() if name in TRAINABLE)
    install_operator(learner, checkpoint['operator'])
    replayed = []
    for i in (20, 113, 359):
        sid = order[i]
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        curves, _ = predict(model, learner, signal)
        with np.load(root/'shots'/f'{i+1:04d}_{sid}.npz', allow_pickle=False) as a:
            assert all(np.array_equal(curves[arm], a[arm]) for arm in ARMS), sid
        replayed.append(sid)
    pairs = {b+'__minus__'+a:flips(rows[a], rows[b]) for a, b in (
        ('cold', 'learned_rule'), ('source_rule', 'learned_rule'),
        ('reverse_feedback', 'learned_rule'), ('negative_shift', 'learned_rule'))}
    result = dict(complete=True, complete_stream_curves_recounted=2160,
        source_and_cold_reference_curves_exact=frozen_matches, zero_feedback_curves_exact=zero_matches,
        first16_unchanged_all_arms=True, metadata_and_archive_verified=True,
        full36_step_fit_and_parameter_tensors_exact=True, cuda_replayed=replayed,
        cuda_replayed_curves_exact=18, metrics=measured, flips=pairs,
        probability_changes_vs_source_rule=probability_changes,
        outcome_authority='HDF5 meta/IsDisrupt and meta/DownTime',
        completed_sha256=sha(root/'completed.json'), audit_script_sha256=sha(Path(__file__)),
        independent_confirmation=False)
    _durable_json(output, result)
    print(json.dumps(dict(verified=True, all_curves=2160, full_fit_exact=True,
                         cuda_replays=replayed, flips=pairs)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
