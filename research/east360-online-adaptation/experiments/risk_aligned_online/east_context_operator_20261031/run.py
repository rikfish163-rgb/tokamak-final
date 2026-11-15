"""Fit the EAST feedback operator after cold20; evaluate six complete-stream arms."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..delayed_recurrent_memory_20261025.learning import original_predictions
from ..fixed_test_protocol import score_fixed_test
from ..lifecycle_stream_20261025.protocol import AccessLedger
from ..lifecycle_stream_20261025.run import STATISTICS, build_adapter, compact
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import fit_prefix_map, transform
from .prepare import TRAINABLE, fit_operator, prepare_examples, real_preflight
from .stream import ARMS, predict

REFERENCE = io23.ROOT/'probe_outputs/current_order_context360_v2'
REFERENCE_SHA = 'cb9cf19197ab25fb6e825687ec002bcad6135834dca1cb7dc3371234eb7883a9'
COLD_SHA = '31ccf7a12567d3af4d4df93d41894339a81077e89d6d47aebcc24306673173a7'
PLAN = io23.ROOT/'EAST_OPERATOR_PLAN_20261031.md'


def read(path):
    return json.loads(Path(path).read_text())


def configure():
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)


def install_operator(model, state):
    if set(state) != TRAINABLE:
        raise ValueError('exact operator-only state required')
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != parameters[name].shape or not torch.isfinite(value).all():
                raise ValueError('invalid operator parameter: '+name)
            parameters[name].copy_(value)


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    if sha(REFERENCE/'completed.json') != REFERENCE_SHA or sha(REFERENCE/'models/cold.pt') != COLD_SHA:
        raise ValueError('fixed original cold parent changed')
    p = read(REFERENCE/'protocol.json')
    order = read(REFERENCE/'order.json')
    ids = [r['shot'] for r in order]
    if len(ids) != 360 or len(set(ids)) != 360:
        raise ValueError('invalid reference roster')
    if len(ids) != 360 or set(ids) != set(p['contract']['shot_ids']):
        raise ValueError('complete unchanged360 required')
    expected_inputs = read(REFERENCE/'input_hashes.json')
    ledger = AccessLedger(ids, 20)
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    threshold = float(parent['threshold'])
    if threshold != p['threshold'] or p['source_sha256'] != PARENT_SHA256:
        raise ValueError('source/threshold drift')
    reference = build_adapter(base, readout='full', feedback='progressive', context=True)
    memory_receipt = load_context_memory(reference, Path(p['source_memory']['path']))
    assert memory_receipt == p['source_memory']
    source_stats = read(STATISTICS)
    assert sha(STATISTICS) == p['source_statistics_sha256']
    root.mkdir(parents=True)
    for name in ('shots', 'models', 'training_steps'):
        (root/name).mkdir()
    _durable_json(root/'order.json', order)
    files = set(Path(__file__).parent.glob('*.py')) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None):
            files.add(Path(module.__file__))
    protocol = dict(scope='DEVELOPMENT_ONLY_COMPLETE360_OPERATOR_INTERVENTION',
        source_checkpoint=str(DEFAULT_PARENT), source_sha256=PARENT_SHA256,
        reference=str(REFERENCE), reference_completed_sha256=REFERENCE_SHA,
        reference_cold_sha256=COLD_SHA, source_memory=memory_receipt,
        order_sha256=sha(root/'order.json'), source_statistics_sha256=sha(STATISTICS),
        plan_sha256=sha(PLAN), seed=131, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        contract=p['contract'], arms=ARMS, warmup=20, warmup_in_primary_score=True,
        threshold=threshold, lead_strictly_greater_ms=30, endpoint_policy='full_window',
        target_loss='disruptive_positive_only', epochs=2, optimizer_steps=36,
        trainable_parameters=7001, trainable_names=sorted(TRAINABLE),
        parent_cold_steps_already_performed=20, operator_learning_rate=.001,
        gradient_clip=1., suffix_persistent_updates=0, external_east_labels=0,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        sorting_is_offline_prefix_proxy=True, physical_units_certified=False,
        independent_confirmation=False, final_epoch_not_selected_on_suffix=True)
    _durable_json(root/'protocol.json', protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f.relative_to(io23.ROOT)))
    pending, map_records = [], []
    mapping = learned = fitted = None
    rows = {arm: [] for arm in ARMS}
    receipt_hashes = {}
    for i, sid in enumerate(ids):
        raw_path = io23.path_for('EAST', sid)
        assert sha(raw_path) == expected_inputs[str(sid)], sid
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = {k:raw[k] for k in ('values', 'available')}
        if mapping is None:
            original = original_predictions(base, signal)
            curves = {arm:original.copy() for arm in ARMS}
            info = dict(stage='source_only', updates=0)
        else:
            curves, info = predict(reference, learned, transform(signal, mapping))
        reference_npz = REFERENCE/'shots'/f'{i+1:04d}_{sid}.npz'
        assert sha(reference_npz) == read(REFERENCE/'shots'/f'{i+1:04d}_{sid}_seal.json')['sha256']
        with np.load(reference_npz, allow_pickle=False) as old:
            assert np.array_equal(curves['cold'], old['cold']), ('cold', sid)
            assert np.array_equal(curves['source_rule'], old['intra']), ('source_rule', sid)
        assert np.array_equal(curves['zero_feedback'], curves['cold']), ('zero_feedback', sid)
        assert all(np.array_equal(curves[arm][:16], curves['cold'][:16]) for arm in ARMS)
        prefix = root/'shots'/f'{i+1:04d}_{sid}'
        path = prefix.with_suffix('.npz')
        with path.open('xb') as f:
            np.savez_compressed(f, times=raw['times'], **curves)
            f.flush()
            os.fsync(f.fileno())
        digest = sha(path)
        ledger.seal(sid, digest)
        _durable_json(Path(str(prefix)+'_seal.json'), dict(shot=sid, index=i+1, sha256=digest,
            outcome_available=False, model_version=0 if i < 20 else 1))
        ledger.reveal(sid)
        outcome = simple_data.attach_outcome(raw, 'EAST', {})
        valid = outcome['valid']
        for arm in ARMS:
            rows[arm].append(dict(shot=sid, times=raw['times'][valid], scores=curves[arm][valid],
                                  is_disrupt=outcome['is_disrupt'], down=outcome['down']))
        _durable_json(Path(str(prefix)+'_audit.json'), dict(shot=sid, index=i+1, fast=info,
            revealed_after_prediction_sha256=digest, suffix_optimizer_steps=0))
        receipt_hashes[str(sid)] = expected_inputs[str(sid)]
        if i < 20:
            pending.append((outcome, curves['cold']))
            map_records.append({k:raw[k] for k in ('shot', 'device', 'values', 'available')})
        if i == 19:
            mapping = fit_prefix_map(source_stats, map_records, order=ids, count=20, policy='density_gain')
            assert mapping == read(REFERENCE/'input_map.json')
            _durable_json(root/'input_map.json', mapping)
            reference.restore(torch.load(REFERENCE/'models/cold.pt', map_location='cuda', weights_only=True))
            learned = copy.deepcopy(reference)
            examples, manifest = prepare_examples(learned, pending, mapping)
            _durable_json(root/'training_manifest.json', manifest)
            print(json.dumps(dict(stage='real_preflight', revealed_shots=20,
                                  seconds=time.monotonic()-start)), flush=True)
            preflight = real_preflight(learned, examples, pending, mapping)
            _durable_json(root/'real_preflight.json', preflight)
            print(json.dumps(dict(stage='operator_fit', revealed_shots=20, parameters=7001,
                                  seconds=time.monotonic()-start)), flush=True)

            def progress(row):
                _durable_json(root/'training_steps'/f"{row['step']:03d}.json", row)
                p = dict(stage='operator_fit', step=row['step'], total_steps=36,
                         loss=row['loss'], seconds=time.monotonic()-start)
                _durable_json(root/'progress.json', p)
                print(json.dumps(p), flush=True)

            fitted = fit_operator(learned, examples, progress=progress)
            _durable_json(root/'training.json', fitted)
            operator = {n:p.detach().cpu().clone() for n, p in learned.named_parameters() if n in TRAINABLE}
            torch.save(dict(operator=operator, source_sha256=PARENT_SHA256,
                            reference_cold_sha256=COLD_SHA, fit_ids=ids[:20], steps=36),
                       root/'models/operator.pt')
            _durable_json(root/'activation.json', dict(after_revealed_shots=20, fit_ids=ids[:20],
                operator_sha256=sha(root/'models/operator.pt'), cold_sha256=COLD_SHA,
                first_affected_stream_rank=21, source_checkpoint_unchanged=sha(DEFAULT_PARENT)==PARENT_SHA256))
            fixed_operator = {n:p.detach().clone() for n, p in learned.named_parameters() if n in TRAINABLE}
            pending, map_records = [], []
        if (i+1) % 10 == 0:
            p = dict(stage='stream', shots=i+1, total=360, seconds=time.monotonic()-start)
            _durable_json(root/'progress.json', p)
            print(json.dumps(p), flush=True)
    assert len(ledger.history) == 360 and fitted is not None
    assert all(torch.equal(fixed_operator[n], p) for n, p in learned.named_parameters() if n in TRAINABLE)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    result, per_shot = {}, {}
    for arm in ARMS:
        scored = score_fixed_test('EAST', rows[arm], threshold)
        per_shot[arm] = scored['predictions']
        result[arm] = dict(full360=compact(rows[arm], threshold), warmup20=compact(rows[arm][:20], threshold),
                          suffix340=compact(rows[arm][20:], threshold),
                          blocks20=[compact(rows[arm][i:i+20], threshold) for i in range(0, 360, 20)])
    _durable_json(root/'per_shot.json', per_shot)
    _durable_json(root/'input_hashes.json', receipt_hashes)
    m, b = result['learned_rule']['full360'], result['cold']['full360']
    _durable_json(root/'completed.json', dict(status='COMPLETED', metrics=result,
        seconds=time.monotonic()-start, original_cold_curves_exact=360, original_intra_curves_exact=360,
        no_feedback_identical_to_cold=True, source_checkpoint_unchanged=True,
        suffix_persistent_updates=0, operator_parameters_unchanged_during_suffix=True,
        raw_intra_gate=bool(m['J'] > b['J'] and m['fpr'] <= .20),
        scientific_goal_achieved=False, independent_confirmation=False))
    print(json.dumps({k:v['full360'] for k, v in result.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
