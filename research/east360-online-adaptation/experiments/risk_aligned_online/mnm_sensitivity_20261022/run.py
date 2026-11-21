"""Restore one actually observed HDF5 field; do not change its semantics.

This is an input-availability sensitivity experiment, not certification of
EAST's upstream poloidal mode extraction. The global reader stays unchanged.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from ..warmstart_alignment_20261022.alignment import SIGNAL_KEYS, fit_prefix_map, transform
from ..warmstart_alignment_20261022.run import predict_many, public, snapshot

REFERENCE = io23.ROOT/'experiments/risk_aligned_online/runs/warmstart_alignment_test360_v2_20261022'


def restore_mnm(signal, raw_values, raw_available):
    if set(signal) != {'values', 'available'}:
        raise ValueError('signal-only input required')
    x, m = signal['values'], signal['available']
    if (x.ndim != 2 or x.shape[1] != 23 or x.dtype != np.float32
            or m.shape != x.shape or m.dtype != bool
            or raw_values.shape != x.shape or raw_available.shape != x.shape
            or raw_available.dtype != bool):
        raise ValueError('matching float32 flat23 arrays and boolean masks required')
    c = io23.RAW23.index('MNM')
    if not np.isfinite(raw_values[raw_available[:, c], c]).all():
        raise ValueError('observed MNM must be finite')
    out, mask = x.copy(), m.copy()
    out[:, c] = np.where(raw_available[:, c], raw_values[:, c], 0.)
    mask[:, c] = raw_available[:, c]
    return dict(values=out, available=mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    start = time.monotonic()
    contract = load_test_contract('EAST'); order = contract['shot_ids']
    base, parent = load_parent(DEFAULT_PARENT)
    model = PairedFeedback(base, np.ones(23)).cuda(); model.set_stage('frozen')
    original = {k: v.cpu().clone() for k, v in model.base.state_dict().items()}
    threshold = float(parent['threshold'])
    args.output.mkdir(parents=True); (args.output/'shots').mkdir()
    hashes = snapshot(args.output)
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'a', zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__, str(Path(__file__).relative_to(io23.ROOT)))
    hashes[str(Path(__file__).relative_to(io23.ROOT))] = sha(__file__)
    source_path = REFERENCE/'source_statistics.json'
    source = json.loads(source_path.read_text())
    if source['shot_ids'] != io23.source_roles()['train']:
        raise AssertionError('source-statistics TRAIN identity changed')
    _durable_json(args.output/'protocol.json', dict(
        source_checkpoint=str(DEFAULT_PARENT), source_sha256=PARENT_SHA256,
        threshold=threshold, contract=contract, code_sha256=hashes,
        source_statistics_path=str(source_path), source_statistics_sha256=sha(source_path),
        arms=['masked', 'restored', 'masked_density20', 'restored_density20'],
        target_fit='exact first20 ended signals; fit only after prediction seals',
        semantics_certified=False, neural_updates=0, independent_confirmation=False,
        intervention='MNM value and availability restored; other22 fields identical'))
    rows = {k: [] for k in ('masked', 'restored', 'masked_density20', 'restored_density20')}
    pending, receipts, mapping = [], [], None
    for index, sid in enumerate(order):
        shot = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        raw = io23.read_signals('EAST', sid, rate_contract='nominal')
        if not np.array_equal(raw['times'], shot['times']): raise AssertionError('clock changed')
        signal = {k: shot[k] for k in ('values', 'available')}
        restored = restore_mnm(signal, raw['v'], raw['a'])
        cols = [i for i in range(23) if i != io23.RAW23.index('MNM')]
        for key in signal:
            if not np.array_equal(signal[key][:, cols], restored[key][:, cols]):
                raise AssertionError('non-MNM input changed')
        inputs = dict(masked=signal, restored=restored)
        if mapping is not None:
            inputs.update(masked_density20=transform(signal, mapping),
                          restored_density20=transform(restored, mapping))
        predictions = predict_many(model, inputs, device='cuda')
        if mapping is None:
            predictions.update(masked_density20=predictions['masked'].copy(),
                               restored_density20=predictions['restored'].copy())
        with np.load(REFERENCE/'shots'/f'{sid}.npz', allow_pickle=False) as ref:
            for name, key in (('masked', 'static'), ('masked_density20', 'density_gain_n20')):
                if not np.array_equal(predictions[name], ref[key]):
                    raise AssertionError(f'baseline parity failed: {sid} {name}')
        path = args.output/'shots'/f'{sid}.npz'
        np.savez_compressed(path, times=shot['times'], **predictions)
        c = io23.RAW23.index('MNM'); v = raw['v'][raw['a'][:, c], c]
        receipts.append(dict(shot=sid, prediction_sha256=sha(path),
            available_mnm=len(v), nonzero_mnm=int(np.count_nonzero(v)),
            constant_mnm=bool(len(v) and v.min()==v.max()), mapping_active=mapping is not None))
        if index < 20: pending.append({k: shot[k] for k in SIGNAL_KEYS})
        if index == 19:
            mapping = fit_prefix_map(source, pending, order=order, count=20, policy='density_gain')
            mapping['preceding_prediction_seals'] = receipts.copy()
            _durable_json(args.output/'density_map.json', mapping)
            pending = []
        # No target outcome is accessed until all arms' current traces are sealed.
        outcome = simple_data.attach_outcome(shot, 'EAST', {})
        for name in rows:
            valid = outcome['valid']
            rows[name].append(dict(shot=sid, times=shot['times'][valid], scores=predictions[name][valid],
                                  is_disrupt=outcome['is_disrupt'], down=outcome['down']))
        if (index+1) % 60 == 0:
            print(json.dumps(dict(shots=index+1, seconds=time.monotonic()-start)), flush=True)
    full = {k: score_fixed_test('EAST', v, threshold) for k, v in rows.items()}
    result = {k: dict(full360=public(full[k]), common_suffix340=public(metrics(v[20:], threshold, lead=30.)))
              for k, v in rows.items()}
    for k, v in original.items():
        if not torch.equal(v, model.base.state_dict()[k].cpu()): raise AssertionError('parent changed')
    if sha(DEFAULT_PARENT) != PARENT_SHA256: raise AssertionError('archive changed')
    _durable_json(args.output/'prediction_receipts.json', receipts)
    _durable_json(args.output/'per_shot.json', full)
    _durable_json(args.output/'completed.json', dict(status='COMPLETED', metrics=result,
        parent_bitwise_unchanged=True, baseline_traces_bitwise_equal=True,
        seconds=time.monotonic()-start, physical_semantics_certified=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
