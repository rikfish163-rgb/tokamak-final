"""Matched diagnostic: same existing uniform in-shot update after input alignment.

This is not a new update policy. The 20-shot, density-only zero-preserving
mapping was chosen after the alignment comparison and is development-selected.
Both arms predict the warmup; statistics become usable only after shot20 ends.
The uniform update runs on EVERY shot, including warmup, exactly as before.
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
from ..context_feedback_20261022.train import model_from_warm
from ..context_feedback_20261022.stream import predict_pair
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from .alignment import SIGNAL_KEYS, fit_prefix_map, transform
from .run import public, snapshot

ALIGNMENT_RUN = io23.ROOT/'experiments/risk_aligned_online/runs/warmstart_alignment_test360_v2_20261022'
RAW_RUN = io23.ROOT/'experiments/risk_aligned_online/runs/context_feedback_east_uniform_20261022'
WARM_CHECKPOINT = io23.ROOT/'experiments/risk_aligned_online/runs/context_feedback_warm_20261022/warm.pt'


def predict_aligned_pair(model, signal, mapping, *, index, device):
    if type(index) is not int or index < 0:
        raise ValueError('nonnegative stream index required')
    if index < 20 and mapping is not None:
        raise ValueError('target statistics not yet available')
    if index >= 20 and (mapping is None or mapping['first_affected_index'] != 20):
        raise ValueError('exact ended prefix mapping required')
    x = signal if mapping is None else transform(signal, mapping)
    return predict_pair(model, x, device=device, use_event=False, uniform=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131); start = time.monotonic()
    prior = json.loads((ALIGNMENT_RUN/'protocol.json').read_text())
    verified = json.loads((ALIGNMENT_RUN/'verification.json').read_text())
    if verified['status'] != 'VERIFIED' or sha(ALIGNMENT_RUN/'completed.json') != verified['completed_sha256']:
        raise ValueError('unaltered verified alignment run required')
    raw_protocol = json.loads((RAW_RUN/'protocol.json').read_text())
    if sha(WARM_CHECKPOINT) != raw_protocol['checkpoint_sha256']:
        raise ValueError('same previous uniform update checkpoint required')
    ck = torch.load(WARM_CHECKPOINT, map_location='cuda', weights_only=True)
    if ck['config']['mode'] != 'warm': raise ValueError('uniform control only')
    model, threshold = model_from_warm(); model.load_state_dict(ck['model']); model.freeze_base()
    state_before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    contract = load_test_contract('EAST'); order = contract['shot_ids']
    if contract != prior['contract'] or threshold != prior['threshold']:
        raise ValueError('parent threshold or test contract changed')
    source = json.loads((ALIGNMENT_RUN/'source_statistics.json').read_text())
    args.output.mkdir(parents=True); (args.output/'shots').mkdir()
    code = snapshot(args.output)
    extra = [Path(__file__).parents[1]/f'context_feedback_20261022/{n}.py'
             for n in ('model', 'stream', 'train')]
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'a', zipfile.ZIP_DEFLATED) as z:
        for file in extra:
            name = str(file.relative_to(io23.ROOT)); z.write(file, name); code[name] = sha(file)
    protocol = dict(contract=contract, arms=['intra'], threshold=threshold,
        source_checkpoint=str(DEFAULT_PARENT), source_sha256=PARENT_SHA256,
        checkpoint=str(WARM_CHECKPOINT), checkpoint_sha256=sha(WARM_CHECKPOINT),
        alignment_reference=str(ALIGNMENT_RUN), alignment_verification_sha256=sha(ALIGNMENT_RUN/'verification.json'),
        source_statistics_sha256=sha(ALIGNMENT_RUN/'source_statistics.json'),
        mapping='density_gain_n20; chosen after development alignment comparison; not independent',
        update='unchanged uniform 2SGD.01; arrived128-point window; next32 predictions; reset each window/shot',
        warmup='both arms use raw input for first20; intra updates also run during warmup',
        outcome_access='only after complete current-shot predictions sealed; scoring only',
        inference_only_controls='same original risk head, source threshold, first-crossing alarm',
        inter_updates=0, code_sha256=code, independent_confirmation=False)
    _durable_json(args.output/'protocol.json', protocol)
    reference_receipts = {r['shot']:r for r in json.loads((ALIGNMENT_RUN/'prediction_receipts.json').read_text())}
    rows = {k:[] for k in ('static','intra')}; receipts=[]; pending=[]; mapping=None
    for index, sid in enumerate(order):
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        scores, info = predict_aligned_pair(model, {k:raw[k] for k in ('values','available')},
                                            mapping, index=index, device='cuda')
        reference_path = ALIGNMENT_RUN/'shots'/f'{sid}.npz'
        if sha(reference_path) != reference_receipts[sid]['prediction_sha256']:
            raise ValueError('alignment reference trace changed')
        with np.load(reference_path, allow_pickle=False) as ref:
            if not np.array_equal(raw['times'], ref['times']): raise AssertionError('clock changed')
            if not np.array_equal(scores['static'], ref['density_gain_n20']):
                raise AssertionError(f'aligned static probabilities not bitwise identical: {sid}')
        if index < 20:
            with np.load(RAW_RUN/'shots'/f'{sid}.npz', allow_pickle=False) as ref:
                if not np.array_equal(scores['intra'], ref['intra']):
                    raise AssertionError('raw warmup update changed')
        path = args.output/'shots'/f'{sid}.npz'
        np.savez_compressed(path, times=raw['times'], **scores)
        _durable_json(args.output/'shots'/f'{sid}_updates.json', info)
        receipts.append(dict(shot=sid, prediction_sha256=sha(path),
                             update_sha256=sha(args.output/'shots'/f'{sid}_updates.json'),
                             mapped=mapping is not None, update_count=len(info['updates'])))
        _durable_json(args.output/'shots'/f'{sid}_receipt.json', receipts[-1])
        if index < 20: pending.append({k:raw[k] for k in SIGNAL_KEYS})
        if index == 19:
            mapping = fit_prefix_map(source, pending, order=order, count=20, policy='density_gain')
            ref_map = json.loads((ALIGNMENT_RUN/'maps'/'density_gain_n20.json').read_text())
            ref_map.pop('preceding_prediction_seals')
            if mapping != ref_map: raise AssertionError('input statistics changed')
            mapping['preceding_prediction_seals'] = {str(r['shot']):r['prediction_sha256'] for r in receipts}
            _durable_json(args.output/'mapping.json', mapping); pending=[]
        # No outcome enters the helper, optimizer, mapping or model state.
        outcome = simple_data.attach_outcome(raw, 'EAST', {})
        for arm in rows:
            v = outcome['valid']
            rows[arm].append(dict(shot=sid, times=raw['times'][v], scores=scores[arm][v],
                                  is_disrupt=outcome['is_disrupt'], down=outcome['down']))
        if (index+1) % 30 == 0:
            progress = dict(shots=index+1, total=len(order), seconds=time.monotonic()-start)
            _durable_json(args.output/'progress.json', progress); print(json.dumps(progress), flush=True)
    if any(not torch.equal(v, model.state_dict()[k].cpu()) for k,v in state_before.items()):
        raise AssertionError('persistent state changed')
    if sha(DEFAULT_PARENT) != PARENT_SHA256: raise AssertionError('source checkpoint changed')
    full = {k:score_fixed_test('EAST', r, threshold) for k,r in rows.items()}
    result = {k:dict(full360=public(full[k]), common_suffix340=public(metrics(r[20:],threshold,lead=30.)),
                    warmup20=public(metrics(r[:20],threshold,lead=30.))) for k,r in rows.items()}
    _durable_json(args.output/'per_shot.json', full)
    _durable_json(args.output/'prediction_receipts.json', receipts)
    completed = dict(status='COMPLETED', metrics=result,
        persistent_state_bitwise_unchanged=True, static_traces_bitwise_identical=True,
        recorded_updates=sum(r['update_count'] for r in receipts), inter_updates=0,
        seconds=time.monotonic()-start, independent_confirmation=False)
    _durable_json(args.output/'completed.json', completed); print(json.dumps(completed), flush=True)


if __name__ == '__main__': main()
