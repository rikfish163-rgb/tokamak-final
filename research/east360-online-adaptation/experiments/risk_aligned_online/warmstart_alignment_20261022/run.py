"""Frozen source, first 10/20 ended EAST shots for signal-only alignment.

All arms predict the complete locked TEST360 stream. Before their own warmup
ends they use the original input. Target statistics are fitted ONLY after
the warmup shots' predictions have been written. Primary matched comparison
also reports the common untouched-by-statistics suffix of 340 shots.
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
from ..paired_feedback_20261022.model import PairedFeedback, initial_encoder
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from .alignment import POLICIES, SIGNAL_KEYS, fit_statistics, fit_prefix_map, transform

REFERENCE = io23.ROOT/'experiments/risk_aligned_online/runs/context_feedback_east_uniform_20261022'


def predict_many(model, signals, *, device, batch_size=32):
    """Same causal windows and classifier; omit only unused forecast output.

Each affine map is pointwise and frozen before the current shot. Batching
different arms/endpoints never supplies a later observation to a window.
    The functional gradient-enabled math path matches the previous reference;
    no backward/optimizer operation is performed. Arms are NOT concatenated
    on the batch axis: GEMM batch-size roundoff changed a threshold crossing
    in real shot94258 by 2.4e-7. The deployed runner uses the original batch32.
    This is a numerical reproducibility constraint, not relaxed validation.
"""
    names = list(signals)
    if not names or type(batch_size) is not int or batch_size < 1:
        raise ValueError('nonempty arms and positive endpoint batch size required')
    n = len(signals[names[0]]['values'])
    windows, masks = {}, {}
    for name, signal in signals.items():
        if (set(signal) != {'values', 'available'} or len(signal['values']) != n
                or signal['values'].shape != (n, 23) or signal['available'].shape != (n, 23)):
            raise ValueError('matching signal-only arms required')
        for key, output in (('values', windows), ('available', masks)):
            padded = np.pad(signal[key], ((model.cfg.window-1, 0), (0, 0)))
            output[name] = np.lib.stride_tricks.sliding_window_view(
                padded, model.cfg.window, axis=0).swapaxes(1, 2)
    fast = initial_encoder(model)
    scores = {name: [] for name in names}
    for name in names:
        for lo in range(0, n, batch_size):
            hi = min(n, lo+batch_size)
            x = torch.as_tensor(np.array(windows[name][lo:hi], copy=True), device=device)
            m = torch.as_tensor(np.array(masks[name][lo:hi], copy=True), device=device)
            h = model.predict_fast(x, m, fast, mode='risk_state')['risk_state']
            p = model.base.risk_head(h).squeeze(-1).sigmoid().detach().cpu().numpy()
            if not np.isfinite(p).all(): raise FloatingPointError('nonfinite risk trace')
            scores[name].append(p)
    return {k: np.concatenate(v) for k, v in scores.items()}


def public(result):
    return {k: v for k, v in result.items() if k != 'predictions'}


def snapshot(output):
    local = Path(__file__).parent
    files = list(local.glob('*.py')) + [local.parent/n for n in (
        'simple_data.py', 'simple23.py', 'io23.py', 'causal_input.py', 'simple_train.py',
        'fixed_test_protocol.py', 'simple_online.py', 'paired_feedback_20261022/model.py',
        'masked_state_20261022/model.py', 'redesign_20261004/prepare_prior.py')]
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, str(f.relative_to(io23.ROOT)))
    return {str(f.relative_to(io23.ROOT)): sha(f) for f in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    start = time.monotonic()
    contract = load_test_contract('EAST'); order = contract['shot_ids']
    base, parent = load_parent(DEFAULT_PARENT)
    model = PairedFeedback(base, np.ones(23)).cuda(); model.set_stage('frozen')
    original = {k: v.cpu().clone() for k, v in model.base.state_dict().items()}
    threshold = float(parent['threshold'])
    arms = {f'{policy}_n{n}': (n, policy) for n in (10, 20) for policy in POLICIES}
    args.output.mkdir(parents=True); (args.output/'shots').mkdir(); (args.output/'maps').mkdir()
    protocol = dict(format='warmstart-input-alignment-v1', contract=contract,
        source_checkpoint=str(DEFAULT_PARENT), source_sha256=PARENT_SHA256, threshold=threshold,
        arms=arms, calibration_order='first 10/20 shots of fixed TEST360 in ascending shot ID',
        intervention='pointwise input affine map only; no source-scaler replacement, masking change or neural training',
        source_statistics='all TRAIN1353 signals; equal shot weight; max128 observed points/channel/shot',
        target_access='fit ONLY after prediction sealing of the exact ended prefix; no outcomes for fitting',
        comparison='complete prequential TEST360 plus common suffix340; warmup cost retained',
        class_selection='none: neither known D nor known safe used to select calibration shots',
        exclusions=['MNM', 'mode_number_n'], learned_alarm_updates=0, neural_parameter_updates=0,
        independent_confirmation=False, reproduction='EFD-inspired; not paper131 PGFE-U/FLS/z-score source reproduction',
        static_reference=str(REFERENCE), static_reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        code_sha256=snapshot(args.output))
    _durable_json(args.output/'protocol.json', protocol)
    roles = io23.source_roles()
    source_rows = [{k: s[k] for k in SIGNAL_KEYS} for s in
                   (simple_data.read_shot('J-TEXT', sid) for sid in roles['train'])]
    source = fit_statistics(source_rows, device='J-TEXT', allowed_ids=roles['train'])
    _durable_json(args.output/'source_statistics.json', source)
    del source_rows
    pending = []; maps = {}; rows = {k: [] for k in ('static', *arms)}; receipts = []
    reference_error = 0.
    for index, sid in enumerate(order):
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = {k: raw[k] for k in ('values', 'available')}
        inputs = dict(static=signal)
        for name, mapping in maps.items(): inputs[name] = transform(signal, mapping)
        predictions = predict_many(model, inputs, device='cuda')
        for name in arms:
            if name not in predictions: predictions[name] = predictions['static'].copy()
        with np.load(REFERENCE/'shots'/f'{sid}.npz', allow_pickle=False) as ref:
            if not np.array_equal(ref['times'], raw['times']): raise ValueError('reference clock changed')
            error = float(abs(ref['static']-predictions['static']).max())
            if error > 2e-6: raise AssertionError(f'frozen parent numerical parity failed: {sid} {error}')
            if not np.array_equal(ref['static'] >= threshold, predictions['static'] >= threshold):
                raise AssertionError('baseline threshold crossings changed')
        reference_error = max(reference_error, error)
        path = args.output/'shots'/f'{sid}.npz'
        np.savez_compressed(path, times=raw['times'], **predictions)
        receipts.append(dict(shot=sid, prediction_sha256=sha(path), static_max_error=error,
                             active_maps=sorted(maps)))
        _durable_json(args.output/'shots'/f'{sid}_receipt.json', receipts[-1])
        # Inference for ALL arms is sealed before statistics or outcomes.
        if index < 20: pending.append({k: raw[k] for k in SIGNAL_KEYS})
        for name, (count, policy) in arms.items():
            if index+1 == count:
                mapping = fit_prefix_map(source, pending, order=order, count=count, policy=policy)
                mapping['preceding_prediction_seals'] = {str(r['shot']): r['prediction_sha256'] for r in receipts}
                _durable_json(args.output/'maps'/f'{name}.json', mapping)
                maps[name] = mapping
        if index == 19: pending = []
        outcome = simple_data.attach_outcome(raw, 'EAST', {})
        for name in rows:
            v = outcome['valid']
            rows[name].append(dict(shot=sid, times=raw['times'][v], scores=predictions[name][v],
                                   is_disrupt=outcome['is_disrupt'], down=outcome['down']))
        if (index+1) % 30 == 0:
            progress = dict(shots=index+1, total=len(order), seconds=time.monotonic()-start)
            _durable_json(args.output/'progress.json', progress); print(json.dumps(progress), flush=True)
    result = {}; full = {}
    for name, arm_rows in rows.items():
        full[name] = score_fixed_test('EAST', arm_rows, threshold)
        result[name] = dict(full360=public(full[name]),
            common_suffix340=public(metrics(arm_rows[20:], threshold, lead=30.)),
            warmup20=public(metrics(arm_rows[:20], threshold, lead=30.)))
        if name in arms:
            n = arms[name][0]
            result[name]['own_suffix'] = public(metrics(arm_rows[n:], threshold, lead=30.))
    for key, value in original.items():
        if not torch.equal(value, model.base.state_dict()[key].cpu()):
            raise AssertionError('source state changed')
    if sha(DEFAULT_PARENT) != PARENT_SHA256: raise AssertionError('original checkpoint changed')
    _durable_json(args.output/'prediction_receipts.json', receipts)
    _durable_json(args.output/'per_shot.json', full)
    summary = dict(status='COMPLETED', metrics=result, threshold=threshold,
        static_reference_max_abs_error=reference_error, source_bitwise_unchanged=True,
        neural_parameter_updates=0, seconds=time.monotonic()-start, independent_confirmation=False)
    _durable_json(args.output/'completed.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
