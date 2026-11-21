"""Full-shot destructive-input controls following the startup sensitivity audit.

Not a proposed predictor and not a neural adaptation experiment. These controls
test whether lowering selected window scores survives whole-shot accounting.
All actual startup errors and >30ms timely alarms are counted; no alarm retuning.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from . import io23, simple_data
from .ended_context_experiment import side_metrics
from .operating_curve_audit import load_sealed, operating_curve
from .persistent_ffn import PersistentFfnModel
from .redesign_20261004.intra import _batch_windows
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash
from .startup_sensitivity import REFERENCE
from .verify_event_projection import recount_prediction

MODES = ('static', 'mask_dx_a', 'mask_ip', 'mask_ne_nG', 'history32')


def predict_intervention(model, shot, intra, *, mode, device='cuda',
                         data_api=simple_data, **kwargs):
    if intra or mode not in MODES:
        raise ValueError('declared fixed input control only; no neural updating')
    signals = {key: shot[key] for key in ('values', 'available', 'times')}
    scores = []
    with torch.no_grad():
        for begin in range(0, len(shot['times']), 128):
            ends = range(begin, min(begin+128, len(shot['times'])))
            x, a = _batch_windows(data_api, signals, ends, model, device)
            if mode.startswith('mask_'):
                a[:, :, io23.RAW23.index(mode[5:])] = False
            elif mode == 'history32':
                a[:, :-32] = False
            scores.extend(model(x, a)['logit'].sigmoid().cpu().tolist())
    return np.asarray(scores, np.float32), dict(
        intra=False, mode=mode, accepted_updates=0, attempted_updates=0,
        fast_discarded=True, score_count=len(scores),
        model_parameter_change_norm=0., model_buffer_change_norm=0.,
        input_intervention=True, scientific_role='destructive diagnostic control, not neural adaptation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output directory required')
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4); torch.manual_seed(131)
    order = json.loads((REFERENCE/'order.json').read_text())
    _durable_json(args.output/'order.json', order)
    base, parent = load_parent(DEFAULT_PARENT)
    model = PersistentFfnModel(base).cuda().eval().requires_grad_(False)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    threshold = float(parent['threshold'])
    root = Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for name in ('startup_intervention_replay.py', 'startup_sensitivity.py', 'persistent_ffn.py',
                     'ffn_intra_20261005/model.py', 'simple23.py', 'simple_data.py', 'io23.py',
                     'causal_input.py', 'simple_online.py', 'simple_train.py',
                     'redesign_20261004/intra.py', 'redesign_20261004/run.py'):
            z.write(root/name, name)
    _durable_json(args.output/'protocol.json', dict(
        scope='POSTHOC DEVELOPMENT DESTRUCTIVE INPUT CONTROLS, NOT A LEARNED POLICY',
        parent_sha256=PARENT_SHA256, order_sha256=_hash(REFERENCE/'order.json'),
        modes=list(MODES), threshold=threshold, no_neural_updates=True,
        channel_choice='three large startup sensitivities; chosen AFTER sensitivity audit, not preregistered',
        history32='one diagnostic history length, not threshold or alarm tuning',
        reference=str(REFERENCE.resolve()), no_target_labels_enter_forward=True,
        all_startup_errors_counted=True, no_physical_intervention_claim=True,
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip')))
    provider = SimpleNamespace(read_shot=partial(simple_data.read_shot, rate_contract='nominal'),
                               attach_outcome=simple_data.attach_outcome, window=simple_data.window)
    signals = None
    results, curves, verification = {}, {}, {}
    started = time.monotonic()
    for mode in MODES:
        folder = args.output/mode
        folder.mkdir()
        result = run_arm(model, order=order, arm='static', threshold=threshold, output_dir=folder/'stream',
            data_api=provider, predictor=partial(predict_intervention, mode=mode))
        metrics = side_metrics(folder/'stream', order, threshold, 0, len(order))
        if any(result['metrics'][k] != metrics[k] for k in ('tp','fp','tpr','fpr','J')):
            raise AssertionError('whole-shot replay and reread metrics differ')
        if signals is None:
            # All first-arm outcomes already revealed. They remain EVALUATION
            # inputs only; subsequent predictors receive the label-free API.
            signals = [simple_data.attach_outcome(provider.read_shot('EAST', r['shot']), 'EAST', {})
                       for r in order]
        records, hashes = load_sealed(folder/'stream', order, signals, threshold)
        curve = operating_curve(records, threshold)
        if any(curve['fixed_source_threshold'][k] != metrics[k] for k in ('tp','fp','tpr','fpr','J')):
            raise AssertionError('independent operational maxima recount differs')
        decisions = [recount_prediction(row, threshold) for row in records]
        if sum(r['timely'] for r in decisions) != metrics['tp'] or sum(
                r['alarm'] and not r['is_disrupt'] for r in decisions) != metrics['fp']:
            raise AssertionError('independent first-alarm recount differs')
        max_error = None
        if mode == 'static':
            previous, _ = load_sealed(REFERENCE/'static/stream', order, signals, threshold)
            max_error = max(float(np.max(np.abs(a['scores']-b['scores'])))
                            for a,b in zip(records, previous))
            if max_error > 2e-6: raise AssertionError('same-parent full static replay differs')
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v.detach().cpu(), initial[k], rtol=0, atol=0)
        results[mode] = {k:v for k,v in metrics.items() if k != 'predictions'}
        curves[mode] = curve
        verification[mode] = dict(checked_seals=len(hashes), accepted_updates=0,
            parameters_bitwise_unchanged=True, original_static_max_error=max_error)
        _durable_json(folder/'independent_recount.json', dict(verification=verification[mode],
            metrics=results[mode], decisions=decisions, seals=hashes))
        _durable_json(args.output/'progress.json', dict(results=results, verification=verification))
        print(json.dumps(dict(completed=mode, metrics=results[mode])), flush=True)
    assert _hash(DEFAULT_PARENT) == PARENT_SHA256 and not model.memory.any()
    _durable_json(args.output/'operating_curves.json', curves)
    _durable_json(args.output/'completed.json', dict(results=results, verification=verification,
        elapsed_seconds=time.monotonic()-started, independent_recount=True,
        source_and_registered_tensors_unchanged=True, no_adaptation_gain_claim=True))


if __name__ == '__main__': main()
