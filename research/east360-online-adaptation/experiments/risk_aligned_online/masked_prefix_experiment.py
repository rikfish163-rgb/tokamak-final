"""Five fixed-parent, source-only-input EAST240 masked-prefix comparisons."""
import argparse
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import torch

from . import simple_data
from .ended_context_experiment import REFERENCE, side_metrics
from .masked_prefix_feedback import MaskedPrefixModel, predict_shot
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash

MODES = ('static', 'signal_stale', 'signal_fresh', 'risk_stale', 'risk_fresh')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new experiment directory required')
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    training = json.loads((args.heads/'completed.json').read_text())
    assert training['base_bitwise_unchanged'] and not training['east_data_used']
    order = json.loads((REFERENCE/'order.json').read_text())
    assert len(order) == 240 and len({r['shot'] for r in order}) == 240
    torch.set_num_threads(4); torch.manual_seed(131)
    args.output.mkdir(parents=True)
    root = Path(__file__).parent
    files = ['masked_prefix_experiment.py', 'masked_prefix_feedback.py', 'masked_prefix_train.py',
             'persistent_ffn.py', 'ffn_intra_20261005/model.py', 'fast_update.py', 'simple23.py',
             'simple_data.py', 'simple_train.py', 'simple_online.py', 'io23.py', 'causal_input.py',
             'redesign_20261004/intra.py', 'redesign_20261004/run.py', 'redesign_20261004/prepare_prior.py',
             'ended_context_experiment.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in files: archive.write(root/name, name)
    _durable_json(args.output/'order.json', order)
    _, parent = load_parent(DEFAULT_PARENT)
    protocol = dict(scope='DEVELOPMENT_ONLY_SOURCE_ONLY_INPUT_SINGLE_SHOT_ADAPTATION',
        modes=list(MODES), source_sha256=PARENT_SHA256, order_sha256=_hash(REFERENCE/'order.json'),
        threshold=float(parent['threshold']), seed=131, selected_shots=240,
        source_classifier_changed=False, alarm_tuning=False, event_labels_used=False,
        target_support_used=False, past_target_normalization=False, persistent_neural_updates=False,
        target_head_training=False, targets='only masked coordinates of already observed same-shot signals',
        source_heads=str(args.heads.resolve()), source_head_sha256=training['decoder_sha256'],
        normalization='immutable source scaler; existing JTEXT ip A conversion; no fitted EAST mapping',
        input_missingness='existing reader excludes EAST MNM; CIII=-10 sentinel unavailable; no target-derived masks',
        temporary_site='original final Transformer FFN', temporary_parameters=16576,
        temporary_reset_each_shot=True, temporary_retained=False,
        start_update_index=152, update_stride=100, step=.05, radius=1.,
        feedback_lag_points=[0, 25], mask_patches=4, total_patches=16, patch_length=8,
        acceptance='strict reduction of same observed masked loss; not an outcome-based guard',
        prediction_rule='risk from original unmasked real inputs; update at t affects t+1 onward',
        lead_ms_strict=30., source_threshold_fixed=True,
        gain_gate='full-stream TP>static, FPR<=.20 and J>static; threshold caps descriptive only',
        independent_future_validation=False, source_checkpoint_not_MAE_pretrained=True,
        equal_update_opportunities=True, equal_wall_clock_budget=False,
        includes_all_startup_errors=True, includes_short_shots_without_updates=True,
        first30='descriptive subset only; no adaptation bootstrap or selection on these outcomes',
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json', protocol)
    provider = SimpleNamespace(read_shot=partial(simple_data.read_shot, rate_contract='nominal'),
                               attach_outcome=simple_data.attach_outcome, window=simple_data.window)
    started = time.monotonic()
    results = {}
    for mode in MODES:
        output = args.output/mode
        output.mkdir()
        base, _ = load_parent(DEFAULT_PARENT)
        model = MaskedPrefixModel(base).cuda().eval().requires_grad_(False)
        head_kind = 'risk' if mode.startswith('risk') else 'signal'
        checkpoint = args.heads/f'{head_kind}_head.pt'
        assert _hash(checkpoint) == training['decoder_sha256'][head_kind]
        state = torch.load(checkpoint, map_location='cuda', weights_only=True)
        assert state['mode'] == head_kind and state['parent_sha256'] == PARENT_SHA256
        model.decoder.load_state_dict(state['decoder'])
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        lag = 25 if mode.endswith('stale') else 0
        result = run_arm(model, order=order, arm='static' if mode == 'static' else 'intra',
            threshold=float(parent['threshold']), output_dir=output/'stream', data_api=provider,
            predictor=partial(predict_shot, lag=lag), postseal_feedback=False)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v.detach().cpu(), initial[k], atol=0, rtol=0)
        assert not model.memory.detach().count_nonzero() and _hash(DEFAULT_PARENT) == PARENT_SHA256
        periods = {name: side_metrics(output/'stream', order, float(parent['threshold']), lo, hi)
                   for name, lo, hi in [('all', 0, 240), ('first30', 0, 30), ('after30', 30, 240)]}
        _durable_json(output/'comparison.json', periods)
        with (output/'final_state.pt').open('xb') as handle:
            torch.save(dict(state=model.state_dict(), parent_sha256=PARENT_SHA256), handle)
        _durable_json(output/'completed.json', dict(mode=mode, lag=lag, head_kind=head_kind,
            source_and_registered_tensors_unchanged=True, observed_shots=result['observed_count'],
            accepted_intra_updates=result['accepted_intra_updates'], persistent_updates=0,
            auxiliary_head_updates=0, elapsed_seconds=result['elapsed_seconds']))
        results[mode] = {period: {k: v for k, v in point.items() if k != 'predictions'}
                         for period, point in periods.items()}
        _durable_json(args.output/'progress.json', dict(completed_modes=list(results), results=results))
        print(json.dumps(dict(completed=mode, metrics=results[mode])), flush=True)
        del model, initial
        torch.cuda.empty_cache()
    _durable_json(args.output/'completed.json', dict(elapsed_seconds=time.monotonic()-started,
        completed_modes=list(results), source_sha256=PARENT_SHA256))


if __name__ == '__main__': main()
