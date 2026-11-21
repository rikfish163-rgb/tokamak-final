"""Matched EAST test of event-state-preserving disposable FFN updates.

The primary head_feedback=none matrix has no post-shot supervised updates.
The same optional early_fixed auxiliary-head feedback is a separate factor,
not silently included in an arm called pure intra. All first30 errors count.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import zipfile

import torch

from .continual_decoder import RollingDecoder, head_digest
from .ended_context_experiment import REFERENCE, side_metrics
from .ended_decoder import DecoderFfnModel
from .ended_density_replay import EndedDensityReplay
from .event_gradient_projection import EventProtectedFfnModel, make_directions
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash
from .io23 import ROOT


PROBE = ROOT / 'outputs/event_prefix_probe_20261021'
MODES = ('static', 'control', 'event', 'random')


def load_directions(mode):
    protocol = json.loads((PROBE / 'protocol.json').read_text())
    verified = json.loads((PROBE / 'independent_verification.json').read_text())
    if protocol['source_parent_sha256'] != PARENT_SHA256 or verified['status'] != 'PASS':
        raise ValueError('verified same-parent source event probe required')
    head = torch.load(PROBE / 'probe_heads.pt', map_location='cpu', weights_only=True)['latent_true_131']
    return make_directions(head['weight'], head['scale'], mode=mode, seed=131)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--head-feedback', choices=('none', 'early_fixed'), default='none')
    parser.add_argument('--limit', type=int, default=240)
    args = parser.parse_args()
    order = json.loads((REFERENCE / 'order.json').read_text())
    if args.output.exists() or not 30 < args.limit <= len(order):
        parser.error('new output and locked31..240 shot prefix required')
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(131)
    base, parent = load_parent(DEFAULT_PARENT)
    original = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    projected = args.mode in ('event', 'random')
    model = (EventProtectedFfnModel(base, load_directions(args.mode), mode=args.mode)
             if projected else DecoderFfnModel(base)).to('cuda').eval()
    model.requires_grad_(False)
    original_head = deepcopy(model.decoder.state_dict())
    learner = RollingDecoder(model, mode='early_fixed') if args.head_feedback == 'early_fixed' else None
    order = order[:args.limit]
    args.output.mkdir(parents=True)
    provider = EndedDensityReplay(parent['source_center'], parent['source_scale'],
        output_dir=args.output / 'stream', order=[r['shot'] for r in order], preserve_observed_channels=True)
    _durable_json(args.output / 'order.json', order)
    folder = Path(__file__).parent
    source_paths = [Path(__file__), folder / 'event_gradient_projection.py', folder / 'event_prefix_probe.py',
        folder / 'continual_decoder.py', folder / 'ended_decoder.py', folder / 'ended_context_rule.py',
        folder / 'ended_density_replay.py', folder / 'persistent_ffn.py', folder / 'ffn_intra_20261005/model.py',
        folder / 'fast_update.py', folder / 'simple23.py', folder / 'simple_data.py', folder / 'io23.py',
        folder / 'simple_online.py', folder / 'simple_train.py', folder / 'alignment.py', folder / 'causal_input.py',
        folder / 'redesign_20261004/intra.py', folder / 'redesign_20261004/run.py',
        folder / 'redesign_20261004/prepare_prior.py', folder / 'ended_context_experiment.py']
    with zipfile.ZipFile(args.output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, str(path.relative_to(folder)))
    protocol = dict(scope='DEVELOPMENT_ONLY_EVENT_STATE_PROJECTION', mode=args.mode,
        parent_sha256=PARENT_SHA256, order_sha256=_hash(REFERENCE / 'order.json'),
        threshold=float(parent['threshold']), selected_shots=args.limit, seed=131,
        source_retrained=False, risk_head_changed=False, alarm_tuning=False,
        head_feedback=args.head_feedback, supervised_updates_enabled=learner is not None,
        head_learning_rule='archived unprojected risk_after_update; equal in all feedback arms',
        new_projection_is_meta_aligned=False, temporary_site='original last Transformer FFN',
        temporary_parameter_count=model.temporary_dim, temporary_discarded=True,
        persistent_ffn_updates=False, target_outcomes='natural reveal only after durable prediction seal',
        normalization='same ended-first30 density_robust mapping, all observed channels retained',
        mapping_approved_physical_calibration=False, first30_errors_included=True,
        state_coordinates='source event-logit contrast span' if args.mode == 'event' else
                          'fixed random five-dimensional feature span' if args.mode == 'random' else None,
        projection='local Jacobian nullspace at latest arrived128 window; rank cutoff1e-7' if projected else None,
        source_event_probe='latent_true_131 fixed before target comparison' if projected else None,
        source_event_probe_file_sha256=_hash(PROBE / 'probe_heads.pt') if projected else None,
        teacher_provenance='offline whole-JTEXT teacher overlaps historical source splits; not independent validation',
        east_event_labels_used=False, event_pseudo_label_used=False,
        stride=100, horizon=25, intra_step=.05, intra_radius=1.,
        lead_ms_strict=30., compute_matched=False,
        code_snapshot_sha256=_hash(args.output / 'code_snapshot.zip'))
    _durable_json(args.output / 'protocol.json', protocol)

    def updater(m, shot, replay, **kwargs):
        return learner(m, shot, replay, provider=provider, **kwargs)

    summary = run_arm(model, order=order, arm='static' if args.mode == 'static' else 'intra',
        threshold=protocol['threshold'], output_dir=args.output / 'stream', data_api=provider,
        updater=updater if learner is not None else None, postseal_feedback=learner is not None)
    for k, v in model.base.state_dict().items():
        torch.testing.assert_close(v.detach().cpu(), original[k], atol=0, rtol=0)
    if learner is None:
        for k, v in model.decoder.state_dict().items():
            torch.testing.assert_close(v, original_head[k], atol=0, rtol=0)
    model.assert_persistent_inactive()
    if _hash(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('original source checkpoint changed')
    periods = {name: side_metrics(args.output / 'stream', order, protocol['threshold'], lo, hi)
        for name, lo, hi in [('all', 0, len(order)), ('first30', 0, 30), ('after30', 30, len(order))]}
    _durable_json(args.output / 'comparison.json', periods)
    with (args.output / 'final_state.pt').open('xb') as handle:
        torch.save(dict(decoder=model.decoder.state_dict(), parent_sha256=PARENT_SHA256,
            directions=None if not projected else model.state_directions.detach().cpu(),
            ended_ids=provider.ended_ids), handle)
    _durable_json(args.output / 'completed.json', dict(source_file_and_parameters_unchanged=True,
        auxiliary_head_unchanged=learner is None, head_sha256=head_digest(model),
        head_optimizer_steps=0 if learner is None else learner.optimizer_steps,
        elapsed_seconds=time.monotonic() - started, observed_shots=summary['observed_count'],
        accepted_intra_updates=summary['accepted_intra_updates'], memory_updates=0,
        temporary_retained=False, all_metric_seal_hashes_verified=True))
    print(json.dumps(dict(completed=args.mode, head_feedback=args.head_feedback,
        metrics={p: {k: v for k, v in result.items() if k != 'predictions'} for p, result in periods.items()})), flush=True)


if __name__ == '__main__':
    main()
