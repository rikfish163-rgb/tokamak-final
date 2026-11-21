"""Fixed-parent EAST240 comparison of raw versus source-state forecast losses."""
import argparse
import json
from pathlib import Path
import time
import zipfile

import torch

from .ended_context_experiment import REFERENCE, side_metrics
from .ended_decoder import DecoderFfnModel
from .ended_density_replay import EndedDensityReplay
from .event_dynamics import EventDynamicsFfnModel, HORIZONS
from .event_gradient_projection import make_directions
from .event_projection_experiment import PROBE
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash


def load_coordinates(mode):
    protocol = json.loads((PROBE / 'protocol.json').read_text())
    verified = json.loads((PROBE / 'independent_verification.json').read_text())
    if protocol['source_parent_sha256'] != PARENT_SHA256 or verified['status'] != 'PASS':
        raise ValueError('verified source probe of the SAME parent required')
    head = torch.load(PROBE / 'probe_heads.pt', map_location='cpu', weights_only=True)['latent_true_131']
    # Unlike the previous raw-feature Jacobian projection, both new losses
    # use standardized192 source features. Orthonormal contrasts are formed
    # in THAT space, then kept fixed throughout target streaming.
    directions = make_directions(head['weight'], torch.ones_like(head['scale']), mode=mode, seed=131)
    return directions, head['mean'], head['scale']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('static', 'control', 'event', 'random'), required=True)
    p.add_argument('--limit', type=int, default=240)
    args = p.parse_args()
    order = json.loads((REFERENCE / 'order.json').read_text())
    if args.output.exists() or not 30 < args.limit <= len(order):
        p.error('new output and locked31..240 shot prefix required')
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(131)
    base, parent = load_parent(DEFAULT_PARENT)
    original = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    state_loss = args.mode in ('event', 'random')
    model = (EventDynamicsFfnModel(base, *load_coordinates(args.mode), mode=args.mode)
             if state_loss else DecoderFfnModel(base)).to('cuda').eval().requires_grad_(False)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    args.output.mkdir(parents=True)
    order = order[:args.limit]
    provider = EndedDensityReplay(parent['source_center'], parent['source_scale'],
        output_dir=args.output / 'stream', order=[r['shot'] for r in order], preserve_observed_channels=True)
    _durable_json(args.output / 'order.json', order)
    folder = Path(__file__).parent
    files = [Path(__file__), folder / 'event_dynamics.py', folder / 'event_gradient_projection.py',
        folder / 'event_projection_experiment.py', folder / 'ended_decoder.py',
        folder / 'ended_context_rule.py', folder / 'ended_density_replay.py', folder / 'persistent_ffn.py',
        folder / 'ffn_intra_20261005/model.py', folder / 'fast_update.py', folder / 'simple23.py',
        folder / 'simple_data.py', folder / 'io23.py', folder / 'simple_online.py', folder / 'simple_train.py',
        folder / 'alignment.py', folder / 'causal_input.py', folder / 'ended_context_experiment.py',
        folder / 'redesign_20261004/intra.py', folder / 'redesign_20261004/run.py',
        folder / 'redesign_20261004/prepare_prior.py']
    with zipfile.ZipFile(args.output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(folder)))
    protocol = dict(scope='DEVELOPMENT_ONLY_MATURED_STATE_LOSS', mode=args.mode,
        parent_sha256=PARENT_SHA256, order_sha256=_hash(REFERENCE / 'order.json'),
        threshold=float(parent['threshold']), selected_shots=args.limit, seed=131,
        source_retrained=False, risk_head_changed=False, alarm_tuning=False,
        auxiliary_head_changed=False, supervised_updates_enabled=False,
        temporary_site='original last Transformer FFN', temporary_parameter_count=model.temporary_dim,
        temporary_discarded=True, persistent_ffn_updates=False,
        target_outcomes='natural reveal only after durable prediction seal, evaluation only',
        normalization='same ended-first30 density_robust mapping, all observed channels retained',
        mapping_approved_physical_calibration=False, first30_errors_included=True,
        objective=('SmoothL1 in fixed source-standardized event contrast space' if args.mode == 'event' else
                   'SmoothL1 in fixed source-standardized random5 space' if args.mode == 'random' else
                   'original masked source-asinh signal SmoothL1' if args.mode == 'control' else 'none'),
        state_horizons=list(HORIZONS) if state_loss else None,
        state_dimensions=5 if state_loss else None,
        source_probe_file_sha256=_hash(PROBE / 'probe_heads.pt') if state_loss else None,
        teacher='immutable source; actual future target detached; forecast input differentiable' if state_loss else None,
        predicted_signals_used_for_risk=False, autoregressive_rollout=False,
        current_outcome_access=False, east_event_labels_used=False, event_pseudo_label_used=False,
        teacher_provenance='offline whole-JTEXT teacher overlaps historical source splits; diagnostic only',
        stride=100, horizon=25, intra_step=.05, intra_radius=1., lead_ms_strict=30.,
        compute_matched=False, code_snapshot_sha256=_hash(args.output / 'code_snapshot.zip'))
    _durable_json(args.output / 'protocol.json', protocol)
    result = run_arm(model, order=order, arm='static' if args.mode == 'static' else 'intra',
        threshold=protocol['threshold'], output_dir=args.output / 'stream', data_api=provider,
        updater=None, postseal_feedback=False)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v.detach().cpu(), initial[k], rtol=0, atol=0)
    for k, v in model.base.state_dict().items():
        torch.testing.assert_close(v.detach().cpu(), original[k], rtol=0, atol=0)
    model.assert_persistent_inactive()
    if _hash(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('qualified original checkpoint changed')
    periods = {name: side_metrics(args.output / 'stream', order, protocol['threshold'], lo, hi)
        for name, lo, hi in [('all', 0, len(order)), ('first30', 0, 30), ('after30', 30, len(order))]}
    _durable_json(args.output / 'comparison.json', periods)
    with (args.output / 'final_state.pt').open('xb') as handle:
        torch.save(dict(state=model.state_dict(), parent_sha256=PARENT_SHA256,
                        ended_ids=provider.ended_ids), handle)
    _durable_json(args.output / 'completed.json', dict(source_and_all_registered_tensors_unchanged=True,
        elapsed_seconds=time.monotonic() - started, observed_shots=result['observed_count'],
        accepted_intra_updates=result['accepted_intra_updates'], memory_updates=0,
        head_optimizer_steps=0, temporary_retained=False, all_metric_seal_hashes_verified=True))
    print(json.dumps(dict(completed=args.mode,
        metrics={period: {k: v for k, v in m.items() if k != 'predictions'} for period, m in periods.items()})), flush=True)


if __name__ == '__main__':
    main()
