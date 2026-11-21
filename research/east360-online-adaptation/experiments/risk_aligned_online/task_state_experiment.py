"""Fixed-parent EAST240 ablation of task-containing state feedback."""
import argparse
import json
from pathlib import Path
import time
import zipfile

import torch

from .ended_context_experiment import REFERENCE, side_metrics
from .ended_decoder import DecoderFfnModel
from .ended_density_replay import EndedDensityReplay
from .event_dynamics_experiment import load_coordinates, PROBE
from .task_state_feedback import TaskStateFfnModel
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash


MODES = ('static', 'risk', 'event_risk', 'random_risk')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('new output directory required')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    previous = Path(__file__).parent / 'runs/event_dynamics240_20261021'
    assert json.loads((previous / 'independent_verification.json').read_text())['verification'] == 'PASS'
    order = json.loads((REFERENCE / 'order.json').read_text())
    assert len(order) == 240
    torch.set_num_threads(4); torch.manual_seed(131)
    base, parent = load_parent(DEFAULT_PARENT)
    model = (DecoderFfnModel(base) if args.mode == 'static' else
        TaskStateFfnModel(base, *load_coordinates('event'), mode=args.mode)).cuda().eval().requires_grad_(False)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    args.output.mkdir(parents=True)
    _durable_json(args.output / 'order.json', order)
    provider = EndedDensityReplay(parent['source_center'], parent['source_scale'],
        output_dir=args.output / 'stream', order=[r['shot'] for r in order], preserve_observed_channels=True)
    root = Path(__file__).parent
    files = ['task_state_feedback.py', 'task_state_experiment.py', 'event_dynamics.py',
        'event_dynamics_experiment.py', 'event_gradient_projection.py', 'event_projection_experiment.py',
        'ended_decoder.py', 'ended_density_replay.py', 'ended_context_rule.py', 'ended_context_experiment.py',
        'persistent_ffn.py', 'ffn_intra_20261005/model.py', 'fast_update.py', 'simple23.py',
        'simple_data.py', 'simple_train.py', 'simple_online.py', 'io23.py', 'causal_input.py', 'alignment.py',
        'redesign_20261004/intra.py', 'redesign_20261004/run.py', 'redesign_20261004/prepare_prior.py']
    with zipfile.ZipFile(args.output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for name in files: z.write(root / name, name)
    protocol = dict(scope='DEVELOPMENT_ONLY_PURE_INTRA_TASK_STATE', mode=args.mode,
        parent_sha256=PARENT_SHA256, threshold=float(parent['threshold']), seed=131,
        order_sha256=_hash(REFERENCE / 'order.json'), selected_shots=240,
        source_retrained=False, risk_head_changed=False, alarm_tuning=False,
        auxiliary_head_changed=False, persistent_ffn_updates=False, supervised_updates_enabled=False,
        normalization='same ended-first30 density_robust mapping, all observed channels retained',
        mapping_approved_physical_calibration=False, first30_errors_included=True,
        temporary_site='original last Transformer FFN', temporary_parameter_count=model.temporary_dim,
        temporary_discarded=True, predicted_signals_used_for_risk=False, autoregressive_rollout=False,
        current_outcome_access=False, east_event_labels_used=False, event_pseudo_label_used=False,
        state_dimensions=0 if args.mode == 'static' else len(model.state_directions),
        risk_coordinate_count=0 if args.mode == 'static' else 1,
        teacher='immutable source response to arrived diagnostics; no EAST outcome targets',
        target_detached=True, risk_teacher_is_physical_truth=False,
        source_probe_file_sha256=_hash(PROBE / 'probe_heads.pt'),
        source_feature_normalization='same TRAIN event-prefix features; teacher split overlap remains',
        objective='none' if args.mode == 'static' else 'SmoothL1 on same-coordinate predicted/arrived windows',
        coordinate_design='risk first, task-orthogonal event5 or seeded random5; no fitted loss coefficient',
        risk_scale=None if args.mode == 'static' else float(model.risk_coordinate_scale),
        state_horizons=[5, 10, 15, 20, 25], stride=100, horizon=25, intra_step=.05, intra_radius=1.,
        acceptance='strictly lower same matured state loss', lead_ms_strict=30.,
        validation_gate='whole EAST240 TP>static and FPR<=.20, strict positive J gain; no posthoc threshold promotion',
        source_auxiliary_screen_not_target_gain_gate=True, all_prespecified_arms_reported=True,
        previous_reference=str(previous), compute_matched=False,
        code_snapshot_sha256=_hash(args.output / 'code_snapshot.zip'))
    _durable_json(args.output / 'protocol.json', protocol)
    started = time.monotonic()
    result = run_arm(model, order=order, arm='static' if args.mode == 'static' else 'intra',
        threshold=float(parent['threshold']), output_dir=args.output / 'stream',
        data_api=provider, updater=None, postseal_feedback=False)
    for k, v in model.state_dict().items(): torch.testing.assert_close(v.detach().cpu(), initial[k], rtol=0, atol=0)
    model.assert_persistent_inactive()
    assert _hash(DEFAULT_PARENT) == PARENT_SHA256
    periods = {name: side_metrics(args.output / 'stream', order, float(parent['threshold']), lo, hi)
        for name, lo, hi in [('all', 0, 240), ('first30', 0, 30), ('after30', 30, 240)]}
    _durable_json(args.output / 'comparison.json', periods)
    with (args.output / 'final_state.pt').open('xb') as handle:
        torch.save(dict(state=model.state_dict(), parent_sha256=PARENT_SHA256, ended_ids=provider.ended_ids), handle)
    _durable_json(args.output / 'completed.json', dict(source_and_all_registered_tensors_unchanged=True,
        elapsed_seconds=time.monotonic()-started, observed_shots=result['observed_count'],
        accepted_intra_updates=result['accepted_intra_updates'], memory_updates=0,
        head_optimizer_steps=0, temporary_retained=False))
    print(json.dumps(dict(completed=args.mode, metrics={name: {k: v for k, v in r.items() if k != 'predictions'}
        for name, r in periods.items()})), flush=True)


if __name__ == '__main__': main()
