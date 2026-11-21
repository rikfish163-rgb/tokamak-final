"""One real causal arm per process, with immutable source/input/alarm controls.

Early vs continued learning of the H25 auxiliary head; no persistent FFN or
final-score updates. All startup mistakes count. Calibration remains the same
failed numerical density intervention, not a claimed physical unit conversion.
"""
import argparse
import json
from pathlib import Path
import zipfile

import torch

from .continual_decoder import MODES, RollingDecoder, head_digest
from .ended_context_experiment import REFERENCE, side_metrics
from .ended_decoder import DecoderFfnModel
from .ended_density_replay import EndedDensityReplay
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash

ROOT = Path(__file__).parent
ARMS = (*MODES, 'early_continual_nofast')
BASELINE = ROOT/'runs/ended_decoder120_20261010'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--limit', type=int, default=120)
    parser.add_argument('--online-steps', type=int, default=4)
    parser.add_argument('--bootstrap-steps', type=int, default=40)
    args = parser.parse_args()
    order = json.loads((REFERENCE/'order.json').read_text())
    if (args.output.exists() or not 30 < args.limit <= len(order)
            or min(args.online_steps, args.bootstrap_steps) < 1):
        parser.error('new output, limit31..240 and positive declared budgets required')
    order = order[:args.limit]
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(131)
    base, state = load_parent(DEFAULT_PARENT)
    threshold = float(state['threshold'])
    model = DecoderFfnModel(base).to('cuda').eval()
    mode = 'early_continual' if args.arm == 'early_continual_nofast' else args.arm
    learner = RollingDecoder(model, mode=mode, online_steps=args.online_steps,
                             bootstrap_steps=args.bootstrap_steps)
    provider = EndedDensityReplay(state['source_center'], state['source_scale'],
        output_dir=args.output/'stream', order=[r['shot'] for r in order],
        preserve_observed_channels=True)
    _durable_json(args.output/'order.json', order)
    files = [Path(__file__), ROOT/'continual_decoder.py', ROOT/'ended_decoder.py',
        ROOT/'ended_context_rule.py', ROOT/'ended_context_experiment.py',
        ROOT/'ended_density_replay.py', ROOT/'alignment.py', ROOT/'persistent_ffn.py',
        ROOT/'fast_update.py', ROOT/'simple23.py', ROOT/'simple_data.py',
        ROOT/'simple_online.py', ROOT/'causal_input.py', ROOT/'channel_gradient_audit.py',
        ROOT/'ffn_intra_20261005/model.py', ROOT/'ffn_intra_20261005/loss.py',
        ROOT/'redesign_20261004/intra.py', ROOT/'redesign_20261004/run.py',
        ROOT/'redesign_20261004/prepare_prior.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, str(file.relative_to(ROOT)))
    _durable_json(args.output/'protocol.json', dict(scope='DEVELOPMENT CAUSAL SCHEDULE PILOT',
        arm=args.arm, mode=mode, parent_sha256=PARENT_SHA256, threshold=threshold,
        order_sha256=_hash(REFERENCE/'order.json'), selected_shots=args.limit,
        comparison_baseline=str(BASELINE.resolve()), online_steps=args.online_steps,
        bootstrap_steps=args.bootstrap_steps, lr=.0003, seed=131,
        replay_shots=32, natural_postseal_outcomes=True,
        input_mapping='same ended-first30 density mapping, preserve observed constant channels',
        mapping_approved_physical_calibration=False, input_coordinate_transition_index=30,
        transition_action='reset auxiliary head/Adam to source and refit exact ended first30',
        risk_head_changed=False, source_changed=False, persistent_ffn_training=False,
        same_shot_delta_discarded=True, stride=100, horizon=25, step=.05, radius=1.,
        after30_outer='adapted whole-task fixed-peak risk plus .20 actual H25 signal maintenance',
        temporal_peak='same original-head causal reference in current coordinates',
        current_in_each_online_pair=True, opposite_from_only_ended_recent_history=True,
        source_risk_threshold_fixed=True, alarm_tuning=False,
        steps_equal_to_fixed40=False,
        total_budget_caveat='schedule and extra compute both change; not compute-matched superiority proof',
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip')))

    def updater(m, shot, replay, **kwargs):
        return learner(m, shot, replay, provider=provider, **kwargs)

    result = run_arm(model, order=order, arm='inter' if args.arm.endswith('_nofast') else 'both',
        threshold=threshold, output_dir=args.output/'stream', data_api=provider, updater=updater)
    if model.memory.any() or _hash(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('persistent/source state changed')
    with (args.output/'final_head.pt').open('xb') as handle:
        torch.save(dict(decoder=model.decoder.state_dict(),
            optimizer=None if learner.optimizer is None else learner.optimizer.state_dict(),
            optimizer_steps=learner.optimizer_steps, ended_ids=provider.ended_ids,
            parent_sha256=PARENT_SHA256, mode=mode, first_unused_future_index=len(order)+1), handle)
    periods = {period:side_metrics(args.output/'stream', order, threshold, a, b)
               for period,a,b in (('all',0,args.limit),('first30',0,30),('after30',30,args.limit))}
    _durable_json(args.output/'comparison.json', periods)
    _durable_json(args.output/'completed.json', dict(source_sha256_unchanged=True,
        persistent_ffn_unchanged=True, head_sha256=head_digest(model),
        model_meta_optimizer_steps=learner.optimizer_steps, observed_shots=result['observed_count'],
        all_metric_seal_hashes_verified=True, temporary_retained=False))
    print(json.dumps(dict(completed=args.arm, metrics={p:{k:v for k,v in m.items() if k!='predictions'}
        for p,m in periods.items()})), flush=True)


if __name__ == '__main__':
    main()
