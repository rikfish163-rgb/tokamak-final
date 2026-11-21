"""Matched ended-EAST decoder pilot; no risk-head, threshold or source changes.

First30 naturally revealed shots learn the auxiliary H25 head once. Compare
signal-only fitting to task-after-adaptation plus signal maintenance. The new
head is frozen before shot31; temporary last-FFN states are discarded each shot.
This is inter-trained intra, not pure source-only intra and not a full system.
"""
import argparse
from functools import partial
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .ended_context_experiment import REFERENCE, side_metrics
from .ended_context_rule import causal_peak_logit, peak_index
from .ended_decoder import DecoderFfnModel, train_pair
from .ended_density_replay import EndedDensityReplay
from .redesign_20261004.intra import predict_shot
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash

ROOT = Path(__file__).parent
ARMS = ('static', 'original_intra', 'signal_after30', 'meta_after30', 'meta_static')


def fit_ended_prefix(model, history, *, provider, output, head_steps, threshold, mode):
    if (provider.pending is not None or len(provider.ended_ids) != 30
            or [s['shot'] for s in history] != provider.ended_ids
            or any('_sealed_scores' not in s for s in history)):
        raise ValueError('exact first30 sealed and naturally revealed shots required')
    start = time.monotonic()
    episodes = []
    # The head is still the original source head here. Both learned arms use
    # the SAME original-intra training peaks, selected only from ENDED shots.
    for ended in history:
        shot = provider.reencode(ended)
        scores, _ = predict_shot(model, shot, intra=True, device='cuda')
        query = peak_index(shot, scores)
        logit, info = causal_peak_logit(model, shot, query)
        if abs(float(logit.detach().sigmoid())-float(scores[query])) > 2e-6:
            raise AssertionError('training peak replay differs from causal stream')
        episodes.append((shot, query))
        if len(episodes) % 5 == 0:
            print(json.dumps(dict(mode=mode, prepared_ended=len(episodes),
                elapsed_seconds=time.monotonic()-start)), flush=True)
    pools = [[e for e in episodes if int(e[0]['is_disrupt']) == label] for label in (0, 1)]
    if not all(pools):
        raise ValueError('ended prefix needs both classes; no silent skipped fitting')
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=.0003)
    rng = np.random.default_rng(131)
    records = []
    boundary = math.log(threshold/(1-threshold))
    for iteration in range(head_steps):
        pair = [pool[int(rng.integers(len(pool)))] for pool in pools]
        row = train_pair(model, pair, optimizer, boundary=boundary, mode=mode)
        records.append(dict(iteration=iteration+1, **row))
        if (iteration+1) % 5 == 0:
            print(json.dumps(dict(mode=mode, head_step=iteration+1,
                objective=row['objective'], risk_gradient_norm=row['risk_gradient_norm'],
                elapsed_seconds=time.monotonic()-start)), flush=True)
    model.decoder.requires_grad_(False)
    with (output/'ended_decoder.pt').open('xb') as handle:
        torch.save(dict(decoder=model.decoder.state_dict(), parent_sha256=PARENT_SHA256,
            fit_ids=provider.ended_ids, first_deployed_shot_index=31, mode=mode,
            base_changed=False, temporary_retained=False), handle)
    _durable_json(output/'head_training.json', dict(mode=mode, fit_ids=provider.ended_ids,
        n_disrupt=len(pools[1]), n_safe=len(pools[0]),
        fixed_peaks=[dict(shot=s['shot'], label=int(s['is_disrupt']), query=q) for s, q in episodes],
        objective='0.20 actual mature H25 signal maintenance plus adapted risk only in meta arm',
        peak_selection='same initial causal intra peaks; D lead>30, S all valid',
        peak_reselection=False, second_order_unroll_depth=4, actual_burn_in=True,
        learning_rate=.0003, steps=head_steps, records=records, seed=131,
        elapsed_seconds=time.monotonic()-start, artifact_sha256=_hash(output/'ended_decoder.pt')))
    return dict(accepted=True, reason='ended_prefix_auxiliary_head_fitted',
                head_steps=head_steps, source_changed=False, persistent_ffn_changed=False)


def prefix_updater(model, shot, replay, *, provider, output, head_steps, threshold, mode, **kwargs):
    if len(provider.ended_ids) == 30:
        return fit_ended_prefix(model, [*replay, shot], provider=provider, output=output,
                                head_steps=head_steps, threshold=threshold, mode=mode)
    return dict(accepted=False, reason='auxiliary_head_fixed_outside_first30_boundary')


def curve_error(left, right, order):
    errors = []
    for index, row in enumerate(order, 1):
        name = f'{index:04d}_{row["shot"]}_sealed.npz'
        with np.load(left/name) as a, np.load(right/name) as b:
            if not np.array_equal(a['times'], b['times']):
                raise AssertionError('control time axes differ')
            errors.append(float(np.max(np.abs(a['scores']-b['scores']))))
    return max(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=120)
    parser.add_argument('--steps', type=int, default=40)
    args = parser.parse_args()
    full_order = json.loads((REFERENCE/'order.json').read_text())
    if args.output.exists() or not 30 < args.limit <= len(full_order) or args.steps < 1:
        parser.error('new output, limit31..240 and positive head steps required')
    order = full_order[:args.limit]
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(131)
    _, state = load_parent(DEFAULT_PARENT)
    threshold = float(state['threshold'])
    _durable_json(args.output/'order.json', order)
    files = [Path(__file__), ROOT/'ended_decoder.py', ROOT/'ended_context_rule.py',
        ROOT/'ended_context_experiment.py', ROOT/'ended_density_replay.py',
        ROOT/'persistent_ffn.py', ROOT/'fast_update.py', ROOT/'simple23.py',
        ROOT/'simple_data.py', ROOT/'simple_online.py', ROOT/'causal_input.py',
        ROOT/'channel_gradient_audit.py', ROOT/'ffn_intra_20261005/model.py',
        ROOT/'ffn_intra_20261005/loss.py', ROOT/'redesign_20261004/intra.py',
        ROOT/'redesign_20261004/run.py', ROOT/'redesign_20261004/prepare_prior.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, str(file.relative_to(ROOT)))
    _durable_json(args.output/'protocol.json', dict(scope='DEVELOPMENT MATCHED PREFIX PILOT',
        parent_sha256=PARENT_SHA256, threshold=threshold, order_sha256=_hash(REFERENCE/'order.json'),
        selected_shots=args.limit, warmup=30, head_steps=args.steps, head_lr=.0003,
        inputs='same existing ended-first30 density mapping; immutable source asinh buffers',
        input_mapping_approved_physical_calibration=False, source_changed=False,
        alarm_tuning=False, transport=False, persistent_ffn_training=False,
        temporary_ffn_parameters=16576, stride=100, horizon=25, step=.05, radius=1.,
        auxiliary_decoder_parameters=sum(p.numel() for p in DecoderFfnModel(load_parent(DEFAULT_PARENT)[0]).decoder.parameters()),
        head_changes_zero_fast_risk=False, head_freezes_after_first30=True,
        arms=list(ARMS), code_snapshot_sha256=_hash(args.output/'code_snapshot.zip')))
    results = {}
    for name in ARMS:
        torch.manual_seed(131)
        base, state = load_parent(DEFAULT_PARENT)
        model = DecoderFfnModel(base).to('cuda').eval()
        provider = EndedDensityReplay(state['source_center'], state['source_scale'],
            output_dir=args.output/name, order=[r['shot'] for r in order],
            preserve_observed_channels=True)
        mode = 'signal_only' if name == 'signal_after30' else 'risk_after_update'
        updater = partial(prefix_updater, provider=provider, output=args.output/name,
                          head_steps=args.steps, mode=mode)
        arm = 'static' if name == 'static' else 'intra' if name == 'original_intra' else (
            'inter' if name == 'meta_static' else 'both')
        run_arm(model, order=order, arm=arm, threshold=threshold,
                output_dir=args.output/name, data_api=provider, updater=updater)
        if model.memory.any() or _hash(DEFAULT_PARENT) != PARENT_SHA256:
            raise AssertionError('persistent or source state changed')
        results[name] = {period: side_metrics(args.output/name, order, threshold, a, b)
            for period, a, b in (('all', 0, args.limit), ('first30', 0, 30), ('after30', 30, args.limit))}
        _durable_json(args.output/'comparison.json', results)
        print(json.dumps(dict(completed=name, metrics={period:{k:v for k,v in m.items() if k!='predictions'}
            for period,m in results[name].items()})), flush=True)
    controls = dict(meta_static_vs_static=curve_error(args.output/'static', args.output/'meta_static', order),
        signal_startup=curve_error(args.output/'original_intra', args.output/'signal_after30', order[:30]),
        meta_startup=curve_error(args.output/'original_intra', args.output/'meta_after30', order[:30]))
    if any(controls.values()):
        raise AssertionError('auxiliary-head direct-risk or startup causal control failed')
    training = [json.loads((args.output/name/'head_training.json').read_text())
                for name in ('signal_after30', 'meta_after30', 'meta_static')]
    if any(t['steps'] != args.steps or len(t['records']) != args.steps for t in training):
        raise AssertionError('actual auxiliary budget differs from declared protocol')
    if any(t['fixed_peaks'] != training[0]['fixed_peaks'] for t in training[1:]):
        raise AssertionError('learned controls used different training peaks')
    for i in range(args.steps):
        if any([r['shot'] for r in t['records'][i]['records']] !=
               [r['shot'] for r in training[0]['records'][i]['records']] for t in training[1:]):
            raise AssertionError('learned controls used different sampled shot pairs')
    _durable_json(args.output/'verification.json', dict(controls=controls,
        head_fit_after_index=30, first_deployment_index=31,
        declared_head_steps=args.steps, executed_head_steps=[len(t['records']) for t in training],
        same_training_peaks_and_pair_order=True, source_sha256_unchanged=True,
        persistent_ffn_unchanged=True, all_metric_seal_hashes_verified=True,
        no_physical_law_or_independent_validation_claim=True))


if __name__ == '__main__':
    main()
