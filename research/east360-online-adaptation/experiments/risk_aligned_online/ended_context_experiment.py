"""Fixed-prefix pilot: uniform intra vs an ended-EAST learned frozen rule.

Same qualified source, density input intervention, FFN site, H25/stride100,
step.05/radius1 and original threshold. No source-gradient transport in ANY arm.
The density mapping is an existing failed input control, not approved physics
calibration. All startup shots count; later results are separated descriptively.
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

from . import simple_data
from .ended_context_rule import ContextFfnModel, causal_peak_logit, peak_index, train_pair
from .ended_density_replay import EndedDensityReplay
from .redesign_20261004.intra import predict_shot
from .redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash
from .simple_train import metrics

ROOT = Path(__file__).parent
REFERENCE = ROOT/'runs/gradient_transport_commission240_20261009'


def fit_ended_prefix(model, history, *, provider, output, steps, threshold):
    if (provider.pending is not None or len(provider.ended_ids) != 30
            or [s['shot'] for s in history] != provider.ended_ids
            or any('_sealed_scores' not in s for s in history)):
        raise ValueError('exact first30 sealed and naturally revealed shots required')
    start = time.monotonic()
    snapshots = {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items()}
    episodes = []
    # Re-encode the already ended history in the now-active input coordinates.
    # Rescored curves are TRAINING ONLY; they do not replace the first30 seals.
    for ended in history:
        shot = provider.reencode(ended)
        scores, audit = predict_shot(model, shot, intra=True, device='cuda')
        query = peak_index(shot, scores)
        logit, info = causal_peak_logit(model, shot, query)
        if abs(float(logit.detach().sigmoid())-float(scores[query])) > 2e-6:
            raise AssertionError('training prefix state differs from causal stream state')
        episodes.append((shot, query))
        print(json.dumps(dict(training_peak=shot['shot'], query=query,
            full_updates=audit['accepted_updates'], replay=info)), flush=True)
    pools = [[e for e in episodes if int(e[0]['is_disrupt']) == label] for label in (0, 1)]
    if not all(pools):
        raise ValueError('ended prefix requires both classes; not silently skipping rule training')
    boundary = math.log(threshold/(1-threshold))
    optimizer = torch.optim.Adam(model.rule.parameters(), lr=.003)
    rng = np.random.default_rng(131)
    records = []
    for iteration in range(steps):
        pair = [pool[int(rng.integers(len(pool)))] for pool in pools]
        row = train_pair(model, pair, optimizer, boundary=boundary)
        records.append(dict(iteration=iteration+1, **row))
        if (iteration+1) % 5 == 0:
            print(json.dumps(dict(rule_step=iteration+1, objective=row['objective'],
                gradient_norm=row['gradient_norm'], elapsed_seconds=time.monotonic()-start)), flush=True)
    model.rule.requires_grad_(False)
    for k, v in model.base.state_dict().items():
        torch.testing.assert_close(v.detach().cpu(), snapshots[k], rtol=0, atol=0)
    with (output/'ended_rule.pt').open('xb') as handle:
        torch.save(dict(rule=model.rule.state_dict(), parent_sha256=PARENT_SHA256,
            fit_ids=provider.ended_ids, first_deployed_shot_index=31,
            history_label_access='postseal natural reveal', source_changed=False,
            temporary_retained=False), handle)
    _durable_json(output/'rule_training.json', dict(
        fit_ids=provider.ended_ids, n_disrupt=len(pools[1]), n_safe=len(pools[0]),
        fixed_peaks=[dict(shot=s['shot'], label=int(s['is_disrupt']), query=q) for s, q in episodes],
        outer='adapted-risk softplus about frozen boundary, at fixed training causal peak',
        peak_selection='first30 only; D all lead>30, S all valid; not125ms point labels',
        changing_peak_limitation=True, second_order_unroll_depth=4, actual_burn_in=True,
        learning_rate=.003, steps=steps, seed=131, records=records,
        elapsed_seconds=time.monotonic()-start, artifact_sha256=_hash(output/'ended_rule.pt')))
    return dict(accepted=True, reason='ended_prefix_context_rule_fitted',
                rule_steps=steps, original_parent_changed=False, persistent_ffn_changed=False)


def prefix_updater(model, shot, replay, *, provider, output, rule_steps, threshold, **kwargs):
    if len(provider.ended_ids) == 30:
        return fit_ended_prefix(model, [*replay, shot], provider=provider, output=output,
                                steps=rule_steps, threshold=threshold)
    return dict(accepted=False, reason='rule_fixed_outside_first30_training_boundary')


def side_metrics(folder, order, threshold, start, stop):
    rows = []
    for index, item in enumerate(order[start:stop], start+1):
        sid = item['shot']
        audit = json.loads((folder/f'{index:04d}_{sid}_audit.json').read_text())
        receipt = json.loads((folder/f'{index:04d}_{sid}_seal.json').read_text())
        with np.load(folder/receipt['npz'], allow_pickle=False) as trace:
            # Evaluation only AFTER replay. Re-read natural outcome; no label
            # can feed backwards into training or mutate an emitted score.
            raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
            ended = simple_data.attach_outcome(raw, 'EAST', {})
            valid = ended['valid']
            if ended['is_disrupt']:
                valid = valid & (trace['times'] <= ended['down'])
            rows.append(dict(shot=sid, times=trace['times'][valid], scores=trace['scores'][valid],
                             is_disrupt=ended['is_disrupt'], down=ended['down']))
        if audit['seal_sha256'] != receipt['sha256'] or _hash(folder/receipt['npz']) != receipt['sha256']:
            raise AssertionError('durable seal identity changed')
    result = metrics(rows, threshold, lead=30.)
    result['early_over_3000'] = sum((p['lead_ms'] or 0) > 3000.
                                   for p in result['predictions'] if p['timely'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=120)
    parser.add_argument('--steps', type=int, default=40)
    args = parser.parse_args()
    order = json.loads((REFERENCE/'order.json').read_text())
    if args.output.exists() or not 30 < args.limit <= len(order) or args.steps < 1:
        parser.error('new output, limit31..240, positive rule steps required')
    order = order[:args.limit]
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(131)
    base, state = load_parent(DEFAULT_PARENT)
    threshold = float(state['threshold'])
    _durable_json(args.output/'order.json', order)
    files = [Path(__file__), ROOT/'ended_context_rule.py', ROOT/'ended_density_replay.py',
        ROOT/'persistent_ffn.py', ROOT/'fast_update.py', ROOT/'simple23.py', ROOT/'simple_data.py',
        ROOT/'channel_gradient_audit.py', ROOT/'simple_online.py', ROOT/'causal_input.py',
        ROOT/'ffn_intra_20261005/model.py', ROOT/'ffn_intra_20261005/loss.py',
        ROOT/'redesign_20261004/intra.py', ROOT/'redesign_20261004/run.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for file in files:
            z.write(file, str(file.relative_to(ROOT)))
    _durable_json(args.output/'protocol.json', dict(
        scope='DEVELOPMENT CAUSAL PREFIX PILOT, NOT independent future confirmation',
        parent_sha256=PARENT_SHA256, threshold=threshold,
        order_sha256=_hash(REFERENCE/'order.json'), selected_shots=args.limit,
        reference=str(REFERENCE.resolve()), warmup=30, rule_steps=args.steps, rule_lr=.003,
        inputs='same existing ended-first30 density mapping; source asinh buffers unchanged',
        input_mapping_approved_physical_calibration=False, source_model_changed=False,
        transport=False, alarm_tuning=False, temporary_ffn_discarded=True,
        fast_parameters=16576, stride=100, horizon=25, step=.05, radius=1.,
        acceptance='same fixed-context weighted signal loss decreases',
        pure_intra_arm='uniform', inter_trained_intra_arm='learned_after30',
        no_persistent_ffn_training=True, code_snapshot_sha256=_hash(args.output/'code_snapshot.zip')))
    results = {}
    for name in ('static', 'uniform', 'learned_after30'):
        torch.manual_seed(131)
        base, state = load_parent(DEFAULT_PARENT)
        model = ContextFfnModel(base).to('cuda').eval()
        provider = EndedDensityReplay(state['source_center'], state['source_scale'],
            output_dir=args.output/name, order=[r['shot'] for r in order],
            preserve_observed_channels=True)
        updater = partial(prefix_updater, provider=provider, output=args.output/name, rule_steps=args.steps)
        run_arm(model, order=order, arm='static' if name=='static' else 'both' if name=='learned_after30' else 'intra',
            threshold=threshold, output_dir=args.output/name, data_api=provider, updater=updater)
        if model.memory.any() or _hash(DEFAULT_PARENT) != PARENT_SHA256:
            raise AssertionError('persistent or source state changed')
        results[name] = {period: side_metrics(args.output/name, order, threshold, a, b)
            for period, a, b in (('all', 0, args.limit), ('first30', 0, 30), ('after30', 30, args.limit))}
        _durable_json(args.output/'comparison.json', results)
        print(json.dumps(dict(completed=name, metrics={period:{k:v for k,v in m.items() if k!='predictions'}
            for period,m in results[name].items()})), flush=True)
    # Fit labels must not change the predictions already sealed in startup.
    errors = []
    for index, item in enumerate(order[:30], 1):
        with np.load(args.output/'uniform'/f'{index:04d}_{item["shot"]}_sealed.npz') as a, \
             np.load(args.output/'learned_after30'/f'{index:04d}_{item["shot"]}_sealed.npz') as b:
            errors.append(float(np.max(np.abs(a['scores']-b['scores']))))
    if max(errors) != 0.:
        raise AssertionError('future rule learning changed a startup trace')
    training = json.loads((args.output/'learned_after30/rule_training.json').read_text())
    if training['steps'] != args.steps or len(training['records']) != args.steps:
        raise AssertionError('executed rule budget does not match declared protocol')
    _durable_json(args.output/'verification.json', dict(first30_max_difference=max(errors),
        source_sha256_unchanged=_hash(DEFAULT_PARENT)==PARENT_SHA256,
        rule_fit_after_index=30, rule_first_deployment_index=31,
        declared_rule_steps=args.steps, executed_rule_steps=len(training['records']),
        all_metric_seals_hash_verified=True, persistent_ffn_unchanged=True,
        no_claim_of_physical_law_learning=True))


if __name__ == '__main__':
    main()
