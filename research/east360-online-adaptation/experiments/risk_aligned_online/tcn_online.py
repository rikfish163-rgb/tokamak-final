"""Qualified TCN source -> zero-label EAST replay with unchanged causal gates.

This is an architecture baseline, not a claimed successful intra-shot method.
The forecast-only maintenance deliberately avoids borrowing the old learned
Transformer gradient rule. All arms retain the same source threshold.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import torch

from . import io23
from .forecast_replay import ForecastRetrainer
from .simple_online import run_arm, _durable_json, _hash
from .source_gate import qualify_source
from .tcn23 import load_checkpoint


def validate_order(order, expected):
    ids = [row['shot'] for row in order]
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('complete unique locked cohort required before prefix selection')
    if any(not math.isfinite(row['proxy']) or row['proxy'] < 0 for row in order):
        raise ValueError('finite nonnegative ordering proxy required')
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--order', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--arms', nargs='+', choices=('static', 'intra', 'inter', 'both'),
                        default=['static', 'intra', 'inter', 'both'])
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.limit <= 360 or len(set(args.arms)) != len(args.arms):
        raise ValueError('new output, unique arms and prefix limit in [0,360] required')
    gate = qualify_source(args.checkpoint)
    if not gate['qualified']:
        raise ValueError(f'TCN source failed acceptance: {gate["reasons"]}')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required')
    order = validate_order(json.loads(args.order.read_text()), io23.cohort_ids('EAST'))
    torch.set_num_threads(4)
    model = load_checkpoint(args.checkpoint, 'cuda')
    if not model.forecast_trained:
        raise ValueError('forecast-trained source required for four-arm comparison')
    args.output.mkdir(parents=True)
    _durable_json(args.output/'order.json', order)
    _durable_json(args.output/'protocol.json', dict(
        architecture='tcn23-v1', parent_sha256=_hash(args.checkpoint), source_gate=gate,
        order_sha256=_hash(args.order), threshold=model.threshold,
        requested_arms=args.arms, requested_limit=args.limit,
        supervised_steps=16, forecast_steps=4, maintenance='forecast',
        source_labels='max(25ms,AdvancedTime)', east_labels='fixed125ms',
        evaluation='strict lead>30ms; first alarm; every startup shot counted',
        evidence='DEVELOPMENT_ONLY; screening is not architecture superiority',
        code_sha256={n:_hash(Path(__file__).with_name(n)) for n in (
            'tcn_online.py', 'tcn23.py', 'simple_online.py', 'simple_data.py', 'io23.py',
            'forecast_replay.py', 'replay_retraining.py', 'simple23.py', 'simple_train.py',
            'fast_update.py', 'stage_metrics.py', 'source_gate.py')}))
    results = {}
    final_states = {}
    for arm in args.arms:
        net = deepcopy(model)
        updater = (ForecastRetrainer(net, supervised_steps=16, forecast_steps=4,
                                     maintenance='forecast')
                   if arm in ('inter', 'both') else None)
        print(f'starting {arm}', flush=True)
        results[arm] = run_arm(net, order=order, arm=arm, threshold=model.threshold,
                              output_dir=args.output/arm, device='cuda',
                              limit=args.limit, startup_budget=30, persistent_updater=updater)
        state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        torch.save(state, args.output/f'{arm}_final.pt')
        final_states[arm] = state
        metric = results[arm].get('metrics_full', results[arm].get('metrics_prefix'))
        print(arm, json.dumps({k:v for k,v in metric.items() if k != 'predictions'}), flush=True)
        _durable_json(args.output/'summary.json', results)
    if {'inter', 'both'}.issubset(results):
        for a, b in zip(results['inter']['rows'], results['both']['rows']):
            for field in ('sample_sha256', 'forecast_sample_sha256', 'classifier_sha256'):
                if a['persistent'][field] != b['persistent'][field]:
                    raise AssertionError(f'paired persistent training differs: {field}')
        equal = all(torch.equal(v, final_states['both'][k]) for k,v in final_states['inter'].items())
        if not equal:
            raise AssertionError('paired final state differs')
        _durable_json(args.output/'pairing.json', dict(persistent_pairing_verified=True,
                      full_state_equal=True, observed_count=len(results['inter']['rows'])))


if __name__ == '__main__':
    main()
