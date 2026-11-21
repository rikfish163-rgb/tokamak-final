"""Full EAST paired development replay: change only within-shot loss weights.

Reuse a completed uniform run only after verifying parent/order/threshold,
each postshot training fingerprint, and the final full persistent state.
"""
import argparse
import json
from pathlib import Path

import torch

from . import io23
from .forecast_replay import ForecastRetrainer
from .simple23 import Config, Simple23
from .simple_online import run_arm, _durable_json, _hash
from .source_gate import qualify_source
from .ended_channel import EndedChannelLearner
from .state_loss import StateLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'order', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--weights', type=Path)
    choice.add_argument('--state-rule', type=Path)
    parser.add_argument('--adapt-ended', action='store_true')
    args = parser.parse_args()
    if args.state_rule and args.adapt_ended:
        parser.error('state-rule postshot learning is not implemented')
    if args.output.exists():
        raise FileExistsError(args.output)
    gate = qualify_source(args.checkpoint)
    if not gate['qualified']:
        raise ValueError(gate['reasons'])
    weight_path = args.state_rule or args.weights
    weights = torch.load(weight_path, map_location='cpu', weights_only=False)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    protocol = json.loads((args.reference/'protocol.json').read_text())
    reference = json.loads((args.reference/'summary.json').read_text())
    weight_parent = weights['protocol']['parent_sha256'] if args.state_rule else weights['parent_sha256']
    if weight_parent != _hash(args.checkpoint) or protocol['parent_sha256'] != _hash(args.checkpoint):
        raise ValueError('parent identity mismatch')
    if protocol['order_sha256'] != _hash(args.order) or protocol['threshold'] != source['threshold']:
        raise ValueError('order or threshold differs')
    if protocol['maintenance'] != 'aligned' or protocol['forecast_steps'] != 4 or protocol['steps'] != 16:
        raise ValueError('reference maintenance contract differs')
    order = json.loads(args.order.read_text())
    ids = [r['shot'] for r in order]
    if len(ids) != 360 or len(set(ids)) != 360 or set(ids) != set(io23.cohort_ids('EAST')):
        raise ValueError('complete locked EAST360 required')
    if [r['shot'] for r in reference['both']['rows']] != ids:
        raise ValueError('reference shot identity mismatch')
    channel_logits, loss_rule = None, None
    if args.state_rule:
        loss_rule = StateLoss().cuda().eval().requires_grad_(False)
        loss_rule.load_state_dict(weights['state'])
        if not all(torch.isfinite(v).all() for v in loss_rule.state_dict().values()):
            raise ValueError('finite state rule required')
    else:
        channel_logits = weights['channel_logits']
        if channel_logits.shape != (23,) or not torch.isfinite(channel_logits).all():
            raise ValueError('finite 23 source-trained weights required')
        channel_logits = channel_logits.cuda()
    torch.set_num_threads(4)
    model = Simple23(Config(**source['model_config']), source['source_center'], source['source_scale']).cuda()
    model.load_state_dict(source['model'])
    args.output.mkdir(parents=True)
    _durable_json(args.output/'protocol.json', dict(
        parent_sha256=_hash(args.checkpoint), weights_sha256=_hash(weight_path), order_sha256=_hash(args.order),
        weighting='state-dependent' if args.state_rule else 'global',
        reference_summary_sha256=_hash(args.reference/'summary.json'), reference=str(args.reference.resolve()),
        source_gate=gate, threshold=source['threshold'], scope='DEVELOPMENT_ONLY; all360 including startup',
        change='channel weights used ONLY for temporary signal updates',
        adapt_ended=args.adapt_ended, channel_meta_steps=4 if args.adapt_ended else 0,
        code_sha256={n:_hash(Path(__file__).with_name(n)) for n in
                     ('channel_east.py', 'channel_loss.py', 'state_loss.py', 'ended_channel.py', 'sequence_fast.py', 'sequence_train.py',
                      'simple_online.py', 'forecast_replay.py',
                      'replay_retraining.py', 'fast_meta.py', 'joint_meta.py', 'simple23.py', 'fast_update.py',
                      'simple_data.py', 'simple_train.py', 'io23.py', 'stage_metrics.py')}))
    updater = ForecastRetrainer(model, maintenance='aligned', forecast_steps=4)
    if args.adapt_ended:
        updater = EndedChannelLearner(updater, channel_logits)
    result = run_arm(model, order=order, arm='both', threshold=source['threshold'],
                     output_dir=args.output/'both', device='cuda', startup_budget=30,
                     persistent_updater=updater, channel_logits=channel_logits, loss_rule=loss_rule)
    torch.save(model.state_dict(), args.output/'both_final.pt')
    if channel_logits is not None:
        torch.save(channel_logits.detach().cpu(), args.output/'channel_logits_final.pt')
    else:
        torch.save(loss_rule.state_dict(), args.output/'state_rule_final.pt')
    for old, new in zip(reference['both']['rows'], result['rows']):
        for key in ('sample_sha256', 'forecast_sample_sha256', 'classifier_sha256'):
            if old['persistent'][key] != new['persistent'][key]:
                raise AssertionError(f'persistent update differs at {new["shot"]}: {key}')
    old_state = torch.load(args.reference/'both_final.pt', map_location='cpu', weights_only=True)
    if not all(torch.equal(v.detach().cpu(), old_state[k]) for k,v in model.state_dict().items()):
        raise AssertionError('final persistent state differs')
    flips = [dict(shot=b['shot'], is_disrupt=b['is_disrupt'],
                  before_alarm=a['alarm'], after_alarm=b['alarm'],
                  before_timely=a['timely'], after_timely=b['timely'])
             for a,b in zip(reference['both']['rows'], result['rows'])
             if (a['alarm'],a['timely']) != (b['alarm'],b['timely'])]
    report = dict(persistent_pairing_verified=True, full_state_equal=True, flips=flips,
                  metrics=result['metrics_full'], reference_metrics=reference['both']['metrics_full'])
    _durable_json(args.output/'paired_result.json', report)
    print(json.dumps({k:v for k,v in result['metrics_full'].items() if k != 'predictions'}), flush=True)
    print(json.dumps(dict(flips=flips, persistent_pairing_verified=True)), flush=True)


if __name__ == '__main__':
    main()
