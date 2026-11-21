"""Paired zero-EAST-label static/intra replay, no alarm or post-shot updates.

This explicitly isolates within-shot neural representation changes. The
accepted low-to-high order is a commissioning PROXY, not historical order.
"""
import argparse
import json
from pathlib import Path

import torch

from . import io23, simple_data, simple_train
from .diagnostic_adapter import DiagnosticAdapter
from .diagnostic_online import predict_shot
from .diagnostic_training import load_parent
from .simple_online import build_order, seal_predictions
from .source_gate import qualify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=360)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.limit <= 360:
        raise ValueError('fresh output and 1..360 shots required')
    torch.set_num_threads(4)
    stored = torch.load(args.adapter, map_location='cpu', weights_only=False)
    protocol = stored['protocol']
    parent_path = Path(protocol['parent'])
    gate = qualify_source(parent_path)
    if gate['checkpoint_sha256'] != protocol['parent_sha256'] or not gate['qualified']:
        raise ValueError('matched qualified source parent required')
    parent, _ = load_parent(parent_path)
    adapter = DiagnosticAdapter(parent.cfg.dim, parent.cfg.horizon).cuda()
    adapter.load_state_dict(stored['adapter'])
    adapter.requires_grad_(False)
    ids = io23.cohort_ids('EAST')
    order = build_order('EAST', ids)[:args.limit]
    replay_protocol = dict(protocol)
    replay_protocol.update(adapter_sha256=simple_train.sha(args.adapter),
                           code_replay_sha256=simple_train.sha(__file__), domain='EAST',
                           labels='fixed125ms', timely_lead='strict >30ms',
                           arms=['static', 'intra'], outcome_access='after prediction seal only',
                           post_shot_updates=False, alarm_parameter_updates=False,
                           evidence='DEVELOPMENT_ONLY', order=order)
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(replay_protocol, indent=2))
    rows, summary = {}, {}
    for arm in ('static', 'intra'):
        output = args.output/arm
        output.mkdir()
        rows[arm] = []
        for index, ordered in enumerate(order, 1):
            shot = simple_data.read_shot('EAST', ordered['shot'])
            scores, events = predict_shot(parent, adapter, shot, intra=arm == 'intra')
            receipt = seal_predictions(output, index, shot['shot'], shot['times'], scores,
                                       arm=arm, threshold=protocol['threshold'])
            # Only now may the evaluator read IsDisrupt/DownTime.
            labeled = simple_data.attach_outcome(shot, 'EAST', {})
            valid = labeled['valid']
            rows[arm].append(dict(shot=shot['shot'], is_disrupt=labeled['is_disrupt'],
                                  down=labeled['down'], times=shot['times'][valid], scores=scores[valid]))
            (output/f'{index:04d}_{shot["shot"]}_audit.json').write_text(json.dumps(
                dict(seal=receipt, events=events, no_persistent_update=True), indent=2, allow_nan=False))
            if index % 30 == 0:
                result = simple_train.metrics(rows[arm], protocol['threshold'], lead=30.)
                print(json.dumps(dict(arm=arm, through=index,
                                      **{k: v for k, v in result.items() if k != 'predictions'})), flush=True)
        summary[arm] = simple_train.metrics(rows[arm], protocol['threshold'], lead=30.)
        (output/'summary.json').write_text(json.dumps(summary[arm], indent=2))
    summary['paired_delta_J'] = summary['intra']['J']-summary['static']['J']
    before = {row['shot']: row for row in summary['static']['predictions']}
    summary['changed_shots'] = [dict(before=before[row['shot']], after=row)
                                for row in summary['intra']['predictions']
                                if (row['alarm'], row['timely']) != (before[row['shot']]['alarm'], before[row['shot']]['timely'])]
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(dict(paired_delta_J=summary['paired_delta_J'], changed_shots=len(summary['changed_shots']))), flush=True)


if __name__ == '__main__':
    main()
