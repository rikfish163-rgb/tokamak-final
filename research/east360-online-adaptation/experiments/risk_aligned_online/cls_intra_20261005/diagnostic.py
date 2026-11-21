"""Post-fit audit of actual next-input benefit; never changes online decisions."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..simple_online import _hash, _durable_json
from ..redesign_20261004.prepare_prior import load_parent, load_source_train, DEFAULT_PARENT
from ..redesign_20261004.intra import _batch_windows
from ..partial_intra_20261005.source import risk_from_probability
from ..partial_intra_20261005.stream import snapshot
from .feedback import segment_step, feedback_loss
from .model import ClsModel
from .stream import predict_shot


def diagnostic_group(shot, query):
    if not shot['valid'][query]: return None
    if not shot['is_disrupt']: return 'S'
    lead = round(float((shot['down'] - shot['times'][query]) * 1000.), 9)
    if lead <= 30.: return None
    return 'D_lead_gt125' if lead > 125. else 'D_lead_30_125'


def audit_shot(model, shot, *, use_task, threshold):
    if shot['device'] != 'J-TEXT': raise ValueError('this audit is offline source only')
    signals = {k: shot[k] for k in ('values', 'available')}
    device = model.center.device; delta = model.initial_delta(); rows = []
    boundary = math.log(threshold / (1 - threshold))
    for origin in range(0, len(shot['times']), 26):
        x, a = _batch_windows(simple_data, signals, [origin], model, device)
        early = None
        for segment, offset in [('early', 5), ('late', 25)]:
            end = origin + offset; query = end + 1
            if end >= len(shot['times']): continue
            if segment == 'early':
                early = snapshot(signals, origin, segment, device, model.center.dtype)
                truth, observed = early
            else:
                late = snapshot(signals, origin, segment, device, model.center.dtype)
                truth = torch.cat((early[0], late[0][:, 1:]), dim=1)
                observed = torch.cat((early[1], late[1][:, 1:]), dim=1)
            after, update = segment_step(model, delta, x, a, truth, observed, segment, use_task=use_task)
            after = after.detach().requires_grad_()
            if query < len(shot['times']):
                # Both states see the SAME actual next endpoint; neither is
                # offered to an online acceptance/rejection rule.
                qx, qa = _batch_windows(simple_data, signals, [query], model, device)
                pb = model(qx, qa, delta)['logit'].detach().sigmoid().reshape(())
                pa = model(qx, qa, after)['logit'].detach().sigmoid().reshape(())
                _, before_loss = feedback_loss(model, model(x, a, delta)['forecast'], x, a,
                                               truth, observed, segment, use_task=True)
                _, after_loss = feedback_loss(model, model(x, a, after)['forecast'], x, a,
                                              truth, observed, segment, use_task=True)
                group = diagnostic_group(shot, query)
                utility = None
                if group is not None:
                    utility = float(risk_from_probability(pb, shot['is_disrupt'], boundary)
                                    - risk_from_probability(pa, shot['is_disrupt'], boundary))
                mask_change = bool((observed[:, 1:] != observed[:, :-1]).any()
                                   or (observed[:, 0] != a[:, -1]).any())
                rows.append(dict(shot=shot['shot'], origin_index=origin, target_end=end,
                    query_index=query, segment=segment, group=group, utility=utility,
                    state_changed=update['state_changed'], mask_changed=mask_change,
                    probability_before=float(pb), probability_after=float(pa),
                    saturated_before=bool(pb <= 1e-7 or pb >= 1-1e-7),
                    saturated_after=bool(pa <= 1e-7 or pa >= 1-1e-7),
                    task_gain=before_loss['task_loss'] - after_loss['task_loss'],
                    signal_gain=before_loss['signal_loss'] - after_loss['signal_loss']))
            delta = after
    return rows


def aggregate(rows):
    groups = {}
    for group in ('S', 'D_lead_gt125', 'D_lead_30_125'):
        selected = [r for r in rows if r['group'] == group]
        shots = sorted({r['shot'] for r in selected})
        averages = [float(np.mean([r['utility'] for r in selected if r['shot'] == sid])) for sid in shots]
        groups[group] = dict(shots=len(shots), events=len(selected),
            shot_equal_mean_utility=float(np.mean(averages)) if averages else None,
            proxy_better_task_worse=sum(r['task_gain'] > 1e-7 and r['utility'] < -1e-7 for r in selected),
            risk_saturated=sum(r['saturated_before'] or r['saturated_after'] for r in selected))
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); args = p.parse_args()
    summary = json.loads((args.run/'summary.json').read_text())
    if not summary['source_all_state_bit_exact']: raise ValueError('completed source screen required')
    output = args.run/'next_input_audit.json'
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(4)
    shots, _ = load_source_train(); lookup = {s['shot']: s for s in shots}
    protocol = summary['protocol']; ids = [e['shot'] for e in protocol['validation_episodes']]
    base, _ = load_parent(DEFAULT_PARENT); model = ClsModel(base.to('cuda')); model.requires_grad_(False)
    all_rows = {}; reports = {}
    for arm, use_task in [('C', False), ('A', True)]:
        checkpoint = torch.load(args.run/f'{arm}_final60.pt', map_location='cpu', weights_only=True)
        if checkpoint['protocol'] != protocol or checkpoint['arm'] != arm: raise ValueError('checkpoint mismatch')
        model.decoder.load_state_dict(checkpoint['decoder']); all_rows[arm] = []
        for sid in ids:
            shot = lookup[sid]
            rows = audit_shot(model, shot, use_task=use_task, threshold=protocol['threshold'])
            scores, _ = predict_shot(model, shot, use_task=use_task, offline_source=True)
            if any(abs(r['probability_after'] - float(scores[r['query_index']])) > 2e-6 for r in rows):
                raise RuntimeError('audited next-input state differs from actual stream')
            all_rows[arm] += rows
        reports[arm] = aggregate(all_rows[arm])
        print(json.dumps(dict(arm=arm, groups=reports[arm])), flush=True)
    _durable_json(output, dict(summary_sha256=_hash(args.run/'summary.json'),
        diagnostic_code_sha256=_hash(Path(__file__)), groups=reports, records=all_rows,
        decision_influence=False, source_only=True, actual_stream_match_tolerance=2e-6,
        meaning='positive utility lowers known-label risk on the same next real input; not an online oracle'))


if __name__ == '__main__': main()
