"""Source-only post-hoc class/clock audit at FIXED step60; no new selection."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..simple_online import _hash, _durable_json
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..ffn_intra_20261005.model import FfnModel
from ..decoder_intra_20261005.whole_meta import select_peak
from .source import risk_from_probability
from .stream import predict_shot


def compare_rows(old, new):
    a = {r['shot']: r for r in old}; b = {r['shot']: r for r in new}
    if len(a) != len(old) or len(b) != len(new) or a.keys() != b.keys() or any(a[k]['label'] != b[k]['label'] for k in a):
        raise ValueError('unique matching shots and labels required')
    rows = [dict(shot=k, label=a[k]['label'], loss_change=b[k]['loss']-a[k]['loss'],
        first_alarm_changed=b[k]['first_alarm'] != a[k]['first_alarm']) for k in a]
    classes = {}
    for label in sorted({r['label'] for r in rows}):
        changes = [r['loss_change'] for r in rows if r['label'] == label]
        classes[str(label)] = dict(n=len(changes), mean_loss_change=float(np.mean(changes)),
            improved=sum(v < 0 for v in changes), worsened=sum(v > 0 for v in changes), unchanged=sum(v == 0 for v in changes))
    return dict(classes=classes, changed_first_alarms=sum(r['first_alarm_changed'] for r in rows),
                records=sorted(rows, key=lambda r: r['loss_change']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args(); output = args.run/'class_clock_audit.json'
    if output.exists(): raise FileExistsError(output)
    summary = json.loads((args.run/'summary.json').read_text()); protocol = summary['protocol']
    if protocol['parent_sha256'] != _hash(DEFAULT_PARENT) or not protocol['source_only']:
        raise ValueError('fixed source lineage required')
    train = set(simple_data.source_roles()['train']); episodes = protocol['validation_episodes']
    if not {e['shot'] for e in episodes} <= train: raise ValueError('canonical source TRAIN only')
    advanced = simple_data.advanced_times()
    shots = [simple_data.attach_outcome(simple_data.read_shot('J-TEXT', e['shot']), 'J-TEXT', advanced) for e in episodes]
    torch.set_num_threads(4)
    base, state = load_parent(DEFAULT_PARENT); base = base.to(args.device); model = FfnModel(base)
    original = {k: v.detach().clone() for k, v in base.forecast_head.state_dict().items()}
    candidate_path = args.run/'candidate_060.pt'; candidate_hash = _hash(candidate_path)
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if candidate['iteration'] != 60 or candidate['protocol'] != protocol: raise ValueError('fixed step60 identity required')
    threshold = state['threshold']; boundary = math.log(threshold/(1-threshold)); results = {}
    for name, head, mode in [('static', original, 'static'), ('original_partial', original, 'partial'),
                              ('original_delayed', original, 'delayed'),
                              ('risk60_partial', candidate['forecast_head'], 'partial'),
                              ('risk60_delayed', candidate['forecast_head'], 'delayed')]:
        base.forecast_head.load_state_dict(head); rows = []
        for shot in shots:
            scores, audit = predict_shot(model, shot, mode=mode, offline_source=True)
            peak = select_peak(shot, scores)
            hits = np.flatnonzero(shot['valid'] & (scores >= threshold))
            first = int(hits[0]) if len(hits) else None
            rows.append(dict(shot=shot['shot'], label=int(shot['is_disrupt']), peak=peak,
                loss=float(risk_from_probability(torch.tensor(scores[peak]), shot['is_disrupt'], boundary)),
                first_alarm=first, updates=audit['accepted_updates'], final_delta_sha256=audit['final_delta_sha256'],
                complete_final_cycle=(len(shot['times'])-1)%26 == 25))
        results[name] = dict(records=rows, classes={str(label): dict(n=sum(r['label'] == label for r in rows),
            mean_loss=float(np.mean([r['loss'] for r in rows if r['label'] == label]))) for label in (0, 1)})
        print(json.dumps(dict(stage='partial_class_audit', name=name, classes=results[name]['classes'])), flush=True)
    comparisons = {}
    for old, new in [('original_partial', 'risk60_partial'), ('risk60_delayed', 'risk60_partial'), ('static', 'risk60_partial')]:
        comparisons[f'{new}_minus_{old}'] = compare_rows(results[old]['records'], results[new]['records'])
    if _hash(DEFAULT_PARENT) != protocol['parent_sha256'] or _hash(candidate_path) != candidate_hash:
        raise RuntimeError('fixed source/candidate changed')
    _durable_json(output, dict(source_only=True, posthoc_no_new_selection=True, results=results, comparisons=comparisons,
        parent_sha256=protocol['parent_sha256'], candidate_sha256=candidate_hash,
        summary_sha256=_hash(args.run/'summary.json'), interpretation='fixed32 development source; not EAST evidence'))


if __name__ == '__main__':
    main()
