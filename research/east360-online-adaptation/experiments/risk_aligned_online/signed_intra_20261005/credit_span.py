"""Two outcome-selected positive controls: single vs repeated internal action.

This is a retrospective mechanism probe, NOT an online model or a performance
estimate. No rule fitting, source training, action/step search or new data block.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import simple_data
from ..simple_online import _hash, _durable_json, seal_predictions
from ..redesign_20261004.prepare_prior import (
    DEFAULT_PARENT, PARENT_SHA256, load_parent, classifier_state, assert_classifier_unchanged,
)
from ..ffn_intra_20261005.model import FfnModel
from .finite_probe import whole_cost, reference_order, fixed_sign_curve
from .policy import plan, score_plan


SHOTS = (54244, 54272)


def single_event_plan(signals, center, scale, sign):
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError('one signed action required')
    return plan(signals, torch.zeros((3, 117), dtype=torch.float64),
                center.double().cpu(), scale.double().cpu(), intervention=(0, sign))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--static', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('same local CUDA path required')
    torch.set_num_threads(4); torch.manual_seed(131); started = time.monotonic()
    previous = json.loads((args.history/'summary.json').read_text())
    if (_hash(args.direction) != previous['protocol']['direction_sha256'] or
            previous['protocol']['parent_sha256'] != PARENT_SHA256):
        raise ValueError('same completed finite probe direction required')
    order = reference_order(args.static)
    base, parent = load_parent(DEFAULT_PARENT); base = base.cuda(); model = FfnModel(base)
    original = classifier_state(base); threshold = float(parent['threshold'])
    direction = torch.load(args.direction, map_location='cpu', weights_only=True)['direction'].to(model.center)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parents[1]
    files = [Path(__file__), Path(__file__).with_name('policy.py'), Path(__file__).with_name('context.py'),
             Path(__file__).with_name('finite_probe.py'), root/'ffn_intra_20261005/model.py',
             root/'simple23.py', root/'simple_data.py', root/'io23.py',
             root/'simple_train.py', root/'simple_online.py', root/'redesign_20261004/intra.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in files: z.write(path, str(path.relative_to(root)))
    protocol = dict(shots=SHOTS, selection='outcome-selected known positive controls; NOT representative',
        single='first legal signed action, then hold current q forever', repeated='reuse sealed full fixed-sign trajectories',
        parent_sha256=PARENT_SHA256, direction_sha256=_hash(args.direction),
        history_summary_sha256=_hash(args.history/'summary.json'), threshold=threshold,
        device=torch.cuda.get_device_name(), parity_tolerance=3e-6, training=False,
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json', protocol)
    results = []
    for sid in SHOTS:
        index = next(i for i, item in enumerate(order, 1) if item['shot'] == sid)
        if sid not in previous['protocol']['shot_ids']: raise ValueError('shot outside original history')
        item = order[index-1]; stem = f'{index:04d}_{sid}'
        if _hash(Path(item['provenance']['path'])) != item['provenance']['sha256']:
            raise ValueError('raw input provenance changed')
        signals = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        archived = {}
        for sign in (0, -1, 1):
            folder = args.history/str(sign)
            receipt = json.loads((folder/f'{stem}_seal.json').read_text())
            if receipt['threshold'] != threshold or _hash(folder/receipt['npz']) != receipt['sha256']:
                raise ValueError('historical finite trace differs')
            with np.load(folder/receipt['npz'], allow_pickle=False) as old:
                if int(old['shot']) != sid or not np.array_equal(old['times'], signals['times']):
                    raise ValueError('historical trace identity/time mismatch')
                archived[sign] = old['scores'].copy()
        zero = score_plan(model, signals, direction, np.zeros(len(signals['times'])))
        parity = float(np.max(np.abs(zero[1]-archived[0])))
        if parity > 3e-6 or np.any((zero[1] >= threshold) != (archived[0] >= threshold)):
            raise RuntimeError('zero-state risk parity failed')
        traces = {}; states = {}
        for sign in (-1, 1):
            q, events = single_event_plan(signals, model.center, model.scale, sign)
            effect = events[0]['effect']
            if effect is None: raise ValueError('first action has no later prediction')
            logits, scores = score_plan(model, signals, direction, q, past=zero, start=effect)
            if not np.array_equal(scores[:effect], zero[1][:effect]):
                raise AssertionError('already emitted prefix changed')
            folder = args.output/str(sign); folder.mkdir(exist_ok=True)
            seal_predictions(folder, index, sid, signals['times'], scores,
                             arm=f'first_event_only_{sign}', threshold=threshold)
            np.savez_compressed(folder/f'{stem}_states.npz', q=q, logits=logits)
            repeated_q = fixed_sign_curve(signals['available'] & np.isfinite(signals['values']), sign)
            traces[sign] = scores
            states[str(sign)] = dict(first_effect=effect, action=sign, final_q=float(q[-1]),
                                     repeated_final_q=float(repeated_q[-1]))
        # This deliberately selected historical diagnosis does not claim fresh label blindness.
        shot = simple_data.attach_outcome(signals, 'EAST', {})
        row = dict(shot=sid, disruptive=bool(shot['is_disrupt']), source_parity=parity, states=states, results={})
        for name, scores in [('hold', zero[1]), *[(f'single_{a}', traces[a]) for a in (-1, 1)],
                             *[(f'repeated_{a}', archived[a]) for a in (-1, 1)]]:
            _, cost, prediction = whole_cost(shot, scores, threshold)
            row['results'][name] = dict(error=cost[0], risk=cost[1], **prediction)
        results.append(row)
    assert_classifier_unchanged(base, original)
    if _hash(DEFAULT_PARENT) != PARENT_SHA256: raise RuntimeError('source file changed')
    report = dict(protocol=protocol, results=results, elapsed_seconds=time.monotonic()-started,
                  classifier_unchanged=True, new_online_candidate=False)
    _durable_json(args.output/'summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
