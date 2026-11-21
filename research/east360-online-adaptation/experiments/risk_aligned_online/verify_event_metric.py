"""Independently recount and replay all four SOURCE VAL160 metric arms."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import simple_data, simple_train
from .ended_decoder import DecoderFfnModel
from .event_dynamics import EventDynamicsFfnModel
from .event_dynamics_experiment import load_coordinates
from .event_metric import EventMetricModel
from .redesign_20261004.intra import predict_source_diagnostic
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('preserve old verification')
    torch.set_num_threads(4)
    protocol = json.loads((args.root / 'protocol.json').read_text())
    completed = json.loads((args.root / 'completed.json').read_text())
    saved_metrics = json.loads((args.root / 'validation.json').read_text())
    roles = simple_data.source_roles()
    assert protocol['parent_sha256'] == _hash(DEFAULT_PARENT) == PARENT_SHA256
    assert protocol['val_ids'] == roles['val'] and protocol['train_ids'] == roles['train']
    assert not protocol['east_access'] and not protocol['test_access'] and not protocol['source_classifier_changed']
    assert protocol['code_snapshot_sha256'] == _hash(args.root / 'code_snapshot.zip')
    base, parent = load_parent(DEFAULT_PARENT)
    shots = simple_train.load_shots(roles['val'], simple_data.advanced_times())
    results, maxima, changed, attempts, accepted, hashes = {}, {}, {}, {}, {}, {}
    for arm in ('static', 'identity', 'signal', 'risk'):
        if arm == 'static':
            model = DecoderFfnModel(base)
        elif arm == 'identity':
            # Reproduce using the PRIOR event loss, not new metric code.
            model = EventDynamicsFfnModel(base, *load_coordinates('event'), mode='event')
        else:
            model = EventMetricModel(base, *load_coordinates('event'))
            state = torch.load(args.root / f'{arm}.pt', map_location='cpu', weights_only=True)
            assert state['parent_sha256'] == PARENT_SHA256 and state['arm'] == arm
            metric = state['metric']
            assert metric.shape == (5, 5) and torch.isfinite(metric).all()
            torch.testing.assert_close(metric.square().sum(), torch.tensor(5.), rtol=1e-6, atol=1e-6)
            changed[arm] = float((metric-torch.eye(5)).norm())
            with torch.no_grad(): model.metric.copy_(metric)
        model = model.cuda().eval().requires_grad_(False)
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rows = []; delta = 0.; attempts[arm] = accepted[arm] = 0; hashes[arm] = []
        for i, shot in enumerate(shots):
            path = args.root / arm / f'{shot["shot"]}.npz'
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert _hash(path) == receipt['curve_sha256']
            with np.load(path, allow_pickle=False) as f:
                scores = f['scores'].copy()
                assert int(f['shot']) == shot['shot'] and bool(f['is_disrupt']) == shot['is_disrupt']
                assert np.array_equal(f['times'], shot['times']) and np.array_equal(f['valid'], shot['valid'])
            a = receipt['audit']
            assert a['scope'] == 'OFFLINE_SOURCE_DIAGNOSTIC_NOT_JTEXT_ONLINE'
            assert a['fast_discarded'] and a['model_parameter_change_norm'] == a['model_buffer_change_norm'] == 0
            assert a['attempted_updates'] == len(a['attempts'])
            assert a['accepted_updates'] == sum(r['accepted'] for r in a['attempts'])
            if a['attempts']: assert a['attempts'][0]['fast_before']['l2'] == 0
            for r in a['attempts']:
                assert r['maturity_index'] == r['origin_index'] + 25 == r['predictions_sealed_through_index']
                if r['accepted']:
                    assert r['effective_index'] == r['maturity_index'] + 1 and r['loss_after'] < r['loss_before']
                if 'matured_objective' in r:
                    m = r['matured_objective']
                    assert m['target_detached'] and not m['outcome_access'] and not m['predicted_signals_used_for_risk']
            replay, ra = predict_source_diagnostic(model, shot, intra=arm != 'static', device='cuda')
            error = float(np.max(np.abs(scores-replay))); delta = max(delta, error)
            assert error == 0 and ra['accepted_updates'] == a['accepted_updates']
            rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'], down=shot['down'],
                times=shot['times'][shot['valid']], scores=scores[shot['valid']]))
            hashes[arm].append(dict(shot=shot['shot'], sha256=_hash(path)))
            attempts[arm] += a['attempted_updates']; accepted[arm] += a['accepted_updates']
            if (i+1) % 80 == 0: print(json.dumps(dict(arm=arm, replayed=i+1, max_error=delta)), flush=True)
        # Explicit independent timely/false-alarm counts, not only metrics().
        tp = fp = 0
        for r in rows:
            if r['is_disrupt']:
                eligible = np.round((r['down']-r['times'])*1000., 9) > 10.
                tp += bool(np.any(r['scores'][eligible] >= parent['threshold']))
            else: fp += bool(np.any(r['scores'] >= parent['threshold']))
        counted = simple_train.metrics(rows, parent['threshold'], lead=10.)
        assert counted == saved_metrics[arm] and counted['tp'] == tp and counted['fp'] == fp
        results[arm] = {k: v for k, v in counted.items() if k != 'predictions'}; maxima[arm] = delta
        for k, v in model.state_dict().items(): torch.testing.assert_close(v.cpu(), before[k], rtol=0, atol=0)
    r, s = results['risk'], results['static']
    passed = (r['tp'] >= s['tp'] and r['fp'] <= s['fp'] and
              all(r['J'] > results[n]['J'] for n in ('static', 'identity', 'signal')))
    assert passed == completed['source_gate_passed']
    _durable_json(args.output, dict(verification='PASS', scope='SOURCE DEVELOPMENT REPLAY NOT EAST GAIN',
        curves_replayed=640, max_errors=maxima, independent_counts=results, metric_change_norm=changed,
        attempts=attempts, accepted=accepted, source_gate_passed=passed,
        source_sha256=_hash(DEFAULT_PARENT), verification_code_sha256=_hash(Path(__file__)),
        hashes=hashes, verification_tensors_unchanged=True,
        training_freeze_evidence='runner compares registered tensors; verifier only independently replays saved metrics'))
    print(json.dumps(dict(verification='PASS', curves_replayed=640, source_gate_passed=passed, results=results)), flush=True)


if __name__ == '__main__': main()
