"""Independent sealed-curve recount and bounded real-model replay of five arms."""
import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from . import simple_data
from .causal_input import apply_input_contract
from .ended_context_experiment import REFERENCE
from .masked_prefix_experiment import MODES
from .masked_prefix_feedback import MaskedPrefixModel, predict_shot
from .operating_curve_audit import load_sealed, operating_curve
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash
from .verify_event_projection import recount_prediction
from .verify_operational_memory import flips


def verify_audit(audit, *, mode):
    assert mode in MODES
    assert audit['intra'] == (mode != 'static')
    lag = 25 if mode.endswith('stale') else 0
    assert audit['lag'] == lag and audit['fast_discarded']
    assert audit['fast_initial']['l2'] == 0
    assert not audit['outcome_access'] and not audit['auxiliary_decoder_updated']
    assert audit['persistent'] is None
    assert audit['model_parameter_change_norm'] == audit['model_buffer_change_norm'] == 0
    rows = audit['attempts']
    assert audit['attempted_updates'] == len(rows)
    assert audit['accepted_updates'] == sum(r['accepted'] for r in rows)
    ticks = list(range(152, audit['score_count']-1, 100)) if mode != 'static' else []
    assert [r['update_index'] for r in rows] == ticks
    if rows: assert rows[0]['fast_before']['l2'] == 0
    for index, row in enumerate(rows):
        assert row['feedback_end_index'] == row['update_index']-lag
        assert row['predictions_sealed_through_index'] == row['update_index']
        assert row['effective_index'] == row['update_index']+1
        assert row['phase'] == index % 4 and 0 <= row['hidden_targets'] <= 4*8*23
        assert 0 <= row['fast_after']['l2'] <= 1.00001
        if row['accepted']:
            assert row['hidden_targets'] > 0 and row['loss_after'] < row['loss_before']
            assert row['reason'] == 'same_observed_loss_decreased'
        else:
            assert row['fast_after'] == row['fast_before']
            assert row['reason'] in ('no_observed_hidden_targets', 'same_observed_loss_not_decreased')


def early_alarm_floor(rows, threshold, *, last_unadapted_index):
    """A necessary bound, not an estimate of rescueability or future accuracy.

    Applies only if the model uses the static scores through this index and
    the first-alarm rule is irrevocable. All epochs remain in the denominator.
    """
    if type(last_unadapted_index) is not int or last_unadapted_index < 0:
        raise ValueError('nonnegative last unadapted endpoint index required')
    safe = [row for row in rows if not row['is_disrupt']]
    if not safe: raise ValueError('safe shots required')
    locked = []
    for row in safe:
        take = row['valid'] & (np.asarray(row['scores'], dtype=np.float64) >= threshold)
        indices = np.flatnonzero(take)
        if len(indices) and int(indices[0]) <= last_unadapted_index:
            index = int(indices[0])
            locked.append(dict(shot=row['shot'], first_alarm_index=index,
                ms_from_record_start=float((row['times'][index]-row['times'][0])*1000.)))
    allowed = int(np.floor(.20*len(safe)+1e-12))
    return dict(last_unadapted_index=last_unadapted_index, safe_count=len(safe),
        locked_false_alarms=len(locked), fpr_lower_bound=len(locked)/len(safe),
        allowed_false_alarms_at_fpr20=allowed,
        impossible_at_fpr_20_under_this_timing=len(locked) > allowed,
        condition='same source probabilities through this endpoint; immutable first-alarm decisions',
        minimum_good_later_updates_cannot_undo_past_alarms=True, locked_shots=locked)


def verify(root):
    assert (root/'completed.json').is_file()
    protocol = json.loads((root/'protocol.json').read_text())
    order = json.loads((root/'order.json').read_text())
    assert len(order) == 240 and order == json.loads((REFERENCE/'order.json').read_text())
    assert protocol['modes'] == list(MODES)
    assert protocol['source_sha256'] == _hash(DEFAULT_PARENT) == PARENT_SHA256
    assert protocol['order_sha256'] == _hash(REFERENCE/'order.json')
    assert protocol['code_snapshot_sha256'] == _hash(root/'code_snapshot.zip')
    for key in ('source_classifier_changed', 'alarm_tuning', 'event_labels_used', 'target_support_used',
                'past_target_normalization', 'persistent_neural_updates', 'target_head_training',
                'temporary_retained', 'independent_future_validation'):
        assert not protocol[key]
    with zipfile.ZipFile(root/'code_snapshot.zip') as archive:
        for name in archive.namelist():
            assert archive.read(name) == (Path(__file__).parent/name).read_bytes(), name
    base, parent = load_parent(DEFAULT_PARENT)
    threshold = float(parent['threshold'])
    assert protocol['threshold'] == threshold
    heads = Path(protocol['source_heads'])
    train_protocol = json.loads((heads/'protocol.json').read_text())
    assert not train_protocol['east_data_used'] and not train_protocol['source_classifier_changed']
    roles = simple_data.source_roles()
    train_ids, valid_ids = set(train_protocol['train_ids']), set(train_protocol['holdout_ids'])
    assert not train_ids & valid_ids and train_ids | valid_ids == set(roles['train'])
    assert not (train_ids | valid_ids) & (set(roles['test']) | set(roles['val']))
    signals = [simple_data.attach_outcome(simple_data.read_shot('EAST', r['shot'], rate_contract='nominal'),
                                          'EAST', {}) for r in order]
    results, predictions, diagnostics, opportunities, traces, seals = {}, {}, {}, {}, {}, {}
    model_replays = []
    provider = SimpleNamespace(window=simple_data.window)
    for mode in MODES:
        folder = root/mode
        complete = json.loads((folder/'completed.json').read_text())
        assert complete['source_and_registered_tensors_unchanged'] and complete['observed_shots'] == 240
        assert complete['persistent_updates'] == complete['auxiliary_head_updates'] == 0
        kind = 'risk' if mode.startswith('risk') else 'signal'
        assert complete['head_kind'] == kind
        assert _hash(heads/f'{kind}_head.pt') == protocol['source_head_sha256'][kind]
        state = torch.load(heads/f'{kind}_head.pt', weights_only=True, map_location='cpu')
        model = MaskedPrefixModel(base)
        model.decoder.load_state_dict(state['decoder'])
        saved = torch.load(folder/'final_state.pt', weights_only=True, map_location='cpu')
        assert saved['state'].keys() == model.state_dict().keys()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(saved['state'][name], value, atol=0, rtol=0)
        records, seals[mode] = load_sealed(folder/'stream', order, signals, threshold)
        traces[mode] = records
        results[mode] = {}
        for period, start, stop in [('all', 0, 240), ('first30', 0, 30), ('after30', 30, 240)]:
            # Compact output; all posthoc threshold choices remain labelled.
            point = operating_curve(records[start:stop], threshold)
            point.pop('curve')
            results[mode][period] = point
        predictions[mode] = [recount_prediction(row, threshold) for row in records]
        comparison = json.loads((folder/'comparison.json').read_text())
        attempts = accepted = 0
        reasons = Counter()
        opportunities[mode] = []
        for index, pred in enumerate(predictions[mode], 1):
            audit = json.loads((folder/'stream'/f'{index:04d}_{pred["shot"]}_audit.json').read_text())
            verify_audit(audit, mode=mode)
            for key in ('shot', 'is_disrupt', 'alarm', 'timely'):
                assert pred[key] == audit['prediction'][key] == comparison['all']['predictions'][index-1][key]
            if pred['lead_ms'] is not None:
                assert abs(pred['lead_ms']-audit['prediction']['lead_ms']) < 1e-6
            attempts += audit['attempted_updates']; accepted += audit['accepted_updates']
            reasons.update(r['reason'] for r in audit['attempts'])
            opportunities[mode].append([r['update_index'] for r in audit['attempts']])
        assert accepted == complete['accepted_intra_updates']
        for period in comparison:
            assert all(comparison[period][k] == results[mode][period]['fixed_source_threshold'][k]
                       for k in ('tp', 'fp', 'tpr', 'fpr', 'J'))
        diagnostics[mode] = dict(attempts=attempts, accepted=accepted, reasons=dict(reasons),
                                 elapsed_seconds=complete['elapsed_seconds'])
        # Bounded execution replay, in addition to checking all 1200 seals.
        # Not represented as a full 240-shot neural rerun.
        model = model.cuda().eval().requires_grad_(False)
        lag = 25 if mode.endswith('stale') else 0
        for index in (0, 100):
            raw = apply_input_contract(simple_data.read_shot('EAST', order[index]['shot'], rate_contract='nominal'))
            score, audit = predict_shot(model, raw, intra=mode != 'static', lag=lag, device='cuda', data_api=provider)
            error = float(np.max(np.abs(score-records[index]['scores'])))
            assert error == 0., (mode, order[index]['shot'], error)
            model_replays.append(dict(mode=mode, shot=order[index]['shot'], max_score_error=error))
        model.cpu()
    assert all(opportunities[m] == opportunities['signal_stale'] for m in MODES[1:])
    for mode in MODES[1:]:
        for before, after in zip(traces['static'], traces[mode]):
            np.testing.assert_array_equal(before['scores'][:153], after['scores'][:153])
    startup = early_alarm_floor(traces['static'], threshold, last_unadapted_index=152)
    latency_sweep = [early_alarm_floor(traces['static'], threshold, last_unadapted_index=k)
                     for k in (0, 7, 15, 24, 31, 49, 99, 127, 152)]
    baseline = results['static']['all']['fixed_source_threshold']
    gates = {}
    for mode in MODES[1:]:
        point = results[mode]['all']['fixed_source_threshold']
        gate = dict(tp_increased=point['tp'] > baseline['tp'], fpr_within_20=point['fpr'] <= .20,
                    J_increased=point['J'] > baseline['J'])
        gates[mode] = dict(gate, passed=all(gate.values()))
    pairs = [('static', m) for m in MODES[1:]] + [('signal_stale', 'signal_fresh'), ('risk_stale', 'risk_fresh')]
    return dict(verification='PASS', verification_method='all sealed curves recounted; 10 bounded neural replays',
        full_model_replay=False, sealed_curves_checked=1200, neural_replays=model_replays,
        source_sha256=PARENT_SHA256, code_sha256=_hash(Path(__file__)),
        all_registered_state_bitexact_initial=True, source_train_role_split_verified=True,
        source_only_input_processing=True, target_label_use='postseal evaluation only',
        equal_update_opportunities=True, no_inter_updates=True, scientific_gain_gate=gates,
        startup_error_floor=startup, static_prefix_latency_bounds=latency_sweep,
        primary_fixed_threshold=True, posthoc_caps_are_not_deployment_performance=True,
        results=results, diagnostics=diagnostics, seals=seals,
        paired_flips={f'{b}_vs_{a}': flips(predictions[a], predictions[b]) for a, b in pairs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new verification artifact required')
    torch.set_num_threads(4)
    result = verify(args.root)
    _durable_json(args.output, result)
    print(json.dumps(dict(verification=result['verification'], gates=result['scientific_gain_gate'],
                         metrics={k: v['all']['fixed_source_threshold'] for k, v in result['results'].items()})), flush=True)


if __name__ == '__main__': main()
