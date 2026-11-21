"""Recount sealed EAST240 curves; not a full neural execution replay.

Compare the four predeclared arms, frozen tensors, update receipts and exact
archived static curves. Threshold-cap searches remain posthoc diagnostics.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import simple_data
from .ended_context_experiment import REFERENCE
from .ended_decoder import DecoderFfnModel
from .event_dynamics_experiment import load_coordinates, PROBE
from .operating_curve_audit import load_sealed, operating_curve
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash
from .task_state_experiment import MODES
from .task_state_feedback import TaskStateFfnModel
from .verify_event_projection import recount_prediction
from .verify_operational_memory import flips


def verify_audit(audit, *, mode):
    assert mode in MODES
    assert audit['intra'] == (mode != 'static') and audit['fast_discarded']
    assert audit['persistent'] is None
    assert audit['model_parameter_change_norm'] == audit['model_buffer_change_norm'] == 0
    attempts = audit['attempts']
    assert audit['attempted_updates'] == len(attempts)
    assert audit['accepted_updates'] == sum(r['accepted'] for r in attempts)
    if mode == 'static': assert not attempts
    if attempts:
        assert attempts[0]['fast_at_origin']['l2'] == attempts[0]['fast_before']['l2'] == 0
    for row in attempts:
        assert row['maturity_index'] == row['origin_index'] + 25
        assert row['maturity_index'] < audit['score_count']
        assert row['predictions_sealed_through_index'] == row['maturity_index']
        assert 'gradient_projection' not in row
        if row['accepted']:
            assert row['effective_index'] == row['maturity_index'] + 1
            assert row['loss_after'] < row['loss_before']
            assert row['fast_change_norm'] > 0 and row['fast_norm_after'] < 1.00001
        if row.get('reason') == 'no_valid_targets':
            assert not row['accepted']
            continue
        assert 'matured_objective' in row
        state = row['matured_objective']
        assert state['mode'] == mode and state['horizons'] == [5, 10, 15, 20, 25]
        assert state['state_dimensions'] == (1 if mode == 'risk' else 6)
        assert state['risk_coordinate_count'] == 1
        assert math.isfinite(state['risk_coordinate_scale']) and state['risk_coordinate_scale'] > 0
        assert not state['risk_teacher_is_physical_truth']
        assert state['target_detached'] and not state['outcome_access']
        assert not state['predicted_signals_used_for_risk'] and not state['hard_event_pseudo_labels']
        assert 0 <= state['valid_common_targets'] <= 575


def verify(root):
    folders = {mode: root / mode for mode in MODES}
    assert all((folder / 'completed.json').is_file() for folder in folders.values())
    order = json.loads((folders['static'] / 'order.json').read_text())
    assert len(order) == 240 and order == json.loads((REFERENCE / 'order.json').read_text())
    assert _hash(DEFAULT_PARENT) == PARENT_SHA256
    base, parent = load_parent(DEFAULT_PARENT)
    threshold = float(parent['threshold'])
    signals = [simple_data.attach_outcome(
        simple_data.read_shot('EAST', row['shot'], rate_contract='nominal'), 'EAST', {}) for row in order]
    results, seals, predictions, snapshots, curves, diagnostics, protocols = {}, {}, {}, {}, {}, {}, {}
    for mode, folder in folders.items():
        protocol = json.loads((folder / 'protocol.json').read_text())
        protocols[mode] = protocol
        assert protocol['mode'] == mode and protocol['parent_sha256'] == PARENT_SHA256
        assert protocol['threshold'] == threshold and protocol['order_sha256'] == _hash(REFERENCE / 'order.json')
        assert protocol['source_probe_file_sha256'] == _hash(PROBE / 'probe_heads.pt')
        assert all(not protocol[key] for key in (
            'source_retrained', 'risk_head_changed', 'alarm_tuning', 'auxiliary_head_changed',
            'supervised_updates_enabled', 'persistent_ffn_updates', 'predicted_signals_used_for_risk',
            'autoregressive_rollout', 'current_outcome_access', 'east_event_labels_used',
            'event_pseudo_label_used', 'risk_teacher_is_physical_truth'))
        assert json.loads((folder / 'order.json').read_text()) == order
        assert _hash(folder / 'code_snapshot.zip') == protocol['code_snapshot_sha256']
        with zipfile.ZipFile(folder / 'code_snapshot.zip') as archive:
            snapshots[mode] = {name: archive.read(name) for name in archive.namelist()}
        expected = (DecoderFfnModel(base) if mode == 'static' else
                    TaskStateFfnModel(base, *load_coordinates('event'), mode=mode))
        assert protocol['state_dimensions'] == (0 if mode == 'static' else len(expected.state_directions))
        saved = torch.load(folder / 'final_state.pt', map_location='cpu', weights_only=True)
        assert saved['parent_sha256'] == PARENT_SHA256 and saved['ended_ids'] == [row['shot'] for row in order]
        assert saved['state'].keys() == expected.state_dict().keys()
        for name, tensor in expected.state_dict().items():
            torch.testing.assert_close(saved['state'][name], tensor, atol=0, rtol=0)
        complete = json.loads((folder / 'completed.json').read_text())
        assert complete['source_and_all_registered_tensors_unchanged']
        assert complete['head_optimizer_steps'] == complete['memory_updates'] == 0
        assert complete['observed_shots'] == 240 and not complete['temporary_retained']
        records, seals[mode] = load_sealed(folder / 'stream', order, signals, threshold)
        curves[mode] = records
        results[mode] = {period: operating_curve(records[start:stop], threshold)
            for period, start, stop in (('all', 0, 240), ('first30', 0, 30), ('after30', 30, 240))}
        predictions[mode] = [recount_prediction(row, threshold) for row in records]
        comparison = json.loads((folder / 'comparison.json').read_text())
        accepted = attempts = 0
        reasons = Counter()
        for index, prediction in enumerate(predictions[mode], 1):
            audit = json.loads((folder / 'stream' / f'{index:04d}_{prediction["shot"]}_audit.json').read_text())
            verify_audit(audit, mode=mode)
            for key in ('shot', 'is_disrupt', 'alarm', 'timely'):
                assert prediction[key] == audit['prediction'][key] == comparison['all']['predictions'][index-1][key]
            if prediction['lead_ms'] is not None:
                assert abs(prediction['lead_ms'] - audit['prediction']['lead_ms']) < 1e-6
            accepted += audit['accepted_updates']
            attempts += audit['attempted_updates']
            reasons.update(row['reason'] for row in audit['attempts'])
        assert complete['accepted_intra_updates'] == accepted
        for period in comparison:
            point = results[mode][period]['fixed_source_threshold']
            assert all(point[key] == comparison[period][key] for key in ('tp', 'fp', 'tpr', 'fpr', 'J'))
        diagnostics[mode] = dict(attempts=attempts, accepted=accepted, reasons=dict(reasons),
                                 elapsed_seconds=complete['elapsed_seconds'])
    assert all(snapshots[mode] == snapshots['static'] for mode in MODES)
    assert all(protocols[mode]['normalization'] == protocols['static']['normalization'] for mode in MODES)
    archived = root.parent / 'event_dynamics240_20261021' / 'static' / 'stream'
    old, seals['archived_static'] = load_sealed(archived, order, signals, threshold)
    error = max(float(np.max(np.abs(a['scores'] - b['scores']))) for a, b in zip(old, curves['static']))
    assert error == 0., 'the original static numerical path changed'
    baseline = results['static']['all']['fixed_source_threshold']
    gate = {}
    for mode in MODES[1:]:
        point = results[mode]['all']['fixed_source_threshold']
        gate[mode] = dict(tp_increased=point['tp'] > baseline['tp'], fpr_within_20=point['fpr'] <= .20,
                          J_increased=point['J'] > baseline['J'])
        gate[mode]['passed'] = all(gate[mode].values())
    return dict(verification='PASS', scope='DEVELOPMENT_ONLY_PURE_INTRA_TASK_STATE',
        verification_method='sealed-curve recount, frozen-state comparison and update-log audit',
        full_model_replay=False, scientific_gain_gate=gate,
        source_sha256=PARENT_SHA256, order_sha256=_hash(REFERENCE / 'order.json'),
        verification_code_sha256=_hash(Path(__file__)), sealed_curve_checks=1200,
        all_registered_parameters_and_teachers_bitexact_initial=True,
        baseline_curve_max_error=error, postshot_supervised_steps=0,
        temporary_reset_and_discard_audited=True, primary_fixed_threshold=True,
        full_stream_threshold_caps_posthoc_only=True, compute_matched=False,
        results=results, diagnostics=diagnostics, seals=seals,
        paired_flips={f'{other}_vs_{reference}': flips(predictions[reference], predictions[other])
            for reference, other in (('static', 'risk'), ('static', 'event_risk'),
                                     ('static', 'random_risk'), ('random_risk', 'event_risk'))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new verification output required')
    torch.set_num_threads(4)
    result = verify(args.root)
    _durable_json(args.output, result)
    print(json.dumps(dict(verification=result['verification'], checks=result['sealed_curve_checks'],
        metrics={mode: row['all']['fixed_source_threshold'] for mode, row in result['results'].items()},
        scientific_gain_gate=result['scientific_gain_gate'], paired_flips=result['paired_flips'])), flush=True)


if __name__ == '__main__': main()
