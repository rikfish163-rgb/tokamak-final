"""Recount sealed curves of the fixed-parent matured-state-loss experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import simple_data
from .ended_context_experiment import REFERENCE
from .ended_decoder import DecoderFfnModel
from .event_dynamics import EventDynamicsFfnModel
from .event_dynamics_experiment import load_coordinates
from .operating_curve_audit import load_sealed, operating_curve
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash
from .verify_event_projection import recount_prediction
from .verify_operational_memory import flips


MODES = ('static', 'control', 'event', 'random')


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
    for r in attempts:
        assert r['maturity_index'] == r['origin_index'] + 25
        assert r['maturity_index'] < audit['score_count']
        assert r['predictions_sealed_through_index'] == r['maturity_index']
        if r['accepted']:
            assert r['effective_index'] == r['maturity_index'] + 1
            assert r['loss_after'] < r['loss_before']
            assert 0 < r['fast_change_norm'] and r['fast_norm_after'] < 1.00001
        assert 'gradient_projection' not in r
        if mode in ('event', 'random') and r.get('reason') != 'no_valid_targets':
            s = r['matured_objective'] if 'matured_objective' in r else None
            assert s is not None and s['mode'] == mode
            assert s['horizons'] == [5, 10, 15, 20, 25] and s['state_dimensions'] == 5
            assert s['target_detached'] and not s['outcome_access']
            assert not s['predicted_signals_used_for_risk'] and not s['hard_event_pseudo_labels']
            assert 0 <= s['valid_common_targets'] <= 575
        else:
            assert 'matured_objective' not in r


def verify(root):
    folders = {m: root / m for m in MODES}
    assert all((f / 'completed.json').is_file() for f in folders.values())
    order = json.loads((folders['static'] / 'order.json').read_text())
    assert len(order) == 240 and order == json.loads((REFERENCE / 'order.json').read_text())
    assert _hash(DEFAULT_PARENT) == PARENT_SHA256
    base, parent = load_parent(DEFAULT_PARENT)
    threshold = float(parent['threshold'])
    signals = [simple_data.attach_outcome(simple_data.read_shot('EAST', r['shot'], rate_contract='nominal'),
                                         'EAST', {}) for r in order]
    results, seals, predictions, snapshots, curves, diagnostics, protocols = {}, {}, {}, {}, {}, {}, {}
    for mode, folder in folders.items():
        p = json.loads((folder / 'protocol.json').read_text())
        protocols[mode] = p
        assert p['mode'] == mode and p['parent_sha256'] == PARENT_SHA256 and p['threshold'] == threshold
        assert p['order_sha256'] == _hash(REFERENCE / 'order.json')
        assert all(not p[k] for k in ('source_retrained', 'risk_head_changed', 'alarm_tuning',
            'auxiliary_head_changed', 'supervised_updates_enabled', 'persistent_ffn_updates',
            'predicted_signals_used_for_risk', 'autoregressive_rollout', 'current_outcome_access',
            'east_event_labels_used', 'event_pseudo_label_used'))
        assert json.loads((folder / 'order.json').read_text()) == order
        assert _hash(folder / 'code_snapshot.zip') == p['code_snapshot_sha256']
        with zipfile.ZipFile(folder / 'code_snapshot.zip') as z:
            snapshots[mode] = {name: z.read(name) for name in z.namelist()}
        saved = torch.load(folder / 'final_state.pt', map_location='cpu', weights_only=True)
        assert saved['parent_sha256'] == PARENT_SHA256 and saved['ended_ids'] == [r['shot'] for r in order]
        expected = (EventDynamicsFfnModel(base, *load_coordinates(mode), mode=mode)
                    if mode in ('event', 'random') else DecoderFfnModel(base))
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
        predictions[mode] = [recount_prediction(r, threshold) for r in records]
        comparison = json.loads((folder / 'comparison.json').read_text())
        accepted = attempts = 0
        reasons = Counter()
        for index, pred in enumerate(predictions[mode], 1):
            a = json.loads((folder / 'stream' / f'{index:04d}_{pred["shot"]}_audit.json').read_text())
            verify_audit(a, mode=mode)
            for key in ('shot', 'is_disrupt', 'alarm', 'timely'):
                assert pred[key] == a['prediction'][key] == comparison['all']['predictions'][index - 1][key]
            if pred['lead_ms'] is not None:
                assert abs(pred['lead_ms'] - a['prediction']['lead_ms']) < 1e-6
            accepted += a['accepted_updates']
            attempts += a['attempted_updates']
            reasons.update(r['reason'] for r in a['attempts'])
        assert complete['accepted_intra_updates'] == accepted
        for period in comparison:
            point = results[mode][period]['fixed_source_threshold']
            assert all(point[k] == comparison[period][k] for k in ('tp', 'fp', 'tpr', 'fpr', 'J'))
        diagnostics[mode] = dict(attempts=attempts, accepted=accepted, reasons=dict(reasons),
                                 elapsed_seconds=complete['elapsed_seconds'])
    assert all(snapshots[m] == snapshots['static'] for m in MODES)
    assert all(protocols[m]['normalization'] == protocols['static']['normalization'] for m in MODES)
    archive_errors = {}
    for mode in ('static', 'control'):
        old_folder = root.parent / 'event_projection240_20261021' / mode / 'stream'
        old, seals['archived_' + mode] = load_sealed(old_folder, order, signals, threshold)
        delta = max(float(np.max(np.abs(a['scores'] - b['scores']))) for a, b in zip(old, curves[mode]))
        assert delta == 0., 'new objective hook changed the archived control numerical path'
        archive_errors[mode] = delta
    return dict(verification='PASS', scope='DEVELOPMENT_ONLY_PURE_INTRA_MATURED_STATE_LOSS',
        source_sha256=PARENT_SHA256, order_sha256=_hash(REFERENCE / 'order.json'),
        verification_code_sha256=_hash(Path(__file__)), sealed_curve_checks=1440,
        all_registered_parameters_and_teachers_bitexact_initial=True,
        baseline_curve_max_errors=archive_errors, postshot_supervised_steps=0,
        temporary_reset_and_discard_audited=True, primary_fixed_threshold=True,
        full_stream_threshold_caps_posthoc_only=True, compute_matched=False,
        results=results, diagnostics=diagnostics, seals=seals,
        paired_flips={f'{b}_vs_{a}': flips(predictions[a], predictions[b])
            for a, b in (('static', 'control'), ('static', 'event'), ('static', 'random'),
                         ('control', 'event'), ('random', 'event'))})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('preserve existing verification; new path required')
    result = verify(args.root)
    _durable_json(args.output, result)
    print(json.dumps(dict(verification=result['verification'], sealed_curve_checks=result['sealed_curve_checks'],
        metrics={m: r['all']['fixed_source_threshold'] for m, r in result['results'].items()},
        paired_flips=result['paired_flips'])), flush=True)


if __name__ == '__main__':
    main()
