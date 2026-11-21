"""Independent recount of the completed pure-intra event-projection pilot.

Posthoc development analysis only. No thresholds, models or sealed curves are
changed. Projection diagnostics describe local numerical behavior, not physics.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import simple_data
from .ended_context_experiment import REFERENCE
from .operating_curve_audit import load_sealed, operating_curve
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash
from .verify_operational_memory import flips


MODES = ('static', 'control', 'event', 'random')


def verify_prediction_audit(audit, *, mode):
    assert audit['intra'] == (mode != 'static') and audit['fast_discarded']
    assert audit['persistent'] is None
    assert audit['model_parameter_change_norm'] == audit['model_buffer_change_norm'] == 0
    attempts = audit['attempts']
    assert audit['attempted_updates'] == len(attempts)
    assert audit['accepted_updates'] == sum(r['accepted'] for r in attempts)
    if mode == 'static':
        assert not attempts
    if attempts:
        assert attempts[0]['fast_at_origin']['l2'] == 0
        assert attempts[0]['fast_before']['l2'] == 0
    for r in attempts:
        assert r['maturity_index'] == r['origin_index'] + 25
        assert r['maturity_index'] < audit['score_count']
        assert r['predictions_sealed_through_index'] == r['maturity_index']
        if r['accepted']:
            assert r['effective_index'] == r['maturity_index'] + 1
            assert r['loss_after'] < r['loss_before']
            assert 0 < r['fast_change_norm'] and r['fast_norm_after'] < 1.00001
        if 'gradient_projection' in r:
            assert mode in ('event', 'random')
            p = r['gradient_projection']
            assert p['mode'] == mode and not p['outcome_access'] and not p['event_pseudo_label_used']
            assert 0 <= p['rank'] <= 5 and 0 <= p['retained_norm_fraction'] <= 1.000001
            assert p['state_derivative_projected_norm'] <= 1e-8 * max(1., p['state_derivative_original_norm'])
        elif 'gradient' in r and r['grad_norm'] > 0:
            assert mode == 'control'


def distribution(values):
    x = np.asarray(values, dtype=float)
    assert len(x) and np.isfinite(x).all()
    return dict(count=len(x), mean=float(x.mean()),
                q05=float(np.quantile(x, .05)), median=float(np.median(x)),
                q95=float(np.quantile(x, .95)))


def recount_prediction(row, threshold):
    hits = np.flatnonzero(row['valid'] & (row['scores'].astype(np.float64) >= threshold))
    alarm = bool(len(hits))
    lead = (float(np.round((row['down'] - row['times'][hits[0]]) * 1000., 9))
            if row['is_disrupt'] and alarm else None)
    return dict(shot=row['shot'], is_disrupt=bool(row['is_disrupt']), alarm=alarm,
                lead_ms=lead, timely=bool(lead is not None and lead > 30.))


def verify(root):
    folders = {m: root / m for m in MODES}
    assert all((f / 'completed.json').is_file() for f in folders.values())
    order = json.loads((folders['static'] / 'order.json').read_text())
    assert len(order) == 240 and order == json.loads((REFERENCE / 'order.json').read_text())
    assert _hash(DEFAULT_PARENT) == PARENT_SHA256
    base, _ = load_parent(DEFAULT_PARENT)
    original_head = base.forecast_head.state_dict()
    protocols = {m: json.loads((f / 'protocol.json').read_text()) for m, f in folders.items()}
    threshold = protocols['static']['threshold']
    signals = [simple_data.attach_outcome(simple_data.read_shot('EAST', r['shot'], rate_contract='nominal'),
                                        'EAST', {}) for r in order]
    results, seals, predictions, snapshots, curves, diagnostics = {}, {}, {}, {}, {}, {}
    for mode, folder in folders.items():
        p = protocols[mode]
        assert p['mode'] == mode and p['parent_sha256'] == PARENT_SHA256
        assert p['threshold'] == threshold and p['order_sha256'] == _hash(REFERENCE / 'order.json')
        assert p['head_feedback'] == 'none' and not p['supervised_updates_enabled']
        assert not p['alarm_tuning'] and not p['risk_head_changed'] and not p['source_retrained']
        assert p['normalization'] == protocols['static']['normalization']
        assert json.loads((folder / 'order.json').read_text()) == order
        assert _hash(folder / 'code_snapshot.zip') == p['code_snapshot_sha256']
        with zipfile.ZipFile(folder / 'code_snapshot.zip') as z:
            snapshots[mode] = {name: z.read(name) for name in z.namelist()}
        state = torch.load(folder / 'final_state.pt', map_location='cpu', weights_only=True)
        assert state['parent_sha256'] == PARENT_SHA256 and state['ended_ids'] == [r['shot'] for r in order]
        for name, value in original_head.items():
            torch.testing.assert_close(state['decoder'][name], value, atol=0, rtol=0)
        complete = json.loads((folder / 'completed.json').read_text())
        assert complete['source_file_and_parameters_unchanged'] and complete['auxiliary_head_unchanged']
        assert complete['head_optimizer_steps'] == complete['memory_updates'] == 0
        assert complete['observed_shots'] == 240 and not complete['temporary_retained']
        records, seals[mode] = load_sealed(folder / 'stream', order, signals, threshold)
        curves[mode] = records
        results[mode] = {period: operating_curve(records[start:stop], threshold)
            for period, start, stop in (('all', 0, 240), ('first30', 0, 30), ('after30', 30, 240))}
        predictions[mode] = [recount_prediction(r, threshold) for r in records]
        comparison = json.loads((folder / 'comparison.json').read_text())
        accepted = 0
        projection_rows = []
        for index, pred in enumerate(predictions[mode], 1):
            a = json.loads((folder / 'stream' / f'{index:04d}_{pred["shot"]}_audit.json').read_text())
            verify_prediction_audit(a, mode=mode)
            for key in ('shot', 'is_disrupt', 'alarm', 'timely'):
                assert pred[key] == a['prediction'][key] == comparison['all']['predictions'][index-1][key]
            if pred['lead_ms'] is not None:
                assert abs(pred['lead_ms'] - a['prediction']['lead_ms']) < 1e-6
            accepted += a['accepted_updates']
            projection_rows.extend(r['gradient_projection'] for r in a['attempts'] if 'gradient_projection' in r)
        assert complete['accepted_intra_updates'] == accepted
        for period in comparison:
            point = results[mode][period]['fixed_source_threshold']
            assert all(point[k] == comparison[period][k] for k in ('tp', 'fp', 'tpr', 'fpr', 'J'))
        diagnostics[mode] = dict(accepted_updates=accepted, elapsed_seconds=complete['elapsed_seconds'])
        if projection_rows:
            diagnostics[mode].update(rank_counts=dict(Counter(r['rank'] for r in projection_rows)),
                values={k: distribution([r[k] for r in projection_rows]) for k in
                    ('retained_norm_fraction', 'state_change_unprojected', 'state_change_projected',
                     'risk_logit_change_unprojected', 'risk_logit_change_projected')},
                finite_step_state_change_reduced=sum(r['state_change_projected'] < r['state_change_unprojected']
                                                    for r in projection_rows),
                total_projection_attempts=len(projection_rows))
    assert all(snapshots[m] == snapshots['static'] for m in MODES)
    old_folder = root.parent / 'operational_intra_pair240_v2_20261010/mapped_static/stream'
    old, seals['archived_static'] = load_sealed(old_folder, order, signals, threshold)
    static_error = max(float(np.max(np.abs(a['scores'] - b['scores'])))
                       for a, b in zip(old, curves['static']))
    assert static_error == 0., 'zero-update path must reproduce archived static curves exactly'
    return dict(verification='PASS', scope='DEVELOPMENT_ONLY_PURE_INTRA_PILOT',
        source_sha256=PARENT_SHA256, order_sha256=_hash(REFERENCE / 'order.json'),
        verification_code_sha256=_hash(Path(__file__)), sealed_curve_checks=1200,
        auxiliary_heads_bitexact_source=True, static_max_error_vs_archive=static_error,
        temporary_reset_and_discard_audited=True, postshot_supervised_steps=0,
        same_model_order_threshold_and_input_rule=True, projection_not_compute_matched=True,
        primary_fixed_threshold=True, full_stream_threshold_caps_posthoc_only=True,
        results=results, diagnostics=diagnostics, seals=seals,
        paired_flips={f'{b}_vs_{a}': flips(predictions[a], predictions[b])
            for a, b in (('static', 'control'), ('static', 'event'), ('static', 'random'),
                         ('control', 'event'), ('random', 'event'))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('new output required; existing verification is immutable')
    result = verify(args.root)
    _durable_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('results', 'seals', 'diagnostics')}))
    print(json.dumps({a: r['all']['fixed_source_threshold'] for a, r in result['results'].items()}))


if __name__ == '__main__':
    main()
