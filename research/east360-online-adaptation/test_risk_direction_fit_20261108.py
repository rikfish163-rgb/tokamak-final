"""CPU-only, tiny actual-network checks for the paired source teaching fit."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import run_joint_trajectory_20261107 as j
from risk_feature_intra_20261107 import FeatureIntra
from risk_meta_sequence_20261108 import sequence_from_note, teacher_sequence
from risk_meta_step_20261107 import MetaOrderedTrajectory
from test_joint_stream_20261107 import base_model, signal


@pytest.fixture
def tiny_reference(tmp_path, monkeypatch):
    """Build a two-task old-Adam reference; production always defaults to 512."""
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield from _tiny_reference(tmp_path, monkeypatch)
    torch.use_deterministic_algorithms(old_deterministic)
    torch.set_num_threads(old_threads)


def _tiny_reference(tmp_path, monkeypatch):
    base = base_model()
    base_before = deepcopy(base.state_dict())
    torch.manual_seed(89)
    initial = MetaOrderedTrajectory(base.center, base.scale, dim=base.cfg.dim, scenarios=4).eval()
    initial_root = tmp_path / 'initial'
    reference = tmp_path / 'reference'
    cold_root = tmp_path / 'cold'
    for path in (initial_root, reference, cold_root):
        path.mkdir()
    torch.save(initial.state_dict(), initial_root / 'source_predictor.pt')

    shot = dict(signal(400), shot=1, valid=np.ones(400, dtype=bool),
                y=(np.arange(400) >= 290).astype(np.float32))
    def make_note(first, group):
        origins = [first + 32*k for k in range(4)]
        query = origins[-1] + 26
        return dict(shot=1, origins=origins, query=query, label_time=query+25, group=group)
    notes = [make_note(127, '0_to_0'), make_note(160, '0_to_1')]
    # Pooled notes are intentionally different so selecting the wrong history is detectable.
    history = [dict(shot=999, recorded_note=deepcopy(note)) for note in notes]
    excluded = list(range(1000, 1046))
    ids = [1, *excluded]
    source = [shot, *[dict(shot=sid) for sid in excluded]]
    order = list(range(200, 222))
    cold = {sid: signal(160) for sid in order[:20]}
    protocol = dict(status='POSITIVE_SHOT_POOL_CONTRAST',
                    source_train_ids=ids, excluded_outer_shots=excluded,
                    parent_sha256=j.PARENT_SHA256, source_steps=2, sequence_steps=4,
                    deterministic_algorithms=True,
                    initial_predictor=str(initial_root / 'source_predictor.pt'),
                    cold20=order[:20], cold_steps=128, cold_batch=32,
                    risk_classifier_training=False, east_label_fitting=False,
                    source_val_or_test_selection=False)
    (reference / 'protocol.json').write_text(json.dumps(protocol))
    (reference / 'fit.json').write_text(json.dumps(
        dict(status='SOURCE_TEACHER_FIT_COMPLETE', history=history,
             deterministic_algorithms=True, source_classifier_unchanged=True)))
    (cold_root / 'input_map.json').write_text(json.dumps(dict(calibration_ids=order[:20])))

    # Reference replay uses the existing Adam teacher directly, not the new fit.
    control = FeatureIntra(base, initial)
    control.decoder.requires_grad_(True)
    opt = torch.optim.AdamW(control.decoder.parameters(), lr=3e-4, weight_decay=0.)
    with sdpa_kernel(SDPBackend.MATH):
        for note in notes:
            control.zero_grad(set_to_none=True)
            episodes, qx, qa, label, _ = sequence_from_note(shot, note, 'cpu')
            result = teacher_sequence(control, episodes, qx, qa, label, credit='last')
            (result['risk_loss'] + result['signal_loss']).backward()
            torch.nn.utils.clip_grad_norm_(control.decoder.parameters(), 1., error_if_nonfinite=True)
            opt.step()
    source_expected = deepcopy(control.decoder.state_dict())
    torch.save(source_expected, reference / 'source_recorded_sampling.pt')

    actual_fit_predictor = j.fit_predictor
    preparation_calls = []
    def tiny_preparation(base_arg, decoder, shots, steps, seed, label):
        assert base_arg is base and steps == 128 and seed == j.SEED+1
        assert j.FIT_BATCH == 32
        assert len(shots) == 20 and all(set(s) == {'values', 'available'} for s in shots)
        preparation_calls.append((shots, steps, seed, label))
        # Exercise the real neural preparation on CPU without a 128 x 32 fixture.
        with monkeypatch.context() as local:
            local.setattr(j, 'FIT_BATCH', 2)
            return actual_fit_predictor(base_arg, decoder, shots, 1, seed, label)
    with sdpa_kernel(SDPBackend.MATH):
        tiny_preparation(base, control.decoder, list(cold.values()), 128, j.SEED+1, 'expected')
    prepared_expected = deepcopy(control.decoder.state_dict())
    torch.save(prepared_expected, reference / 'prepared_recorded_sampling.pt')
    preparation_calls.clear()

    raw_calls = []
    def raw_for(sid, hashes):
        raw_calls.append(sid)
        assert hashes == 'fixed-hashes'
        return sid
    def prepare_actual(sid, rank, mapping):
        assert rank == 21 and mapping['calibration_ids'] == order[:20]
        # The fit must forbid outcome reads throughout raw loading/preparation.
        with pytest.raises(RuntimeError, match='outcomes unavailable'):
            j.r.io23.read_outcome('EAST', sid)
        return cold[sid]
    monkeypatch.setattr(j.r, 'configure', lambda: None)
    monkeypatch.setattr(j.r, 'load_parent', lambda _: (base, {}))
    monkeypatch.setattr(j.r, 'COLD', cold_root)
    monkeypatch.setattr(j, 'load_source_train', lambda: (source, ids))
    monkeypatch.setattr(j, 'cohort', lambda _: (None, dict(order=order, raw_sha256='fixed-hashes'),
                                                None, (None, None, 20)))
    monkeypatch.setattr(j, 'raw_for', raw_for)
    monkeypatch.setattr(j.r, 'prepare_actual', prepare_actual)
    monkeypatch.setattr(j, 'fit_predictor', tiny_preparation)
    yield dict(base=base, base_before=base_before, initial=initial, reference=reference,
               initial_root=initial_root, notes=notes, order=order, cold=cold,
               source_expected=source_expected, prepared_expected=prepared_expected,
               preparation_calls=preparation_calls, raw_calls=raw_calls)


def test_paired_actual_network_fit_replays_control_and_forbids_east_outcomes(
        tiny_reference, tmp_path, monkeypatch):
    import run_risk_direction_teaching_20261108 as run

    fixture = tiny_reference
    calls = []
    real_teacher = run.teacher_sequence
    def checked_teacher(model, episodes, qx, qa, label, *, credit, step_rule):
        assert credit == 'last' and len(episodes) == 4
        assert all('y' not in e and 'label' not in e for e in episodes)
        assert all(p.grad is None for p in model.parameters())
        calls.append((step_rule, float(label.item())))
        return real_teacher(model, episodes, qx, qa, label, credit=credit, step_rule=step_rule)
    monkeypatch.setattr(run, 'teacher_sequence', checked_teacher)
    out = tmp_path / 'paired'
    run.fit(out, reference_fit=fixture['reference'], initial_root=fixture['initial_root'],
            device='cpu', steps=2)

    protocol = json.loads((out / 'protocol.json').read_text())
    fit = json.loads((out / 'fit.json').read_text())
    assert [entry['recorded_note'] for entry in fit['history']] == fixture['notes']
    assert calls == [('adam', 0.), ('momentum_direction', 0.),
                     ('adam', 1.), ('momentum_direction', 1.)]
    assert protocol['cold20'] == fixture['order'][:20]
    assert not protocol['risk_classifier_training'] and not protocol['east_label_fitting']
    assert not protocol['jtext_online']
    assert fixture['raw_calls'] == fixture['order'][:20]
    assert len(fixture['preparation_calls']) == 2
    assert fit['source_control_exact'] and fit['prepared_control_exact']
    assert fit['source_classifier_unchanged']
    for phase in ('source', 'prepared'):
        expected = fixture[f'{phase}_expected']
        control = torch.load(out / f'{phase}_control.pt', map_location='cpu', weights_only=True)
        assert set(control) == set(expected)
        assert all(torch.equal(control[k], expected[k]) for k in expected)
        assert (out / f'{phase}_direction.pt').is_file()
    assert all(torch.equal(v, fixture['base_before'][k])
               for k, v in fixture['base'].state_dict().items())
    for entry in fit['history']:
        assert len(entry['recorded_note']['origins']) == 4
        assert entry['feedback_horizons'] == [
            dict(origin=origin, start=origin+1, end=origin+25)
            for origin in entry['recorded_note']['origins']]
        assert entry['native_current_label'] in (0., 1.)
        assert entry['native_future_label'] in (0., 1.)
        for stats in entry['losses'].values():
            assert np.isfinite(stats['gradient_norm'])
            assert np.isfinite(stats['signal']) and np.isfinite(stats['risk'])


@pytest.mark.parametrize('damage', ['missing_task', 'missing_note', 'missing_checkpoint'])
def test_incomplete_reference_is_rejected_before_output(tiny_reference, tmp_path, damage):
    import run_risk_direction_teaching_20261108 as run

    reference = tiny_reference['reference']
    path = reference / 'fit.json'
    done = json.loads(path.read_text())
    if damage == 'missing_task':
        done['history'].pop()
        path.write_text(json.dumps(done))
    elif damage == 'missing_note':
        del done['history'][-1]['recorded_note']
        path.write_text(json.dumps(done))
    else:
        (reference / 'prepared_recorded_sampling.pt').unlink()
    out = tmp_path / 'must-not-exist'
    with pytest.raises((ValueError, FileNotFoundError)):
        run.fit(out, reference_fit=reference, initial_root=tiny_reference['initial_root'],
                device='cpu', steps=2)
    assert not out.exists()


def test_control_mismatch_fails_without_claiming_completion(tiny_reference, tmp_path):
    import run_risk_direction_teaching_20261108 as run

    reference = tiny_reference['reference']
    path = reference / 'source_recorded_sampling.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    key = next(k for k, value in state.items() if value.is_floating_point())
    state[key].view(-1)[0] += 1
    torch.save(state, path)
    out = tmp_path / 'mismatch'
    with pytest.raises(ValueError, match='exactly reproduce'):
        run.fit(out, reference_fit=reference, initial_root=tiny_reference['initial_root'],
                device='cpu', steps=2)
    assert not (out / 'fit.json').exists()
    assert not tiny_reference['raw_calls']


def test_direction_zero_momentum_error_propagates_without_fallback(tiny_reference, tmp_path, monkeypatch):
    import run_risk_direction_teaching_20261108 as run

    real_teacher = run.teacher_sequence
    def failing_direction(*args, credit, step_rule):
        if step_rule == 'momentum_direction':
            raise FloatingPointError('zero momentum')
        return real_teacher(*args, credit=credit, step_rule=step_rule)
    monkeypatch.setattr(run, 'teacher_sequence', failing_direction)
    out = tmp_path / 'zero-momentum'
    with pytest.raises(FloatingPointError, match='zero momentum'):
        run.fit(out, reference_fit=tiny_reference['reference'],
                initial_root=tiny_reference['initial_root'], device='cpu', steps=2)
    assert not (out / 'fit.json').exists()
    assert not tiny_reference['raw_calls']
