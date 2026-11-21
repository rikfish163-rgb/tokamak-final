"""Paired source-only auxiliary teaching with a matched update-direction control.

Run CUDA with CUBLAS_WORKSPACE_CONFIG=:4096:8 set before starting Python.
This does not train the qualified risk classifier or run J-TEXT online learning.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import run_joint_trajectory_20261107 as j
from risk_feature_intra_20261107 import FeatureIntra
from risk_meta_sequence_20261108 import sequence_from_note, teacher_sequence
from risk_meta_step_20261107 import MetaOrderedTrajectory
from run_joint_motion_20261107 import FIT as INITIAL
from run_risk_positive_pool_20261108 import FIT as REFERENCE, census


OUT = j.r.ROOT / 'probe_outputs/risk_direction_source_20261108_v1'
STEPS = 512
RULES = {'control': 'adam', 'direction': 'momentum_direction'}
NOTE_KEYS = ('shot', 'origins', 'query', 'label_time', 'group')


def _same_state(actual, expected):
    return (set(actual) == set(expected)
            and all(torch.equal(actual[key], expected[key]) for key in actual))


def fit(out=OUT, *, reference_fit=REFERENCE, initial_root=INITIAL, device='cuda', steps=STEPS):
    """Run the paired fit; the shorter `steps` override is for CPU fixtures only."""
    started = time.monotonic()
    out, reference_fit, initial_root = Path(out), Path(reference_fit), Path(initial_root)
    if out.exists():
        raise FileExistsError(f'exclusive output already exists: {out}')
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError('positive reference task count required')
    prior, done = (j.r.read(reference_fit / f'{name}.json') for name in ('protocol', 'fit'))
    if (prior.get('status') != 'POSITIVE_SHOT_POOL_CONTRAST'
            or done.get('status') != 'SOURCE_TEACHER_FIT_COMPLETE'
            or not prior.get('deterministic_algorithms')
            or not done.get('deterministic_algorithms')
            or not done.get('source_classifier_unchanged')
            or prior.get('parent_sha256') != j.PARENT_SHA256
            or prior.get('source_steps') != steps or prior.get('sequence_steps') != 4
            or prior.get('cold_steps') != 128 or prior.get('cold_batch') != 32
            or prior.get('risk_classifier_training') is not False
            or prior.get('east_label_fitting') is not False
            or prior.get('source_val_or_test_selection') is not False
            or Path(prior.get('initial_predictor', '')).resolve()
               != (initial_root / 'source_predictor.pt').resolve()):
        raise ValueError('complete deterministic matched source-teaching reference required')
    history = done.get('history')
    if (not isinstance(history, list) or len(history) != steps
            or any(not isinstance(item, dict) or not isinstance(item.get('recorded_note'), dict)
                   or any(key not in item['recorded_note'] for key in NOTE_KEYS) for item in history)):
        raise ValueError('complete recorded source task history required')
    recorded = [deepcopy(item['recorded_note']) for item in history]
    excluded_list = prior.get('excluded_outer_shots', [])
    if len(excluded_list) != 46 or len(set(excluded_list)) != 46:
        raise ValueError('the reference 46 excluded source shots are required')
    for path in (initial_root / 'source_predictor.pt', reference_fit / 'source_recorded_sampling.pt',
                 reference_fit / 'prepared_recorded_sampling.pt'):
        if not path.is_file():
            raise FileNotFoundError(f'incomplete source-teaching reference: {path}')

    j.r.configure()
    torch.use_deterministic_algorithms(True)
    base, _ = j.r.load_parent(j.r.DEFAULT_PARENT)
    base.to(device)
    original = deepcopy(base.state_dict())
    source, ids = j.load_source_train()
    excluded = set(excluded_list)
    if ids != prior.get('source_train_ids') or not excluded.issubset(ids):
        raise ValueError('canonical source TRAIN membership and exclusions changed')
    by_id = {shot['shot']: shot for shot in source if shot['shot'] not in excluded}
    if any(note['shot'] not in by_id for note in recorded):
        raise ValueError('recorded task is outside the eligible source TRAIN shots')
    # Validate every recorded episode, including its clock, valid mask and native labels,
    # before creating an output directory or running any teaching step.
    for note in recorded:
        sequence_from_note(by_id[note['shot']], note, 'cpu')
    _, cfg, _, (_, _, warm) = j.cohort('fixed360')
    mapping = j.r.read(j.r.COLD / 'input_map.json')
    cold20 = cfg['order'][:20]
    if (warm != 20 or len(cold20) != 20 or mapping.get('calibration_ids') != cold20
            or prior.get('cold20') != cold20 or j.FIT_BATCH != 32 or j.FIT_LR != 3e-4):
        raise ValueError('unchanged original20 signal preparation and fixed mapping required')
    decoder = MetaOrderedTrajectory(base.center, base.scale, dim=base.cfg.dim, scenarios=4).to(device).eval()
    decoder.load_state_dict(torch.load(initial_root / 'source_predictor.pt',
                                       map_location=device, weights_only=True))
    template = decoder.state_dict()
    references = {phase: torch.load(reference_fit / f'{phase}_recorded_sampling.pt',
                                    map_location=device, weights_only=True)
                  for phase in ('source', 'prepared')}
    for state in references.values():
        if (set(state) != set(template) or any(not isinstance(state[k], torch.Tensor)
                or state[k].shape != template[k].shape or state[k].dtype != template[k].dtype
                for k in template)):
            raise ValueError('incomplete or incompatible recorded-control checkpoint')
    models = {name: FeatureIntra(base, decoder) for name in RULES}
    for model in models.values():
        model.decoder.requires_grad_(True)
    optimizers = {name: torch.optim.AdamW(model.decoder.parameters(), lr=3e-4, weight_decay=0.)
                  for name, model in models.items()}

    out.mkdir(parents=True, exist_ok=False)
    files = (Path(__file__), j.r.ROOT / 'risk_meta_sequence_20261108.py',
             j.r.ROOT / 'risk_meta_adam_20261108.py', j.r.ROOT / 'risk_feature_intra_20261107.py',
             j.r.ROOT / 'risk_meta_step_20261107.py', j.r.ROOT / 'run_joint_trajectory_20261107.py')
    j.r.write_new(out / 'protocol.json', dict(status='SOURCE_DIRECTION_TEACHING',
        parent_sha256=j.PARENT_SHA256, source_train_ids=ids, excluded_outer_shots=sorted(excluded),
        reference_fit=str(reference_fit), initial_predictor=str(initial_root / 'source_predictor.pt'),
        source_steps=steps, sequence_steps=4, batch=1, lr=3e-4, weight_decay=0.,
        initial_predictor_source_seed=j.SEED, source_tasks='replay recorded notes; no resampling',
        step_rules=RULES, task_census=census(recorded),
        objective='four-feedback mean signal energy plus last-step native future-label risk loss',
        feedback_horizon=25, source_labels='native J-TEXT; y(query) and y(query+25)',
        intra_steps='temporary risk weight; matched AdamW length, clip1, .01 shot-start prior',
        cold20=cold20, cold_steps=128, cold_batch=32, cold_seed=j.SEED+1,
        risk_classifier_training=False, east_label_fitting=False, jtext_online=False,
        source_val_or_test_selection=False, deterministic_algorithms=True,
        independent_confirmation=False, source_files={str(path): path.read_text() for path in files}))

    records = []
    with sdpa_kernel(SDPBackend.MATH):
        for step, note in enumerate(recorded, 1):
            shot = by_id[note['shot']]
            episodes, qx, qa, label, _ = sequence_from_note(shot, note, device)
            losses = {}
            for name, rule in RULES.items():
                model = models[name]
                model.zero_grad(set_to_none=True)
                result = teacher_sequence(model, episodes, qx, qa, label,
                                          credit='last', step_rule=rule)
                objective = result['risk_loss'] + result['signal_loss']
                if not torch.isfinite(objective):
                    raise FloatingPointError('nonfinite source-direction teaching objective')
                objective.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1., error_if_nonfinite=True)
                optimizers[name].step()
                losses[name] = dict(signal=float(result['signal_loss'].detach()),
                                    risk=float(result['risk_loss'].detach()), gradient_norm=float(norm))
            records.append(dict(step=step, recorded_note=note,
                native_current_label=float(shot['y'][note['query']]),
                native_future_label=float(shot['y'][note['label_time']]),
                feedback_horizons=[dict(origin=origin, start=origin+1, end=origin+25)
                                   for origin in note['origins']], losses=losses))
            if step % 64 == 0:
                print('paired direction source', step, '/', steps, losses, flush=True)
        if not _same_state(models['control'].decoder.state_dict(), references['source']):
            raise ValueError('control decoder does not exactly reproduce the recorded source reference')
        for name, model in models.items():
            with (out / f'source_{name}.pt').open('xb') as handle:
                torch.save(model.decoder.state_dict(), handle)
        with patch.object(j.r.io23, 'read_outcome',
                          side_effect=RuntimeError('EAST outcomes unavailable for auxiliary preparation')):
            cold = [j.r.prepare_actual(j.raw_for(sid, cfg['raw_sha256']), 21, mapping) for sid in cold20]
            preparation = {name: j.fit_predictor(base, model.decoder, cold, 128, j.SEED+1, f'ended20-{name}')
                           for name, model in models.items()}
    if not _same_state(models['control'].decoder.state_dict(), references['prepared']):
        raise ValueError('control decoder does not exactly reproduce the recorded ended20 reference')
    if any(not _same_state(model.state_dict(), original)
           for model in (base, *(model.risk for model in models.values()))):
        raise ValueError('qualified source risk template changed during auxiliary teaching')
    for name, model in models.items():
        with (out / f'prepared_{name}.pt').open('xb') as handle:
            torch.save(model.decoder.state_dict(), handle)
    j.r.write_new(out / 'fit.json', dict(status='SOURCE_DIRECTION_TEACHER_FIT_COMPLETE',
        history=records, preparation=preparation, source_control_exact=True, prepared_control_exact=True,
        source_classifier_unchanged=True, seconds=time.monotonic()-started))
    print('paired direction source fit complete', round(time.monotonic()-started, 2), 's', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', nargs='?', type=Path, default=OUT)
    fit(parser.parse_args().out)
