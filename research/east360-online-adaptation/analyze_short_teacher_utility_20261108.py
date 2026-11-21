"""Paired CPU diagnostic of prepared signal teachers on excluded source shots."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import run_joint_trajectory_20261107 as j
from experiments.risk_aligned_online import io23, simple_data
from joint_motion_20261107 import components
from risk_feature_intra_20261107 import FeatureIntra, update
from risk_meta_step_20261107 import MetaOrderedTrajectory
from risk_short_context_20261108 import GROUPS, context_pools, sequence_from_note


ROOT = Path(__file__).resolve().parent
FIT = ROOT / 'probe_outputs/risk_short_context_source_20261108_v1'
OUTPUT = FIT / 'utility_cpu.json'
SEED = 147
PER_GROUP = 16
COMMAND = (
    "env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "
    "MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 "
    "/home/hetaisheng/miniconda3/envs/rl_env/bin/python -B "
    "analyze_short_teacher_utility_20261108.py"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_source_shots(protocol, fit):
    roles = simple_data.source_roles()
    train = roles['train']
    excluded = protocol['excluded_outer_shots']
    require(len(excluded) == len(set(excluded)) == 46, '46 distinct excluded shots required')
    require(train == protocol['source_train_ids'], 'canonical TRAIN membership or order changed')
    require(set(excluded) <= set(train), 'excluded shot is not in canonical TRAIN')
    require(protocol['sequence_steps'] == 4 and protocol['source_steps'] == 512,
            'four-step completed source fit required')
    require(protocol['parent_sha256'] == j.PARENT_SHA256, 'qualified parent mismatch')
    require(not protocol['jtext_online'] and not protocol['risk_classifier_training']
            and not protocol['east_label_fitting'] and not protocol['source_val_or_test_selection'],
            'unexpected source fit scope')
    require(fit['status'] == 'SOURCE_SHORT_CONTEXT_FIT_COMPLETE'
            and fit['source_control_exact'] and fit['prepared_control_exact']
            and fit['source_classifier_unchanged'], 'completed matched source fit required')
    history = fit['history']
    require(len(history) == protocol['source_steps'], 'incomplete source fit history')
    excluded_set = set(excluded)
    for arm in ('control', 'short_context'):
        used = {int(row[arm]['shot']) for row in history}
        require(used <= set(train) and not used & excluded_set,
                f'{arm} fit history violates source exclusion')

    all_advanced = simple_data.advanced_times()
    advanced = {sid: all_advanced[sid] for sid in excluded if sid in all_advanced}
    shots, audit = [], []
    for sid in sorted(excluded):
        shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        require(shot['shot'] == sid and shot['device'] == 'J-TEXT', 'shot identity changed')
        horizon, raw_advanced, fallback = io23.effective_horizon_ms(sid, advanced)
        require(shot['horizon'] == horizon and np.isfinite(shot['times']).all()
                and np.all(np.diff(shot['times']) > 0), 'invalid native label horizon or clock')
        if shot['is_disrupt']:
            remaining_ms = (shot['down'] - shot['times']) * 1000
            valid = remaining_ms >= -1e-6
            expected_y = ((remaining_ms <= horizon + 1e-6) & valid).astype(np.float32)
        else:
            valid = np.ones(len(shot['times']), dtype=bool)
            expected_y = np.zeros(len(shot['times']), dtype=np.float32)
        require(np.array_equal(shot['valid'], valid) and np.array_equal(shot['y'], expected_y),
                'native HDF5 outcome label contract mismatch')
        shots.append(shot)
        audit.append(dict(shot=sid, is_disrupt=bool(shot['is_disrupt']),
                          down_time_s=shot['down'], advanced_time_ms=raw_advanced,
                          advanced_time_missing=bool(fallback), horizon_ms=float(horizon),
                          samples=len(shot['times'])))
    return shots, audit


def verify_single_hold_loss(decoder):
    # K=1, constant available issuance values, and nonconstant observed truth.
    truth = torch.linspace(-1., 1., 25).reshape(1, 25, 1).expand(1, 25, 23).clone()
    hold = torch.zeros(1, 1, 25, 23)
    probabilities = torch.ones(1, 1)
    mask = torch.ones(1, 25, 23, dtype=torch.bool)
    mask[:, ::3, 1::2] = False
    direct = decoder.loss(hold, probabilities, truth, mask)
    level, change = components(hold, probabilities, truth, mask)
    torch.testing.assert_close(direct, level + decoder.motion_weight * change, rtol=1e-6, atol=1e-7)
    return dict(scenarios=1, score=float(direct), level=float(level), lag8=float(change),
                available_count=int(mask.sum()))


def sample_notes(shots):
    pools = context_pools(shots, minimum_origin=7)
    rng = np.random.default_rng(SEED)
    notes = []
    for i in range(PER_GROUP * len(GROUPS)):
        group = GROUPS[i % len(GROUPS)]
        rows = pools[group]
        shot_index, origins = rows[int(rng.integers(len(rows)))]
        first = int(rng.choice(origins))
        q = first + 122
        notes.append(dict(shot=shots[shot_index]['shot'],
                          origins=[first + 32 * k for k in range(4)],
                          query=q, label_time=q + 25, group=group))
    require(all(sum(n['group'] == group for n in notes) == PER_GROUP for group in GROUPS),
            'unbalanced or incomplete prespecified task budget')
    return notes, {group: len(pools[group]) for group in GROUPS}


def risk_metrics(model, query, query_mask, current, future):
    with torch.enable_grad():
        logit = model.risk(query, query_mask)['logit']
        current_label = torch.full_like(logit, float(current))
        future_label = torch.full_like(logit, float(future))
        result = dict(probability=float(logit.sigmoid().detach()),
                      current_bce=float(F.binary_cross_entropy_with_logits(logit, current_label).detach()),
                      future_bce=float(F.binary_cross_entropy_with_logits(logit, future_label).detach()))
    require(all(math.isfinite(v) for v in result.values()), 'nonfinite risk metric')
    return result


def measured_scores(model, episode):
    require(set(episode) == {'x', 'a', 'truth', 'mask'}, 'inner episode must not contain labels')
    with torch.enable_grad():
        output = model(episode['x'], episode['a'])
        truth = episode['truth']
        mask = episode['mask'] & output['available'] & torch.isfinite(truth)
        issued = episode['a'][:, -1] & torch.isfinite(episode['x'][:, -1])
        hold_mask = episode['mask'] & issued[:, None, :] & torch.isfinite(truth)
        require(torch.equal(mask, hold_mask), 'G and hold availability masks differ')
        require(int(mask.sum()) > 0 and int((mask[:, 8:] & mask[:, :-8]).sum()) > 0,
                'observed H25 and lag-8 targets required')
        center, scale = model.risk.center, model.risk.scale
        target = torch.asinh((torch.where(mask, truth, center) - center) / scale)
        signal = model.decoder.loss(output['encoded'], output['probabilities'], target, mask).mean()
        observed_issuance = torch.where(issued, episode['x'][:, -1], center)
        hold = torch.asinh((observed_issuance - center) / scale)[:, None, None, :]
        hold = hold.expand(-1, 1, truth.shape[1], -1)
        hold_probabilities = torch.ones((len(truth), 1), dtype=truth.dtype, device=truth.device)
        hold_score = model.decoder.loss(hold, hold_probabilities, target, mask).mean()

    level, change = components(output['encoded'].detach(), output['probabilities'].detach(),
                               target.detach(), mask)
    hold_level, hold_change = components(hold.detach(), hold_probabilities, target.detach(), mask)
    weight = model.decoder.motion_weight
    torch.testing.assert_close(signal.detach(), (level + weight * change).mean(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(hold_score.detach(), (hold_level + weight * hold_change).mean(),
                               rtol=1e-6, atol=1e-7)
    result = dict(signal_energy=float(signal.detach()), hold_energy=float(hold_score.detach()),
                  signal_level=float(level.mean()), signal_lag8=float(change.mean()),
                  hold_level=float(hold_level.mean()), hold_lag8=float(hold_change.mean()),
                  issued_channels=int(issued.sum()), observed_elements=int(mask.sum()),
                  observed_lag8_pairs=int((mask[:, 8:] & mask[:, :-8]).sum()))
    require(all(math.isfinite(v) for v in result.values()), 'nonfinite H25 score or counts')
    return result


def run_arm(base, decoder, episodes, query, query_mask, current, future):
    model = FeatureIntra(base, decoder)
    require(torch.equal(model.weight, base.encoder.layers[0].linear2.weight),
            'episode does not start from the qualified parent')
    require([name for name, p in model.named_parameters() if p.requires_grad]
            == ['risk.encoder.layers.0.linear2.weight'], 'unexpected trainable episode state')
    optimizer = torch.optim.AdamW([model.weight], lr=j.INTRA_LR, weight_decay=0.)
    before = risk_metrics(model, query, query_mask, current, future)
    steps = []
    for episode in episodes:
        row = measured_scores(model, episode)
        actual = update(model, optimizer, episode)
        difference = abs(row['signal_energy'] - actual['signal_loss'])
        require(difference <= 1e-7, 'measured signal differs from actual update dispatch')
        require(math.isfinite(actual['gradient_norm']) and actual['gradient_norm'] > 0,
                'finite nonzero update gradient required')
        row.update(update=actual, measurement_update_abs_error=difference)
        steps.append(row)
    after = risk_metrics(model, query, query_mask, current, future)
    displacement = float((model.weight - model.initial_weight).norm().detach())
    require(math.isfinite(displacement) and displacement > 0, 'nonzero finite W displacement required')
    require(len(steps) == 4, 'exactly four updates required')
    return dict(before=before, after=after, weight_displacement_l2=displacement,
                mean_signal_energy=float(np.mean([v['signal_energy'] for v in steps])),
                mean_hold_energy=float(np.mean([v['hold_energy'] for v in steps])),
                steps=steps)


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    require(len(values) > 0 and np.isfinite(values).all(), 'empty or nonfinite summary')
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else None)


def direction_counts(values):
    return dict(improved=sum(v < 0 for v in values), worsened=sum(v > 0 for v in values),
                unchanged=sum(v == 0 for v in values))


def summarize(records):
    result = {}
    for group in ('all', *GROUPS):
        selected = records if group == 'all' else [v for v in records if v['group'] == group]
        out = dict(tasks=len(selected), distinct_shots=len({v['shot'] for v in selected}),
                   first_origin_below_127=sum(v['origins'][0] < 127 for v in selected), arms={})
        for arm in ('control', 'short_context'):
            rows = [v['arms'][arm] for v in selected]
            signal_deltas = [v['mean_signal_energy'] - v['mean_hold_energy'] for v in rows]
            future_deltas = [v['after']['future_bce'] - v['before']['future_bce'] for v in rows]
            current_deltas = [v['after']['current_bce'] - v['before']['current_bce'] for v in rows]
            out['arms'][arm] = dict(
                signal_energy=mean_sd([v['mean_signal_energy'] for v in rows]),
                hold_energy=mean_sd([v['mean_hold_energy'] for v in rows]),
                signal_minus_hold=mean_sd(signal_deltas),
                signal_vs_hold=direction_counts(signal_deltas),
                future_bce_before=mean_sd([v['before']['future_bce'] for v in rows]),
                future_bce_after=mean_sd([v['after']['future_bce'] for v in rows]),
                future_bce_after_minus_before=mean_sd(future_deltas),
                future_bce_direction=direction_counts(future_deltas),
                current_bce_before=mean_sd([v['before']['current_bce'] for v in rows]),
                current_bce_after=mean_sd([v['after']['current_bce'] for v in rows]),
                current_bce_after_minus_before=mean_sd(current_deltas),
                current_bce_direction=direction_counts(current_deltas),
                probability_before=mean_sd([v['before']['probability'] for v in rows]),
                probability_after=mean_sd([v['after']['probability'] for v in rows]),
                weight_displacement_l2=mean_sd([v['weight_displacement_l2'] for v in rows]))
        out['short_minus_control'] = dict(
            signal_energy=mean_sd([v['arms']['short_context']['mean_signal_energy']
                                   - v['arms']['control']['mean_signal_energy'] for v in selected]),
            future_bce_delta=mean_sd([
                (v['arms']['short_context']['after']['future_bce']
                 - v['arms']['short_context']['before']['future_bce'])
                - (v['arms']['control']['after']['future_bce']
                   - v['arms']['control']['before']['future_bce']) for v in selected]))
        result[group] = out
    return result


def main():
    require(len(sys.argv) == 1 and Path.cwd().resolve() == ROOT, 'run from the worktree without arguments')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU must be disabled for this CPU diagnostic')
    require(not OUTPUT.exists(), 'refusing to overwrite an existing output')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    started = time.monotonic()
    protocol = json.loads((FIT / 'protocol.json').read_text())
    fit = json.loads((FIT / 'fit.json').read_text())
    shots, shot_audit = native_source_shots(protocol, fit)
    notes, eligible_shots = sample_notes(shots)
    by_id = {s['shot']: s for s in shots}
    base, _ = j.r.load_parent(j.r.DEFAULT_PARENT)
    base.to('cpu')
    decoders = {}
    for arm in ('control', 'short_context'):
        decoder = MetaOrderedTrajectory(base.center, base.scale, dim=base.cfg.dim, scenarios=4).eval()
        decoder.load_state_dict(torch.load(FIT / f'prepared_{arm}.pt', map_location='cpu', weights_only=True))
        decoder.requires_grad_(False)
        decoders[arm] = decoder
    templates = dict(base=base, **decoders)
    original = {name: deepcopy(model.state_dict()) for name, model in templates.items()}
    single_hold_check = verify_single_hold_loss(decoders['control'])
    records = []
    with sdpa_kernel(SDPBackend.MATH):
        for index, note in enumerate(notes, 1):
            shot = by_id[note['shot']]
            episodes, qx, qa, future, rebuilt = sequence_from_note(shot, note, 'cpu')
            require(rebuilt == note, 'episode identity changed')
            current = float(shot['y'][note['query']])
            future_y = float(shot['y'][note['label_time']])
            require(float(future.item()) == future_y
                    and (current, future_y) == tuple(map(float, note['group'].split('_to_'))),
                    'native labels do not match stratification')
            arms = {name: run_arm(base, decoder, episodes, qx, qa, current, future_y)
                    for name, decoder in decoders.items()}
            require(arms['control']['before'] == arms['short_context']['before'],
                    'paired arms do not share the same source risk before updates')
            for a, b in zip(arms['control']['steps'], arms['short_context']['steps'], strict=True):
                require(abs(a['hold_energy'] - b['hold_energy']) <= 1e-7
                        and a['observed_elements'] == b['observed_elements']
                        and a['observed_lag8_pairs'] == b['observed_lag8_pairs'],
                        'paired episode hold or observation masks differ')
            records.append(dict(**note, origin_times_s=[float(shot['times'][o]) for o in note['origins']],
                                query_time_s=float(shot['times'][note['query']]),
                                label_time_s=float(shot['times'][note['label_time']]),
                                current_label=int(current), future_label=int(future_y), arms=arms))
            if index % 6 == 0:
                print(f'CPU source utility: {index}/{len(notes)} tasks, '
                      f'{time.monotonic() - started:.1f}s', flush=True)
    unchanged = {name: (set(model.state_dict()) == set(original[name])
                        and all(torch.equal(v, original[name][key])
                                for key, v in model.state_dict().items()))
                 for name, model in templates.items()}
    require(all(unchanged.values()), 'a qualified base or decoder template changed')
    require(len(records) == 48 and all(len(v['arms'][a]['steps']) == 4
            for v in records for a in decoders), 'incomplete task/update count')
    max_error = max(step['measurement_update_abs_error'] for row in records
                    for arm in row['arms'].values() for step in arm['steps'])
    result = dict(
        status='SOURCE_TRAIN_EXCLUDED_SHORT_TEACHER_CPU_UTILITY_COMPLETE',
        metadata=dict(command=COMMAND, cwd=str(ROOT), seed=SEED, device='cpu',
                      torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                      mha_fastpath_enabled=torch.backends.mha.get_fastpath_enabled(), sdp_backend='MATH',
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      fit=str(FIT / 'fit.json'), protocol=str(FIT / 'protocol.json'),
                      parent=str(j.r.DEFAULT_PARENT), parent_sha256=protocol['parent_sha256'],
                      checkpoints={arm: str(FIT / f'prepared_{arm}.pt') for arm in decoders},
                      requested_tasks=48, actual_tasks=len(records), requested_per_group=PER_GROUP,
                      budget_reason='full prespecified budget', sampling='seed147; round-robin group, uniform shot then origin',
                      first_origin_minimum=7, step_stride=32, steps_per_task=4, horizon=25,
                      optimizer='AdamW lr=1e-4 weight_decay=0, clip norm=1, W-start prior coefficient=0.01',
                      signal_score='H25 joint energy level + motion_weight * lag8-change energy, source asinh',
                      hold='one scenario: repeat observed issuance values; same issuance/future availability mask',
                      aggregation='mean four steps per task, then equal mean over tasks; SD is across sampled tasks',
                      probability='same query risk probability assessed against current and q+25 native labels',
                      label_contract='HDF5 IsDisrupt/DownTime and native max(25ms, AdvancedTime), loader clock',
                      source_hdf5_shots_loaded=len(shots), eligible_shots_by_group=eligible_shots,
                      source_role_ids_read_for_membership_only=True, source_val_test_hdf5_loaded=False,
                      labels_in_inner_episode=False, source_online=False, east_stream_read=False,
                      original_signal_pretraining_saw_all_train=True,
                      prepared_decoders_used_ended_east20_signals=True,
                      independent_confirmation=False, east_efficacy_evaluation=False,
                      elapsed_seconds=time.monotonic() - started),
        verification=dict(canonical_train_ids_match_fit=True, excluded_count=46,
                          all_excluded_in_canonical_train=True,
                          fit_history_excludes_all_46_for_both_arms=True,
                          native_labels_reconstructed_from_hdf5_outcomes=True,
                          single_scenario_hold=single_hold_check,
                          same_episodes_queries_labels_and_masks=True,
                          all_384_update_gradients_finite_nonzero=True,
                          measurement_update_max_abs_error=max_error,
                          templates_torch_equal_unchanged=unchanged),
        source_shot_label_audit=shot_audit, summary=summarize(records), records=records)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with OUTPUT.open('x', encoding='utf-8') as handle:
        handle.write(serialized)
    print(f'completed: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
