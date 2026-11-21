"""Post-seal four-step local direction diagnostic on the fixed static errors.

The oracle is a hindsight-labelled reference, not an online rule or an upper
bound. All adaptation starts from the same fresh qualified source weights.
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
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
from risk_feature_intra_20261107 import FeatureIntra, update
from risk_meta_step_20261107 import MetaOrderedTrajectory


ROOT = Path(__file__).resolve().parent
SEALED = ROOT / 'probe_outputs/risk_positive_pool360_20261108_v2'
DIAGNOSTIC = SEALED / 'error_margin_diagnostic_20261108.json'
FIT = ROOT / 'probe_outputs/risk_short_context_source_20261108_v1'
OUTPUT = FIT / 'east_local_direction_cpu.json'
THRESHOLD = 0.7258946895599365
ARMS = ('SSL_control', 'SSL_short', 'oracle_4steps')
WEIGHT = 'encoder.layers.0.linear2.weight'
EXPECTED = Counter({'tardy': 38, 'miss': 26, 'fp': 29})


def need(condition, message):
    if not condition:
        raise ValueError(message)


def configure_cpu():
    for key, value in {'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                       'MKL_NUM_THREADS': '1'}.items():
        need(os.environ.get(key) == value, f'{key} must be {value!r}')
    random.seed(149)
    np.random.seed(149)
    torch.manual_seed(149)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def check_state(module, reference, *, excluded=()):
    state = module.state_dict()
    need(set(state) == set(reference), 'template tensor keys changed')
    for name, old in reference.items():
        if name not in excluded:
            need(torch.equal(state[name], old), f'template tensor changed: {name}')


def checkpoint_files():
    return {'source': j.r.DEFAULT_PARENT,
            'control': FIT / 'prepared_control.pt',
            'short': FIT / 'prepared_short_context.pt'}


def load_inputs():
    diagnostic = j.r.read(DIAGNOSTIC)
    protocol = j.r.read(SEALED / 'protocol.json')
    completed = j.r.read(SEALED / 'completed.json')
    fit_protocol = j.r.read(FIT / 'protocol.json')
    fit_done = j.r.read(FIT / 'fit.json')
    _, cfg, archive, (_, _, warm) = j.cohort('fixed360')
    mapping_path = j.r.COLD / 'input_map.json'
    mapping = j.r.read(mapping_path)
    need(diagnostic['status'] == 'SEALED_FIXED360_CPU_ERROR_MARGIN_DIAGNOSTIC_COMPLETE',
         'the error diagnostic is not sealed')
    need(protocol['status'] == 'RISK_TEACHER_INTRA360'
         and completed['status'] == 'SEALED_RISK_TEACHER360', 'reference run is not sealed')
    need(fit_done['status'] == 'SOURCE_SHORT_CONTEXT_FIT_COMPLETE'
         and fit_done['source_control_exact'] and fit_done['prepared_control_exact']
         and fit_done['source_classifier_unchanged'], 'source fit is not complete')
    need(protocol['parent_sha256'] == fit_protocol['parent_sha256'] == j.PARENT_SHA256,
         'qualified parent differs')
    need(protocol['threshold'] == diagnostic['definitions']['threshold'] == THRESHOLD
         and protocol['warm'] == warm == 20, 'threshold or preparation period differs')
    need(protocol['order'] == cfg['order'] == [v['shot'] for v in archive['sealed']]
         == [v['shot'] for v in completed['records']]
         == [v['shot'] for v in diagnostic['shots']], 'fixed cohort order differs')
    need(mapping['calibration_ids'] == cfg['order'][:20]
         and j.r.sha(mapping_path) == cfg['input_sha256'][str(mapping_path)],
         'fixed original input map differs')
    need(fit_protocol['cold20'] == cfg['order'][:20]
         and not fit_protocol['risk_classifier_training']
         and not fit_protocol['east_label_fitting'], 'source fit contract differs')
    errors = [row for row in diagnostic['shots'] if row['static_score']['status'] in EXPECTED]
    need(Counter(row['static_score']['status'] for row in errors) == EXPECTED,
         'static error population differs from 38 tardy, 26 miss, 29 fp')
    need(all(row['rank'] == cfg['order'].index(row['shot']) + 1 for row in errors),
         'diagnostic rank differs from fixed order')
    return cfg, mapping, errors


def choose_and_verify(row, cfg):
    """Verify original HDF/clock and the prescribed sealed peak, even if excluded."""
    rank, shot = row['rank'], row['shot']
    category = row['static_score']['status']
    truth, clock = j.raw_truth(j.r.io23.path_for('EAST', shot), shot)
    need(truth['disruptive'] == row['hdf_is_disrupt']
         and truth['down'] == row['hdf_down_time_s'], f'HDF outcome differs: {shot}')
    need(truth['disruptive'] == (category != 'fp'), f'error category/outcome mismatch: {shot}')
    raw = j.raw_for(shot, cfg['raw_sha256'])
    np.testing.assert_array_equal(raw['times'], clock, err_msg=f'original clock differs: {shot}')
    need(len(clock) == row['clock_points'], f'clock length differs: {shot}')
    with np.load(SEALED / 'shots' / f'{rank:04d}_{shot}.npz', allow_pickle=False) as saved:
        static = saved['static'].copy()
    need(static.shape == clock.shape and np.isfinite(static).all()
         and ((0 <= static) & (static <= 1)).all(), f'invalid sealed static curve: {shot}')
    need(j.score(clock, static, truth, THRESHOLD) == row['static_score'],
         f'sealed static classification differs: {shot}')

    if truth['disruptive']:
        leads = np.round((truth['down'] - clock) * 1000., 9)
        indices = np.flatnonzero((leads > 30.) & (leads <= 125.))
        window = row['windows']['early_positive']
        selection = 'disruptive_static_peak_30_lt_lead_le_125_ms'
    else:
        leads = None
        indices = np.arange(len(clock))
        window = row['windows']['eligible_history_or_safe_full']
        selection = 'safe_static_peak_full_observation'
    need(window is not None and window['point_count'] == len(indices),
         f'sealed peak window differs: {shot}')
    q = int(indices[np.argmax(static[indices])]) if len(indices) else None
    peak = window['static']
    if q is not None:
        lead = float(leads[q]) if leads is not None else None
        need(peak['max_index'] == q and peak['max_probability'] == float(static[q])
             and peak['max_time_s'] == float(clock[q]) and peak['max_lead_ms'] == lead,
             f'sealed peak differs from HDF clock or static curve: {shot}')
        y, valid = j.r.io23.labels_at(clock[q:q+1], truth['disruptive'], truth['down'], 125.)
        need(bool(valid[0]) and int(y[0]) == int(truth['disruptive']),
             f'query is not a valid native EAST label: {shot}')
        native_y = int(y[0])
        need((float(static[q]) < THRESHOLD) if native_y else (float(static[q]) >= THRESHOLD),
             f'query is not on the static error side: {shot}')
    else:
        lead, native_y = None, None
        need(peak is None, f'empty peak window has a peak: {shot}')
    reasons = []
    if rank <= 20:
        reasons.append('rank_le_20_no_policy_updates')
    if q is None:
        reasons.append('empty_prescribed_query_window')
    elif q < 129:
        reasons.append('query_lt_129_four_step_history_unavailable')
    case = dict(rank=rank, shot=shot, static_status=category,
                hdf_is_disrupt=truth['disruptive'], hdf_down_time_s=truth['down'],
                query_selection=selection, query_index=q,
                query_time_s=float(clock[q]) if q is not None else None,
                query_lead_ms=lead, native_y_at_query=native_y,
                sealed_static_probability=float(static[q]) if q is not None else None,
                exclusion_reasons=reasons)
    return case, raw


def make_episode(signal, origin, q):
    need(0 <= origin and origin + 25 < q, 'feedback target is not strictly before query')
    x, a = j.simple_data.window(signal, origin, 128)
    truth = signal['values'][origin+1:origin+26]
    mask = signal['available'][origin+1:origin+26]
    need(truth.shape == mask.shape == (25, 23), 'incomplete actual H25 target')
    return {name: torch.as_tensor(value[None]) for name, value in
            {'x': x, 'a': a, 'truth': truth, 'mask': mask}.items()}


def output_at_query(model, x, a, label):
    with torch.no_grad():
        logit = model.risk(x, a)['logit']
        loss = F.binary_cross_entropy_with_logits(logit, label)
        return {'logit': float(logit.item()), 'probability': float(logit.sigmoid().item()),
                'bce': float(loss.item())}


def cosine(a, b):
    a, b = a.detach().double().reshape(-1), b.detach().double().reshape(-1)
    denominator = a.norm() * b.norm()
    return float(torch.dot(a, b) / denominator) if denominator > 0 else None


def run_case(case, raw, mapping, base, decoders, source_state, decoder_states):
    q, rank = case['query_index'], case['rank']
    # Deliberately present only the prefix through q to the original fixed map.
    prefix = dict(raw, values=raw['values'][:q+1], available=raw['available'][:q+1],
                  times=raw['times'][:q+1])
    signal = j.r.prepare_actual(prefix, rank, mapping)
    need(len(signal['values']) == q+1 and not signal['available'][:, 2].any(),
         'prefix or native EAST MNM mask changed')
    origins = [q-122, q-90, q-58, q-26]
    need(origins[0] >= 7 and all(b-a == 32 for a, b in zip(origins, origins[1:])),
         'invalid four-step clock')
    episodes = [make_episode(signal, origin, q) for origin in origins]
    x, a = j.simple_data.window(signal, q, 128)
    x, a = torch.as_tensor(x[None]), torch.as_tensor(a[None])
    label = torch.tensor([float(case['native_y_at_query'])], dtype=torch.float32)
    models = {'SSL_control': FeatureIntra(base, decoders['control']),
              'SSL_short': FeatureIntra(base, decoders['short']),
              'oracle_4steps': FeatureIntra(base, decoders['control'])}
    before = {name: output_at_query(model, x, a, label) for name, model in models.items()}
    need(before['SSL_control'] == before['SSL_short'] == before['oracle_4steps'],
         f'arm starting outputs differ: {case["shot"]}')
    reference = case['sealed_static_probability']
    difference = abs(before['SSL_control']['probability'] - reference)
    need(difference <= 5e-6 and
         (before['SSL_control']['probability'] >= THRESHOLD) == (reference >= THRESHOLD),
         f'CPU starting probability fails sealed parity: {case["shot"]}: {difference}')
    for model in models.values():
        need(torch.equal(model.weight, source_state[WEIGHT])
             and torch.equal(model.initial_weight, source_state[WEIGHT]),
             'arm does not start at qualified source W')

    steps, first_gradients = {}, {}
    for name in ('oracle_4steps', 'SSL_control', 'SSL_short'):
        model = models[name]
        optimizer = torch.optim.AdamW([model.weight], lr=1e-4, weight_decay=0., foreach=False)
        arm_steps = []
        for index in range(4):
            weight_before = model.weight.detach().clone()
            captured = []
            hook = model.weight.register_hook(lambda grad: captured.append(grad.detach().clone()))
            try:
                if name == 'oracle_4steps':
                    optimizer.zero_grad(set_to_none=True)
                    logit = model.risk(x, a)['logit']
                    bce = F.binary_cross_entropy_with_logits(logit, label)
                    prior = (model.weight-model.initial_weight).square().sum()
                    objective = bce + .01 * prior
                    need(bool(torch.isfinite(objective)), 'nonfinite oracle objective')
                    objective.backward()
                    norm = torch.nn.utils.clip_grad_norm_([model.weight], 1., error_if_nonfinite=True)
                    optimizer.step()
                    stats = {'query_bce': float(bce.detach()), 'prior': float(prior.detach()),
                             'objective': float(objective.detach()), 'gradient_norm': float(norm)}
                else:
                    stats = update(model, optimizer, episodes[index], feedback='real')
            finally:
                hook.remove()
            need(len(captured) == 1 and torch.isfinite(captured[0]).all(), 'invalid actual gradient')
            captured_norm = float(captured[0].norm())
            need(abs(captured_norm-stats['gradient_norm']) <= 1e-6 * max(1., captured_norm),
                 'recorded gradient norm differs from actual unclipped gradient')
            if index == 0:
                first_gradients[name] = captured[0]
            stats.update(step=index+1, origin=origins[index] if name != 'oracle_4steps' else None,
                         gradient_norm_after_clip=float(model.weight.grad.norm()),
                         actual_step_norm=float((model.weight.detach()-weight_before).norm()))
            need(all(np.isfinite(v) for v in stats.values() if isinstance(v, float)),
                 'nonfinite update statistic')
            arm_steps.append(stats)
        steps[name] = arm_steps

    arms = {}
    for name, model in models.items():
        after = output_at_query(model, x, a, label)
        initial = before[name]
        positive = bool(case['native_y_at_query'])
        gap = (THRESHOLD-reference if positive else reference-THRESHOLD)
        cpu_gap = (THRESHOLD-initial['probability'] if positive
                   else initial['probability']-THRESHOLD)
        progress = (after['probability']-initial['probability'] if positive
                    else initial['probability']-after['probability'])
        correct = (after['probability'] >= THRESHOLD if positive
                   else after['probability'] < THRESHOLD)
        arms[name] = dict(before=initial, after=after, local_cross=bool(correct),
                          initial_probability_gap=float(gap), starting_cpu_probability_gap=float(cpu_gap),
                          signed_gap_closed_fraction=float(progress/gap) if gap > 0 else None,
                          weight_displacement_norm=float((model.weight-model.initial_weight).norm().detach()),
                          first_ssl_vs_query_gradient_cosine=(
                              cosine(first_gradients[name], first_gradients['oracle_4steps'])
                              if name != 'oracle_4steps' else None), steps=steps[name])
        check_state(model.risk, source_state, excluded=(WEIGHT,))
        key = 'short' if name == 'SSL_short' else 'control'
        check_state(model.decoder, decoder_states[key])
    return dict(**case, original_peak_probability_difference=difference,
                origins=origins, actual_H25_target_ranges=[[o+1, o+25] for o in origins],
                arms=arms)


def distribution(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {'n': 0, 'min': None, 'median': None, 'mean': None, 'max': None}
    return {'n': len(values), 'min': min(values), 'median': float(np.median(values)),
            'mean': float(np.mean(values)), 'max': max(values)}


def summarize(cases):
    summaries = {}
    for group in ('all', 'tardy', 'miss', 'fp'):
        rows = [v for v in cases if group == 'all' or v['static_status'] == group]
        summaries[group] = {}
        for name in ARMS:
            items = [row['arms'][name] for row in rows]
            fractions = [v['signed_gap_closed_fraction'] for v in items]
            summaries[group][name] = dict(
                n=len(items), local_cross=sum(v['local_cross'] for v in items),
                bce_improved=sum(v['after']['bce'] < v['before']['bce'] for v in items),
                toward_correct_side=sum(f is not None and f > 0 for f in fractions),
                away_from_correct_side=sum(f is not None and f < 0 for f in fractions),
                signed_gap_closed_fraction=distribution(fractions),
                probability_change=distribution([v['after']['probability']-v['before']['probability'] for v in items]),
                weight_displacement_norm=distribution([v['weight_displacement_norm'] for v in items]),
                first_ssl_vs_query_gradient_cosine=distribution(
                    [v['first_ssl_vs_query_gradient_cosine'] for v in items]))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='check one D and one safe FP; do not write')
    args = parser.parse_args()
    started = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    configure_cpu()
    need(not OUTPUT.exists(), f'refusing to overwrite {OUTPUT}')
    cfg, mapping, errors = load_inputs()
    files = checkpoint_files()
    file_hashes = {name: j.r.sha(path) for name, path in files.items()}
    base, parent = j.r.load_parent(files['source'])
    need(parent['threshold'] == THRESHOLD, 'qualified parent threshold differs')
    decoders = {}
    for name in ('control', 'short'):
        decoder = MetaOrderedTrajectory(base.center, base.scale, dim=base.cfg.dim, scenarios=4).eval()
        decoder.load_state_dict(torch.load(files[name], map_location='cpu', weights_only=True), strict=True)
        decoders[name] = decoder.requires_grad_(False)
    source_state = snapshot(base)
    decoder_states = {name: snapshot(model) for name, model in decoders.items()}
    cases, excluded, preflight_groups = [], [], set()
    with sdpa_kernel(SDPBackend.MATH):
        for row in errors:
            # A no-write preflight checks one eligible D and one eligible safe FP.
            if args.preflight:
                group = 'fp' if row['static_score']['status'] == 'fp' else 'disruptive'
                window = row['windows']['eligible_history_or_safe_full' if group == 'fp' else 'early_positive']
                if group in preflight_groups or row['rank'] <= 20 or not window['point_count']:
                    continue
                if window['static']['max_index'] < 129:
                    continue
            case, raw = choose_and_verify(row, cfg)
            if case['exclusion_reasons']:
                excluded.append(case)
                continue
            cases.append(run_case(case, raw, mapping, base, decoders, source_state, decoder_states))
            if args.preflight:
                preflight_groups.add(group)
                if len(preflight_groups) == 2:
                    break
            elif len(cases) % 10 == 0:
                print(f'eligible cases completed: {len(cases)}; elapsed {time.monotonic()-started:.1f}s', flush=True)
    check_state(base, source_state)
    for name, model in decoders.items():
        check_state(model, decoder_states[name])
    need({name: j.r.sha(path) for name, path in files.items()} == file_hashes,
         'a source or G checkpoint file changed during the run')
    if args.preflight:
        need(preflight_groups == {'disruptive', 'fp'}, 'preflight did not cover both HDF outcomes')
        print(json.dumps({'status': 'PREFLIGHT_PASSED_NO_OUTPUT',
                          'cases': [(v['rank'], v['shot'], v['query_index']) for v in cases],
                          'max_static_probability_difference': max(v['original_peak_probability_difference'] for v in cases),
                          'seconds': time.monotonic()-started}), flush=True)
        return
    need(len(cases) + len(excluded) == len(errors), 'not all static errors accounted for')
    need(Counter(v['static_status'] for v in cases + excluded) == EXPECTED, 'error counts do not reconcile')
    elapsed = time.monotonic() - started
    result = dict(
        status='COMPLETE_CPU_POSTHOC_LOCAL_DIRECTION',
        command='PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=\'\' OPENBLAS_NUM_THREADS=1 '
                'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ' + sys.executable + ' ' + ' '.join(sys.argv),
        cwd=os.getcwd(), started_utc=start_utc, seconds=elapsed,
        inputs=dict(error_diagnostic=str(DIAGNOSTIC), sealed_curve_directory=str(SEALED / 'shots'),
                    fixed_map=str(j.r.COLD / 'input_map.json'),
                    source_and_G={name: str(path) for name, path in files.items()},
                    source_and_G_sha256=file_hashes),
        contract=dict(threshold=THRESHOLD, seed=149, cpu_threads=1, deterministic_algorithms=True,
                      mha_fastpath=False, sdp_backend='MATH', horizon=25, native_east_horizon_ms=125,
                      plastic_parameter=WEIGHT, steps=4, optimizer='torch AdamW lr=1e-4 wd=0',
                      clip_norm=1., shot_start_squared_weight_prior=.01,
                      feedback_origins='q-122,q-90,q-58,q-26; each actual H25 target ends before q',
                      native_query_only=True, fresh_base_no_prior_TTT=True,
                      signed_gap_closed_fraction='CPU before-to-after movement toward correct side divided by sealed static probability gap; not clipped',
                      local_cross='query crosses from static wrong side to correct side; not a shot-level metric',
                      gradient_cosine='first actual SSL objective gradient versus first actual query BCE gradient; positive means aligned gradient descent directions',
                      interpretation='posthoc error-peak selection; oracle uses query label; no full-stream claim, upper bound, online algorithm, or independent validation'),
        verification=dict(expected_static_errors=dict(EXPECTED), accounted=len(cases)+len(excluded),
                          eligible=len(cases), excluded=len(excluded),
                          original_hdf_truth_clock_and_raw_hash_checked_for_all_errors=True,
                          sealed_static_score_and_prescribed_peak_checked_for_all_errors=True,
                          starting_probability_tolerance=5e-6,
                          max_starting_probability_difference=max(v['original_peak_probability_difference'] for v in cases),
                          same_before_all_arms=True, prefix_only_to_fixed_map_and_four_step_updates=True,
                          source_and_decoder_templates_tensor_unchanged=True,
                          each_arm_nonplastic_source_and_decoder_tensors_unchanged=True,
                          source_and_decoder_checkpoint_files_unchanged=True),
        summary=summarize(cases), cases=cases, exclusions=excluded)
    with OUTPUT.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'eligible': len(cases), 'excluded': len(excluded),
                      'seconds': elapsed, 'output': str(OUTPUT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
