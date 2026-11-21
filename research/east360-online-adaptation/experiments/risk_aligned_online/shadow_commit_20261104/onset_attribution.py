"""CPU-only first-RLS-update attribution for four explicitly authorized shots.

This reads 23 signal datasets only as [:136]; no HDF outcome or full signal is
read. Labels are consulted only from the previously sealed metadata after all
four attributions have been computed. No parameters are fitted or updated.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile

import h5py
import numpy as np
import torch

from .. import io23
from ..causal_input import apply_input_contract
from ..direct_fast_memory_20261104.memory import dynamic_key, innovation
from ..direct_fast_memory_20261104.model import Injection, levels, predict
from ..direct_fast_memory_20261104.rls import initial_covariance, ridge_step
from ..direct_fast_memory_20261104.timeline import feedback, window_batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..warmstart_alignment_20261022.alignment import transform


ROOT = io23.ROOT
RUN = ROOT / 'probe_outputs/direct_fast_memory_ridge360_20261104_v1'
OUTPUT = ROOT / 'probe_outputs/shadow_onset_attribution_20261104_v1'
META = ROOT / 'probe_outputs/hdf5_outcome_authority_20261104_v3'
RUN_DONE_SHA = 'faeeaa386e3d160b4da4a2582adda79d3a3b226029c0767fe187bcdd90e732f6'
META_DONE_SHA = '5887d42ceed9cc7859b7f92abfff7f50f58338992956912348d7e6e0fd027e4d'
SHOTS = (87689, 70258, 64562, 69768)
QUERY = 135
ORIGIN = 127
PREFIX = 136
PARITY_ATOL = 2e-5
IG_ATOL = 1e-5
IG_RTOL = 1e-4
CHANNELS = tuple(name for i, name in enumerate(io23.RAW23) if i not in (2, 10))
USED_CODE = (
    'experiments/risk_aligned_online/io23.py',
    'experiments/risk_aligned_online/simple_data.py',
    'experiments/risk_aligned_online/simple23.py',
    'experiments/risk_aligned_online/causal_input.py',
    'experiments/risk_aligned_online/warmstart_alignment_20261022/alignment.py',
    'experiments/risk_aligned_online/redesign_20261004/prepare_prior.py',
    'experiments/risk_aligned_online/direct_fast_memory_20261104/memory.py',
    'experiments/risk_aligned_online/direct_fast_memory_20261104/model.py',
    'experiments/risk_aligned_online/direct_fast_memory_20261104/rls.py',
    'experiments/risk_aligned_online/direct_fast_memory_20261104/timeline.py',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def integrated_gradients(function, vector, *, steps):
    """Straight-line integrated logit gradient, trapezoid, zero baseline.

    ``function`` accepts a batch of independent vectors and returns one scalar
    per vector.  The returned tensor has the same shape as the input vector.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if not isinstance(vector, torch.Tensor) or vector.ndim != 1 or not vector.is_floating_point():
        raise ValueError('a one-dimensional floating vector is required')
    if not torch.isfinite(vector).all():
        raise ValueError('finite vector required')
    with torch.enable_grad():
        alpha = torch.linspace(0., 1., steps + 1, dtype=vector.dtype, device=vector.device)
        batch = (alpha[:, None] * vector.detach()[None, :]).requires_grad_(True)
        values = function(batch)
        if values.shape != (steps + 1,) or not torch.isfinite(values).all():
            raise ValueError('function must return a finite scalar per batch row')
        (gradient,) = torch.autograd.grad(values.sum(), batch)
    mean_gradient = (gradient[0] * .5 + gradient[-1] * .5 + gradient[1:-1].sum(0)) / steps
    result = vector.detach() * mean_gradient
    if not torch.isfinite(result).all():
        raise FloatingPointError('non-finite path attribution')
    return result.detach()


def first_update_from_tensors(initial, key_origin, target, mask, key_current):
    """One archived ridge_step from W0/P0=I; project innovation at current key."""
    state0 = initial.unsqueeze(0)
    covariance = initial_covariance(1, state0)
    state1, _, loss = ridge_step(
        state0, covariance, key_origin.unsqueeze(0), target.unsqueeze(0),
        mask.unsqueeze(0), origin=state0, radius=2.,
    )
    delta = innovation(state1, state0, key_current.unsqueeze(0))
    return delta[0], state1[0], loss[0]


def read_prefix_group(data_group, *, shot):
    """Reproduce io23/simple_data on 23 datasets, with ONLY [:136] reads."""
    ip = data_group['ip']
    rate = float(np.asarray(ip.attrs['SampleRate']).reshape(-1)[0])
    start = float(np.asarray(ip.attrs['StartTime']).reshape(-1)[0])
    if not math.isfinite(rate) or not math.isfinite(start) or not math.isclose(
        rate, 1000., rel_tol=0., abs_tol=io23.rate_tolerance_hz('EAST', 'nominal')
    ):
        raise ValueError('nominal EAST clock required')
    total = ip.shape[0]
    if total < PREFIX:
        raise ValueError('at least 136 signal samples required')
    columns = []
    for name in io23.RAW23:
        ds = data_group[name]
        if len(ds.shape) != 1 or ds.shape[0] != total:
            raise ValueError(f'unaligned channel shape: {name}')
        for attr, expected in (('SampleRate', rate), ('StartTime', start)):
            if attr not in ds.attrs or not np.isclose(
                float(np.asarray(ds.attrs[attr]).reshape(-1)[0]), expected
            ):
                raise ValueError(f'unaligned channel clock: {name}/{attr}')
        # Explicit bounded hyperslab. Never use [:], [()], or a full-file hash.
        column = np.asarray(ds[:PREFIX], dtype=np.float32)
        if column.shape != (PREFIX,):
            raise ValueError('incomplete prefix')
        columns.append(column)
    values = np.stack(columns, axis=1)
    available = np.isfinite(values)
    values = np.where(available, values, 0.).astype(np.float32)
    available[:, io23.RAW23.index('MNM')] = False
    values[~available] = 0.
    return dict(shot=int(shot), device='EAST', values=values, available=available,
                times=start + np.arange(PREFIX) / rate, rate=rate, start=start)


def read_authorized_prefix(shot):
    if type(shot) is not int or shot not in SHOTS:
        raise ValueError('only the four explicitly authorized EAST shots may be read')
    path = io23.path_for('EAST', shot)
    before = path.stat()
    with h5py.File(path, 'r') as handle:
        result = read_prefix_group(handle['data'], shot=shot)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('raw file changed during bounded prefix read')
    return result, dict(size=after.st_size, mtime_ns=after.st_mtime_ns)


def verify_inputs():
    """Verify completed run, required files, archived code, parent and map."""
    if sha(RUN / 'completed.json') != RUN_DONE_SHA:
        raise ValueError('RLS completion manifest changed')
    done = read_json(RUN / 'completed.json')
    if (done['status'] != 'COMPLETED_DIRECT_MEMORY_INTRA_DIAGNOSTIC'
            or not done['source_unchanged'] or not done['native_exact_baseline']):
        raise ValueError('completed frozen-source RLS run required')
    artifacts = done['artifact_sha256']
    for name in ('code.zip', 'fit.json', 'fitted.pt', 'protocol.json', 'per_shot.json'):
        if sha(RUN / name) != artifacts[name]:
            raise ValueError(f'RLS artifact hash mismatch: {name}')
    protocol = read_json(RUN / 'protocol.json')
    fit = read_json(RUN / 'fit.json')
    order = [row['shot'] for row in protocol['order']]
    if (len(order) != 360 or len(set(order)) != 360 or any(order.index(s) < 20 for s in SHOTS)
            or protocol['update_rule'] != 'ridge' or protocol['horizon_samples'] != 8
            or protocol['update_interval_samples'] != 32
            or protocol['source_sha256'] != PARENT_SHA256
            or fit['label_shots'] != order[:20] or done['fit_label_shots'] != order[:20]):
        raise ValueError('exact source, RLS cadence and ended-20 label set required')
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise ValueError('qualified source checkpoint mismatch')
    mapping_path = ROOT / 'probe_outputs/stratified_cold20_context360_v1/input_map.json'
    if sha(mapping_path) != protocol['input_sha256'][str(mapping_path)]:
        raise ValueError('prefix map hash mismatch')
    mapping = read_json(mapping_path)
    if mapping['calibration_ids'] != order[:20]:
        raise ValueError('mapping must be fit on the same exact ended prefix')
    with zipfile.ZipFile(RUN / 'code.zip') as archive:
        for relative in USED_CODE:
            actual = sha(ROOT / relative)
            archived = hashlib.sha256(archive.read(relative)).hexdigest()
            if actual != protocol['code_sha256'][str(ROOT / relative)] or actual != archived:
                raise ValueError(f'current dependency differs from archived RLS source: {relative}')
    for shot in SHOTS:
        index = order.index(shot) + 1
        prefix = f'shots/{index:04d}_{shot}'
        for suffix in ('.npz', '_seal.json', '_updates.json'):
            name = prefix + suffix
            if sha(RUN / name) != artifacts[name]:
                raise ValueError(f'RLS shot artifact hash mismatch: {name}')
        seal = read_json(RUN / (prefix + '_seal.json'))
        if (seal['shot'] != shot or seal['curve_sha256'] != artifacts[prefix + '.npz']
                or seal['current_outcome_used_in_prediction'] is not False):
            raise ValueError('sealed curve does not bind current shot')
    if sha(META / 'completed.json') != META_DONE_SHA:
        raise ValueError('v3 metadata completion manifest changed')
    meta_done = read_json(META / 'completed.json')
    if sha(META / 'hdf5_metadata.json') != meta_done['artifact_sha256']['hdf5_metadata.json']:
        raise ValueError('v3 metadata hash mismatch')
    return protocol, fit, mapping, order, done


def attributions(base, injection, x, available, delta):
    """Compute a converged straight-path logit IG matrix and diagnostics."""
    def function(vectors):
        return predict(base, injection, x.expand(len(vectors), -1, -1),
                       available.expand(len(vectors), -1, -1), vectors)

    with torch.no_grad():
        endpoints = function(torch.stack((torch.zeros_like(delta), delta)))
    difference = float((endpoints[1] - endpoints[0]).item())
    attempts = []
    result = None
    for steps in (32, 64, 128, 256):
        candidate = integrated_gradients(function, delta, steps=steps)
        total = float(candidate.double().sum().item())
        error = abs(total - difference)
        attempts.append(dict(steps=steps, sum=total, abs_completeness_error=error))
        result = candidate
        if steps >= 64 and error <= IG_ATOL + IG_RTOL * abs(difference):
            break
    else:
        raise ValueError(f'integrated-gradient completeness not met: {attempts}')
    return endpoints.detach(), result.reshape(8, 21), attempts


def run():
    if OUTPUT.exists():
        raise FileExistsError('refusing to overwrite the attribution output')
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol, fit, mapping, order, done = verify_inputs()
    base, parent = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    if state_digest(base) != protocol['source_state_sha256'] or float(parent['threshold']) != protocol['threshold']:
        raise ValueError('loaded source state or threshold does not match RLS run')
    fitted = torch.load(RUN / 'fitted.pt', map_location='cpu', weights_only=True)
    if (fitted['label_shots'] != order[:20] or fitted['source_sha256'] != PARENT_SHA256
            or fitted['update_rule'] != 'ridge' or fitted['initial'].shape != (168, 70)
            or not torch.isfinite(fitted['initial']).all()):
        raise ValueError('fitted W0/source/rule/ended-label binding mismatch')
    initial = fitted['initial'].detach().clone()
    injection = Injection().cpu().eval().requires_grad_(False)
    injection.load_state_dict(fitted['injection'], strict=True)
    if not torch.isfinite(injection.weight).all():
        raise ValueError('finite frozen injection required')
    before = dict(source=state_digest(base), injection=state_digest(injection), initial=initial.clone())
    results = []
    stats = {}
    for shot in SHOTS:
        raw, stats[shot] = read_authorized_prefix(shot)
        actual = apply_input_contract(raw)
        prepared = transform({key: actual[key] for key in ('values', 'available')}, mapping)
        prepared['shot'] = shot
        with torch.no_grad():
            key_origin, target, observed = feedback(base, prepared, ORIGIN, QUERY, device='cpu')
            x, available = window_batch(prepared, [QUERY], device='cpu')
            key_current = dynamic_key(levels(base, x, available), available)
            delta, state1, loss = first_update_from_tensors(
                initial, key_origin[0], target[0], observed[0], key_current[0]
            )
            before_prediction = key_origin[0] @ initial.T
            weight_displacement = float(torch.linalg.vector_norm(state1 - initial).item())
        index = order.index(shot) + 1
        prefix = RUN / f'shots/{index:04d}_{shot}'
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as archive:
            if not np.array_equal(raw['times'], archive['times'][:PREFIX]):
                raise ValueError(f'prefix clock mismatch: {shot}')
            saved_native = float(archive['native'][QUERY])
            saved_real = float(archive['real'][QUERY])
            firsts = {}
            for arm in ('native', 'real'):
                hits = np.flatnonzero(archive[arm] >= protocol['threshold'])
                firsts[arm] = int(hits[0]) if len(hits) else None
        history = read_json(prefix.with_name(prefix.name + '_updates.json'))['real'][0]
        if (history['origin'] != ORIGIN or history['arrival'] != QUERY
                or history['observed'] != int(observed.sum())):
            raise ValueError(f'first feedback receipt mismatch: {shot}')
        reconstructed_mse = float(loss.item() * 2.)
        if (abs(reconstructed_mse - history['pre_update_mse']) > PARITY_ATOL
                or abs(weight_displacement - history['weight_displacement']) > PARITY_ATOL):
            raise ValueError(f'CPU/GPU first ridge step mismatch: {shot}')
        with torch.no_grad():
            endpoint_check = predict(
                base, injection, x.expand(2, -1, -1), available.expand(2, -1, -1),
                torch.stack((torch.zeros_like(delta), delta)),
            )
        probability = endpoint_check.sigmoid().numpy()
        parity = dict(native_abs=float(abs(probability[0] - saved_native)),
                      real_abs=float(abs(probability[1] - saved_real)))
        if max(parity.values()) > PARITY_ATOL:
            raise ValueError(f'CPU/GPU probability mismatch: {shot} {parity}')
        if firsts['real'] != QUERY or saved_native >= protocol['threshold'] or saved_real < protocol['threshold']:
            raise ValueError(f'not the specified first-update crossing: {shot}')
        endpoints, matrix, attempts = attributions(base, injection, x, available, delta)
        if not torch.allclose(endpoints, endpoint_check, atol=1e-6, rtol=0.):
            raise ValueError(f'integration endpoint mismatch: {shot}')
        channel = matrix.double().sum(dim=0).tolist()
        horizon = matrix.double().sum(dim=1).tolist()
        residual = (target[0] - before_prediction).masked_fill(~observed[0], 0.)
        results.append(dict(
            shot=shot, order_index=index, time=float(raw['times'][QUERY]),
            threshold=protocol['threshold'], saved_native=saved_native, saved_real=saved_real,
            cpu_native=float(probability[0]), cpu_real=float(probability[1]),
            native_logit=float(endpoints[0]), real_logit=float(endpoints[1]),
            delta_logit=float(endpoints[1] - endpoints[0]), parity_abs=parity,
            first_alarm_indices=firsts, observed=int(observed.sum()),
            ciii_available_prefix=int(actual['available'][:, 1].sum()),
            mnm_available_prefix=int(actual['available'][:, 2].sum()),
            first_update_pre_mse=reconstructed_mse,
            first_update_pre_mse_saved=history['pre_update_mse'],
            first_update_weight_displacement=weight_displacement,
            first_update_weight_displacement_saved=history['weight_displacement'],
            ig_attempts=attempts, ig_by_horizon_channel=matrix.double().tolist(),
            ig_by_channel=dict(zip(CHANNELS, channel, strict=True)),
            ig_by_horizon=horizon,
            innovation_by_horizon_channel=delta.reshape(8, 21).double().tolist(),
            target_residual_by_horizon_channel=residual.reshape(8, 21).double().tolist(),
            feedback_observed_by_horizon_channel=observed.reshape(8, 21).tolist(),
        ))
    if (state_digest(base) != before['source'] or state_digest(injection) != before['injection']
            or not torch.equal(initial, before['initial'])
            or any(parameter.grad is not None for parameter in (*base.parameters(), *injection.parameters()))):
        raise AssertionError('frozen source, injection or W0 changed')
    # Read sealed outcomes only after all forward/gradient computations.
    metadata = read_json(META / 'hdf5_metadata.json')
    outcomes = {row['shot']: row for row in metadata['EAST'] if row['shot'] in SHOTS}
    saved_scores = read_json(RUN / 'per_shot.json')
    for result in results:
        shot = result['shot']
        outcome = outcomes[shot]
        if stats[shot] != {key: outcome[key] for key in ('size', 'mtime_ns')}:
            raise ValueError(f'raw file stat differs from sealed v3 metadata: {shot}')
        for arm in ('native', 'real'):
            row = saved_scores[arm][result['order_index'] - 1]
            if (row['shot'] != shot or row['disruptive'] != outcome['disruptive']
                    or row['down_s'] != outcome['down']):
                raise ValueError(f'sealed outcome mismatch: {shot}')
            result[f'{arm}_sealed_outcome'] = dict(alarm=row['alarm'], timely=row['timely'],
                                                    lead_ms=row['lead_ms'])
        result['disruptive_v3'] = outcome['disruptive']
    verify_inputs()
    report = dict(
        scope='OFFLINE_FOUR_CASE_MODEL_ATTRIBUTION_NOT_CAUSAL_PHYSICS',
        device='cpu', shots=list(SHOTS), raw_hyperslab='each of 23 channels [:136] only',
        raw_file_sha256_not_recomputed='full raw read prohibited',
        current_outcomes_used_before_all_gradients=False,
        source_sha256=PARENT_SHA256, source_state_sha256=before['source'],
        rls_completed_sha256=RUN_DONE_SHA, fitted_sha256=done['artifact_sha256']['fitted.pt'],
        archived_code_sha256=done['artifact_sha256']['code.zip'],
        v3_metadata_completed_sha256=META_DONE_SHA, label_shots=fit['label_shots'],
        map_sha256=protocol['input_sha256'][str(ROOT / 'probe_outputs/stratified_cold20_context360_v1/input_map.json')],
        origin=ORIGIN, arrival=QUERY, target_points=[128, 129, 130, 131, 132, 133, 134, 135],
        method='logit IG, zero to actual 168-vector, trapezoid, 32 then 64-256 intervals',
        parity_atol=PARITY_ATOL, ig_atol=IG_ATOL, ig_rtol=IG_RTOL,
        source_and_injection_unchanged=True, channels=list(CHANNELS), results=results,
        script_sha256=sha(__file__),
    )
    OUTPUT.mkdir(parents=False, exist_ok=False)
    result_path = OUTPUT / 'results.json'
    with result_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with (OUTPUT / 'completed.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(status='COMPLETED_CPU_FOUR_CASE_ATTRIBUTION',
                       results_sha256=sha(result_path), script_sha256=sha(__file__)), stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status='COMPLETED_CPU_FOUR_CASE_ATTRIBUTION',
                          output=str(OUTPUT), results_sha256=sha(result_path))))


if __name__ == '__main__':
    run()
