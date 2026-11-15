"""Exactly replay selected completed context-vector suffix shots on the GPU.

The saved full360 report chooses the default shots; prediction itself receives
only signals. This is a bounded reproducibility check, not a new evaluation.
"""

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import audit_full as audit
from . import run_full as runner
from .stream import NAMES


FIRST_SUFFIX_SHOT = 70323


def select_shots(order, comparisons, requested=None):
    """Select paired changes versus cold, or a distinct explicit suffix subset."""
    if (len(order) != 360 or len(set(order)) != 360
            or any(type(sid) is not int or sid <= 0 for sid in order)):
        raise ValueError('distinct ordered full360 shot IDs required')
    if order[20] != FIRST_SUFFIX_SHOT:
        raise ValueError('first suffix shot must be 70323')
    suffix = set(order[20:])
    if requested is not None:
        if (not requested or any(type(sid) is not int for sid in requested)
                or len(set(requested)) != len(requested) or not set(requested) <= suffix):
            raise ValueError('distinct nonempty suffix shot IDs required')
        selected = set(requested)
    else:
        if not isinstance(comparisons, dict):
            raise ValueError('completed paired comparisons required')
        full, part = comparisons['full360'], comparisons['suffix340']
        if set(full) != set(NAMES) or set(part) != set(NAMES):
            raise ValueError('paired comparison arm coverage differs')
        selected = {FIRST_SUFFIX_SHOT}
        for name in NAMES:
            item = full[name]
            if (set(item) != {'rescued', 'harmed'} or item != part[name]
                    or any(not isinstance(item[k], list) for k in ('rescued', 'harmed'))):
                raise ValueError(f'paired comparison sections differ: {name}')
            rescued, harmed = item['rescued'], item['harmed']
            if (any(type(sid) is not int for sid in rescued + harmed)
                    or len(set(rescued)) != len(rescued) or len(set(harmed)) != len(harmed)
                    or set(rescued) & set(harmed) or not set(rescued + harmed) <= suffix):
                raise ValueError(f'paired comparison contains invalid suffix shots: {name}')
            selected.update(rescued)
            selected.update(harmed)
        if full['cold'] != {'rescued': [], 'harmed': []}:
            raise ValueError('cold versus cold paired comparison is not empty')
    return [sid for sid in order[20:] if sid in selected]


def check_output_path(directory, output):
    directory, output = Path(directory).absolute(), Path(output).absolute()
    if output.parent.resolve() != directory.resolve() or output.suffix != '.json':
        raise ValueError('output must be a JSON file directly inside the completed run')
    if os.path.lexists(output):
        raise FileExistsError(f'will not overwrite replay output: {output}')
    return output


@contextmanager
def no_outcomes():
    """Deny outcome, label, and class-manifest helpers throughout the replay."""
    def deny(*_args, **_kwargs):
        raise AssertionError('outcome, label and class manifest access blocked during replay')

    with ExitStack() as stack:
        for module, name in ((runner.io23, 'read_outcome'), (runner.io23, 'labels_at'),
                             (runner.io23, 'cohort_ids'), (runner.simple_data, 'attach_outcome'),
                             (runner.fixed_test_protocol, 'load_test_contract')):
            stack.enter_context(patch.object(module, name, side_effect=deny))
        yield


def _array_error(actual, saved, label):
    actual, saved = np.asarray(actual), np.asarray(saved)
    if actual.shape != saved.shape or actual.dtype != saved.dtype or not actual.size:
        raise ValueError(f'{label} shape or dtype differs')
    if not np.isfinite(actual).all() or not np.isfinite(saved).all():
        raise ValueError(f'{label} contains nonfinite values')
    error = float(np.max(np.abs(actual.astype(np.float64) - saved.astype(np.float64))))
    if not np.array_equal(actual, saved) or actual.tobytes() != saved.tobytes():
        raise ValueError(f'{label} differs exactly; maximum absolute error={error}')
    return error


def compare_exact(times, curves, updates, saved, sealed_updates):
    if set(curves) != set(NAMES) or set(saved) != {'times', *NAMES}:
        raise ValueError('exactly 22 curve arms and timestamps required')
    times_error = _array_error(times, saved['times'], 'time')
    if np.asarray(times).ndim != 1:
        raise ValueError('time must be one-dimensional')
    errors = {}
    for name in NAMES:
        values = np.asarray(curves[name])
        if (values.shape != np.asarray(times).shape or not np.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()):
            raise ValueError(f'curve {name} must have one finite probability per time')
        errors[name] = _array_error(values, saved[name], f'curve {name}')
    try:
        actual = json.dumps(updates, sort_keys=True, separators=(',', ':'), allow_nan=False)
        expected = json.dumps(sealed_updates, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('receipt is not finite JSON') from exc
    if actual != expected:
        raise ValueError('update receipt differs exactly from its seal')
    return dict(times_max_abs_error=times_error, curves_max_abs_error=errors,
                receipt_max_abs_error=0., receipts_exact=True)


def _module_snapshot(module):
    return dict(state={k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
                training={k: v.training for k, v in module.named_modules()},
                requires_grad={k: v.requires_grad for k, v in module.named_parameters()})


def _same_tensor(actual, expected):
    actual = actual.detach().cpu()
    return actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(actual, expected)


def _same_module(module, saved):
    state = module.state_dict()
    return (set(state) == set(saved['state'])
            and all(_same_tensor(state[k], value) for k, value in saved['state'].items())
            and {k: v.training for k, v in module.named_modules()} == saved['training']
            and {k: v.requires_grad for k, v in module.named_parameters()} == saved['requires_grad'])


def snapshot_frozen(net, slow, policies):
    return dict(net=_module_snapshot(net), slow=slow.detach().cpu().clone(),
                policies={v: dict(controller=_module_snapshot(p['controller']),
                                  mean_context=p['mean_context'].detach().cpu().clone(),
                                  scale={k: x.detach().cpu().clone() for k, x in p['scale'].items()},
                                  radius=p['radius']) for v, p in policies.items()})


def assert_frozen(net, slow, policies, saved):
    if not _same_module(net, saved['net']) or not _same_tensor(slow, saved['slow']):
        raise AssertionError('frozen source model or slow state changed')
    if set(policies) != set(saved['policies']):
        raise AssertionError('frozen policy identities changed')
    for variant, policy in policies.items():
        old = saved['policies'][variant]
        if (not _same_module(policy['controller'], old['controller'])
                or not _same_tensor(policy['mean_context'], old['mean_context'])
                or policy['radius'] != old['radius'] or set(policy['scale']) != set(old['scale'])
                or any(not _same_tensor(policy['scale'][k], x) for k, x in old['scale'].items())):
            raise AssertionError('frozen policy changed: ' + variant)


def replay_shot(directory, previous, index, sid, barrier, old_barrier, raw_hashes,
                mapping, net, slow, policies):
    stem = f'{index + 1:04d}_{sid}'
    shot_path = directory / 'shots' / (stem + '.npz')
    seal_path = directory / 'shots' / (stem + '_seal.json')
    old_path = previous / 'shots' / (stem + '.npz')
    old_seal_path = previous / 'shots' / (stem + '_seal.json')
    if (barrier['seals'][index]['shot'] != sid or old_barrier['seals'][index]['shot'] != sid
            or audit.sha(seal_path) != barrier['receipt_sha256'][index]):
        raise ValueError(f'shot/receipt barrier changed: {sid}')
    seal = audit.read_json(seal_path)
    if (seal['shot'] != sid or seal['index'] != index + 1
            or seal['sha256'] != barrier['seals'][index]['sha256']
            or seal['archive_seal_sha256'] != audit.sha(old_seal_path)
            or seal['startup_archived'] is not False or seal['outcomes_used_in_prediction'] is not False):
        raise ValueError(f'sealed suffix receipt differs: {sid}')
    saved = audit.verified_npz(shot_path, barrier['seals'][index]['sha256'])
    old = audit.verified_npz(old_path, old_barrier['seals'][index]['sha256'])
    if set(saved) != {'times', *NAMES}:
        raise ValueError('sealed 22-curve archive fields differ')
    audit.verify_curve_contract(saved['times'], {name: saved[name] for name in NAMES},
                                old['times'], old['cold'], seal['updates'], False)
    runner.validate_receipts(seal['updates'], len(saved['times']), policies)
    raw_path = runner.io23.path_for('EAST', sid)
    if audit.sha(raw_path) != raw_hashes[str(sid)]:
        raise ValueError(f'raw signal file hash differs: {sid}')
    raw = runner.apply_input_contract(runner.simple_data.read_shot('EAST', sid, rate_contract='nominal'))
    _array_error(raw['times'], saved['times'], 'raw time')
    signal = runner.transform({k: raw[k] for k in ('values', 'available')}, mapping)
    curves, updates = runner.predict(net, signal, slow, policies)
    runner.validate_receipts(updates, len(raw['times']), policies)
    result = compare_exact(raw['times'], curves, updates, saved, seal['updates'])
    if (audit.sha(raw_path) != raw_hashes[str(sid)]
            or audit.sha(shot_path) != barrier['seals'][index]['sha256']
            or audit.sha(old_path) != old_barrier['seals'][index]['sha256']
            or audit.sha(seal_path) != barrier['receipt_sha256'][index]
            or audit.sha(old_seal_path) != seal['archive_seal_sha256']):
        raise ValueError(f'raw input, prediction or seal changed during replay: {sid}')
    return dict(shot=sid, index=index + 1, prediction_sha256=seal['sha256'],
                receipt_sha256=barrier['receipt_sha256'][index], **result)


def replay(directory, requested=None):
    directory = Path(directory).absolute()
    with no_outcomes():
        # Nothing from the archive, report comparisons, or shot data is read
        # until the current run's completion and 360-shot barrier are verified.
        complete, protocol, barrier = audit.require_completed_barrier(directory)
        completion_hash = audit.sha(directory / 'complete.json')
        reference, old_barrier, _ = audit.verify_provenance(directory, complete, protocol)
        audit.verify_all_seals(directory, protocol, barrier, old_barrier)
        if protocol['score_batch_per_arm'] != 32 or protocol['padded_rows_per_variant'] != 9:
            raise ValueError('original 9 by 32 padded scoring protocol required')
        if not {'order.json', 'input_map.json', 'input_hashes.json', 'models/cold.pt'} <= set(
                protocol['reference_hashes']):
            raise ValueError('required reference input and model hashes are missing')
        shots = select_shots(protocol['order'], complete['comparisons'] if requested is None else None,
                             requested)
        previous = Path(protocol['previous'])
        old_protocol = audit.read_json(previous / 'protocol.json')
        mapping, raw_hashes = (audit.read_json(reference / name)
                               for name in ('input_map.json', 'input_hashes.json'))
        torch.set_num_threads(2)
        torch.manual_seed(131)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('rl_env CUDA required for exact GPU replay')
        policies, policy_hashes = runner.load_policies(Path(protocol['policy']), old_protocol, device='cuda')
        if policy_hashes != protocol['policy_hashes']:
            raise ValueError('loaded policy provenance differs from the sealed protocol')
        base, parent = runner.load_parent(runner.DEFAULT_PARENT)
        net = runner.MatrixAttention(base.cuda()).eval()
        if float(parent['threshold']) != protocol['threshold']:
            raise ValueError('source threshold differs from sealed protocol')
        slow = torch.load(reference / 'models/cold.pt', map_location='cuda', weights_only=True)['slow']
        frozen = snapshot_frozen(net, slow, policies)
        indices = {sid: i for i, sid in enumerate(protocol['order'])}
        results = [replay_shot(directory, previous, indices[sid], sid, barrier, old_barrier,
                               raw_hashes, mapping, net, slow, policies) for sid in shots]
        assert_frozen(net, slow, policies, frozen)
        if (audit.sha(directory / 'complete.json') != completion_hash
                or audit.sha(directory / 'protocol.json') != complete['artifact_sha256']['protocol.json']
                or audit.sha(directory / 'all_predictions_sealed.json')
                != complete['artifact_sha256']['all_predictions_sealed.json']):
            raise ValueError('completed run metadata changed during replay')
        # Recheck every saved code, model, policy and input-map hash after use.
        audit.verify_provenance(directory, complete, protocol)
        return dict(replay_passed=True, scope='bounded exact GPU re-execution; not independent confirmation',
                    complete_sha256=completion_hash,
                    barrier_sha256=complete['artifact_sha256']['all_predictions_sealed.json'],
                    selection='explicit' if requested is not None else 'all paired changes plus 70323',
                    shots=shots, arm_count=len(NAMES), score_batch_per_arm=32, padded_rows_per_variant=9,
                    gpu=torch.cuda.get_device_name(), original_gpu=protocol['gpu'],
                    torch=str(torch.__version__), original_torch=protocol['torch'],
                    frozen_parameters_unchanged=True,
                    selection_uses_completed_paired_report=requested is None,
                    io23_outcome_and_label_access_blocked=True,
                    times_max_abs_error=max(item['times_max_abs_error'] for item in results),
                    curves_max_abs_error=max(max(item['curves_max_abs_error'].values()) for item in results),
                    receipt_max_abs_error=max(item['receipt_max_abs_error'] for item in results),
                    results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=audit.RUN)
    parser.add_argument('--output', type=Path, required=True,
                        help='new JSON file directly inside the completed run')
    parser.add_argument('--shots', type=int, nargs='+', help='explicit distinct suffix shot IDs')
    args = parser.parse_args()
    output = check_output_path(args.run, args.output)
    result = replay(args.run, args.shots)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with output.open('x', encoding='utf-8') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"Exact GPU replay passed for {len(result['shots'])} shots; report: {output}", flush=True)


if __name__ == '__main__':
    main()
