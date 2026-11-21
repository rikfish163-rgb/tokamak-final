"""Independent original-batch witness replay and local, first-order geometry."""
import argparse
import copy
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..delayed_recurrent_memory_20261025.learning import batch
from ..east_context_operator_20261031.run import configure, install_operator, read
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .prepare import enumerate_errors


def geometry(a, g):
    a, g = np.asarray(a, dtype=np.float64), np.asarray(g, dtype=np.float64)
    if (a.ndim != 2 or not min(a.shape) or g.shape != (a.shape[0],)
            or not np.isfinite(a).all() or not np.isfinite(g).all()):
        raise ValueError('finite matrix and matching gradient required')
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    radius, norm = float(s[0]), float(np.linalg.norm(g))
    rank = int((s > max(a.shape)*np.finfo(float).eps*radius).sum())
    fraction = float(np.linalg.norm(u[:, :rank].T @ g)/norm) if norm else None
    context = float(np.linalg.norm(a.T @ g))
    free = radius*norm
    ratio = context/free if free else None
    assert fraction is None or fraction <= 1+1e-10
    assert ratio is None or ratio <= 1+1e-10
    return dict(radius=radius, rank=rank, singular_values=s.tolist(),
                last_token_logit_gradient_norm=norm, gradient_subspace_fraction=fraction,
                local_context_logit_change_bound=context,
                local_free_last_logit_change_bound=free,
                local_context_to_free_ratio=ratio)


def direct_risk(model, h, valid, shift):
    """Explicit block-boundary replay, independent of optimize.risk_with_shift."""
    last = h[:, -2] + model.slow(h[:, -2])*valid[:, -2, None]
    data = torch.cat((h[:, :-2], last[:, None]), dim=1)
    data = data + shift*valid[:, :-1, None]
    changed = torch.cat((data, h[:, -1:]), dim=1)
    assert torch.equal(changed[:, -1], h[:, -1]), 'CLS was directly changed'
    with sdpa_kernel(SDPBackend.MATH):
        encoded = model.base.encoder.layers[1](changed, src_key_padding_mask=~valid)
    gain, bias = model.base.initial_fast(len(h)).chunk(2, -1)
    encoded = model.base.norm(encoded)*(1+.1*gain[:, None].tanh())+.1*bias[:, None].tanh()
    return model.base.risk_head(encoded[:, -1]).squeeze(-1)


def main(root):
    if (root/'verification.json').exists():
        raise FileExistsError(root/'verification.json')
    done, p = read(root/'completed.json'), read(root/'protocol.json')
    assert done['status'] == 'COMPLETED_POSTHOC_DIAGNOSTIC' and done['not_online_performance']
    parent = Path(p['parent'])
    assert sha(parent/'completed.json') == p['parent_completed_sha256']
    assert sha(parent/'models/cold.pt') == p['parent_cold_sha256']
    assert sha(DEFAULT_PARENT) == p['source_sha256']
    with zipfile.ZipFile(root/'code_snapshot.zip') as z:
        for name, digest in p['code_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
            assert sha(io23.ROOT/name) == digest
    errors, cases = enumerate_errors(parent)
    assert errors == read(root/'all88_errors.json') and cases == read(root/'cases78.json')
    assert sum(not r['enabled'] for r in cases) == 5
    assert sum(r['prior_alarm_before_first_effect'] for r in cases) == 7
    assert sha(root/'prepared.pt') == done['prepared_sha256']
    assert sha(root/'witness_replays.json') == done['witness_replays_sha256']
    saved = read(root/'witness_replays.json')
    assert len(saved) == 468
    lookup = {(r['shot'], r['operator'], r['path']):r for r in saved}
    configure()
    base, _ = load_parent(DEFAULT_PARENT)
    source = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    load_context_memory(source, Path(p['source_memory']['path']))
    source.restore(torch.load(parent/'models/cold.pt', map_location='cuda', weights_only=True))
    learned = copy.deepcopy(source)
    operator = io23.ROOT/'probe_outputs/east_operator360_v1/models/operator.pt'
    assert sha(operator) == p['target_operator_sha256']
    install_operator(learned, torch.load(operator, map_location='cuda', weights_only=True)['operator'])
    models = dict(source_operator=source, learned_operator=learned)
    for model in models.values():
        model.requires_grad_(False)
        model.eval()
    prepared = torch.load(root/'prepared.pt', map_location='cuda', weights_only=True)
    outputs = {name:torch.load(root/f'{name}.pt', map_location='cuda', weights_only=True) for name in models}
    mapping = read(parent/'input_map.json')
    max_probability_error = 0.
    for i, case in enumerate(cases):
        raw = apply_input_contract(simple_data.read_shot('EAST', case['shot'], rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        with torch.no_grad():
            x, mask = batch(signal, range(case['original_batch_start'], case['original_batch_stop']), 'cuda')
            h, valid = source.tokens(x, mask)
            j = case['endpoint']-case['original_batch_start']
            assert torch.equal(h[j], prepared['h'][i]) and torch.equal(valid[j], prepared['valid'][i])
            for name, model in models.items():
                result = outputs[name]
                previous = None
                for path, found in result['paths'].items():
                    record = lookup[(case['shot'], name, path)]
                    shift = found['best_shift'][i]
                    radius = float(result['radius'][i])
                    assert float(shift.norm()) <= radius+2e-6*max(1., radius)
                    assert not torch.count_nonzero(shift[~valid[j, :-1]])
                    if not case['enabled']:
                        assert not torch.count_nonzero(shift)
                    if path == 'context8':
                        c = found['best_parameter'][i]
                        assert float(c.norm()) <= 1.+1e-6
                        b = prepared['operators'][name]['basis'][i]
                        expected = model.memory.output(b @ c)*valid[j, -2]
                        assert torch.allclose(expected, shift[-1], atol=2e-6, rtol=2e-6)
                        assert not torch.count_nonzero(shift[:-1])
                    elif path == 'free_last':
                        assert not torch.count_nonzero(shift[:-1])
                    full = h.new_zeros(len(h), 16, 64)
                    full[j] = shift
                    z = direct_risk(model, h, valid, full)[j]
                    probability = float(z.sigmoid())
                    error = abs(probability-record['witness_probability'])
                    max_probability_error = max(max_probability_error, error)
                    assert error <= 2e-6
                    met = probability >= p['threshold'] if case['disruptive'] else probability < p['threshold']
                    assert met == record['satisfies_selected_endpoint']
                    loss = float(torch.nn.functional.binary_cross_entropy_with_logits(z, prepared['labels'][i]))
                    assert previous is None or loss <= previous+2e-6
                    previous = loss
    # These are local derivatives at zero shift, not global nonlinear bounds.
    shift = torch.zeros(len(cases), 16, 64, device='cuda', requires_grad=True)
    logits = direct_risk(source, prepared['h'], prepared['valid'], shift)
    gradient, = torch.autograd.grad(logits.sum(), shift)
    g = gradient.detach().double().cpu().numpy()
    geometry_rows = []
    for name, model in models.items():
        w = model.memory.output.weight.detach().double().cpu().numpy()
        info = prepared['operators'][name]
        bases = info['basis'].double().cpu().numpy()
        contexts = info['actual_context'].double().cpu().numpy()
        for i, case in enumerate(cases):
            a = w @ bases[i]
            result = geometry(a, g[i, -1])
            effective_radius = result['radius'] if case['enabled'] else 0.
            assert np.isclose(effective_radius, float(outputs[name]['radius'][i]), atol=2e-6, rtol=2e-6)
            actual_shift = a @ contexts[i] if case['enabled'] else np.zeros(64)
            norm = float(np.linalg.norm(actual_shift))
            desired = g[i, -1]*(1 if case['disruptive'] else -1)
            denominator = norm*np.linalg.norm(desired)
            cosine = float(actual_shift @ desired/denominator) if denominator else None
            if not case['enabled']:
                result['gradient_subspace_fraction'] = None
                result['local_context_to_free_ratio'] = None
            geometry_rows.append(dict(shot=case['shot'], operator=name, disruptive=case['disruptive'],
                enabled=case['enabled'], actual_shift_norm=norm,
                actual_shift_to_radius=norm/effective_radius if effective_radius else None,
                actual_shift_cosine_to_label_desired_gradient=cosine,
                all_data_token_gradient_norm=float(np.linalg.norm(g[i])), **result))
    _durable_json(root/'local_geometry.json', dict(scope='LOCAL_LOGIT_DERIVATIVE_NOT_GLOBAL_BOUND',
                                               rows=geometry_rows, audit_script_sha256=sha(Path(__file__))))
    summaries = {}
    for name in models:
        summaries[name] = {}
        for path in ('context8', 'free_last', 'free_sequence'):
            group = [r for r in saved if r['operator'] == name and r['path'] == path]
            counts = dict(positive=sum(r['disruptive'] and r['satisfies_selected_endpoint'] for r in group),
                          negative=sum(not r['disruptive'] and r['satisfies_selected_endpoint'] for r in group))
            reported = done['summaries'][name][path]
            assert counts['positive'] == reported['positive_endpoint_targets_met']
            assert counts['negative'] == reported['negative_endpoint_targets_met']
            summaries[name][path] = counts
    result = dict(verified=True, scope='POSTHOC_ENDPOINT_WITNESSES_ONLY_NOT_ONLINE_GAIN',
        original_error_ledger_recomputed=88, startup_errors_not_optimized=10,
        chosen_endpoint_cases=78, independently_replayed_witnesses=468,
        original_batch_direct_forward_max_probability_difference=max_probability_error,
        all_threshold_sides_reproduced=True, norm_mask_clock_and_nested_witness_checks=True,
        summaries=summaries, local_geometry_rows=156,
        completed_sha256=sha(root/'completed.json'), audit_script_sha256=sha(Path(__file__)),
        operator_results_sha256={name:sha(root/f'{name}.pt') for name in models},
        entire_optimizer_trajectories_not_reexecuted=True,
        independent_confirmation=False, scientific_goal_achieved=False)
    _durable_json(root/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
