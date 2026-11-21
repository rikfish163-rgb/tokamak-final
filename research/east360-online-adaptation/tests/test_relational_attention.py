"""CPU contracts for measured temporal-relation Q/K adaptation."""

from types import SimpleNamespace

import pytest
import torch

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.relational_attention_20261030.learning import (
    attention_kl,
    bounded_step,
    fit_reference,
    normalized,
    relation_target,
    restrict_target,
)


U = [io23.RAW23.index(name) for name in ('Bt', 'ip')]
C = [io23.RAW23.index(name) for name in
     ('Mir_avg_amp', 'sxr_mean', 'xuv_ratio', 'ne_nG')]


def make_shot(seed, rows=256):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(rows, 23, generator=generator, dtype=torch.float64)
    u = torch.randn(rows, 2, generator=generator, dtype=torch.float64) * 2
    slopes = torch.tensor([[2., -1., .5, 1.], [-.5, 2., 1., -1.]], dtype=z.dtype)
    z[:, U] = u
    z[:, C] = torch.tensor([1., 2., -1., .5], dtype=z.dtype) + u @ slopes
    z[:, C] += torch.randn(rows, 4, generator=generator, dtype=z.dtype)
    return z, torch.ones_like(z, dtype=torch.bool)


def manual_reference(shots):
    """Independent loop implementation of the specified shot-weighted fit."""
    designs, responses, usable = [], [], []
    for z, mask in shots:
        valid = mask[:, U + C].all(dim=1)
        design = torch.cat((torch.ones(int(valid.sum()), 1, dtype=z.dtype),
                            z[valid][:, U]), dim=1)
        designs.append(design)
        responses.append(z[valid][:, C])
        usable.append(valid)
    gram = torch.stack([x.T @ x / len(x) for x in designs]).mean(0)
    rhs = torch.stack([x.T @ y / len(x) for x, y in zip(designs, responses)]).mean(0)
    beta = torch.linalg.solve(gram + torch.diag(torch.tensor([0., 1., 1.], dtype=gram.dtype)), rhs)
    variances, counts = [], []
    for (z, _), valid in zip(shots, usable):
        means = []
        for start in range(0, 256, 8):
            indices = torch.arange(start, start + 8)[valid[start:start + 8]]
            if len(indices) < 6:
                means.append(None)
                continue
            x = torch.cat((torch.ones(len(indices), 1, dtype=z.dtype), z[indices][:, U]), dim=1)
            means.append((z[indices][:, C] - x @ beta).mean(0))
        diffs = [means[i] - means[i - 1] for i in range(1, 32)
                 if means[i] is not None and means[i - 1] is not None]
        counts.append(len(diffs))
        variances.append(torch.stack(diffs).square().mean(0))
    return beta, torch.stack(variances).mean(0).clamp_min(1e-4), counts


def test_normalized_matches_frozen_source_asinh_and_finite_availability():
    center = torch.linspace(-1, 1, 23, dtype=torch.float64)
    scale = torch.linspace(.5, 2, 23, dtype=torch.float64)
    net = SimpleNamespace(base=SimpleNamespace(center=center, scale=scale))
    values = torch.arange(2 * 3 * 23, dtype=torch.float64).reshape(2, 3, 23) / 10
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[0, 0, 0] = False
    values[0, 0, 0] = float('nan')
    values[1, 0, 1] = float('inf')
    values[1, 0, 2] = -float('inf')
    z, valid = normalized(net, values, mask)
    expected_valid = mask & torch.isfinite(values)
    expected = torch.asinh((torch.where(expected_valid, values, center) - center) / scale)
    assert torch.equal(valid, expected_valid)
    torch.testing.assert_close(z, expected, rtol=0, atol=0)
    assert torch.count_nonzero(z[~valid]) == 0
    assert torch.equal(center, net.base.center) and torch.equal(scale, net.base.scale)
    single_z, single_valid = normalized(net, values[0], mask[0])
    torch.testing.assert_close(single_z, expected[0], rtol=0, atol=0)
    assert torch.equal(single_valid, expected_valid[0])


def test_fit_reference_matches_equal_shot_weighted_ridge_and_difference_scale():
    first, second = make_shot(12), make_shot(29)
    second[1][96:, U[0]] = False
    expected_beta, expected_scale, counts = manual_reference([first, second])
    ref = fit_reference([first, second])
    assert ref['beta'].shape == (3, 4) and ref['scale'].shape == (4,)
    torch.testing.assert_close(ref['beta'], expected_beta, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ref['scale'], expected_scale, rtol=1e-12, atol=1e-12)
    assert ref['support']['shot_count'] == 2
    assert ref['support']['joint_valid_counts'] == [256, 96]
    assert ref['support']['valid_difference_counts'] == counts == [31, 11]
    assert ref['support']['design_rank'] == 3
    # Pooling observations would silently give the first shot more influence.
    single_first = fit_reference([first])['beta']
    assert not torch.allclose(ref['beta'], single_first, rtol=1e-5, atol=1e-5)


def test_fit_reference_has_scale_floor_and_uses_only_supplied_records(monkeypatch):
    shot = make_shot(18)
    shot[0][:, C] = torch.tensor([2., -1., .5, 4.], dtype=shot[0].dtype)
    monkeypatch.setattr(io23, 'read_signals', lambda *args, **kwargs: pytest.fail('unexpected data read'))
    ref = fit_reference([shot])
    torch.testing.assert_close(ref['beta'][0], shot[0][0, C], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ref['beta'][1:], torch.zeros(2, 4, dtype=shot[0].dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(ref['scale'], torch.full((4,), 1e-4, dtype=shot[0].dtype), rtol=0, atol=0)
    changed = (shot[0].clone(), shot[1].clone())
    changed[0][:, C] += 1
    assert not torch.equal(fit_reference([changed])['beta'], ref['beta'])


def test_covariate_only_signal_is_removed_by_fitted_conditional_relation():
    def covariate_only(seed, rows):
        generator = torch.Generator().manual_seed(seed)
        z = torch.zeros(rows, 23, dtype=torch.float64)
        z[:, U] = torch.randn(rows, 2, generator=generator, dtype=z.dtype) * 5000
        beta = torch.tensor([[1., -2., .5, 3.], [2., -1., .5, 1.],
                             [-.5, 2., 1., -1.]], dtype=z.dtype)
        z[:, C] = torch.cat((torch.ones(rows, 1, dtype=z.dtype), z[:, U]), 1) @ beta
        return z, torch.ones_like(z, dtype=torch.bool), beta

    a, am, expected_beta = covariate_only(2, 256)
    b, bm, _ = covariate_only(3, 256)
    window, mask, _ = covariate_only(4, 128)
    ref = fit_reference([(a, am), (b, bm)])
    # Ridge shrinks slopes slightly, which also shifts the fitted intercept
    # when the finite-sample covariate mean is not exactly zero.
    torch.testing.assert_close(ref['beta'], expected_beta, rtol=1e-6, atol=3e-5)
    target, edges, rows = relation_target(window[None], mask[None], ref)
    uniform = edges / edges.sum(-1, keepdim=True).clamp_min(1)
    assert rows.all()
    torch.testing.assert_close(target, uniform.to(target.dtype), rtol=0, atol=1e-3)
    unconditioned, _, _ = relation_target(window[None], mask[None], ref, mode='unconditioned')
    assert (unconditioned - uniform).abs().max() > .02


@pytest.mark.parametrize('kind', ['empty', 'short', 'long', 'too_few', 'rank', 'no_pairs', 'nonfinite'])
def test_fit_reference_rejects_inadequate_or_nonfinite_support(kind):
    z, mask = make_shot(7)
    if kind == 'empty':
        shots = []
    elif kind == 'short':
        shots = [(z[:255], mask[:255])]
    elif kind == 'long':
        shots = [(torch.cat((z, z[:1])), torch.cat((mask, mask[:1])))]
    else:
        if kind == 'too_few':
            mask[31:, U[0]] = False
        elif kind == 'rank':
            z[:, U[1]] = 2 * z[:, U[0]]
        elif kind == 'no_pairs':
            mask.reshape(32, 8, 23)[:, 4:, U[0]] = False
        elif kind == 'nonfinite':
            z[0, io23.RAW23.index('CIII')] = float('nan')
        shots = [(z, mask)]
    with pytest.raises((ValueError, FloatingPointError)):
        fit_reference(shots)


def test_reference_and_targets_ignore_values_at_masked_positions():
    z, mask = make_shot(31)
    mask[::7, U[0]] = False
    mask[3::11, C[2]] = False
    altered = z.clone()
    altered[~mask] = float('nan')
    ref = fit_reference([(z, mask)])
    dirty_ref = fit_reference([(altered, mask)])
    torch.testing.assert_close(ref['beta'], dirty_ref['beta'], rtol=0, atol=0)
    torch.testing.assert_close(ref['scale'], dirty_ref['scale'], rtol=0, atol=0)
    clean_result = relation_target(z[None, :128], mask[None, :128], ref)
    dirty_result = relation_target(altered[None, :128], mask[None, :128], ref)
    for expected, actual in zip(clean_result, dirty_result):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_relation_edges_patch_threshold_and_stop_gradient():
    z, mask = make_shot(9, 128)
    z.requires_grad_()
    mask[0:3, U[0]] = False  # Patch 0 has five joint-valid rows.
    mask[8:10, C[0]] = False  # Patch 1 has six and is valid.
    ref = {'beta': torch.zeros(3, 4, dtype=z.dtype), 'scale': torch.ones(4, dtype=z.dtype)}
    target, edge, row = relation_target(z[None], mask[None], ref)
    assert target.shape == edge.shape == (1, 4, 16) and row.shape == (1, 4)
    assert not target.requires_grad and target.grad_fn is None
    assert not edge[:, :, 0].any() and not edge[:, :, 1].any()
    assert edge[0, 0, 2]  # Six joint-valid rows in patch 1 are sufficient.
    assert not edge[0, 0, 11:].any()  # Query patch 12 requires j <= 10.
    assert row.all() and (edge.sum(-1) >= 4).all()
    torch.testing.assert_close(target.sum(-1), torch.ones(1, 4, dtype=z.dtype))
    assert torch.count_nonzero(target[~edge]) == 0


def test_relation_target_matches_independent_clipped_similarity_oracle():
    z = torch.zeros(1, 128, 23, dtype=torch.float64)
    mask = torch.ones_like(z, dtype=torch.bool)
    changes = torch.tensor([[((i % 5) - 2) * .9, ((i % 3) - 1) * .7,
                             ((i % 4) - 1.5) * .5, ((i % 7) - 3) * .3]
                            for i in range(16)], dtype=z.dtype)
    levels = changes.cumsum(dim=0)
    for patch in range(16):
        z[0, patch * 8:(patch + 1) * 8, C] = levels[patch]
    scale = torch.tensor([.5, 2., .3, .7], dtype=z.dtype)
    ref = {'beta': torch.zeros(3, 4, dtype=z.dtype), 'scale': scale}
    target, edges, rows = relation_target(z, mask, ref)
    saw_positive_clip = saw_negative_clip = False
    expected = torch.zeros_like(target)
    expected_edges = torch.zeros_like(edges)
    for row, query in enumerate(range(12, 16)):
        keys = list(range(1, query - 1))
        scores = torch.stack([(changes[query] * changes[key] / scale).sum() for key in keys])
        saw_positive_clip |= bool((scores > 4).any())
        saw_negative_clip |= bool((scores < -4).any())
        expected[0, row, keys] = torch.softmax(scores.clamp(-4, 4), dim=0)
        expected_edges[0, row, keys] = True
    assert saw_positive_clip and saw_negative_clip and rows.all()
    assert torch.equal(edges, expected_edges)
    torch.testing.assert_close(target, expected, rtol=1e-12, atol=1e-12)


def test_reverse_only_reverses_last_25_values_and_masks_without_mutation():
    z, mask = make_shot(47, 128)
    z, mask = z[None], mask[None]
    mask[:, 104:110, U[0]] = False
    original_z, original_mask = z.clone(), mask.clone()
    ref = {'beta': torch.zeros(3, 4, dtype=z.dtype), 'scale': torch.ones(4, dtype=z.dtype)}
    actual = relation_target(z, mask, ref, mode='reverse')
    flipped_z = torch.cat((z[:, :-25], z[:, -25:].flip(1)), 1)
    flipped_mask = torch.cat((mask[:, :-25], mask[:, -25:].flip(1)), 1)
    expected = relation_target(flipped_z, flipped_mask, ref)
    wrong_mask = relation_target(flipped_z, mask, ref)
    for observed, wanted in zip(actual, expected):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
    assert not torch.equal(actual[1], wrong_mask[1])
    assert torch.equal(z, original_z) and torch.equal(mask, original_mask)


def test_unconditioned_uses_zero_beta_and_same_scale():
    z, mask = make_shot(21, 128)
    ref = {'beta': torch.randn(3, 4, dtype=z.dtype), 'scale': torch.tensor([.1, .2, .3, .4], dtype=z.dtype)}
    zero = {'beta': torch.zeros_like(ref['beta']), 'scale': ref['scale']}
    unconditioned = relation_target(z[None], mask[None], ref, mode='unconditioned')
    expected = relation_target(z[None], mask[None], zero)
    for observed, wanted in zip(unconditioned, expected):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


def test_common_support_is_renormalized_and_rows_with_under_four_keys_are_zero():
    z, mask = make_shot(41, 128)
    ref = {'beta': torch.zeros(3, 4, dtype=z.dtype), 'scale': torch.ones(4, dtype=z.dtype)}
    target, edges, _ = relation_target(z[None], mask[None], ref)
    common = torch.zeros_like(edges)
    for row, keep in enumerate((3, 4, 6, 0)):
        common[0, row, edges[0, row].nonzero().flatten()[:keep]] = True
    restricted = restrict_target(target, common)
    effective = common & (common.sum(-1, keepdim=True) >= 4)
    assert effective.sum(-1).tolist() == [[0, 4, 6, 0]]
    assert torch.count_nonzero(restricted[~effective]) == 0
    for row in (1, 2):
        expected = target[0, row] * common[0, row]
        torch.testing.assert_close(restricted[0, row], expected / expected.sum())
    assert not restricted.requires_grad


def test_invalid_and_all_missing_targets_have_no_invented_mass():
    z, mask = make_shot(13, 128)
    mask.fill_(False)
    z.fill_(float('nan'))
    ref = {'beta': torch.zeros(3, 4, dtype=z.dtype), 'scale': torch.ones(4, dtype=z.dtype)}
    target, edges, valid = relation_target(z[None], mask[None], ref)
    assert torch.isfinite(target).all() and not edges.any() and not valid.any()
    assert torch.count_nonzero(target) == 0
    restricted = restrict_target(target, torch.ones_like(edges))
    assert torch.count_nonzero(restricted) == 0 and torch.isfinite(restricted).all()


def test_attention_kl_exact_masked_toy_and_invalid_rows_are_differentiable():
    q = torch.zeros(2, 17, 64, dtype=torch.float64, requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    with torch.no_grad():
        q[0, 12, 0] = 4
        k[0, 1:5, 0] = torch.arange(4, dtype=q.dtype)
        k[0, 5, 0] = 1e6  # Outside the support, must not enter log-softmax.
        q[:, :, 16:] = 1000  # Other heads must not enter the KL.
        k[:, :, 16:] = 1000
    edge = torch.zeros(2, 4, 16, dtype=torch.bool)
    edge[0, 0, 1:5] = True
    target = torch.zeros(2, 4, 16, dtype=q.dtype, requires_grad=True)
    with torch.no_grad():
        target[0, 0, 1:5] = torch.tensor([.1, .2, .3, .4], dtype=q.dtype)
    loss = attention_kl(q, k, target, edge)
    p = target.detach()[0, 0, 1:5]
    expected = (p * (p.log() - torch.log_softmax(torch.arange(4, dtype=q.dtype), 0))).sum()
    torch.testing.assert_close(loss, torch.stack((expected, expected * 0)), rtol=1e-12, atol=1e-12)
    loss.sum().backward()
    assert target.grad is None
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
    assert torch.count_nonzero(q.grad[1]) == torch.count_nonzero(k.grad[1]) == 0
    assert torch.count_nonzero(q.grad[:, :, 16:]) == torch.count_nonzero(k.grad[:, :, 16:]) == 0
    assert torch.count_nonzero(k.grad[0, 5]) == 0


def test_attention_kl_averages_valid_rows_and_accepts_zero_target_entries():
    q = torch.zeros(1, 17, 64, dtype=torch.float64, requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    edges = torch.zeros(1, 4, 16, dtype=torch.bool)
    edges[0, :2, 1:5] = True
    target = torch.zeros(1, 4, 16, dtype=q.dtype)
    target[0, 0, 1] = 1
    target[0, 1, 1:5] = .25
    loss = attention_kl(q, k, target, edges)
    torch.testing.assert_close(loss, torch.tensor([torch.log(torch.tensor(4., dtype=q.dtype)) / 2]))
    loss.sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()


def test_bounded_step_uses_only_first_head_and_exact_per_example_norm():
    g = torch.arange(1, 3 * 8192 + 1, dtype=torch.float64).reshape(3, 8192)
    g[1] = 0
    g[2].reshape(2, 64, 64)[:, :16] = 0  # Nonzero only outside the permitted head.
    original = g.clone()
    step = bounded_step(g)
    matrices = step.reshape(3, 2, 64, 64)
    assert torch.count_nonzero(matrices[:, :, 16:]) == 0
    assert torch.count_nonzero(step[1:]) == 0
    assert step[0].norm().item() == pytest.approx(.05, abs=1e-15)
    selected = original[0].reshape(2, 64, 64).clone()
    selected[:, 16:] = 0
    torch.testing.assert_close(step[0], (-.05 * selected / selected.norm()).flatten(), rtol=1e-12, atol=1e-15)
    assert torch.equal(g, original)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_bounded_step_rejects_nonfinite_even_outside_selected_head(value):
    g = torch.zeros(1, 8192)
    g[0, 4095] = value
    with pytest.raises((ValueError, FloatingPointError)):
        bounded_step(g)


def test_bounded_step_handles_tiny_nonzero_gradients_without_fabricating_zero():
    g = torch.zeros(1, 8192, dtype=torch.float64)
    g[0, 0] = 1e-300
    g[0, 4096] = -2e-300
    step = bounded_step(g)
    assert torch.isfinite(step).all()
    assert step.norm().item() == pytest.approx(.05, abs=1e-15)
    assert step[0, 0] < 0 < step[0, 4096]


def test_cpu_qk_adapter_integration_updates_only_head_zero():
    from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    torch.manual_seed(131)
    net = MatrixAttention(Simple23(Config(), torch.zeros(23), torch.ones(23)))
    frozen = {name: value.clone() for name, value in net.state_dict().items()}
    reference_data = torch.randn(256, 23)
    reference_mask = torch.ones_like(reference_data, dtype=torch.bool)
    reference_z, reference_valid = normalized(net, reference_data, reference_mask)
    ref = fit_reference([(reference_z, reference_valid)])
    values = torch.randn(1, 128, 23)
    masks = torch.ones_like(values, dtype=torch.bool)
    z, valid = normalized(net, values, masks)
    target, edges, supported = relation_target(z, valid, ref)
    assert supported.all()
    state = torch.zeros(1, 8192, requires_grad=True)
    _, packed, _ = net.first(values, masks, state, 'qk')
    loss = attention_kl(packed['query'], packed['key'], target, edges)
    gradient, = torch.autograd.grad(loss.sum(), state)
    step = bounded_step(gradient)
    assert step.norm().item() == pytest.approx(.05, abs=1e-7)
    _, updated, _ = net.first(values, masks, step, 'qk')
    for name in ('query', 'key'):
        delta = updated[name] - packed[name]
        assert torch.count_nonzero(delta[..., :16]) > 0
        assert torch.count_nonzero(delta[..., 16:]) == 0
    assert all(torch.equal(tensor, frozen[name]) for name, tensor in net.state_dict().items())


def test_cpu_driver_uses_tensor_targets_common_edges_and_zero_support():
    from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
    from experiments.risk_aligned_online.relational_attention_20261030 import run
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    torch.manual_seed(314)
    net = MatrixAttention(Simple23(Config(), torch.zeros(23), torch.ones(23)))
    reference_data = torch.randn(256, 23)
    reference_mask = torch.ones_like(reference_data, dtype=torch.bool)
    ref = fit_reference([normalized(net, reference_data, reference_mask)])
    values = torch.randn(2, 128, 23)
    masks = torch.ones_like(values, dtype=torch.bool)
    masks[1] = False
    data = {'x': values, 'mask': masks}
    mean, mean_gradient = run.fixed_controls(net, {1: data}, [1], ref, None)
    targets, edges = run.targets(net, data, ref, mean)
    assert targets.keys() == {'real', 'reverse', 'unconditioned', 'mean_target'}
    assert all(isinstance(target, torch.Tensor) for target in targets.values())
    assert edges[0].any() and not edges[1].any()
    assert bool(((edges.sum(-1) >= 4) | (edges.sum(-1) == 0)).all())
    for target in targets.values():
        torch.testing.assert_close(target.sum(-1), edges.any(-1).to(target.dtype))
    result = run.evaluate(net, data, data, ref, mean, mean_gradient, None)
    assert result['support'].tolist() == [True, False]
    for arm in run.ARMS:
        assert result[arm + '_norm'][1] == 0
        assert result[arm][1] == result['cold'][1]


def test_helper_caller_slices_future_before_fitting_and_constructing_targets():
    history, history_mask = make_shot(50, 320)
    changed = history.clone()
    changed[256:] = 1e10  # Must be excluded by the caller before the fit.
    ref_a = fit_reference([(history[:256], history_mask[:256])])
    ref_b = fit_reference([(changed[:256], history_mask[:256])])
    torch.testing.assert_close(ref_a['beta'], ref_b['beta'], rtol=0, atol=0)
    torch.testing.assert_close(ref_a['scale'], ref_b['scale'], rtol=0, atol=0)
    record, mask = make_shot(51, 320)  # Distinct held shot from the reference.
    endpoint = 200
    changed = record.clone()
    changed[endpoint + 1:] = -1e10
    window = slice(endpoint - 127, endpoint + 1)
    a = relation_target(record[None, window], mask[None, window], ref_a)
    b = relation_target(changed[None, window], mask[None, window], ref_a)
    for observed, wanted in zip(a, b):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
