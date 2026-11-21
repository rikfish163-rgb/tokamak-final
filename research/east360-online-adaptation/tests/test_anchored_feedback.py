"""CPU contracts for mean-anchored targets and fit-shot donor selection."""

import pytest
import torch

from experiments.risk_aligned_online.anchored_feedback_20261030.learning import (
    AnchoredTarget,
    make_reference_context,
    nearest_donor_indices,
)
from experiments.risk_aligned_online.context_vector_20261030.learning import (
    ContextTarget,
    representation_step,
)


@pytest.fixture(scope='module', autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def deterministic_torch():
    with torch.random.fork_rng():
        torch.manual_seed(131)
        yield


def test_identical_initialization_parameter_count_and_state_dict_compatibility():
    torch.manual_seed(51)
    old = ContextTarget()
    torch.manual_seed(51)
    anchored = AnchoredTarget()
    assert isinstance(anchored, ContextTarget)
    assert sum(parameter.numel() for parameter in anchored.parameters()) == 14316
    assert anchored.state_dict().keys() == old.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(anchored.state_dict()[name], value)
    with torch.no_grad():
        old.network[3].weight.normal_(0, .2)
        old.network[3].bias.normal_(0, .2)
    anchored.load_state_dict(old.state_dict(), strict=True)
    restored = ContextTarget()
    restored.load_state_dict(anchored.state_dict(), strict=True)
    for name, value in old.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)


def test_reference_keeps_origins_and_uses_only_detached_mean_feedback():
    context = torch.randn(3, 358, requires_grad=True)
    mean = torch.randn(358, requires_grad=True)
    original_context, original_mean = context.detach().clone(), mean.detach().clone()
    reference = make_reference_context(context, mean)
    assert reference.shape == context.shape
    assert not reference.requires_grad
    assert torch.equal(reference[:, :128], context[:, :128])
    assert torch.equal(reference[:, 128:], mean[None, 128:].expand(3, -1))
    changed_mean = mean.detach().clone()
    changed_mean[:128] += 500
    assert torch.equal(make_reference_context(context, changed_mean), reference)
    reference[0, 0] += 1
    reference[0, 128] += 1
    assert torch.equal(context, original_context)
    assert torch.equal(mean, original_mean)


def test_shared_network_formula_bounds_support_and_no_forward_mutation():
    target = AnchoredTarget().double()
    with torch.no_grad():
        target.network[3].weight.normal_(0, .4)
        target.network[3].bias.normal_(0, .4)
    context = torch.randn(358, 3, dtype=torch.float64).T.requires_grad_()
    mean = torch.randn(358, dtype=torch.float64, requires_grad=True)
    support = torch.tensor([True, False, True])
    originals = [value.detach().clone() for value in (context, mean, support)]
    before = {name: value.clone() for name, value in target.state_dict().items()}
    reference = make_reference_context(context, mean)
    expected = .5 * (target.network(context.detach()) - target.network(reference))
    expected = expected.masked_fill(~support[:, None], 0.)
    actual = target(context, support, mean)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (3, 64)
    assert torch.isfinite(actual).all() and (actual.abs() <= 1).all()
    assert torch.count_nonzero(actual[1]) == 0
    individual = torch.cat([target(context[i:i + 1], support[i:i + 1], mean)
                            for i in range(len(context))])
    torch.testing.assert_close(actual, individual, atol=1e-12, rtol=1e-12)
    for value, original in zip((context, mean, support), originals):
        assert torch.equal(value, original)
    assert all(torch.equal(value, before[name]) for name, value in target.state_dict().items())
    actual.sum().backward()
    assert context.grad is None and mean.grad is None
    assert target.network[3].weight.grad.abs().sum() > 0


def test_zero_initial_adaptation_has_head_gradient_and_optimizer_starts():
    target = AnchoredTarget().double()
    context = torch.randn(3, 358, dtype=torch.float64, requires_grad=True)
    mean = torch.randn(358, dtype=torch.float64, requires_grad=True)
    support = torch.tensor([True, True, False])
    optimizer = torch.optim.SGD(target.parameters(), lr=.1)
    initial_head = target.network[3].weight.detach().clone()

    # An independent linear representation makes the adaptation derivative
    # explicit; no labels, clocks, or per-example optimizer are needed here.
    state = torch.zeros(3, 8192, dtype=torch.float64, requires_grad=True)
    h = 2 * state[:, :64]
    delta = target(context, support, mean)
    assert torch.count_nonzero(delta) == 0
    update = representation_step(h, h.detach(), delta, state, create_graph=True)
    assert torch.count_nonzero(update) == 0
    objective = -update[:, :64].sum()
    objective.backward()
    assert context.grad is None and mean.grad is None
    assert torch.count_nonzero(target.network[1].weight.grad) == 0
    assert target.network[3].weight.grad.abs().sum() > 0
    assert torch.isfinite(target.network[3].weight.grad).all()
    optimizer.step()
    assert not torch.equal(target.network[3].weight, initial_head)
    assert torch.count_nonzero(target(context, support, mean)[:2]) > 0

    # This identity must survive optimization and arbitrary current origins.
    matched = torch.randn(5, 358, dtype=torch.float64)
    matched[:, 128:] = mean.detach()[128:]
    matched_support = torch.ones(5, dtype=torch.bool)
    assert torch.count_nonzero(target(matched, matched_support, mean)) == 0
    optimizer.zero_grad(set_to_none=True)
    target(matched, matched_support, mean).sum().backward()
    assert all(parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
               for parameter in target.parameters())


def test_unsupported_rows_have_zero_parameter_gradient_after_training():
    target = AnchoredTarget()
    with torch.no_grad():
        target.network[3].weight.normal_()
    output = target(torch.randn(2, 358), torch.zeros(2, dtype=torch.bool), torch.randn(358))
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert all(parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
               for parameter in target.parameters())


@pytest.mark.parametrize('defect', [
    'context_type', 'context_shape', 'context_integer', 'context_nan',
    'mean_type', 'mean_shape', 'mean_integer', 'mean_inf', 'mean_dtype',
    'support_type', 'support_shape', 'support_dtype', 'support_device', 'model_dtype',
])
def test_target_rejects_invalid_inputs(defect):
    context, mean, support = torch.randn(2, 358), torch.randn(358), torch.tensor([True, False])
    if defect == 'context_type':
        context = context.tolist()
    elif defect == 'context_shape':
        context = context[:, :-1]
    elif defect == 'context_integer':
        context = context.long()
    elif defect == 'context_nan':
        context[1, 200] = float('nan')
    elif defect == 'mean_type':
        mean = None
    elif defect == 'mean_shape':
        mean = mean[None]
    elif defect == 'mean_integer':
        mean = mean.long()
    elif defect == 'mean_inf':
        mean[0] = float('inf')
    elif defect == 'mean_dtype':
        mean = mean.double()
    elif defect == 'support_type':
        support = [True, False]
    elif defect == 'support_shape':
        support = support[:, None]
    elif defect == 'support_dtype':
        support = support.float()
    elif defect == 'support_device':
        support = torch.empty(2, dtype=torch.bool, device='meta')
    else:
        context, mean = context.double(), mean.double()
    with pytest.raises(ValueError):
        AnchoredTarget()(context, support, mean)


def test_nonfinite_network_result_raises_without_fallback():
    target = AnchoredTarget()
    with torch.no_grad():
        target.network[1].weight[0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='finite'):
        target(torch.randn(2, 358), torch.ones(2, dtype=torch.bool), torch.randn(358))


def donor_example():
    contexts = torch.randn(5, 358)
    contexts[:, :128] = torch.tensor([0., .1, 2., -2., 0.])[:, None]
    return contexts, [11, 11, 22, 33, 44], [33, 22, 11]


def test_donors_exclude_entire_own_shot_and_held_shots_with_original_index_ties():
    contexts, ids, fit_ids = donor_example()
    originals = contexts.clone(), ids.copy(), fit_ids.copy()
    expected = torch.tensor([2, 2, 1, 0, 0], dtype=torch.int64)
    actual = nearest_donor_indices(contexts, ids, fit_ids)
    assert actual.device.type == 'cpu' and actual.dtype == torch.int64
    assert torch.equal(actual, expected)
    assert torch.equal(nearest_donor_indices(contexts, ids, list(reversed(fit_ids))), expected)
    for i, donor in enumerate(actual.tolist()):
        assert ids[donor] in fit_ids and ids[donor] != ids[i]
    assert torch.equal(contexts, originals[0]) and ids == originals[1] and fit_ids == originals[2]


def test_donors_use_mean_squared_distance_over_all_128_origin_coordinates_only():
    contexts = torch.randn(4, 358, dtype=torch.float64, requires_grad=True)
    ids, fit_ids = [10, 20, 30, 40], [10, 20, 30]
    with torch.no_grad():
        contexts[:, :128] = 0
        contexts[0, 0] = 5
        contexts[1, :128] = 1
        contexts[2, :128] = 2
    actual = nearest_donor_indices(contexts, ids, fit_ids)
    assert actual[3] == 0  # 25/128 < 1; using only the first coordinate is wrong.
    for row, shot in enumerate(ids):
        candidates = [i for i, sid in enumerate(ids) if sid in fit_ids and sid != shot]
        expected = min(candidates, key=lambda i: ((contexts[row, :128] - contexts[i, :128]) ** 2).mean().item())
        assert actual[row] == expected
    changed = contexts.detach().clone()
    changed[:, 128:] = torch.randn(4, 230) * 1e6
    assert torch.equal(nearest_donor_indices(changed, ids, fit_ids), actual)
    assert contexts.grad is None and not actual.requires_grad


def test_held_context_changes_cannot_change_fit_donor_pool():
    contexts, ids, fit_ids = donor_example()
    original = nearest_donor_indices(contexts, ids, fit_ids)
    contexts[4, :128] = 2.01
    contexts[4, 128:] = 1000
    changed = nearest_donor_indices(contexts, ids, fit_ids)
    assert torch.equal(changed[:4], original[:4])
    assert changed[4] == 2
    assert not (changed == 4).any()


@pytest.mark.parametrize('fit_ids', [[], [11], [999]])
def test_donors_fail_if_any_row_has_no_other_fit_shot(fit_ids):
    contexts, ids, _ = donor_example()
    with pytest.raises(ValueError, match='candidate'):
        nearest_donor_indices(contexts, ids, fit_ids)


@pytest.mark.parametrize('defect', ['type', 'shape', 'integer', 'nan', 'device', 'ids_length'])
def test_donor_input_validation(defect):
    contexts, ids, fit_ids = donor_example()
    if defect == 'type':
        contexts = contexts.tolist()
    elif defect == 'shape':
        contexts = contexts[:, :-1]
    elif defect == 'integer':
        contexts = contexts.long()
    elif defect == 'nan':
        contexts[0, 200] = float('nan')
    elif defect == 'device':
        contexts = torch.empty(5, 358, device='meta')
    else:
        ids = ids[:-1]
    with pytest.raises(ValueError):
        nearest_donor_indices(contexts, ids, fit_ids)


def test_empty_inputs_preserve_output_shapes():
    context = torch.empty(0, 358)
    assert AnchoredTarget()(context, torch.empty(0, dtype=torch.bool), torch.zeros(358)).shape == (0, 64)
    indices = nearest_donor_indices(context, [], [])
    assert indices.shape == (0,) and indices.dtype == torch.int64
