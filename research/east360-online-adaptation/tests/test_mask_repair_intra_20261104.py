import pytest
import torch

from experiments.risk_aligned_online.mask_repair_20261104 import core, intra
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = load_parent(DEFAULT_PARENT)
    repair = core.TokenRepair().eval().requires_grad_(False)
    with torch.no_grad():
        repair.net[-1].weight.normal_(0, 0.04)
        repair.net[-1].bias.normal_(0, 0.03)
    x = base.center + 0.3 * base.scale * torch.randn(1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    a[:, :, 2] = False
    yield base, repair, x, a
    torch.backends.mha.set_fastpath_enabled(previous)


def test_zero_state_exactly_matches_static_but_keeps_gradient(fixture):
    base, repair, x, a = fixture
    state = intra.initial_state(repair).requires_grad_(True)
    local = intra.LocalRepair(repair, state)
    actual = core.forward(base, local, x, a)
    expected = core.forward(base, repair, x, a)
    assert torch.equal(actual['embedding_after'], expected['embedding_after'])
    assert torch.equal(actual['logit'], expected['logit'])
    gradient, = torch.autograd.grad(actual['embedding_after'].square().mean(), state)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert all(p.grad is None for p in (*base.parameters(), *repair.parameters()))


def test_local_change_is_bounded_without_changing_real_mask(fixture):
    base, repair, x, a = fixture
    before = core.forward(base, repair, x, a)
    state = intra.initial_state(repair) + 0.2
    result = core.forward(base, intra.LocalRepair(repair, state), x, a)
    assert result['embedding_delta'].double().square().mean(-1).sqrt().max() <= 0.25
    difference = result['embedding_delta'] - before['embedding_delta']
    assert difference.double().square().mean(-1).sqrt().max() <= 0.0250001
    assert not a[:, :, 2].any() and a[:, :, 1].all()


def test_support_excludes_padding_and_ciii_only_frames(fixture):
    _, _, _, a = fixture
    a[:, :8] = False
    a[:, 8:16] = False
    a[:, 8:12, 1] = True
    train, held = intra.support(a)
    assert not train[0, :2].any()
    assert not train[0, 12:].any()
    assert held[0, 12:].all() and not held[0, :12].any()


@pytest.mark.parametrize('mode', intra.MODES)
def test_step_only_changes_local_state_and_no_update_discards_it(fixture, mode):
    base, repair, x, a = fixture
    old_x, old_a = x.clone(), a.clone()
    original = {k: v.clone() for k, v in repair.state_dict().items()}
    state, info = intra.step(base, repair, intra.initial_state(repair), x, a, mode=mode)
    assert info['status'] == 'UPDATED' if mode != 'no_update' else info['status'] == 'CONTROL_NO_UPDATE'
    assert info['gradient_norm'] > 0
    assert state.double().norm() <= 0.2500001
    assert torch.count_nonzero(state) > 0 if mode != 'no_update' else torch.count_nonzero(state) == 0
    assert torch.equal(x, old_x) and torch.equal(a, old_a)
    assert all(torch.equal(original[k], v) for k, v in repair.state_dict().items())
    assert all(p.grad is None for p in (*base.parameters(), *repair.parameters()))


def test_no_ciii_cannot_fabricate_target_or_update(fixture):
    base, repair, x, a = fixture
    a[:, :, 1] = False
    state, info = intra.step(base, repair, intra.initial_state(repair), x, a)
    assert info['status'] == 'INSUFFICIENT_OBSERVATIONS'
    assert torch.count_nonzero(state) == 0


def test_new_shot_state_has_no_alias_or_carryover(fixture):
    _, repair, _, _ = fixture
    a, b = intra.initial_state(repair), intra.initial_state(repair)
    a[0] = 0.2
    assert torch.count_nonzero(b) == 0 and a.data_ptr() != b.data_ptr()


@pytest.mark.parametrize('local', [False, True])
def test_embedding_only_is_exactly_the_full_source_intervention(fixture, local):
    base, repair, x, a = fixture
    a[:, :21] = False
    a[:, 43:49, 1] = False
    if local:
        repair = intra.LocalRepair(repair, intra.initial_state(repair).requires_grad_(True))
    result = intra.embedding_after(base, repair, x, a)
    expected = core.forward(base, repair, x, a)['embedding_after']
    assert torch.equal(result, expected)
    if local:
        actual, = torch.autograd.grad(result.square().mean(), repair.state)
        reference, = torch.autograd.grad(expected.square().mean(), repair.state)
        assert torch.equal(actual, reference)
