"""Frozen-path diagnostics must not turn into a hidden model intervention."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.input_path_audit import (
    choose_donor, capture, readout, swap_observed_channel, stratified_indices,
)
from experiments.risk_aligned_online.simple23 import Simple23, Config
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel


def test_strata_selected_without_labels():
    indices = stratified_indices()
    assert len(indices) == 158 and len(set(indices)) == 158
    assert indices[:30] == list(range(30))
    assert indices[-1] == 2952


def test_only_prior_donors_and_target_does_not_control_distance():
    z = np.zeros((3, 128, 23))
    m = np.ones_like(z, bool)
    z[0, :, 0] = .1
    z[0, :, 12] = 3.
    z[1, :, 12] = 4.
    z[2] = z[1]  # Future exact match must not be chosen.
    match = choose_donor(z, m, 1)
    assert match['donor'] == 0 and match['accepted']
    assert match['distance'] == pytest.approx(.1 / np.sqrt(22))


def test_identical_mask_and_overlap_required():
    z = np.zeros((2, 128, 23))
    m = np.ones_like(z, bool)
    m[0, 0, 0] = False
    assert choose_donor(z, m, 1)['reason'] == 'no_same_mask_prior'
    m[:] = False
    assert choose_donor(z, m, 1)['reason'] == 'no_same_mask_prior'


def test_unsupported_and_zero_dose_are_not_effect_estimates():
    z = np.zeros((2, 128, 23))
    m = np.ones_like(z, bool)
    assert choose_donor(z, m, 1)['reason'] == 'insufficient_dose'
    z[0] = 10.
    assert choose_donor(z, m, 1)['reason'] == 'context_distance'
    assert choose_donor(z, m, 0)['reason'] == 'no_same_mask_prior'


def test_swap_does_not_fabricate_missing_or_modify_other_columns():
    x, donor = torch.zeros(1, 128, 23), torch.ones(1, 128, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :100] = False
    changed = swap_observed_channel(x, donor, mask)
    assert torch.equal(changed[:, :, :12], x[:, :, :12])
    assert torch.equal(changed[:, :, 13:], x[:, :, 13:])
    assert not changed[:, :100, 12].any()
    assert changed[:, 100:, 12].eq(1).all()
    assert not x.any()


@pytest.fixture
def model():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    base = Simple23(Config(fast_site='tokens'), np.zeros(23), np.ones(23))
    with torch.no_grad():
        base.fast_origin.uniform_(-.2, .2)
    return FfnModel(base)


def test_capture_identity_matches_original_and_reconstructed_readout(model):
    x = torch.randn(2, 128, 23)
    m = torch.ones_like(x, dtype=torch.bool)
    m[0, :100] = False
    before = {k: v.clone() for k, v in model.state_dict().items()}
    direct = model(x, m, model.initial_delta())['logit'].detach()
    captured = capture(model, x, m)
    torch.testing.assert_close(captured['logit'], direct)
    torch.testing.assert_close(readout(model, captured['h'], captured['r']), direct)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert not model.base.encoder.layers[-1].linear2._forward_hooks
    assert all(p.grad is None for p in model.parameters())


def test_last_patch_ffn_intervention_does_not_change_cls_risk(model):
    x = torch.randn(1, 128, 23)
    m = torch.ones_like(x, dtype=torch.bool)
    direct = capture(model, x, m)['logit']

    def patch_only(_module, _inputs, output):
        altered = output.clone()
        altered[:, :-1] += 7.
        return altered

    handle = model.base.encoder.layers[-1].linear2.register_forward_hook(patch_only)
    try:
        changed = capture(model, x, m)['logit']
    finally:
        handle.remove()
    torch.testing.assert_close(changed, direct, atol=1e-6, rtol=0.)


def test_capture_cleanup_on_error(model):
    with pytest.raises(ValueError):
        capture(model, torch.zeros(1, 2, 23), torch.ones(1, 2, 23, dtype=torch.bool))
    assert not model.base.encoder.layers[-1].linear2._forward_hooks
    assert not model.base.encoder.layers[-1]._forward_hooks
