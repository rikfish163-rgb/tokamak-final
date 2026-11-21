import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_embedding_fast_20261114 as run
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_measured_matrix_is_exactly_a_value_only_embedding_weight_update():
    torch.manual_seed(9)
    x = torch.randn(2, 16, 368)
    v = x.reshape(2, 16, 8, 46)
    v[..., 23:] = torch.randint(2, v[..., 23:].shape)
    v[..., :23] *= v[..., 23:]
    d = torch.randn(2, 64, len(run.VALUES)*8)
    exact = torch.zeros(2, 64, 8, 46)
    exact[..., list(run.VALUES)] = d.unflatten(-1, (8, len(run.VALUES)))
    torch.testing.assert_close(run.measured_values(x) @ d.mT,
                               x @ exact.flatten(-2).mT, atol=1e-5, rtol=1e-5)


def test_reference_projection_preserves_patches_without_ending_update():
    g = torch.Generator().manual_seed(19)
    ref = torch.randn(16, 32, generator=g)
    ref[0] = 0.
    ref[1] = ref[2]
    d = torch.randn(3, 11, 32, generator=g)
    span = run.reference_span(ref)
    assert len(span)==14
    update = run.protect_reference(d, span)
    torch.testing.assert_close(update@ref.mT, torch.zeros(3, 11, 16), atol=8e-6, rtol=0.)
    assert update.norm()>0.
    torch.testing.assert_close(run.protect_reference(update, span),update,atol=1e-6,rtol=0.)


def test_wide_value_write_cannot_consult_unavailable_coordinates_or_mnm():
    cols = run.ALL_AVAILABLE_VALUES
    x = torch.ones(1, 2, 8, 46)
    x[0,0,0,23+cols[0]] = 0.
    source = run.measured_values(x.flatten(-2),cols)
    x[0,0,0,cols[0]] = 100.
    x[..., 2] = 999.
    torch.testing.assert_close(source,run.measured_values(x.flatten(-2),cols))
    assert source.shape==(1,2,176)


def test_projection_does_not_amplify_a_purely_protected_gradient():
    g = torch.Generator().manual_seed(31)
    basis = run.reference_span(torch.randn(16, 32, generator=g))
    in_span = (torch.randn(3, 11, 16, generator=g).double()@basis.double()).float()
    assert run.stable_direction(run.protect_reference(in_span,basis),in_span).count_nonzero()==0


def test_fixed_handoff_support_cannot_read_unreleased_or_missing_packets():
    q = np.array([127, 159, 191, 223, 255])
    previous, good = run.feedback_support(q, np.array([True, True, False, True, True]))
    assert previous.tolist()==[-1, 0, -1, -1, 3]
    ends = np.arange(290)
    slots, scale = run.action_clock(q, good, ends)
    assert slots[:191].max()<=2 and not scale[:191].any()
    assert scale[222:287].max()==0.
    assert scale[191]>0. and scale[198]==1. and scale[287]>0.


def test_shadow_forecast_backprop_reaches_value_matrix_not_classifier_weights():
    torch.manual_seed(11)
    model = Simple23(Config(), torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(3, 128, 23)
    truth = torch.ones_like(x, dtype=torch.bool)
    delta = torch.zeros(3,64,len(run.VALUES)*8, requires_grad=True)
    zero, _ = run.forward_embed(model, x, truth, lambda p:p, delta[None])
    native = model(x, truth)['logit']
    torch.testing.assert_close(native, model.risk_head(zero[1]).squeeze(-1),rtol=0.,atol=3e-7)
    grad, = torch.autograd.grad(zero[1].square().sum(), delta)
    assert grad.norm()>1e-6 and all(p.grad is None for p in model.parameters())
