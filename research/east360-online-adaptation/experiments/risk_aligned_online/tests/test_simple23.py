import torch
from dataclasses import replace

from experiments.risk_aligned_online.simple23 import Simple23, Config, fit_forecaster


torch.set_num_threads(2)


def case():
    torch.manual_seed(131)
    model = Simple23(Config(window=16, patch=4, dim=16, heads=2, horizon=4),
                     torch.zeros(23), torch.ones(23))
    x = torch.randn(2, 16, 23)
    return model, x, torch.ones_like(x, dtype=torch.bool)


def test_single_risk_head_and_direct_forecast():
    model, x, mask = case()
    out = model(x, mask)
    assert out['logit'].shape == (2,)
    assert out['forecast'].shape == (2, 4, 23)
    assert not any(k in n for n, _ in model.named_parameters()
                   for k in ('private', 'shared', 'controller', 'expert'))


def test_forecast_training_cannot_change_risk_predictions():
    model, x, mask = case()
    before = model(x, mask)['logit'].detach().clone()
    target = torch.randn(2, 4, 23)
    optimizer = torch.optim.Adam(model.forecast_head.parameters(), lr=.01)
    fit_forecaster(model, optimizer, x, mask, target, torch.ones_like(target, dtype=torch.bool))
    torch.testing.assert_close(before, model(x, mask)['logit'], rtol=0, atol=0)


def test_fast_signal_step_changes_risk_but_not_persistent_state():
    model, x, mask = case()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    fast = model.initial_fast(2).requires_grad_(True)
    out = model(x, mask, fast)
    grad, = torch.autograd.grad(out['forecast'].square().mean(), fast)
    after = model(x, mask, fast - .1 * grad)
    assert not torch.equal(out['logit'], after['logit'])
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, weights[k], rtol=0, atol=0)
    torch.testing.assert_close(model.initial_fast(2), torch.zeros_like(fast))


def test_missing_input_and_other_batch_member_do_not_affect_sample():
    model, x, mask = case()
    mask[:, :, 4] = False
    altered = x.clone()
    altered[:, :, 4] = float('nan')
    torch.testing.assert_close(model(x, mask)['logit'], model(altered, mask)['logit'])
    torch.testing.assert_close(model(x, mask)['logit'][:1], model(x[:1], mask[:1])['logit'])


def test_forecast_loss_has_no_target_gradient_and_no_missing_signal_update():
    model, x, mask = case()
    truth = torch.randn(2, 4, 23, requires_grad=True)
    loss = model.forecast_loss(model(x, mask)['forecast'], truth, torch.ones_like(truth, dtype=torch.bool))
    loss.backward()
    assert truth.grad is None
    zero = model.forecast_loss(model(x, mask)['forecast'], truth, torch.zeros_like(truth, dtype=torch.bool))
    assert zero.item() == 0


def test_token_fast_neutral_matches_parent_and_changes_encoder_input():
    parent, x, mask = case()
    parent.eval()
    # Test nonzero learned source initialization, not only zero initialization.
    with torch.no_grad():
        parent.fast_origin.fill_(.2)
    model = Simple23(replace(parent.cfg, fast_site='tokens'), torch.zeros(23), torch.ones(23))
    model.load_state_dict(parent.state_dict())
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x, mask)['logit'], parent(x, mask)['logit'], rtol=0, atol=0)
    inputs = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    fast = model.initial_fast(2).detach()
    model(x, mask, fast)
    model(x, mask, fast+.3)
    hook.remove()
    assert not torch.equal(inputs[0][:, :-1], inputs[1][:, :-1])
    torch.testing.assert_close(inputs[0][:, -1], inputs[1][:, -1], rtol=0, atol=0)


def test_token_fast_supports_second_order_forecast_training():
    from experiments.risk_aligned_online.fast_meta import train_step
    model, x, mask = case()
    model = Simple23(replace(model.cfg, fast_site='tokens'), torch.zeros(23), torch.ones(23))
    e = dict(x=x, a=mask, qx=x*.9, qa=mask, truth=torch.randn(2,4,23),
             mask=torch.ones(2,4,23,dtype=torch.bool), y=torch.tensor([0.,1.]))
    before = {k:v.clone() for k,v in model.state_dict().items()}
    out = train_step(model, torch.optim.Adam(model.forecast_head.parameters(), lr=.001), e, mode='meta')
    assert out['head_grad_norm'] > 0
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items() if not k.startswith('forecast_head.'))


def test_window_view_risk_is_invariant_to_channel_units_and_offsets():
    base, x, mask = case()
    model = Simple23(replace(base.cfg, input_view='window'), torch.zeros(23), torch.ones(23)).eval()
    scale = torch.logspace(-5, 5, 23)
    offset = scale * torch.linspace(-2, 2, 23)
    with torch.no_grad():
        a = model(x, mask)['logit']
        b = model(x*scale+offset, mask)['logit']
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_window_view_constant_and_missing_channels_are_finite():
    base, x, mask = case()
    model = Simple23(replace(base.cfg, input_view='window'), torch.zeros(23), torch.ones(23))
    x[:, :, 0] = 1e-9
    mask[:, :, 1] = False
    x[:, :, 1] = float('nan')
    out = model(x, mask)
    assert all(torch.isfinite(v).all() for v in out.values())
    out['logit'].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_window_view_forecast_is_decoded_with_observed_window_statistics():
    base, x, mask = case()
    model = Simple23(replace(base.cfg, input_view='window'), torch.zeros(23), torch.ones(23))
    with torch.no_grad():
        model.forecast_head.weight.zero_()
        model.forecast_head.bias.zero_()
    out = model(x, mask)['forecast']
    expected = x.mean(1)[:,None].asinh().expand_as(out)
    torch.testing.assert_close(out, expected)


def test_hybrid_view_retains_absolute_level_and_relative_change():
    base, x, mask = case()
    model = Simple23(replace(base.cfg, input_view='hybrid'), torch.zeros(23), torch.ones(23))
    seen = []
    handle = model.embedding.register_forward_pre_hook(lambda module,args: seen.append(args[0].detach().clone()))
    model(x, mask)
    model(x+3, mask)
    handle.remove()
    assert model.embedding.in_features == model.cfg.patch*69
    a,b = [v.reshape(2,16,69) for v in seen]
    assert not torch.allclose(a[:,:,:23],b[:,:,:23])
    torch.testing.assert_close(a[:,:,23:46], b[:,:,23:46], rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(a[:,:,46:], b[:,:,46:], rtol=0,atol=0)


def test_zero_relative_input_preserves_parent_outputs_and_receives_gradient():
    from experiments.risk_aligned_online.simple23 import add_relative_input
    parent, x, mask = case()
    parent.eval()
    model = add_relative_input(parent).eval()
    with torch.no_grad():
        for key,value in parent(x,mask).items():
            torch.testing.assert_close(model(x,mask)[key],value,rtol=0,atol=0)
    model(x,mask)['logit'].sum().backward()
    assert model.relative_embedding.weight.grad.norm() > 0
    assert torch.count_nonzero(model.relative_embedding.weight) == 0
    with torch.no_grad():
        model.center.add_(1)
    torch.testing.assert_close(parent.center, torch.zeros(23), rtol=0, atol=0)
