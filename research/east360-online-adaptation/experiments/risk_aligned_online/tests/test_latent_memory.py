import inspect

import torch
from torch.nn import functional as F

torch.set_num_threads(2)

from experiments.risk_aligned_online.latent_memory import (  # noqa: E402
    LatentMemory,
    LatentMemoryConfig,
)


def make_model(chunk_size=4):
    torch.manual_seed(241)
    return LatentMemory(LatentMemoryConfig(
        input_dim=64, state_dim=16, chunk_size=chunk_size,
    ))


def make_inputs(batch=2, length=12):
    torch.manual_seed(242)
    features = torch.randn(batch, length, 64)
    base_logits = torch.randn(batch, length)
    return features, base_logits


def test_contract_static_is_bitwise_and_first_chunk_delta_is_zero():
    model = make_model()
    features, base_logits = make_inputs()
    static = model(features, base_logits, adapt=False)
    adapted = model(features, base_logits, adapt=True)

    assert torch.equal(static['logits'], base_logits)
    assert adapted['logits'].shape == base_logits.shape
    assert adapted['delta'].shape == base_logits.shape
    assert torch.equal(static['delta'], torch.zeros_like(base_logits))
    assert torch.equal(adapted['delta'][:, :4], torch.zeros_like(base_logits[:, :4]))
    assert torch.equal(adapted['logits'][:, :4], base_logits[:, :4])
    assert adapted['state'].W.shape == (2, 16, 16)
    assert adapted['state'].pending_k.shape[1] == 0
    assert not hasattr(model, 'forecast_head')
    assert not any(name.lower().find('film') >= 0 for name, _ in model.named_parameters())
    assert not any(name.lower().find('expert') >= 0 for name, _ in model.named_parameters())
    assert model.readout.weight.abs().sum() > 0


def test_future_latent_input_cannot_change_early_logits():
    model = make_model().eval()
    features, base_logits = make_inputs(batch=1)
    changed = features.clone()
    changed[:, 5:] += 8.

    before = model(features, base_logits)['logits']
    after = model(changed, base_logits)['logits']
    torch.testing.assert_close(before[:, :5], after[:, :5], rtol=0, atol=0)


def test_missing_latent_mask_ignores_nonfinite_feature_values():
    model = make_model().eval()
    features, base_logits = make_inputs()
    observed = torch.ones(2, 12, dtype=torch.bool)
    observed[:, 3] = False
    changed = features.clone()
    changed[:, 3] = float('nan')

    before = model(features, base_logits, observed=observed)
    after = model(changed, base_logits, observed=observed)
    torch.testing.assert_close(before['logits'], after['logits'], rtol=0, atol=0)
    torch.testing.assert_close(before['state'].W, after['state'].W, rtol=0, atol=0)


def test_inner_update_has_no_labels_and_does_not_mutate_parameters():
    model = make_model()
    features, base_logits = make_inputs()
    before = {name: parameter.detach().clone()
              for name, parameter in model.named_parameters()}

    output = model(features, base_logits, adapt=True)

    assert 'labels' not in inspect.signature(model.forward).parameters
    assert 'outcome' not in inspect.signature(model.forward).parameters
    assert torch.isfinite(output['state'].W).all()
    assert not torch.equal(output['state'].W, model.initial_state(2).W)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_each_shot_resets_to_w0_unless_state_is_explicitly_carried():
    model = make_model()
    features, base_logits = make_inputs(batch=1)
    first = model(features, base_logits, adapt=True)
    second = model(features, base_logits, adapt=True)

    torch.testing.assert_close(first['logits'], second['logits'], rtol=0, atol=0)
    torch.testing.assert_close(first['state'].W, second['state'].W, rtol=0, atol=0)
    continued = model(features[:, :2], base_logits[:, :2],
                      state=first['state'], adapt=True)
    assert not torch.equal(continued['state'].W, model.initial_state(1).W)


def test_outer_risk_gradient_reaches_qkv_eta_w0_and_readout():
    model = make_model(chunk_size=4)
    features, base_logits = make_inputs(length=16)
    labels = torch.randint(0, 2, base_logits.shape, dtype=torch.float32)
    output = model(features, base_logits, adapt=True)
    loss = F.binary_cross_entropy_with_logits(output['logits'], labels)
    loss.backward()

    for name in ('key.weight', 'value.weight', 'query.weight',
                 'eta', 'W0', 'readout.weight'):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name


def test_arbitrary_stream_splits_match_full_sequence_and_keep_tail_buffer():
    model = make_model(chunk_size=4).eval()
    features, base_logits = make_inputs(batch=2, length=13)
    full = model(features, base_logits, adapt=True)

    state = None
    logits = []
    start = 0
    for end in (3, 5, 9, 13):
        output = model(features[:, start:end], base_logits[:, start:end],
                       state=state, adapt=True)
        logits.append(output['logits'])
        state = output['state']
        start = end
    streamed = torch.cat(logits, dim=1)

    torch.testing.assert_close(streamed, full['logits'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(state.W, full['state'].W, rtol=1e-6, atol=1e-6)
    assert state.pending_k.shape[1] == 1
    assert state.pending_v.shape[1] == 1


def test_long_sequence_stays_finite_and_respects_w_offset_bound():
    model = make_model(chunk_size=8).eval()
    features, base_logits = make_inputs(batch=2, length=513)
    observed = torch.ones(2, 513, dtype=torch.bool)
    observed[:, ::11] = False
    features[:, ::11, 7] = float('nan')

    output = model(features, base_logits, observed=observed, adapt=True)

    assert torch.isfinite(output['logits']).all()
    assert torch.isfinite(output['state'].W).all()
    offset = output['state'].W - model.W0.unsqueeze(0)
    assert offset.flatten(1).norm(dim=1).max() <= model.cfg.state_radius + 1e-5
