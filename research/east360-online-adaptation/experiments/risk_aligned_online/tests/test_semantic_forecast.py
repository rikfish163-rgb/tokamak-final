import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.semantic_forecast import semantic_loss, cache_events, batch_losses
from experiments.risk_aligned_online.tests.test_anticipatory_readout import Toy
from experiments.risk_aligned_online.persistent_ffn import PersistentFfnModel
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online import simple_data


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_semantic_feedback_transmits_gradient_to_prediction_not_teacher():
    model = Toy().requires_grad_(False)
    x = torch.ones(2, 128, 23); a = torch.ones_like(x, dtype=torch.bool)
    prediction = torch.zeros(2, 25, 23, requires_grad=True)
    teacher = torch.ones(2, requires_grad=True)
    loss = semantic_loss(model, x, a, prediction, teacher)
    assert loss.shape == (2,) and (loss > 0).all()
    loss.mean().backward()
    assert prediction.grad.norm() > 0 and teacher.grad is None
    assert all(p.grad is None for p in model.parameters())


def test_matching_real_future_risk_has_zero_bernoulli_kl():
    model = Toy().requires_grad_(False)
    x = torch.ones(1, 128, 23); a = torch.ones_like(x, dtype=torch.bool)
    prediction = torch.full((1, 25, 23), .4)
    torch.testing.assert_close(semantic_loss(model, x, a, prediction, torch.tensor([.4])),
                               torch.zeros(1), rtol=0, atol=1e-7)


def test_teacher_identity_shape_and_nonfinite_targets_are_rejected():
    model = Toy(); x = torch.ones(1, 128, 23); a = torch.ones_like(x, dtype=torch.bool)
    prediction = torch.zeros(1, 25, 23)
    for teacher in (torch.ones(2), torch.tensor([float('nan')])):
        with pytest.raises(ValueError, match='teacher'):
            semantic_loss(model, x, a, prediction, teacher)


def actual_case():
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'), torch.zeros(23), torch.ones(23)).eval()
    model = PersistentFfnModel(base).requires_grad_(False)
    x = np.arange(61, dtype=np.float32)[:, None].repeat(23, axis=1)/10
    shot = dict(device='J-TEXT', shot=1, values=x, available=np.ones_like(x, bool),
                valid=np.ones(61, bool), times=np.arange(61)/1000., is_disrupt=False)
    return model, shot


def test_source_cache_teacher_is_actual_matured_future_not_a_synthetic_target():
    model, shot = actual_case()
    cache, rows = cache_events(model, [shot])
    assert [r['origin'] for r in rows] == [0, 26]
    torch.testing.assert_close(cache['target'][0], torch.asinh(torch.from_numpy(shot['values'][1:26])))
    torch.testing.assert_close(cache['values'][0, -1], torch.from_numpy(shot['values'][0]))
    # Match the two-window teacher batch rather than comparing different CPU
    # GEMM shapes. This keeps bitwise equality without relaxing tolerance.
    pairs = [simple_data.window(shot, end, 128) for end in (25, 51)]
    x, a = [torch.from_numpy(np.stack(v)) for v in zip(*pairs)]
    with torch.no_grad(): teacher = model(x, a)['logit']
    torch.testing.assert_close(cache['teacher_logits'], teacher, rtol=0, atol=0)
    with pytest.raises(ValueError, match='source'):
        cache_events(model, [dict(shot, device='EAST')])


def test_semantic_training_updates_only_external_head_not_classifier_or_memory():
    model, shot = actual_case()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    cache, _ = cache_events(model, [shot])
    head = copy.deepcopy(model.base.forecast_head).requires_grad_(True)
    before = {k: v.clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.SGD(head.parameters(), lr=.001)
    signal, semantic = batch_losses(model, head, cache, [0, 1])
    optimizer.zero_grad(); (signal+.20*semantic).mean().backward(); optimizer.step()
    assert any(not torch.equal(v, before[k]) for k, v in head.state_dict().items())
    for k, v in model.state_dict().items(): torch.testing.assert_close(v, original[k], rtol=0, atol=0)
    assert not model.memory.any()


def test_source_training_context_gate_excludes_padding_and_observation_gaps():
    model, short = actual_case()
    with pytest.raises(ValueError, match='no fully observed'):
        cache_events(model, [short], require_full_context=True)
    n = 210
    x = np.arange(n, dtype=np.float32)[:, None].repeat(23, axis=1)/10
    shot = dict(short, values=x, available=np.ones_like(x, bool),
                times=np.arange(n)/1000., valid=np.ones(n, bool))
    cache, rows = cache_events(model, [shot], require_full_context=True)
    assert [r['origin'] for r in rows] == [130, 156, 182]
    assert cache['available'].any(-1).all()
    shot['available'][100] = False
    with pytest.raises(ValueError, match='no fully observed'):
        cache_events(model, [shot], require_full_context=True)
