"""CPU contract tests for ended-shot, single-matrix risk supervision."""

from copy import deepcopy
import math

import h5py
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.joint_lora_20261104.training import (
    ended_records, eos_schedule,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23
from risk_feature_inter_20261108 import train_inter
from risk_meta_step_20261107 import WEIGHT_KEY


def base_model():
    torch.manual_seed(71)
    return Simple23(Config(dim=8, heads=4, layers=1, fast_site='tokens'),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)


@pytest.fixture
def ended_data():
    shots = {}
    for sid in (101, 102, 103):
        rng = np.random.default_rng(sid)
        values = rng.normal(0., .15, (280, 23)).astype(np.float32)
        available = np.ones_like(values, dtype=bool)
        available[:, 2] = False
        shots[sid] = dict(shot=sid, device='EAST', values=values,
                          available=available, times=np.arange(280) / 1000.)
    truth = {101: (False, None), 102: (True, .270), 103: (True, .270)}
    current, _ = ended_records({103: shots[103]}, {103: truth[103]}, [103])
    previous, _ = ended_records({sid: shots[sid] for sid in (101, 102)},
                                {sid: truth[sid] for sid in (101, 102)}, [101, 102])
    records, schedule = eos_schedule(current, previous, seed=91)
    return shots, records, schedule


def batch(shots, records, draw):
    selected = [records[i] for i in draw['indices']]
    windows = [simple_data.window(shots[row['shot']], row['query'], 128)
               for row in selected]
    x = torch.from_numpy(np.stack([item[0] for item in windows]))
    available = torch.from_numpy(np.stack([item[1] for item in windows]))
    labels = torch.tensor([row['y'] for row in selected])
    return x, available, labels


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_schedule_windows_labels_and_freezing_are_exact_without_hdf(monkeypatch, ended_data):
    shots, records, schedule = ended_data
    model = base_model()
    before = snapshot(model)
    original_weight = before[WEIGHT_KEY].clone()
    initial_weight = original_weight.clone()
    records_before, schedule_before = deepcopy(records), deepcopy(schedule)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    expected_rows = [[records[i] for i in draw['indices']] for draw in schedule]
    assert len(expected_rows) == 8 and all(len(rows) == 64 for rows in expected_rows)
    assert all(sum(row['shot'] == 103 for row in rows) == 32 for rows in expected_rows)
    expected_windows = [(row['shot'], row['query'], 128)
                        for rows in expected_rows for row in rows]
    observed_windows, observed_labels, observed_logits, model_logits = [], [], [], []
    original_window = simple_data.window
    original_bce = F.binary_cross_entropy_with_logits
    original_adamw = torch.optim.AdamW
    original_clip = torch.nn.utils.clip_grad_norm_
    optimizers, clips = [], []

    def forbidden(*args, **kwargs):
        raise AssertionError('ended-shot update must not perform HDF I/O')

    def watch_window(shot, end, length):
        observed_windows.append((shot['shot'], end, length))
        return original_window(shot, end, length)

    def watch_bce(logits, labels, *args, **kwargs):
        observed_logits.append(logits.detach().clone())
        observed_labels.append(labels.detach().clone())
        return original_bce(logits, labels, *args, **kwargs)

    def watch_forward(module, args, kwargs, output):
        assert len(args) <= 2 and 'fast' not in kwargs
        assert [name for name, p in module.named_parameters() if p.requires_grad] == [WEIGHT_KEY]
        model_logits.append(output['logit'].detach().clone())

    def watch_adamw(*args, **kwargs):
        optimizer = original_adamw(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    def watch_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        clips.append((parameters, max_norm, kwargs.get('error_if_nonfinite')))
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(h5py, 'File', forbidden)
    monkeypatch.setattr(io23, 'read_signals', forbidden)
    monkeypatch.setattr(io23, 'read_outcome', forbidden)
    monkeypatch.setattr(simple_data, 'read_shot', forbidden)
    monkeypatch.setattr(simple_data, 'window', watch_window)
    monkeypatch.setattr(F, 'binary_cross_entropy_with_logits', watch_bce)
    monkeypatch.setattr(torch.optim, 'AdamW', watch_adamw)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', watch_clip)
    hook = model.register_forward_hook(watch_forward, with_kwargs=True)
    try:
        history = train_inter(model, shots, records, schedule, initial_weight)
    finally:
        hook.remove()

    assert observed_windows == expected_windows
    assert len(observed_logits) == len(model_logits) == len(observed_labels) == 8
    for rows, logits, actual_logits, labels in zip(
            expected_rows, observed_logits, model_logits, observed_labels):
        torch.testing.assert_close(logits, actual_logits, rtol=0, atol=0)
        torch.testing.assert_close(labels, torch.tensor([row['y'] for row in rows]), rtol=0, atol=0)
    assert len(optimizers) == 1
    assert optimizers[0].param_groups[0]['lr'] == 1e-4
    assert optimizers[0].param_groups[0]['weight_decay'] == 0.
    assert optimizers[0].param_groups[0]['params'] == [dict(model.named_parameters())[WEIGHT_KEY]]
    assert len(clips) == 8
    assert all(len(params) == 1 and params[0] is dict(model.named_parameters())[WEIGHT_KEY]
               and maximum == 1. and error is True for params, maximum, error in clips)
    assert [item['step'] for item in history] == [draw['step'] for draw in schedule]
    for item in history:
        assert all(isinstance(item[key], float) and math.isfinite(item[key])
                   for key in ('loss', 'risk_loss', 'prior', 'gradient_norm'))
        assert item['gradient_norm'] > 0.
    assert not torch.equal(model.state_dict()[WEIGHT_KEY], original_weight)
    for name, value in model.state_dict().items():
        if name != WEIGHT_KEY:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    torch.testing.assert_close(initial_weight, original_weight, rtol=0, atol=0)
    assert records == records_before and schedule == schedule_before


def test_raw_bce_prior_gradient_and_first_adamw_step_match_math(ended_data):
    shots, records, schedule = ended_data
    model = base_model()
    weight = dict(model.named_parameters())[WEIGHT_KEY]
    initial_weight = weight.detach().clone()
    with torch.no_grad():
        weight.add_(.04)
    start = weight.detach().clone()
    reference = deepcopy(model)
    reference_weight = dict(reference.named_parameters())[WEIGHT_KEY]
    reference_weight.requires_grad_(True)
    x, available, labels = batch(shots, records, schedule[0])
    logits = reference(x, available)['logit']
    raw_bce = F.binary_cross_entropy_with_logits(logits, labels)
    margin_bce = F.binary_cross_entropy_with_logits(logits - math.log(.2 / .8), labels)
    prior = (reference_weight - initial_weight).square().sum()
    objective = raw_bce + .01 * prior
    gradient, = torch.autograd.grad(objective, reference_weight)
    norm = torch.linalg.vector_norm(gradient)
    clipped = gradient * (1. / (norm + 1e-6)).clamp(max=1.)
    expected_weight = start - 1e-4 * clipped / (clipped.abs() + 1e-8)
    initial_before = initial_weight.clone()

    history = train_inter(model, shots, records, schedule[:1], initial_weight)

    assert abs(float((raw_bce - margin_bce).detach())) > .01
    assert history[0]['risk_loss'] == pytest.approx(float(raw_bce.detach()), abs=1e-7)
    assert history[0]['prior'] == pytest.approx(float(prior.detach()), abs=1e-7)
    assert history[0]['loss'] == pytest.approx(float(objective.detach()), abs=1e-7)
    assert history[0]['gradient_norm'] == pytest.approx(float(norm), abs=1e-7)
    torch.testing.assert_close(weight, expected_weight, rtol=0, atol=1e-7)
    torch.testing.assert_close(initial_weight, initial_before, rtol=0, atol=0)


def test_identical_models_and_schedule_are_deterministic(ended_data):
    shots, records, schedule = ended_data
    first = base_model()
    second = deepcopy(first)
    initial_weight = first.state_dict()[WEIGHT_KEY].clone()
    rng_before = torch.random.get_rng_state().clone()

    first_history = train_inter(first, shots, records, schedule, initial_weight)
    second_history = train_inter(second, shots, records, schedule, initial_weight)

    assert first_history == second_history
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)


def test_adamw_is_reset_on_each_ended_shot(ended_data):
    shots, records, schedule = ended_data
    separate = base_model()
    continuous = deepcopy(separate)
    initial_weight = separate.state_dict()[WEIGHT_KEY].clone()

    train_inter(separate, shots, records, schedule, initial_weight)
    train_inter(separate, shots, records, schedule, initial_weight)
    train_inter(continuous, shots, records, schedule + schedule, initial_weight)

    assert not torch.equal(separate.state_dict()[WEIGHT_KEY], continuous.state_dict()[WEIGHT_KEY])


def test_nonfinite_gradient_aborts_and_cleans_up(ended_data):
    shots, records, schedule = ended_data
    model = base_model()
    before = snapshot(model)
    initial_weight = before[WEIGHT_KEY].clone()
    initial_weight[0, 0] = float('nan')

    with pytest.raises(RuntimeError, match='non-finite'):
        train_inter(model, shots, records, schedule[:1], initial_weight)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_initial_weight_must_not_share_storage_with_the_updated_weight(ended_data):
    shots, records, schedule = ended_data
    model = base_model()
    before = snapshot(model)
    initial_weight = dict(model.named_parameters())[WEIGHT_KEY].detach()

    with pytest.raises(ValueError, match='independent'):
        train_inter(model, shots, records, schedule, initial_weight)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_initial_weight_shape_must_match_without_broadcasting(ended_data):
    shots, records, schedule = ended_data
    model = base_model()
    before = snapshot(model)
    initial_weight = before[WEIGHT_KEY][:1].clone()

    with pytest.raises(ValueError, match='shape'):
        train_inter(model, shots, records, schedule, initial_weight)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
