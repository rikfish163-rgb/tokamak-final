import copy

import numpy as np
import pytest
import torch

from ..ffn_intra_20261005.test_meta import case
from ..decoder_intra_20261005.matched_meta import enable_head_only
from .stream import predict_shot
from .source import causal_logit, whole_risk, selection_report, train_pair
from . import source


@pytest.mark.parametrize('mode', ['partial', 'delayed'])
@pytest.mark.parametrize('query', [0, 5, 6, 25, 26, 31, 32, 51, 52, 104, 156])
def test_causal_source_query_matches_real_emitted_trace(mode, query):
    model, shot, _ = case(); enable_head_only(model)
    trace, audit = predict_shot(model, shot, mode=mode, offline_source=True)
    logit, row = causal_logit(model, shot, query, mode=mode, create_graph=True)
    assert float(logit.sigmoid().detach()) == pytest.approx(float(trace[query]), abs=3e-7)
    events = [r for r in audit['events'] if r['update_index'] < query]
    assert row['mature_events'] == len(events)
    assert row['differentiated_updates'] == min(4, len(events))


def test_last_four_executed_not_four_scheduled_slots():
    model, shot, _ = case(); enable_head_only(model)
    shot['available'][27:] = False
    z, row = causal_logit(model, shot, 156, mode='partial', create_graph=True)
    assert row['executed_updates'] == row['differentiated_updates'] == 2
    gradient, = torch.autograd.grad(z, model.base.forecast_head.weight)
    assert gradient.norm() > 0


@pytest.mark.parametrize('mode', ['partial', 'delayed'])
def test_actual_whole_risk_gradient_and_early_no_path(mode):
    model, shot, _ = case(); model = model.double(); enable_head_only(model)
    shot['valid'][:] = False; shot['valid'][32] = True
    p = model.base.forecast_head.weight; original = p.detach().clone()
    torch.manual_seed(77); direction = torch.randn_like(p); direction /= direction.norm()
    def value(offset, graph=False):
        with torch.no_grad(): p.copy_(original + offset * direction)
        return whole_risk(model, shot, mode=mode, boundary=.97, create_graph=graph)[0]
    loss = value(0., True); gradient, = torch.autograd.grad(loss, p)
    analytical = float((gradient * direction).sum())
    numerical = float((value(1e-4) - value(-1e-4)).detach()) / 2e-4
    assert abs(analytical) > 1e-9
    assert analytical == pytest.approx(numerical, rel=.005, abs=3e-8)
    shot['valid'][:] = False; shot['valid'][5] = True
    early = whole_risk(model, shot, mode=mode, boundary=.97, create_graph=True)[0]
    assert torch.autograd.grad(early, p, allow_unused=True)[0] is None


def test_truncated_derivative_holds_burn_state_fixed(monkeypatch):
    model, shot, _ = case(); model = model.double(); enable_head_only(model)
    real_step = source.segment_step; captured = []
    def capture(*args, **kwargs):
        value, row = real_step(*args, **kwargs)
        captured.append((value.detach().clone(), row))
        return value, row
    monkeypatch.setattr(source, 'segment_step', capture)
    z, row = causal_logit(model, shot, 104, mode='partial', create_graph=True)
    burn = row['burn_in_updates']; assert burn == 4
    p = model.base.forecast_head.bias; original = p.detach().clone()
    torch.manual_seed(88); direction = torch.randn_like(p); direction /= direction.norm()
    gradient, = torch.autograd.grad(torch.nn.functional.softplus(-z), p)
    analytical = float((gradient * direction).sum())
    def value(offset):
        counter = [0]
        def fixed(*args, **kwargs):
            j = counter[0]; counter[0] += 1
            return captured[j] if j < burn else real_step(*args, **kwargs)
        monkeypatch.setattr(source, 'segment_step', fixed)
        with torch.no_grad(): p.copy_(original + offset * direction)
        z, _ = causal_logit(model, shot, 104, mode='partial')
        return float(torch.nn.functional.softplus(-z).detach())
    numerical = (value(1e-4) - value(-1e-4)) / 2e-4
    assert abs(analytical) > 1e-9
    assert analytical == pytest.approx(numerical, rel=.005, abs=3e-8)


def test_source_report_and_training_preserve_classifier():
    model, shot, e = case(); enable_head_only(model)
    safe = copy.deepcopy(shot); safe.update(shot=2, is_disrupt=False)
    se = dict(e, shot=2, label=0)
    pairs = [(shot, e), (safe, se)]
    report = selection_report(model, pairs, .7, mode='partial')
    assert len(report['whole']['records']) == len(report['later']['records']) == 2
    expected = np.mean([float(whole_risk(model, s, mode='partial', boundary=np.log(.7/.3))[0].detach()) for s, _ in pairs])
    assert report['whole']['balanced_loss'] == pytest.approx(expected, abs=3e-7)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.base.forecast_head.parameters(), lr=3e-5)
    train_pair(model, pairs, opt, objective='risk_aligned', boundary=.97)
    assert not torch.equal(model.base.forecast_head.weight, before['base.forecast_head.weight'])
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items() if not k.startswith('base.forecast_head.'))
    shot['device'] = 'EAST'
    with pytest.raises(ValueError, match='source'):
        causal_logit(model, shot, 32, mode='partial')
