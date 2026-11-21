import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.source_task_memory_20261026.learning import (
    make_adapter, source_endpoints, loss_from_example, validate_roles,
)


def test_east_loader_rejects_wrong_source_or_training_roster(tmp_path):
    from experiments.risk_aligned_online.source_task_memory_20261026.checkpoint import load_source_memory
    from experiments.risk_aligned_online.redesign_20261004.prepare_prior import PARENT_SHA256
    roles = dict(train=list(range(1353)), val=list(range(2000, 2160)), test=list(range(3000, 3220)))
    m, _ = setup()
    content = dict(kind='learned', source_sha256=PARENT_SHA256, train_ids=roles['train'],
                   val_ids=roles['val'], epoch=2, readout='full', memory=m.memory.state_dict(),
                   protocol_sha256='a'*64)
    path = tmp_path/'memory.pt'
    torch.save(content, path)
    receipt = load_source_memory(m, path, roles=roles)
    assert receipt['kind'] == 'learned' and len(receipt['sha256']) == 64
    torch.save(dict(content, train_ids=roles['train'][1:]+[9999]), path)
    with pytest.raises(ValueError, match='TRAIN'):
        load_source_memory(m, path, roles=roles)
    torch.save(dict(content, source_sha256='bad'), path)
    with pytest.raises(ValueError, match='source'):
        load_source_memory(m, path, roles=roles)


def test_source_role_boundary():
    roles = dict(train=[1, 2], val=[3], test=[4])
    validate_roles(roles, counts=(2, 1, 1))
    with pytest.raises(ValueError, match='overlap'):
        validate_roles(dict(roles, test=[1]), counts=(2, 1, 1))
    with pytest.raises(ValueError, match='count'):
        validate_roles(roles, counts=(1353, 160, 220))


def test_source_endpoints_use_strict_10ms_not_east_30ms():
    t = np.arange(200) * .001
    s = dict(times=t, valid=np.ones(200, bool), is_disrupt=True, down=.199,
             y=(t >= .174).astype('float32'), device='J-TEXT')
    ends = source_endpoints(s)
    assert max(ends) == 188  # 11ms, excludes exactly 10ms
    assert min(ends) == 0  # source model already supports masked left padding
    with pytest.raises(ValueError, match='J-TEXT'):
        source_endpoints(dict(s, device='EAST'))


def test_short_source_discharge_keeps_available_history_instead_of_being_dropped():
    t = .15 + np.arange(52) * .001
    s = dict(times=t, valid=np.ones(52, bool), is_disrupt=True, down=.2022798176837159,
             y=(t >= .1772798176837159).astype('float32'), device='J-TEXT')
    ends = source_endpoints(s)
    assert ends[0] == 0 and ends[-1] == 42 and 40 in ends


def setup():
    torch.manual_seed(131)
    torch.set_num_threads(1)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    np.zeros(23), np.ones(23)).eval()
    m = make_adapter(base, 'learned')
    torch.nn.init.normal_(m.memory.output.weight, std=.02)
    means = torch.randn(24, 23)
    available = torch.ones_like(means, dtype=torch.bool)
    x = torch.randn(4, 128, 23)
    with torch.no_grad():
        h, valid = m.tokens(x, torch.ones_like(x, dtype=torch.bool))
    e = dict(means=means, complete=available, h=h, valid=valid,
             ends=torch.tensor([127, 143, 159, 191]), flag=True)
    return m, e


def test_risk_gradient_reaches_learning_rule_but_not_base():
    m, e = setup()
    loss, _ = loss_from_example(m, e, .7)
    loss.backward()
    for name in ('key.weight', 'prior', 'rate', 'output.weight'):
        g = dict(m.memory.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0
    assert all(p.grad is None for p in m.base.parameters())
    assert m.slow.weight.grad is None
    # Finite-difference check of the gradient through the recurrent write rule.
    analytic = m.memory.rate.grad.item()
    with torch.no_grad():
        m.memory.rate.add_(.002)
        plus = loss_from_example(m, e, .7)[0].item()
        m.memory.rate.sub_(.004)
        minus = loss_from_example(m, e, .7)[0].item()
    assert analytic == pytest.approx((plus-minus)/.004, rel=.08, abs=2e-5)


def test_fixed_task_control_only_learns_same_readout():
    m, e = setup()
    fixed = make_adapter(m.base, 'fixed')
    fixed.memory.load_state_dict(m.memory.state_dict())
    loss, _ = loss_from_example(fixed, e, .7)
    loss.backward()
    assert fixed.memory.output.weight.grad.abs().max() > 0
    assert all(p.grad is None for n, p in fixed.memory.named_parameters() if not n.startswith('output.'))


def test_future_observation_cannot_change_earlier_risk_loss_and_state_resets():
    m, e = setup()
    e = dict(e, ends=e['ends'][:2], h=e['h'][:2], valid=e['valid'][:2])
    first = loss_from_example(m, e, .7)[0]
    changed = dict(e, means=e['means'].clone())
    changed['means'][18:] += 500
    torch.testing.assert_close(first, loss_from_example(m, changed, .7)[0], atol=0, rtol=0)
    torch.testing.assert_close(first, loss_from_example(m, e, .7)[0], atol=0, rtol=0)
