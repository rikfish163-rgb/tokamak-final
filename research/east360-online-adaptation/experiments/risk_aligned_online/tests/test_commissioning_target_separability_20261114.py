import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_target_separability_20261114 as s
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_windows_include_current_never_future_including_cold_prefix():
    x = np.arange(23*155, dtype=np.float32).reshape(155, 23)
    m = np.ones_like(x, dtype=bool)
    start = np.array([0, 126, 127, 134])
    values, valid = s.windows(x, m, start)
    np.testing.assert_array_equal(values[:, -1], x[start])
    assert valid[0, :-1].sum() == 0
    ref = s.windows(x[:135], m[:135], start)
    np.testing.assert_array_equal(values, ref[0])
    np.testing.assert_array_equal(valid, ref[1])


def test_archival_partitions_exclude_every_stream_member_and_hold_out_blocks():
    ids = list(range(350))
    stream = list(range(10, 70))
    train, valid = s.archival_roles(ids, stream)
    assert set(train).isdisjoint(valid) and (set(train) | set(valid)).isdisjoint(stream)
    assert sorted(train+valid) == sorted(set(ids)-set(stream))
    for k in range(0, 350, 50):
        block = set(range(k, min(k+50, 350)))-set(stream)
        assert block.issubset(train) or block.issubset(valid)


def test_label_and_warning_have_different_contracts_and_do_not_extend_the_window():
    # At the exact 125 ms edge the point is positive. Late positives are
    # still part of the training window but never count as timely warnings.
    t = np.array([-.126, -.125, -.031, -.030, -.029, -.001])
    normal, early, late = s.interval(t, True, 0.)
    np.testing.assert_array_equal(normal, [0])
    np.testing.assert_array_equal(early, [1, 2])
    np.testing.assert_array_equal(late, [3, 4, 5])
    negative, early, late = s.interval(t, False, None)
    np.testing.assert_array_equal(negative, np.arange(len(t)))
    assert not len(early) and not len(late)


def test_set_control_ignores_old_patch_order_but_not_near_window():
    torch.manual_seed(11)
    net = Simple23(Config(), torch.zeros(23), torch.ones(23)).eval()
    s.tie_earlier_positions(net)
    x = torch.randn(2, 128, 23)
    mask = torch.rand(2, 128, 23)>.2
    order = torch.cat((torch.arange(11, -1, -1), torch.arange(12, 16)))
    recent = torch.cat((torch.arange(12), torch.arange(15, 11, -1)))

    def change(v, indices):
        return v.reshape(-1, 16, 8, 23)[:, indices].reshape_as(v)

    real = net(x, mask)['logit']
    old = net(change(x, order), change(mask, order))['logit']
    new = net(change(x, recent), change(mask, recent))['logit']
    torch.testing.assert_close(real, old, rtol=0., atol=4e-7)
    assert (real-new).abs().max()>1e-5
    real.sum().backward()
    assert torch.isfinite(net.parametrizations.position.original.grad).all()


def test_location_probes_change_only_the_selected_slot_and_risk_head():
    for k, scope in s.LOCATIONS.items():
        net = Simple23(Config(), torch.zeros(23), torch.ones(23))
        s.train_scope(net, k)
        enabled = [name for name, p in net.named_parameters() if p.requires_grad]
        assert any(name.startswith(scope) for name in enabled)
        assert all(name.startswith(('risk_head.', scope)) for name in enabled)
    block = [sum(p.numel() for name, p in net.named_parameters() if name.startswith(scope))
             for scope in ('encoder.layers.0.', 'encoder.layers.1.')]
    assert block[0]==block[1]


def test_offline_value_scope_freezes_head_and_mask_even_after_optimizer_step():
    for arm,allowed in s.NUMERIC.items():
        torch.manual_seed(16)
        net = Simple23(Config(), torch.zeros(23), torch.ones(23))
        x = torch.randn(3,128,23)
        ok = torch.ones_like(x,dtype=torch.bool)
        old = net(x,ok)['logit'].detach()
        initial = net.embedding.weight.detach().clone()
        old_bias = net.embedding.bias.detach().clone()
        head = net.risk_head.weight.detach().clone()
        s.train_scope(net,arm)
        torch.testing.assert_close(net.embedding.weight,initial,rtol=0.,atol=0.)
        # Freezing enables no-grad attention kernels for the identical weights.
        torch.testing.assert_close(old,net(x,ok)['logit'],rtol=0.,atol=3e-7)
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.01)
        net(x,ok)['logit'].sum().backward()
        opt.step()
        slot = torch.arange(368)%46
        changed = (net.embedding.weight-initial).abs().sum(0)>0.
        assert changed.any()
        assert not (changed & ~torch.isin(slot,torch.tensor(allowed))).any()
        torch.testing.assert_close(net.embedding.bias,old_bias,atol=0.,rtol=0.)
        torch.testing.assert_close(net.risk_head.weight,head,atol=0.,rtol=0.)
