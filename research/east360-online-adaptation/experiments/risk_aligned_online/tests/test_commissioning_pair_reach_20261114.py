import torch

from experiments.risk_aligned_online import commissioning_pair_reach_20261114 as run


def test_teacher_projection_preserves_the_eight_permitted_value_columns_only():
    w = torch.arange(64*8*23).float().reshape(64,8*23)
    all_values, witnessed = run.teacher_columns(w)
    assert all_values.shape == witnessed.shape == (64,8*22)
    expected = w.reshape(64,8,23)[:,:,list(run.UNMASKED)]
    torch.testing.assert_close(all_values.unflatten(1,(8,22)),expected)
    limited = witnessed.unflatten(1,(8,22))
    for i,k in enumerate(run.UNMASKED):
        torch.testing.assert_close(limited[:,:,i],expected[:,:,i] if k in run.pair.VALUES else torch.zeros((64,8)))


def test_postseal_oracle_uses_only_pair_spend_and_keeps_native_norm_separate():
    raw = torch.arange(2*3*4,dtype=torch.float32).reshape(2,3,4)
    d = torch.randn((2,4,4),generator=torch.Generator().manual_seed(63))
    requested = torch.tensor([.2,.7])
    matched,native,amount = run.matched_writes(raw,d,requested)
    assert matched.shape == (2,2,3,4)
    torch.testing.assert_close(matched.flatten(2).norm(dim=-1),requested.expand(2,-1))
    torch.testing.assert_close(native,raw@d[0].T)
    torch.testing.assert_close(amount,native.flatten(1).norm(dim=-1))
    assert (amount>requested).all()
