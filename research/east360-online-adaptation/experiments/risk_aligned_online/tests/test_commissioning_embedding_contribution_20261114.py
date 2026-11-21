import torch

from experiments.risk_aligned_online import commissioning_embedding_contribution_20261114 as a


def test_measurement_and_availability_coefficients_never_mix():
    old = torch.zeros(64, 368)
    new = torch.arange(64*368, dtype=old.dtype).reshape_as(old)
    offset = torch.ones(64)
    values, bias = a.mix_embedding(old, old[:, 0], new, offset, 'measured')
    mask, extra_bias = a.mix_embedding(old, old[:, 0], new, offset, 'mask_and_bias')
    z = new.reshape(64, 8, 46)
    torch.testing.assert_close(values.reshape_as(z)[..., :23], z[..., :23])
    assert not values.reshape_as(z)[..., 23:].count_nonzero()
    assert not mask.reshape_as(z)[..., :23].count_nonzero()
    torch.testing.assert_close(mask.reshape_as(z)[..., 23:], z[..., 23:])
    torch.testing.assert_close(values+mask, new)
    assert not bias.count_nonzero()
    torch.testing.assert_close(extra_bias, offset)
