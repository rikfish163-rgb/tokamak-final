"""Reconstruct the origin+25 window from origin and origin+50 windows."""

import torch


def maturity_window(x, mask, later_x, later_mask):
    """Return detached copies ending at origin+25, preserving all channels.

    For x covering 0..127, later_x covers 50..177. The shared history is
    50..127 and the result covers 25..152. Values and masks at 153..177
    are neither consumed nor validated; unused x history 0..24 is ignored.
    """
    if (not isinstance(x, torch.Tensor) or x.ndim != 3
            or x.shape[1:] != (128, 23) or not x.is_floating_point()):
        raise ValueError('x must be a floating tensor of shape (B,128,23)')
    if (not isinstance(later_x, torch.Tensor) or later_x.shape != x.shape
            or later_x.dtype != x.dtype or later_x.device != x.device):
        raise ValueError('later_x must match x shape, floating dtype and device')
    for name, value in (('mask', mask), ('later_mask', later_mask)):
        if (not isinstance(value, torch.Tensor) or value.shape != x.shape
                or value.dtype != torch.bool or value.device != x.device):
            raise ValueError(f'{name} must be a same-device boolean tensor matching x')

    if not torch.isfinite(x[:, 25:]).all() or not torch.isfinite(later_x[:, :-25]).all():
        raise ValueError('used values and past overlap must be finite')
    if not torch.equal(x[:, 50:], later_x[:, :78]):
        raise ValueError('origin and later window values must match exactly in the overlap')
    if not torch.equal(mask[:, 50:], later_mask[:, :78]):
        raise ValueError('origin and later window masks must match exactly in the overlap')

    mature_x = torch.cat((x[:, 25:].detach(), later_x[:, -50:-25].detach()), dim=1)
    mature_mask = torch.cat((mask[:, 25:].detach(), later_mask[:, -50:-25].detach()), dim=1)
    return mature_x, mature_mask
