"""CLS-only H25 readout with an unchanged, frozen Simple23 risk path.

Only ``decoder`` is trainable for source-only fitting; freeze it for target
replay. The last-FFN delta remains external and disposable, exactly as in
FfnModel. Neither the decoder nor the coordinate-risk metric changes the
classifier's parameters or its official risk output.
"""
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..ffn_intra_20261005.model import FfnModel


class ClsModel(FfnModel):
    def __init__(self, base):
        super().__init__(base)
        self.decoder = nn.Linear(64, 25 * 23, device=self.center.device, dtype=self.center.dtype)
        self.eval()

    def forward(self, values, available, delta):
        """Return logit, forecast and the exact 64-D risk-head input as a dict."""
        captured = []
        # Capture after the final norm AND source fast-origin affine. Do not
        # replace the risk head, detach its input, or retain a persistent hook.
        hook = self.base.risk_head.register_forward_pre_hook(
            lambda module, args: captured.append(args[0]))
        try:
            out = super().forward(values, available, delta)
        finally:
            hook.remove()
        cls, = captured
        forecast = self.decoder(cls).reshape(len(values), self.cfg.horizon, 23)
        return dict(logit=out['logit'], forecast=forecast, cls=cls)

    def risk_from_source_coordinates(self, z, available):
        """Static delta-zero source logit, retaining first/second input gradients.

        ``z`` is already asinh((raw - source_center) / source_scale). Missing
        coordinates are zero-filled and excluded exactly as in Simple23; no
        inverse sinh, per-window statistics, decoder or target delta is used.
        Call with autograd enabled, then detach the real/teacher branch at its
        caller. The predicted branch must remain differentiable.
        """
        if not torch.is_grad_enabled():
            raise RuntimeError('use enabled autograd and detach outputs; fixed attention dispatch required')
        if z.ndim != 3 or z.shape[1:] != (self.cfg.window, 23):
            raise ValueError('z must be [batch, past window, 23]')
        if available.shape != z.shape or available.dtype != torch.bool:
            raise ValueError('matching boolean availability required')
        available = available & torch.isfinite(z)
        level = torch.where(available, z, 0.)

        # Minimal source/tokens forward from just AFTER source normalization.
        # Keep this sequence aligned with Simple23.forward; equivalence tests
        # cover both values and input gradients, including missing/padded data.
        base, batch = self.base, len(z)
        x = torch.cat((level, available.to(level.dtype)), -1)
        x = base.embedding(x.reshape(batch, self.cfg.window // self.cfg.patch, -1))
        # Match FfnModel's differentiable attention dispatch even when z does
        # not require grad. This fixed leaf is not an adaptation parameter.
        fast = base.initial_fast(batch).detach().requires_grad_(True)
        dg, db = (fast - base.initial_fast(batch)).chunk(2, -1)
        x = x * (1 + .1 * dg[:, None].tanh()) + .1 * db[:, None].tanh()
        valid = available.reshape(batch, self.cfg.window // self.cfg.patch, -1).any(-1)
        x = torch.cat((x, base.cls.expand(batch, -1, -1)), 1) + base.position
        valid = torch.cat((valid, torch.ones(batch, 1, dtype=torch.bool, device=x.device)), 1)
        with sdpa_kernel(SDPBackend.MATH):
            x = base.encoder(x, src_key_padding_mask=~valid)
        gain, bias = base.initial_fast(batch).chunk(2, -1)
        x = base.norm(x) * (1 + .1 * gain[:, None].tanh()) + .1 * bias[:, None].tanh()
        x = x.masked_fill(~valid[:, :, None], 0.)
        return base.risk_head(x[:, -1]).squeeze(-1)
