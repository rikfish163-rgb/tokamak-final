"""Isolated, per-endpoint low-rank writes at Simple23's input patch tokens.

The source model is never modified.  Supply a frozen/eval source-view model,
its current 128-sample window and availability, and a basis from that same
model.  Packed data contain only information from those supplied windows;
there is no file, target, future, or persistent-state access in this module.
"""

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.risk_aligned_online.simple23 import Simple23


RANK = 8
WIDTH = 64
MODES = ("motion", "level")
_PATCHES = 16
_PATCH = 8
_CHANNELS = 23
_COORDINATE_FLOOR = 1e-3


def _require(condition, message):
    """Check a scalar tensor without synchronizing an accelerator stream."""
    if condition.device.type == "cpu":
        if not bool(condition):
            raise ValueError(message)
    else:
        torch._assert_async(condition, message)


def _check_base(base):
    if not isinstance(base, Simple23):
        raise ValueError("a frozen source Simple23 is required")
    cfg = base.cfg
    if (
        (cfg.window, cfg.patch, cfg.dim, cfg.horizon) != (128, 8, 64, 25)
        or cfg.input_view != "source"
        or cfg.fast_site != "tokens"
        or base.embedding.weight.shape != (64, 368)
        or base.embedding.bias is None
        or base.embedding.bias.shape != (64,)
        or base.position.shape != (1, 17, 64)
        or base.cls.shape != (1, 1, 64)
        or base.fast_origin.shape != (128,)
        or base.center.shape != (23,)
        or base.scale.shape != (23,)
        or len(base.encoder.layers) != cfg.layers
        or any(module.training for module in base.modules())
        or any(parameter.requires_grad for parameter in base.parameters())
    ):
        raise ValueError("eval/frozen Simple23 128/8/64/H25 source token configuration required")
    for layer in base.encoder.layers:
        if not isinstance(layer, nn.TransformerEncoderLayer):
            raise ValueError("the native pre-norm Simple23 encoder is required")
        attention = layer.self_attn
        if (
            not layer.norm_first
            or not attention.batch_first
            or not attention._qkv_same_embed_dim
        ):
            raise ValueError("the native pre-norm Simple23 encoder is required")


def _value_weight(base):
    return base.embedding.weight.reshape(WIDTH, _PATCH, 2 * _CHANNELS)[:, :, :_CHANNELS].reshape(
        WIDTH, _PATCH * _CHANNELS
    )


@torch.no_grad()
def basis(base):
    """Return sign-oriented top-eight value-embedding left singular vectors.

    The decomposition is deliberately performed on CPU in float64, regardless
    of the source device and dtype.  The result is cast back to the embedding's
    dtype/device.  Like any SVD, a repeated singular value does not determine a
    unique orientation of the corresponding multi-dimensional subspace.
    """
    _check_base(base)
    weight = _value_weight(base).detach().to(device="cpu", dtype=torch.float64)
    _require(torch.isfinite(weight).all(), "finite source value-embedding weights required")
    u, _, _ = torch.linalg.svd(weight, full_matrices=False)
    u = u[:, :RANK]
    columns = torch.arange(RANK, device=u.device)
    pivots = u.abs().argmax(dim=0)
    u = u * u[pivots, columns].sign().unsqueeze(0)
    return u.to(device=base.embedding.weight.device, dtype=base.embedding.weight.dtype)


@dataclass(frozen=True)
class Packed:
    """Native tokens plus read-only, window-local intervention features.

    The final native token is CLS.  ``projection`` is shared by ``take`` and
    is not indexed or copied.  Tensors should be treated as immutable by the
    caller and used only with the same source model that produced them.
    """

    tokens: torch.Tensor
    valid: torch.Tensor
    features: torch.Tensor
    support: torch.Tensor
    projection: torch.Tensor

    def __post_init__(self):
        tensors = (self.tokens, self.valid, self.features, self.support, self.projection)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("Packed fields must be torch tensors")
        if self.tokens.ndim != 3:
            raise ValueError("tokens must be [batch, 17, 64]")
        batch = self.tokens.shape[0]
        if (
            self.tokens.shape != (batch, 17, WIDTH)
            or self.valid.shape != (batch, 17)
            or self.features.shape != (batch, len(MODES), _PATCHES, RANK)
            or self.support.shape != (batch, len(MODES))
            or self.projection.shape != (WIDTH, RANK)
            or self.valid.dtype != torch.bool
            or self.support.dtype != torch.bool
            or not self.tokens.is_floating_point()
            or self.features.dtype != self.tokens.dtype
            or self.projection.dtype != self.tokens.dtype
            or any(value.device != self.tokens.device for value in tensors)
        ):
            raise ValueError("inconsistent Packed shapes, dtypes or devices")

    def take(self, index):
        """Select endpoints by scalar, slice, or 1-D torch/NumPy/list index.

        A scalar index retains a length-one endpoint axis.  Integer indices may
        repeat and boolean masks must match the endpoint count, as in PyTorch.
        """
        if not isinstance(index, slice):
            if isinstance(index, (list, tuple)) and len(index) == 0:
                index = torch.empty(0, dtype=torch.long, device=self.tokens.device)
            else:
                index = torch.as_tensor(index, device=self.tokens.device)
            if index.ndim == 0 and index.dtype != torch.bool:
                index = index.reshape(1)
            if index.ndim != 1:
                raise ValueError("endpoint indices must be scalar or one-dimensional")
            if index.dtype == torch.bool:
                if len(index) != len(self.tokens):
                    raise ValueError("boolean endpoint mask must match the batch")
            elif index.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                index = index.to(dtype=torch.long)
            else:
                raise ValueError("endpoint indices must be integers or booleans")
        return Packed(
            tokens=self.tokens[index],
            valid=self.valid[index],
            features=self.features[index],
            support=self.support[index],
            projection=self.projection,
        )


@torch.no_grad()
def encode(base, x, available, projection):
    """Encode native tokens and common-coordinate adjacent-patch features.

    ``x`` and the boolean availability mask are [B,128,23].  Nonfinite values
    are unavailable, exactly as in Simple23.  A pair is eligible when at least
    eight corresponding time/channel coordinates are observed in both patches.
    For each mode, support additionally requires at least two eligible pairs,
    RMS >= 1e-3 over their shared *source-normalized coordinates*, and a
    nonzero projected latent RMS.  Latent RMS is the square root of summed
    squared latent coordinates divided by the number of eligible pairs.
    """
    _check_base(base)
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 3
        or x.shape[1:] != (128, _CHANNELS)
        or len(x) == 0
        or not x.is_floating_point()
        or x.dtype != base.embedding.weight.dtype
        or x.device != base.embedding.weight.device
        or x.dtype != base.center.dtype
        or x.device != base.center.device
        or x.dtype != base.scale.dtype
        or x.device != base.scale.device
    ):
        raise ValueError("nonempty source-dtype/device floating [batch, 128, 23] windows required")
    if (
        not isinstance(available, torch.Tensor)
        or available.shape != x.shape
        or available.dtype != torch.bool
        or available.device != x.device
    ):
        raise ValueError("matching boolean availability on the input device required")
    if (
        not isinstance(projection, torch.Tensor)
        or projection.shape != (WIDTH, RANK)
        or projection.dtype != x.dtype
        or projection.device != x.device
        or projection.requires_grad
    ):
        raise ValueError("a detached source-dtype/device [64,8] projection is required")

    # An orthonormal projection is essential to the advertised induced-norm bound.
    gram = projection.T @ projection
    tolerance = max(1e-8, 20 * torch.finfo(projection.dtype).eps)
    _require(
        torch.isfinite(projection).all()
        & ((gram - torch.eye(RANK, dtype=x.dtype, device=x.device)).abs() <= tolerance).all()
        & torch.isfinite(base.center).all()
        & torch.isfinite(base.scale).all()
        & (base.scale > 0).all()
        & torch.isfinite(base.embedding.weight).all()
        & torch.isfinite(base.embedding.bias).all(),
        "finite source normalization/embedding and an orthonormal projection are required",
    )

    observed = available & torch.isfinite(x)
    clean = torch.where(observed, x, base.center)
    level = torch.asinh((clean - base.center) / base.scale)
    batch = len(x)
    native = torch.cat((level, observed.to(level.dtype)), dim=-1).reshape(batch, _PATCHES, 368)
    patches = base.embedding(native)
    tokens = torch.cat((patches, base.cls.expand(batch, -1, -1)), dim=1) + base.position
    patch_valid = observed.reshape(batch, _PATCHES, _PATCH * _CHANNELS).any(-1)
    valid = torch.cat((patch_valid, torch.ones(batch, 1, dtype=torch.bool, device=x.device)), dim=1)

    levels = level.reshape(batch, _PATCHES, _PATCH, _CHANNELS)
    masks = observed.reshape(batch, _PATCHES, _PATCH, _CHANNELS)
    common = masks[:, 1:] & masks[:, :-1]
    eligible = common.sum(dim=(-2, -1)) >= _PATCH
    current = torch.where(common, levels[:, 1:], 0)
    previous = torch.where(common, levels[:, :-1], 0)
    coordinates = torch.stack((current - previous, current), dim=1)
    coordinates = coordinates.masked_fill(~eligible[:, None, :, None, None], 0)
    count = eligible.sum(dim=1)
    shared_count = (common.sum(dim=(-2, -1)) * eligible).sum(dim=1).clamp_min(1)
    coordinate_rms = (coordinates.square().sum(dim=(2, 3, 4)) / shared_count[:, None]).sqrt()
    latents = F.linear(coordinates.reshape(batch, len(MODES), _PATCHES - 1, 184), _value_weight(base)) @ projection
    latent_rms = (latents.square().sum(dim=(2, 3)) / count[:, None].clamp_min(1)).sqrt()
    _require(
        torch.isfinite(level).all() & torch.isfinite(tokens).all()
        & torch.isfinite(coordinate_rms).all() & torch.isfinite(latent_rms).all(),
        "finite normalized windows, native tokens and patch latents are required",
    )
    support = (count[:, None] >= 2) & (coordinate_rms >= _COORDINATE_FLOOR) & (latent_rms > 0)
    denominator = torch.where(support, latent_rms, torch.ones_like(latent_rms))
    features = (latents / denominator[:, :, None, None]).masked_fill(~support[:, :, None, None], 0)
    features = torch.cat((features.new_zeros(batch, len(MODES), 1, RANK), features), dim=2)
    return Packed(tokens=tokens, valid=valid, features=features, support=support, projection=projection)


def delta(packed, state, mode):
    """Return [B,16,64] patch-only writes from per-endpoint [B,64] states."""
    if not isinstance(packed, Packed):
        raise ValueError("a Packed window batch is required")
    if not isinstance(mode, Integral) or isinstance(mode, bool) or mode not in (0, 1):
        raise ValueError("mode must be integer 0 (motion) or 1 (level)")
    if (
        not isinstance(state, torch.Tensor)
        or state.shape != (len(packed.tokens), RANK * RANK)
        or state.dtype != packed.tokens.dtype
        or state.device != packed.tokens.device
    ):
        raise ValueError("one source-dtype/device [64] state per endpoint is required")
    _require(torch.isfinite(state).all(), "finite endpoint states required")
    return (packed.features[:, mode] @ state.reshape(-1, RANK, RANK)) @ packed.projection.T


def _math_encoder(base, tokens, valid):
    """Execute native pre-norm layers using MHA's functional math-SDPA path.

    Calling the functional attention implementation avoids PyTorch's global
    inference fastpath without changing global backend flags or module hooks.
    All projections, masks, normalizations, residuals and FFNs remain native.
    Encoder, layer, and self-attention forward hooks are not invoked by this
    copied path; hooks on the constituent modules that are called still run.
    """
    h = tokens
    with sdpa_kernel(SDPBackend.MATH):
        for layer in base.encoder.layers:
            attention = layer.self_attn
            q = layer.norm1(h).transpose(0, 1)
            attended, _ = F.multi_head_attention_forward(
                q, q, q, attention.embed_dim, attention.num_heads,
                attention.in_proj_weight, attention.in_proj_bias,
                attention.bias_k, attention.bias_v, attention.add_zero_attn,
                attention.dropout, attention.out_proj.weight, attention.out_proj.bias,
                training=attention.training, key_padding_mask=~valid,
                need_weights=False, attn_mask=None, is_causal=False,
            )
            h = h + layer.dropout1(attended.transpose(0, 1))
            h = h + layer._ff_block(layer.norm2(h))
        if base.encoder.norm is not None:
            h = base.encoder.norm(h)
    return h


def forward(base, packed, state, mode, *, forecast=False):
    """Return native risk logit and, optionally, H25 forecast with patch writes.

    This is differentiable with respect to ``state``, including at zero; no
    optimizer or fast state is stored.  Output fast modulation is the native
    frozen ``fast_origin``.  The CLS token is never written.
    """
    _check_base(base)
    if not isinstance(packed, Packed) or packed.tokens.device != base.center.device or packed.tokens.dtype != base.center.dtype:
        raise ValueError("Packed windows must match the source dtype and device")
    if not isinstance(forecast, bool):
        raise ValueError("forecast must be a boolean")
    if len(packed.tokens) == 0:
        raise ValueError("forward requires at least one endpoint")
    change = delta(packed, state, mode)
    tokens = torch.cat((packed.tokens[:, :_PATCHES] + change, packed.tokens[:, _PATCHES:]), dim=1)
    h = _math_encoder(base, tokens, packed.valid)
    gain, bias = base.fast_origin.chunk(2)
    h = base.norm(h) * (1 + 0.1 * gain[None, None].tanh()) + 0.1 * bias[None, None].tanh()
    h = h.masked_fill(~packed.valid[:, :, None], 0)
    result = {"logit": base.risk_head(h[:, -1]).squeeze(-1)}
    if forecast:
        result["forecast"] = base.forecast_head(h.flatten(1)).reshape(len(h), 25, _CHANNELS)
    return result
