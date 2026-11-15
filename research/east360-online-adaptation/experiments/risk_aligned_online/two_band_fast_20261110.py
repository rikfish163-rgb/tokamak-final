"""Temporary, two-band modulation of Simple23 patch embeddings."""


def banded_forward(base, values, available, delta):
    """Run the source model with [batch, 2, 2*dim] gain/bias deltas.

    Bands are the earlier and later halves of the ordered patch embeddings.
    The temporary hook is scoped to this forward; calls sharing a base model
    must not overlap while the hook is installed.
    """
    cfg = base.cfg
    patches = cfg.window // cfg.patch
    if cfg.fast_site != "tokens" or cfg.input_view != "source" or patches % 2:
        raise ValueError("source-view token adaptation with an even patch count is required")
    batch = len(values)
    if delta.shape != (batch, 2, 2 * cfg.dim):
        raise ValueError("delta must have shape [batch, 2, 2*dim]")

    def modulate(_module, _inputs, embedding):
        gain, bias = delta.chunk(2, dim=-1)
        grouped = embedding.reshape(batch, 2, patches // 2, cfg.dim)
        adjusted = grouped * (1 + 0.1 * gain[:, :, None, :].tanh())
        adjusted = adjusted + 0.1 * bias[:, :, None, :].tanh()
        return adjusted.reshape_as(embedding)

    hook = base.embedding.register_forward_hook(modulate)
    try:
        return base(values, available)
    finally:
        hook.remove()
