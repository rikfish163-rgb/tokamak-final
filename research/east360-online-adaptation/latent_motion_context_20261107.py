import torch


def assemble(raw, now, previous, older, kind):
    """Assemble detached context from already-arrived source-model states."""
    if kind not in ("summary", "level", "ordered", "stale", "swap", "global"):
        raise ValueError(f"unknown kind: {kind!r}")

    tensors = (raw, now, previous, older)
    names = ("raw", "now", "previous", "older")
    for name, tensor, width in zip(names, tensors, (69, 64, 64, 64)):
        if tensor.ndim != 2 or tensor.shape[1] != width:
            raise ValueError(f"{name} must have shape [B, {width}]")

    for name, tensor in zip(names, tensors):
        if tensor.shape[0] != raw.shape[0]:
            raise ValueError(f"{name} batch size must match raw")
        if tensor.dtype != raw.dtype:
            raise ValueError(f"{name} dtype must match raw")
        if tensor.device != raw.device:
            raise ValueError(f"{name} device must match raw")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")

    raw, now, previous, older = (tensor.detach() for tensor in tensors)
    if kind == "global":
        return raw.new_zeros((raw.shape[0], 261))
    if kind == "summary":
        zero = torch.zeros_like(now)
        return torch.cat((raw, zero, zero, zero), dim=1)
    if kind == "level":
        return torch.cat((raw, now, now, torch.zeros_like(now)), dim=1)
    if kind == "ordered":
        return torch.cat((raw, now, previous, now - previous), dim=1)
    if kind == "stale":
        return torch.cat((raw, previous, older, previous - older), dim=1)
    return torch.cat((raw, previous, now, previous - now), dim=1)
