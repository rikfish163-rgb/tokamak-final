"""Explicit first-FFN plastic directions for the frozen source model."""

import torch

import motion_metric_model_20261107 as motion_model


def _check_basis(basis):
    if basis.shape != (4, 128):
        raise ValueError("a 4x128 first-FFN basis is required")


def encode_windows(base, x, available, basis):
    """Extend the existing cache with the actual first linear2 input."""
    _check_basis(basis)
    activation = []

    def capture(_module, args):
        if not activation:
            activation.append(args[0][:, :16].detach())

    handle = base.encoder.layers[0].linear2.register_forward_pre_hook(capture)
    try:
        pack = motion_model.encode_windows(base, x, available, basis)
    finally:
        handle.remove()
    if not activation:
        raise RuntimeError("the first FFN linear2 input was not captured")
    return {**pack, "activation": activation[0]}


def forward_state(base, pack, state, basis):
    """Apply the existing state forward using the current differentiable basis."""
    _check_basis(basis)
    phi = pack["activation"] @ basis.T / 2
    return motion_model.forward_state(base, {**pack, "phi": phi}, state)


@torch.no_grad()
def project_rows(basis):
    """Reproject a full-row-rank basis in place with a positive QR diagonal."""
    _check_basis(basis)
    q, r = torch.linalg.qr(basis.T, mode="reduced")
    signs = torch.where(r.diagonal() < 0, -1, 1)
    basis.copy_((q * signs).T)
    return basis


def random_basis(reference, seed=131):
    """Generate deterministic orthogonal rows without changing global RNG state."""
    _check_basis(reference)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    basis = torch.randn((4, 128), dtype=torch.float64, device="cpu", generator=generator)
    return project_rows(basis).to(reference)
