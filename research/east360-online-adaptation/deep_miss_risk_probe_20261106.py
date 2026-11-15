"""Train independent bounded residual heads on caller-supplied training arrays.

This module does not construct data splits or read experiment artifacts.  Each
prediction is a logit, not a probability: the native logit plus a residual in
[-4, 4].  The caller is responsible for supplying only approved training data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HIDDEN_DIM = 48
EPOCHS = 15
BATCH_SIZE = 512
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.001
RESIDUAL_CAP = 4.0
MIN_STD = 1e-3
FEATURE_CLIP = 8.0

_PARAMETER_NAMES = ("0.weight", "0.bias", "2.weight", "2.bias")


def _finite_array(value: object, name: str) -> np.ndarray:
    """Convert a real numeric array to float64, rejecting nonfinite values."""
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _weighted_stats(features: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Compute in scaled coordinates to avoid squaring large feature values.
    magnitude = np.maximum(np.max(np.abs(features), axis=0), 1.0)
    scaled = features / magnitude
    total = np.sum(weights)
    mean_scaled = np.sum(scaled * weights[:, None], axis=0) / total
    variance_scaled = np.sum((scaled - mean_scaled) ** 2 * weights[:, None], axis=0) / total
    mean = mean_scaled * magnitude
    scale = np.maximum(np.sqrt(variance_scaled) * magnitude, MIN_STD)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("weighted feature statistics must be finite")
    return mean, scale


def _standardize(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # Halve both terms if their subtraction overflows; the quotient may still
    # lie within the clipping bound when the fitted scale is also very large.
    with np.errstate(over="ignore", invalid="ignore"):
        difference = features - mean
        standardized = difference / scale
        if np.isinf(difference).any():
            fallback = (features / 2 - mean / 2) / (scale / 2)
            standardized = np.where(np.isinf(difference), fallback, standardized)
        standardized = np.clip(standardized, -FEATURE_CLIP, FEATURE_CLIP)
    if not np.isfinite(standardized).all():
        raise ValueError("standardized features must be finite")
    return standardized.astype(np.float32)


def _new_head(width: int, seed: int) -> nn.Sequential:
    # Initialize on CPU with a local RNG context, independently for every arm.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(width, HIDDEN_DIM, dtype=torch.float32, device="cpu"),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1, dtype=torch.float32, device="cpu"),
        )
        nn.init.zeros_(model[2].weight)
        nn.init.zeros_(model[2].bias)
    return model


def _bounded_residual(raw: torch.Tensor) -> torch.Tensor:
    return RESIDUAL_CAP * torch.tanh(raw / RESIDUAL_CAP)


def _weighted_loss(
    model: nn.Module,
    features: torch.Tensor,
    native: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    n_rows: int,
) -> torch.Tensor:
    logits = native + _bounded_residual(model(features).squeeze(-1))
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    # Weights are normalized over the whole training set, not this batch.
    return torch.sum(weights * losses) / (len(weights) / n_rows)


def train_heads(
    features: dict[str, np.ndarray],
    native_logit: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    seed: int,
    device: str | torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Fit one head per arm using identical initialization and epoch shuffles.

    All arms must have the same nonempty [N, D] shape.  Labels must be binary,
    and every training weight must be strictly positive.  Each mini-batch uses
    global weights and the batch-size/N sampling fraction. Diagnostics contain
    the initial and post-epoch weighted BCE, evaluated on these training rows.
    State arrays are numeric, finite, and suitable for NPZ or JSON conversion.
    """
    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a nonempty dictionary of arms")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    arrays = {}
    shape = None
    for name, values in features.items():
        if not isinstance(name, str):
            raise ValueError("arm names must be strings")
        array = _finite_array(values, f"features[{name!r}]")
        if array.ndim != 2 or not all(array.shape):
            raise ValueError("each feature arm must have nonempty [N, D] shape")
        if shape is not None and array.shape != shape:
            raise ValueError("all feature arms must have the same [N, D] shape")
        shape = array.shape
        arrays[name] = array

    n_rows, width = shape
    native = _finite_array(native_logit, "native_logit")
    labels = _finite_array(target, "target")
    weights = _finite_array(weight, "weight")
    if any(array.shape != (n_rows,) for array in (native, labels, weights)):
        raise ValueError("native_logit, target, and weight must each have shape [N]")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("target must contain only 0 and 1")
    if np.any(weights <= 0):
        raise ValueError("weights must be strictly positive")
    with np.errstate(over="ignore"):
        weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("weight sum must be finite and positive")
    normalized_weights = weights / weight_sum
    if np.any(normalized_weights == 0):
        raise ValueError("weight range is too large to normalize")

    device = torch.device(device)
    native_tensor = torch.as_tensor(native, dtype=torch.float64, device=device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float64, device=device)
    weights_tensor = torch.as_tensor(normalized_weights, dtype=torch.float64, device=device)
    states = {}
    diagnostics = {}
    for name, array in arrays.items():
        mean, scale = _weighted_stats(array, normalized_weights)
        x = torch.as_tensor(_standardize(array, mean, scale), dtype=torch.float32, device=device)
        model = _new_head(width, seed).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def evaluate() -> float:
            # Accumulate by batch so diagnostics do not require a full hidden matrix.
            total = torch.zeros((), dtype=torch.float64, device=device)
            with torch.no_grad():
                for start in range(0, n_rows, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, n_rows)
                    logits = native_tensor[start:end] + _bounded_residual(model(x[start:end]).squeeze(-1))
                    losses = F.binary_cross_entropy_with_logits(logits, labels_tensor[start:end], reduction="none")
                    total += torch.sum(weights_tensor[start:end] * losses)
            result = float(total.item() / normalized_weights.sum())
            if not np.isfinite(result):
                raise ValueError("training diagnostics must be finite")
            return result

        initial_loss = evaluate()
        loss_curve = []
        for _ in range(EPOCHS):
            permutation = torch.randperm(n_rows, generator=generator, device="cpu")
            for start in range(0, n_rows, BATCH_SIZE):
                indices = permutation[start:start + BATCH_SIZE].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = _weighted_loss(model, x[indices], native_tensor[indices], labels_tensor[indices],
                                      weights_tensor[indices], n_rows)
                if not torch.isfinite(loss).item():
                    raise ValueError("weighted training loss must be finite")
                loss.backward()
                optimizer.step()
            loss_curve.append(evaluate())

        state = {"mean": mean.copy(), "scale": scale.copy()}
        state.update({key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()})
        if not all(np.isfinite(value).all() for value in state.values()):
            raise ValueError("trained state must be finite")
        states[name] = state
        diagnostics[name] = {
            "n_samples": n_rows,
            "weight_sum": weight_sum,
            "initial_loss": initial_loss,
            "loss_curve": loss_curve,
        }
    return states, diagnostics


def predict(
    state: dict[str, np.ndarray],
    features: np.ndarray,
    native_logit: np.ndarray,
    device: str | torch.device,
) -> np.ndarray:
    """Return finite [N] logit scores using the supplied saved training state."""
    if not isinstance(state, dict) or not {"mean", "scale", *_PARAMETER_NAMES}.issubset(state):
        raise ValueError("state must contain scaling statistics and head parameters")
    mean = _finite_array(state["mean"], "state mean")
    scale = _finite_array(state["scale"], "state scale")
    if mean.ndim != 1 or mean.size == 0 or scale.shape != mean.shape or np.any(scale < MIN_STD):
        raise ValueError("state mean and floored scale must have matching [D] shape")
    x = _finite_array(features, "features")
    native = _finite_array(native_logit, "native_logit")
    if x.ndim != 2 or x.shape[1] != mean.size or native.shape != (x.shape[0],):
        raise ValueError("features must have shape [N, D] and native_logit shape [N]")

    shapes = {"0.weight": (HIDDEN_DIM, mean.size), "0.bias": (HIDDEN_DIM,),
              "2.weight": (1, HIDDEN_DIM), "2.bias": (1,)}
    parameters = {}
    for name, expected_shape in shapes.items():
        value = _finite_array(state[name], f"state {name}")
        if value.shape != expected_shape:
            raise ValueError(f"state {name} has an invalid shape")
        with np.errstate(over="ignore"):
            value = value.astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"state {name} must be finite in float32")
        parameters[name] = torch.as_tensor(value, device=device)

    with torch.no_grad():
        tensor = torch.as_tensor(_standardize(x, mean, scale), device=device)
        hidden = F.silu(F.linear(tensor, parameters["0.weight"], parameters["0.bias"]))
        raw = F.linear(hidden, parameters["2.weight"], parameters["2.bias"]).squeeze(-1)
        residual = _bounded_residual(raw).cpu().numpy().astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        result = native + residual
    if not np.isfinite(result).all():
        raise ValueError("predicted scores must be finite")
    return result
