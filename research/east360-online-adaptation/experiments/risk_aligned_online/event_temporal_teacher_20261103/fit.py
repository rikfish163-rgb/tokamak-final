"""Fit paired CPU-only auxiliary temporal-order teachers on supplied training rows.

The caller must supply already-standardized, row-aligned differences and training
metadata. In particular, comparator row i must retain the anchor metadata of row
i; this interface cannot independently verify comparator provenance or the HDF5
origin of the supplied outcome. No evaluation or checkpoint selection occurs.
"""

from collections import defaultdict
from copy import deepcopy
import math
from numbers import Integral

import numpy as np
import torch
import torch.nn.functional as functional


_ARMS = ("true_order", "shuffled_order", "matched_other_order")
_SEED = 131
_WIDTH = 128
_BATCH_SIZE = 64
_MAX_STEPS = 512


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate(differences, records, steps):
    steps = _integer(steps, "steps")
    if not 1 <= steps <= _MAX_STEPS:
        raise ValueError(f"steps must be between 1 and {_MAX_STEPS}")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list")
    if not isinstance(differences, dict) or set(differences) != set(_ARMS):
        raise ValueError(f"differences must contain exactly {_ARMS}")

    arrays = {}
    for arm in _ARMS:
        values = differences[arm]
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{arm} must be a numpy array")
        if values.shape != (len(records), _WIDTH):
            raise ValueError(f"{arm} must have shape (len(records), {_WIDTH})")
        if values.dtype.kind not in "iuf":
            raise TypeError(f"{arm} must contain real numbers")
        if not np.isfinite(values).all():
            raise ValueError(f"{arm} must contain only finite values")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.array(values, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(array).all():
            raise ValueError(f"{arm} must be representable as finite float32 values")
        arrays[arm] = array

    shot_outcomes = {}
    event_metadata = {}
    event_rows = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError("each record must be a dictionary")
        if not {"shot", "event_id", "family", "disruptive"} <= record.keys():
            raise ValueError(
                "each record must contain shot, event_id, family and disruptive"
            )
        shot = _integer(record["shot"], "shot")
        event_id = _integer(record["event_id"], "event_id")
        family = _integer(record["family"], "family")
        if not 0 <= family <= 5:
            raise ValueError("family must be in 0..5")
        if not isinstance(record["disruptive"], (bool, np.bool_)):
            raise TypeError("disruptive must be a boolean")
        disruptive = bool(record["disruptive"])
        if shot in shot_outcomes and shot_outcomes[shot] != disruptive:
            raise ValueError("disruptive must be consistent for each shot")
        shot_outcomes[shot] = disruptive
        event = (shot, event_id)
        if event in event_metadata and event_metadata[event] != (family, disruptive):
            raise ValueError("family and disruptive must be consistent for each event")
        event_metadata[event] = (family, disruptive)
        event_rows[event].append(index)
    return arrays, event_metadata, event_rows, steps


def _shuffle_signs(event_metadata, event_rows, nrows):
    """Balance event-level flips within family/outcome, pooling singleton cells."""
    rng = np.random.default_rng(_SEED)
    strata = defaultdict(list)
    for event, metadata in sorted(event_metadata.items()):
        strata[metadata].append(event)

    event_signs = {event: 1 for event in event_metadata}
    singleton_pools = defaultdict(list)
    summaries = []

    def balance(events, *, positive_remainder=False):
        count = len(events) // 2
        if len(events) % 2 and not positive_remainder:
            count += int(rng.integers(0, 2))
        order = rng.permutation(len(events))
        for position in order[:count]:
            event_signs[events[int(position)]] = -1
        remainder = (
            events[int(order[-1])] if positive_remainder and len(events) % 2 else None
        )
        return count, remainder

    for (family, disruptive), events in sorted(strata.items()):
        if len(events) == 1:
            singleton_pools[disruptive].extend(events)
            flipped = None
        else:
            flipped, _ = balance(events)
        summaries.append(
            {
                "family": family,
                "disruptive": disruptive,
                "events": len(events),
                "flipped": flipped,
            }
        )

    pool_summaries = []
    forced_positive = []
    for disruptive, events in sorted(singleton_pools.items()):
        flipped, remainder = balance(events, positive_remainder=True)
        if remainder is not None:
            shot, event_id = remainder
            family, _ = event_metadata[remainder]
            forced_positive.append(
                {
                    "shot": shot,
                    "event_id": event_id,
                    "family": family,
                    "disruptive": disruptive,
                }
            )
        pool_summaries.append(
            {"disruptive": disruptive, "events": len(events), "flipped": flipped}
        )

    signs = [1] * nrows
    for event, rows in event_rows.items():
        for row in rows:
            signs[row] = event_signs[event]
    disclosure = {
        "outcome_source": "caller-supplied records[].disruptive (must be HDF-derived)",
        "family_outcome_strata": summaries,
        "pooled_singletons_by_outcome": pool_summaries,
        "forced_positive_singletons": forced_positive,
    }
    return signs, disclosure


def _schedule(event_rows, steps):
    shots = defaultdict(list)
    for shot, event_id in sorted(event_rows):
        shots[shot].append((shot, event_id))
    shot_ids = sorted(shots)
    rng = np.random.default_rng(_SEED)
    schedule = []
    for _ in range(steps):
        batch = []
        for _ in range(_BATCH_SIZE):
            shot = shot_ids[int(rng.integers(len(shot_ids)))]
            events = shots[shot]
            event = events[int(rng.integers(len(events)))]
            rows = event_rows[event]
            batch.append(rows[int(rng.integers(len(rows)))])
        schedule.append(batch)
    return schedule


def _heads():
    # torch.manual_seed also seeds CUDA; seed and restore only the CPU generator.
    state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(_SEED)
        head = torch.nn.Linear(_WIDTH, 1, bias=False, device="cpu", dtype=torch.float32)
    finally:
        torch.random.set_rng_state(state)
    initial = head.weight.detach().reshape(-1).tolist()
    return {arm: deepcopy(head) for arm in _ARMS}, initial


def _train(arrays, signs, schedule):
    # AdamW can create its internal step counter on the ambient default device.
    # Override it locally as well as placing the input and parameters on CPU.
    with torch.device("cpu"):
        models, initial_weights = _heads()
        optimizers = {
            arm: torch.optim.AdamW(
                models[arm].parameters(), lr=0.001, weight_decay=0.001
            )
            for arm in _ARMS
        }
        values = {arm: torch.from_numpy(arrays[arm]) for arm in _ARMS}
        targets = {
            arm: torch.tensor(signs[arm], dtype=torch.float32, device="cpu")
            for arm in _ARMS
        }
        losses = {arm: [] for arm in _ARMS}
        for batch in schedule:
            indices = torch.tensor(batch, dtype=torch.long, device="cpu")
            for arm in _ARMS:
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                logits = models[arm](values[arm].index_select(0, indices)).squeeze(-1)
                loss = functional.softplus(
                    -targets[arm].index_select(0, indices) * logits
                ).mean()
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(f"non-finite training loss for {arm}")
                loss.backward()
                optimizer.step()
                if not torch.isfinite(models[arm].weight).all().item():
                    raise FloatingPointError(f"non-finite teacher weights for {arm}")
                losses[arm].append(loss_value)
        heads = {arm: models[arm].weight.detach().reshape(-1).tolist() for arm in _ARMS}
    return heads, initial_weights, losses


def fit(
    differences: dict[str, np.ndarray], records: list[dict], steps: int = 512
) -> dict:
    """Train three paired linear heads using a common hierarchical row schedule.

    ``steps`` is a positive integer capped at 512; shorter budgets support
    synthetic tests. Signs are flipped per entire event for the shuffled-order
    control only. Families and caller-supplied HDF-derived outcomes are used
    solely for offline sign balancing, never as model inputs or sampling weights.
    The returned JSON-compatible mapping contains each final head, per-row
    signs, shared schedule, per-update training loss, and the actual budget.
    """
    arrays, event_metadata, event_rows, steps = _validate(differences, records, steps)
    shuffled_signs, disclosure = _shuffle_signs(
        event_metadata, event_rows, len(records)
    )
    signs = {
        "true_order": [1] * len(records),
        "shuffled_order": shuffled_signs,
        "matched_other_order": [1] * len(records),
    }
    schedule = _schedule(event_rows, steps)
    heads, initial_weights, losses = _train(arrays, signs, schedule)

    return {
        "seed": _SEED,
        "initial_weight128": initial_weights.copy(),
        "initial_weights": initial_weights,
        "heads": heads,
        "schedule": schedule,
        "signs": signs,
        "training_losses": losses,
        "shuffle_sign_control": disclosure,
        "actual_budget": {
            "steps": steps,
            "batch_size": _BATCH_SIZE,
            "examples_per_arm": steps * _BATCH_SIZE,
            "optimizer_updates_per_arm": steps,
            "final_checkpoint": "last",
            "device": "cpu",
            "learning_rate": 0.001,
            "weight_decay": 0.001,
        },
    }
