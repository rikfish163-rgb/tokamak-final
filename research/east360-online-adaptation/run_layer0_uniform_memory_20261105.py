"""Matched ablation: replace only the contextual memory kernel with uniform weights."""

import argparse
from pathlib import Path
from unittest.mock import patch

import torch

import layer0_local_memory_20261105 as memory
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import run


PLAN = r.ROOT / "LAYER0_UNIFORM_MEMORY_CONTROL_PLAN_20261105.md"
VARIANT = "uniform_global_layer0"


def uniform_memory_actions(keys, episodes, bandwidth):
    memory._matrix(keys, memory.KEY_WIDTH, "query keys")
    memory._bandwidth(bandwidth)
    if not episodes:
        return keys.new_zeros((len(keys), memory.ACTION_WIDTH))
    for episode in episodes:
        if not isinstance(episode, memory.Episode):
            raise ValueError("episode objects required")
        episode.__post_init__()
        memory._same_type(keys, episode.delta)
    residual = torch.stack([v.delta for v in episodes]).mean(dim=0)
    if not torch.isfinite(residual).all():
        raise FloatingPointError("nonfinite uniform memory action")
    return residual[None, :].expand(len(keys), -1)


def run_control(output):
    write = r.write_new
    binding = {str(p.resolve()): r.sha(p) for p in (PLAN, Path(__file__))}

    def recorded_write(path, value):
        if path == output / "protocol.json":
            value = dict(
                value,
                memory_variant=VARIANT,
                input_sha256={**value["input_sha256"], **binding},
            )
        if path == output / "completed.json" and any(
            r.sha(p) != d for p, d in binding.items()
        ):
            raise ValueError("matched control bindings changed during replay")
        write(path, value)

    with (
        patch.object(memory, "memory_actions", new=uniform_memory_actions),
        patch.object(r, "write_new", new=recorded_write),
    ):
        run(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run_control(p.parse_args().output)
