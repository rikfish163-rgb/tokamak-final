"""Corrected control: the same uniform kernel is used by optimization and prediction."""

import argparse
from pathlib import Path
from unittest.mock import patch

import layer0_local_memory_20261105 as memory
import moving_peak_replay_20261105 as r
import run_layer0_query_memory_20261105 as core
from run_layer0_uniform_memory_20261105 import uniform_memory_actions


PLAN = r.ROOT / "LAYER0_UNIFORM_KERNEL_CORRECTION_PLAN_20261105.md"
VARIANT = "uniform_global_matched_kernel"


def uniform_kernel(keys, episodes, bandwidth, *, fixed_prefix=0):
    if type(fixed_prefix) is not int or not 0 <= fixed_prefix <= len(episodes):
        raise ValueError("valid frozen prefix required")
    frozen = tuple(
        memory.Episode(item.keys.detach(), item.delta.detach())
        for item in episodes[:fixed_prefix]
    )
    return uniform_memory_actions(keys, (*frozen, *episodes[fixed_prefix:]), bandwidth)


def run(output):
    write = r.write_new
    binding = {
        str(p.resolve()): r.sha(p)
        for p in (
            PLAN,
            Path(__file__),
            Path(uniform_memory_actions.__code__.co_filename),
        )
    }
    optimizer_calls = 0

    def counted_kernel(keys, episodes, bandwidth, *, fixed_prefix=0):
        nonlocal optimizer_calls
        optimizer_calls += 1
        return uniform_kernel(keys, episodes, bandwidth, fixed_prefix=fixed_prefix)

    def recorded_write(path, value):
        if path == output / "protocol.json":
            value = dict(
                value,
                memory_variant=VARIANT,
                optimizer_kernel="uniform",
                inference_kernel="uniform",
                input_sha256={**value["input_sha256"], **binding},
            )
        if path == output / "completed.json":
            if any(r.sha(p) != digest for p, digest in binding.items()):
                raise ValueError("uniform experiment input changed")
            value = dict(value, optimizer_kernel_calls=optimizer_calls)
        write(path, value)

    with (
        patch.object(memory, "memory_actions", new=uniform_memory_actions),
        patch.object(memory, "_memory_actions", new=counted_kernel),
        patch.object(r, "write_new", new=recorded_write),
    ):
        core.run(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
