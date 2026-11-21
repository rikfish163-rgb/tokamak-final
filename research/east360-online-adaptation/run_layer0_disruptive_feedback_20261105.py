"""Complete sequential replay: safe labels protect history, but do not fit episodes."""

import argparse
from pathlib import Path
from unittest.mock import patch

import moving_peak_replay_20261105 as r
import run_layer0_query_memory_20261105 as core


PLAN = r.ROOT / "LAYER0_DISRUPTIVE_FEEDBACK_PLAN_20261105.md"
VARIANT = "disruptive_feedback_only"


def disruptive_feedback(original):
    def fit(base, current, *args, **kwargs):
        flag = current[2][1]
        if type(flag) is not bool:
            raise ValueError("committed binary HDF5 outcome required")
        if not flag:
            # The replay still appends this safely revealed case to the guard set.
            return None, dict(decision="guard_only_safe_outcome")
        return original(base, current, *args, **kwargs)

    return fit


def run(output):
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
            raise ValueError("disruptive feedback bindings changed during replay")
        write(path, value)

    with (
        patch.object(core, "learn", new=disruptive_feedback(core.learn)),
        patch.object(r, "write_new", new=recorded_write),
    ):
        core.run(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
