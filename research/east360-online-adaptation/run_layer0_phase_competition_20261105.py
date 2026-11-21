"""Causal replays with past within-shot early/precursor competition."""

import argparse
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import layer0_local_memory_20261105 as memory
import layer0_phase_kernel_20261105 as kernel
import layer0_query_protocol_20261105 as protocol
import moving_peak_replay_20261105 as r
import run_layer0_query_memory_20261105 as core


PLAN = r.ROOT / "LAYER0_PHASE_COMPETITION_NEXT_20261105.md"
VARIANTS = {"all": "phase_all_early", "matched": "phase_matched_early"}


def phase_learn(delegate, mode):
    if mode not in VARIANTS:
        raise ValueError("unknown phase reference mode")

    def apply(base, current, guard, episodes, bandwidth, sealed, output, block):
        _, case, (times, flag, down) = current
        if type(flag) is not bool:
            raise ValueError("committed binary HDF5 outcome required")
        if not flag:
            return None, dict(decision="guard_only_safe_outcome")
        old = {k: sealed[k] for k in protocol.PERSISTENT_ARMS}
        previous = {
            k: r.timeline(times, v, flag, down, r.THRESHOLD) for k, v in old.items()
        }
        q = protocol.teaching_indices(times, old, flag, down, r.THRESHOLD)
        if not len(q) or all(v["status"] == "tp" for v in previous.values()):
            return delegate(
                base, current, guard, episodes, bandwidth, sealed, output, block
            )
        n_pos = len(np.unique((q - protocol.FIRST) // protocol.STRIDE))
        anchors = kernel.early_anchor_indices(times, down, n_pos, mode)
        if not len(anchors):
            return None, dict(
                decision="no_eligible_early_reference",
                query_status=previous,
                early_anchor_indices=[],
                phase_mode=mode,
            )
        keys = case[2]
        positions = (anchors - protocol.FIRST) // protocol.STRIDE
        if (positions < 0).any() or (positions >= len(keys)).any():
            raise ValueError("historical phase anchor outside the ended shot")
        bank = keys[torch.as_tensor(positions, device=keys.device)].detach().clone()
        calls = 0

        def fit(source, ctx, fast, current_keys, old_memory, h, indices, outcome):
            nonlocal calls
            if not np.array_equal(indices, q) or outcome is not True:
                raise ValueError(
                    "proposal does not use the same committed teaching indices"
                )
            calls += 1
            return kernel.fit_phase(
                source, ctx, fast, current_keys, old_memory, h, indices, outcome, bank
            )

        with patch.object(memory, "fit_episode", new=fit):
            candidate, info = delegate(
                base, current, guard, episodes, bandwidth, sealed, output, block
            )
        if calls != 1:
            raise RuntimeError("exactly one phase-aware proposal was expected")
        return candidate, dict(
            info, early_anchor_indices=anchors.tolist(), phase_mode=mode
        )

    return apply


def save_state(path, episodes):
    arrays = {"size": np.array(len(episodes), dtype=np.int64)}
    for i, episode in enumerate(episodes):
        if not isinstance(episode, kernel.PhaseEpisode):
            raise ValueError("phase state requires both banks")
        arrays.update(
            {
                f"keys_{i}": episode.keys.detach().cpu().numpy(),
                f"delta_{i}": episode.delta.detach().cpu().numpy(),
                f"early_keys_{i}": episode.early_keys.detach().cpu().numpy(),
            }
        )
    r.save_arrays(path, **arrays)
    return r.sha(path)


def run(mode, output):
    if mode not in VARIANTS:
        raise ValueError("unknown phase reference mode")
    write = r.write_new
    binding = {
        str(p.resolve()): r.sha(p)
        for p in (PLAN, Path(__file__), Path(kernel.__file__))
    }

    def recorded_write(path, value):
        if path == output / "protocol.json":
            value = dict(
                value,
                memory_variant=VARIANTS[mode],
                phase_mode=mode,
                early_min_lead_ms=500,
                optimizer_kernel="phase_competition",
                inference_kernel="phase_competition",
                input_sha256={**value["input_sha256"], **binding},
            )
        if path == output / "completed.json" and any(
            r.sha(p) != digest for p, digest in binding.items()
        ):
            raise ValueError("phase experiment input changed")
        write(path, value)

    with (
        patch.object(
            memory, "memory_actions", new=partial(kernel.phase_actions, fixed_prefix=0)
        ),
        patch.object(core, "learn", new=phase_learn(core.learn, mode)),
        patch.object(core, "save_memory", new=save_state),
        patch.object(r, "write_new", new=recorded_write),
    ):
        core.run(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=VARIANTS, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.mode, a.output)
