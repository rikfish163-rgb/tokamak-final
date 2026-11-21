"""Instrument the unchanged timing replay with a monotonic reveal chain."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import layer_timing_control_20261105 as timing
import moving_peak_replay_20261105 as r


CLOCK_PROTOCOL = "MONOTONIC_SEAL_OUTCOME_REVEAL_CHAIN_V1"


class RevealClock:
    def __init__(self, output, writer, reader):
        self.output = Path(output).resolve()
        self.writer = writer
        self.reader = reader
        self.next_rank = 1
        self.previous_sha = None
        self.pending = None
        self.last_ns = time.monotonic_ns()
        self.initial_ns = self.last_ns

    def _tick(self):
        now = time.monotonic_ns()
        if now < self.last_ns:
            raise RuntimeError("monotonic process clock regressed")
        self.last_ns = now
        return now

    def write(self, path, value):
        path = Path(path).resolve()
        if not path.is_relative_to(self.output):
            raise ValueError("writer target outside the clocked replay")
        if path == self.output / "protocol.json":
            value = dict(
                value,
                clock_protocol=dict(
                    name=CLOCK_PROTOCOL,
                    initial_ns=self.initial_ns,
                    recorder=str(Path(__file__).resolve()),
                    sha256=r.sha(Path(__file__)),
                ),
            )
        elif path.name.endswith("_seal.json"):
            if self.pending is not None:
                raise RuntimeError("new prediction seal before previous reveal")
            sid, rank = value["shot"], value["rank"]
            if rank != self.next_rank or path.name != f"{rank:04d}_{sid}_seal.json":
                raise RuntimeError("unexpected seal ordinal or shot")
            curve = path.with_name(f"{rank:04d}_{sid}.npz")
            curve_sha = r.sha(curve)
            if curve_sha != value["curve_sha256"]:
                raise RuntimeError("curve changed before sealing")
            note = dict(
                ordinal=3 * rank - 2,
                monotonic_ns=self._tick(),
                previous_reveal_sha256=self.previous_sha,
            )
            self.writer(path, dict(value, clock=note))
            self.pending = dict(
                sid=sid,
                curve=curve,
                curve_sha=curve_sha,
                seal=path,
                seal_sha=r.sha(path),
                outcome_ns=None,
            )
            return
        elif path.name.endswith("_revealed.json"):
            p = self.pending
            if p is None or p["outcome_ns"] is None:
                raise RuntimeError("outcome not read through the sealed boundary")
            rank, sid = self.next_rank, p["sid"]
            if (
                value["shot"] != sid
                or path.name != f"{rank:04d}_{sid}_revealed.json"
                or value["after_seal_sha256"] != p["seal_sha"]
            ):
                raise RuntimeError("reveal does not belong to the active seal")
            note = dict(
                outcome_ordinal=3 * rank - 1,
                outcome_read_start_ns=p["outcome_ns"][0],
                outcome_read_end_ns=p["outcome_ns"][1],
                ordinal=3 * rank,
                monotonic_ns=self._tick(),
            )
            self.writer(path, dict(value, clock=note))
            self.previous_sha = r.sha(path)
            self.pending = None
            self.next_rank += 1
            return
        elif path == self.output / "completed.json":
            if self.pending is not None or self.next_rank != value["n_shots"] + 1:
                raise RuntimeError("incomplete reveal chain")
            value = dict(
                value,
                clock_chain=dict(
                    last_reveal_sha256=self.previous_sha,
                    events=3 * (self.next_rank - 1),
                    monotonic_ns=self._tick(),
                ),
            )
        self.writer(path, value)

    def outcome(self, device, sid):
        p = self.pending
        if device != "EAST" or p is None or sid != p["sid"]:
            raise RuntimeError("no matching sealed EAST prediction")
        if p["outcome_ns"] is not None:
            raise RuntimeError("outcome already read for this seal")
        if r.sha(p["curve"]) != p["curve_sha"] or r.sha(p["seal"]) != p["seal_sha"]:
            raise RuntimeError("curve changed or seal mismatch before HDF read")
        start = self._tick()
        result = self.reader(device, sid)
        p["outcome_ns"] = (start, self._tick())
        return result


def replay(output):
    clock = RevealClock(output, r.write_new, r.io23.read_outcome)
    with (
        patch.object(r, "write_new", new=clock.write),
        patch.object(r.io23, "read_outcome", new=clock.outcome),
    ):
        timing.run(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    replay(p.parse_args().output)
