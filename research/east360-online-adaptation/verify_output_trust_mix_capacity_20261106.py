"""Directly replay accepted capacity states on original, complete TRAIN shots."""

import argparse
from pathlib import Path

import numpy as np
import torch

import run_output_trust_table_20261106 as bank
from inspect_peak_contrast_first_alarm_20261106 import first
from verify_confirmation_raw_replay_20261105 import Hashes


def verify(path, out):
    r, h = bank.r, Hashes()
    if out.exists():
        raise FileExistsError(out)
    raw = r.read(path)
    if (
        raw["status"] != "TRAIN_LATE_CONTINUOUS_ORACLE_NOT_POLICY"
        or raw["val_test_east_reads"] != 0
        or not all(h(p) == v for p, v in raw["input_sha256"].items())
    ):
        raise ValueError("same measured capacity evidence required")
    _, original = bank.src.source_inputs(h)
    allowed = {q["shot"]: q["sha256"] for q in original}
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    if parent["threshold"] != r.THRESHOLD:
        raise ValueError("different source threshold")
    shots, results = {}, []
    for mode, cases in raw["results"].items():
        for q in cases:
            for sid in q["reachable"]:
                if sid not in shots:
                    if h(r.io23.path_for("J-TEXT", sid)) != allowed[sid]:
                        raise ValueError(
                            "capacity witness must be original source TRAIN"
                        )
                    shot = r.simple_data.read_shot("J-TEXT", sid)
                    flag, down = r.io23.read_outcome("J-TEXT", sid)
                    x, a = bank.prior.views(base, shot)
                    z = bank.prior.scan(base, x, a).numpy()
                    native = first(shot["times"], z, flag, down, bank.TAU)
                    if not flag or native["timely"]:
                        raise ValueError("original case was not a late disruption")
                    shots[sid] = (shot["times"], down, x, a, z, native)
                t, down, x, a, z, old = shots[sid]
                trials = []
                for row in raw["rows"]:
                    if (row["shot"], row["seed"]) != (sid, q["seed"]):
                        continue
                    if mode == "observed_projection":
                        trials.extend((row["maturity"], v) for v in row[mode])
                    else:
                        trials.extend(
                            (row["maturity"], c)
                            for v in row["trials"][mode]
                            for c in (v["projected"], v["optimized"])
                            if c is not None
                        )
                m, best = max(trials, key=lambda q: q[1]["peak"])
                if best["peak"] != q["peak_by_shot"][str(sid)]:
                    raise ValueError(
                        "chosen capacity witness is not the archived maximum"
                    )
                state = torch.tensor(best["state"], dtype=torch.float32)
                p = np.array([m - 16, m - 8, m])
                at = np.arange(m + 1, min(len(t), m + 33))
                with torch.no_grad():
                    past = [
                        bank.old.risk(
                            base,
                            x[p].contiguous(),
                            a[p].contiguous(),
                            sign * state.expand(3, -1),
                        ).numpy()
                        for sign in (1, -1)
                    ]
                    now = bank.old.risk(
                        base,
                        x[at].contiguous(),
                        a[at].contiguous(),
                        state.expand(len(at), -1),
                    ).numpy()
                err = float(np.abs(np.stack(past) - z[p]).max())
                joined = z.copy()
                joined[at] = now
                replay = first(t, joined, True, down, bank.TAU)
                if (
                    err > bank.tr.LIMIT
                    or not replay["timely"]
                    or float(state.norm()) > bank.SITE_NORM + 5e-7
                ):
                    raise ValueError(
                        "independent full-shot model replay did not verify capacity"
                    )
                results.append(
                    dict(
                        mode=mode,
                        shot=sid,
                        seed=q["seed"],
                        maturity=m,
                        native=old,
                        counterfactual=replay,
                        norm=float(state.norm()),
                        max_abs_past=err,
                    )
                )
    h.check()
    r.write_new(
        out,
        dict(
            status="VERIFIED_COUNTERFACTUAL_NOT_POLICY",
            input_sha256=h(path),
            rows=results,
            verifier_sha256=h(Path(__file__)),
            val_test_east_reads=0,
        ),
    )
    print("independent full-shot capacity witnesses", len(results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    verify(a.input, a.output)
