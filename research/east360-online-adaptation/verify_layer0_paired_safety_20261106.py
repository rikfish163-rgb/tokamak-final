"""Independently verify held first alarms, fold separation and pair geometry."""

import argparse
from pathlib import Path
import time

import numpy as np

import run_layer0_paired_safety_20261106 as trial
from verify_adjacent_patch_intra_20261106 import outcome_and_clock
from verify_confirmation_raw_replay_20261105 import Hashes


b = trial.b


def alarm(t, trace, flag, down, tau):
    exists = np.flatnonzero((trace.astype(float) >= tau) & ((t <= down + 1e-9) if flag else True))
    first = int(exists[0]) if len(exists) else None
    lead = round(float((down - t[first])*1000.), 9) if flag and first is not None else None
    return dict(first=first, lead_ms=lead, timely=lead is not None and lead > 10,
                false_alarm=not flag and first is not None)


def context(view, sid, clock, by, tape):
    q = by[view, sid]
    k = q.blocks.start + np.searchsorted(tape[view].data["clock"][q.blocks], clock)
    if k >= q.blocks.stop or tape[view].data["clock"][k] != clock:
        raise ValueError("training release clock is not the original clock")
    groups = tuple((tape[view].data["ordered"][k, 0, -9:] > 0).reshape(3, 3).any(1))
    t = clock + 8
    feat = np.array([q.native[t], q.times[t] - q.times[0], q.native[t] - q.native[clock - 16]])
    return q, groups, feat


def check_pairs(protocol, tape, by):
    records = []
    unique = {"danger": set(), "safe": set(), "non_timely": set(), "non_timely_before_alarm": set()}
    for fold, note in protocol["fits"].items():
        by_view = {v: [] for v in tape}
        shots = {"danger": set(), "safe": set(), "non_timely": set(), "non_timely_before_alarm": set()}
        after_first = 0
        for view, p, ct, s, cs in note["identities"]:
            if str(protocol["folds"][str(p)]) == fold or str(protocol["folds"][str(s)]) == fold:
                raise ValueError("held physical shot leaked into its fit")
            a, ap, acontext = context(view, p, ct, by, tape)
            n, npat, ncontext = context(view, s, cs, by, tape)
            eligible = (a.down - a.times[ct+9:ct+33])*1000.
            first = a.teacher["baseline"]["first"]
            if (not a.disruptive or n.disruptive or ap != npat or sum(ap) < 2
                    or abs(acontext[0] - ncontext[0]) > 1.00001
                    or abs(acontext[1] - ncontext[1]) > .500001
                    or not ((eligible > 10) & (eligible <= a.horizon + 1e-6)).any()):
                raise ValueError("archived pair violates the preregistered matching variables")
            by_view[view].append((p,s,acontext,ncontext))
            shots["danger"].add(p)
            shots["safe"].add(s)
            after_first += int(first is not None and first <= ct + 8)
            if not a.teacher["baseline"]["timely"]:
                shots["non_timely"].add(p)
                if first is None or first > ct + 8:
                    shots["non_timely_before_alarm"].add(p)
        original, rotated = [], []
        for pairs in by_view.values():
            p, s, d, c = zip(*pairs, strict=True)
            if len(set(p)) != len(p) or len(set(s)) != len(s):
                raise ValueError("physical shot duplicated in same view/fit")
            original.extend(np.abs(np.stack(d) - np.stack(c)))
            rotated.extend(np.abs(np.stack(d) - np.roll(np.stack(c), 1, axis=0)))
        for k in unique:
            unique[k].update(shots[k])
        records.append(dict(fold=int(fold), pairs=len(original), after_first=after_first,
            **{k: len(v) for k,v in shots.items()},
            match_median=np.median(original, axis=0).tolist(),
            shuffle_median=np.median(rotated, axis=0).tolist()))
    return records, {k: sorted(v) for k,v in unique.items()}


def verify(root):
    start = time.monotonic()
    protocol, result = (b.r.read(root / f"{name}.json") for name in ("protocol", "result"))
    parent = root.parent
    tape, _, _ = trial.m.load(parent/"patch_utility_bank_20261106_v1",
        parent/"patch_utility_target_views_20261106_v1",
        parent/"patch_utility_selection_20261106_v1"/"result.json", Hashes())
    by = {(v, q.sid): q for v,t in tape.items() for q in t.shots}
    pair_notes, unique = check_pairs(protocol, tape, by)
    ref = b.r.read(parent/"patch_sustained_margin_20261106_v1"/"result.json")
    late, checked, nonzero = {}, 0, 0
    for view, rows in result["rows"].items():
        if [(x["shot"], x["role"]) for x in rows] != [(x["shot"], x["role"]) for x in ref["rows"][view]]:
            raise ValueError("held diagnostic cohort has changed")
        late[view] = []
        for row in rows:
            q = by[view, row["shot"]]
            with np.load(root/view/f"{q.sid}.npz", allow_pickle=False) as f:
                t, z, c = (f[k] for k in ("time", "logits", "clock"))
            flag, down, clock = outcome_and_clock(b.r.io23.path_for("J-TEXT", q.sid), len(t))
            if (not np.array_equal(clock,t) or not np.array_equal(q.native,z[0])
                    or (flag, down) != (row["disruptive"], row["down"])):
                raise ValueError("held HDF5 truth, time or frozen trace mismatch")
            possible = np.zeros(len(t), bool)
            for m in c:
                possible[m+9:m+33] = True
            if not np.array_equal(z[:, ~possible], np.broadcast_to(z[0, ~possible], (len(z),sum(~possible)))):
                raise ValueError("a trace wrote outside the causal future suffix")
            for arm, trace in zip(protocol["arms"], z, strict=True):
                if alarm(t, trace, flag, down, protocol["tau"]) != row["alarms"][arm]:
                    raise ValueError("archived first alarm differs from independent scorer")
                checked += 1
            nonzero += int((np.abs(z[1]-z[2])>1e-5).any())
            if row["role"] == "late_disruption":
                good = possible & (np.round((down-t)*1000., 9) > 10.)
                margins = (z[:,good].max(1).astype(float)-protocol["tau"]).tolist() if good.any() else None
                late[view].append(dict(shot=q.sid, margins=margins))
    b.r.write_new(root/"verification.json", dict(status="HELD_PAIR_AND_RAW_ALARM_RECOUNT_MATCH",
        pairs=pair_notes, unique=unique, checked_curves=checked, views_changed_by_pairing=nonzero,
        late_peak_margins=late, duration=time.monotonic()-start))
    print("curves", checked, "pair directions differ", nonzero, "unique teachers",
          {k: len(v) for k,v in unique.items()}, "fold matches", pair_notes, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    verify(cli.parse_args().root)
