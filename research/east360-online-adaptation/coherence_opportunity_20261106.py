"""Locate possible correction times relative to existing sealed native alarms."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import raw_for
from run_within_shot_coherence_20261106 import STATES, normalized
from trajectory_replay_20261105 import SOURCE, baseline_for, reference
from verify_confirmation_raw_replay_20261105 import Hashes, score
from within_shot_coherence_20261106 import BLOCK, WARM, observer


PLAN = Path(__file__).with_name("COHERENCE_UNPERTURBED_OPPORTUNITY_PLAN_20261106.md")


def ready_blocks(completed, n):
    """Necessary data support: the most recent warmup+one blocks all completed."""
    completed = np.asarray(completed)
    if completed.shape != (n // BLOCK,) or completed.dtype != bool:
        raise ValueError("complete supported-block flags required")
    total = (n + BLOCK - 1) // BLOCK
    out = np.zeros(total, dtype=bool)
    count = 0
    for i in range(1, total):
        count = count + 1 if completed[i - 1] else 0
        out[i] = count >= WARM + 1
    return out


def opportunity(t, p, disrupted, down, b, ready, threshold):
    t, p, b, ready = map(np.asarray, (t, p, b, ready))
    n = len(t)
    nb = (n + BLOCK - 1) // BLOCK
    if (
        not n
        or p.shape != (n,)
        or t.shape != p.shape
        or not np.isfinite(t).all()
        or not np.isfinite(p).all()
        or (np.diff(t) <= 0).any()
        or (p < 0).any()
        or (p > 1).any()
        or b.shape != (nb, len(STATES))
        or ready.shape != (nb,)
        or b.dtype != bool
        or ready.dtype != bool
        or (disrupted and (down is None or not np.isfinite(down)))
    ):
        raise ValueError(
            "aligned time, probability, causal proposal and outcome required"
        )
    base = score(t, p, dict(shot=0, disruptive=disrupted, down=down), threshold)
    i = base["first_index"]
    active = b[np.arange(n) // BLOCK]
    report = {k: base[k] for k in ("status", "first_index", "lead_ms")}

    def flags(a):
        return dict(zip(STATES, a.tolist(), strict=True))

    if i is not None:
        report["first_active"] = flags(active[i])
        report["first_has_nine_supported_blocks"] = bool(ready[i // BLOCK])
    else:
        report["first_active"] = None
        report["first_has_nine_supported_blocks"] = None
    report["ever_active"] = flags(active.any(0))
    report["active_points"] = dict(zip(STATES, active.sum(0).tolist(), strict=True))
    if disrupted:
        lead = np.round((down - t) * 1000, decimals=9)
        strict = lead > 30.0
        reasonable = strict & (lead <= 3000.0)
        report["strict_any"] = flags(active[strict].any(0))
        report["reasonable_any"] = flags(active[reasonable].any(0))
    return report


def produce(out):
    if out.exists():
        raise FileExistsError(out)
    start, h = time.monotonic(), Hashes()
    order, cfg, baseline = reference()
    mpath = r.COLD / "input_map.json"
    mapping = r.read(mpath)
    if (
        mapping["calibration_ids"] != order[:20]
        or h(mpath) != cfg["input_sha256"][str(mpath)]
    ):
        raise ValueError("original fixed input mapping required")
    base, meta = r.load_parent(r.DEFAULT_PARENT)
    if (
        r.state_digest(base) != cfg["source_state_sha256"]
        or meta["threshold"] != cfg["threshold"]
        or cfg["threshold"] != r.THRESHOLD
    ):
        raise ValueError("unchanged source required")
    inputs = {
        str(p.resolve()): h(p)
        for p in (
            Path(__file__),
            PLAN,
            Path(observer.__code__.co_filename),
            Path(normalized.__code__.co_filename),
            Path(r.prepare_actual.__code__.co_filename),
            Path(raw_for.__code__.co_filename),
            Path(r.simple_data.__file__),
            Path(r.io23.__file__),
            SOURCE / "protocol.json",
            SOURCE / "completed.json",
            mpath,
            r.DEFAULT_PARENT,
        )
    }
    out.mkdir(parents=True)
    observed, avail, index, rows = [], [], [0], []
    with patch.object(
        r.io23, "read_outcome", side_effect=RuntimeError("labels forbidden before seal")
    ):
        for rank, sid in enumerate(order, start=1):
            raw = raw_for(sid, cfg["raw_sha256"])
            t, p = baseline_for(rank, sid, baseline)
            if not np.array_equal(t, raw["times"]) or t.shape != p.shape:
                raise ValueError(
                    "existing static prediction clock differs from raw data"
                )
            shot = r.prepare_actual(raw, rank, mapping)
            v, a = shot["values"], shot["available"]
            states = observer(normalized(base, v, a), a)
            b = np.stack([states[k][::BLOCK].any(1) for k in STATES], axis=1)
            q = ready_blocks(states["completed_supported"], len(t))
            if rank <= 20:
                b.fill(False)
                q.fill(False)
            observed.append(b)
            avail.append(q)
            index.append(index[-1] + len(b))
            rows.append(
                dict(rank=rank, shot=sid, n=len(t), sha256=cfg["raw_sha256"][str(sid)])
            )
            if rank % 60 == 0:
                print(f"sealed original-data support {rank}/360", flush=True)
    h.check()
    r.save_arrays(
        out / "support.npz",
        active=np.concatenate(observed),
        ready=np.concatenate(avail),
        offset=np.asarray(index),
    )
    r.write_new(
        out / "completed.json",
        dict(
            status="SEALED_COHERENCE_OPPORTUNITY_NO_OUTCOMES",
            shots=rows,
            states=STATES,
            labels_read=0,
            new_alarm_predictions=0,
            source_state_sha256=cfg["source_state_sha256"],
            threshold=r.THRESHOLD,
            input_sha256=inputs,
            support_sha256=h(out / "support.npz"),
            seconds=time.monotonic() - start,
        ),
    )


def subset_summary(rows):
    groups = {}
    for name in (
        "tp",
        "fp",
        "tn",
        "miss",
        "tardy",
        "miss_or_tardy",
        "reasonable_tp",
        "early_tp",
    ):
        q = [
            v
            for v in rows
            if v["status"] == name
            or (name == "miss_or_tardy" and v["status"] in ("miss", "tardy"))
            or (
                name == "reasonable_tp" and v["status"] == "tp" and v["lead_ms"] <= 3000
            )
            or (name == "early_tp" and v["status"] == "tp" and v["lead_ms"] > 3000)
        ]
        stats = dict(
            n=len(q),
            first_has_nine_supported_blocks=sum(
                r["first_has_nine_supported_blocks"] is True for r in q
            ),
        )
        for key in ("first_active", "ever_active", "strict_any", "reasonable_any"):
            stats[key] = {
                k: sum(bool(r.get(key) and r[key][k]) for r in q) for k in STATES
            }
        groups[name] = stats
    return groups


def inspect(root, output):
    start, h = time.monotonic(), Hashes()
    done = r.read(root / "completed.json")
    order, cfg, baseline = reference()
    if (
        done["status"] != "SEALED_COHERENCE_OPPORTUNITY_NO_OUTCOMES"
        or done["labels_read"]
        or done["new_alarm_predictions"]
        or not all(h(p) == digest for p, digest in done["input_sha256"].items())
        or done["support_sha256"] != h(root / "support.npz")
        or [x["shot"] for x in done["shots"]] != order
        or done["threshold"] != cfg["threshold"]
        or tuple(done["states"]) != STATES
    ):
        raise ValueError(
            "sealed causal coverage and fixed references required before revealing truth"
        )
    with np.load(root / "support.npz", allow_pickle=False) as a:
        b, ready, o = a["active"], a["ready"], a["offset"]
    if (
        o.shape != (len(order) + 1,)
        or o.dtype.kind not in "iu"
        or o[0] != 0
        or o[-1] != len(b)
        or len(ready) != len(b)
        or (np.diff(o) <= 0).any()
    ):
        raise ValueError("invalid sealed support inventory")
    rows = []
    for k, rec in enumerate(done["shots"]):
        rank, sid = k + 1, rec["shot"]
        if rank != rec["rank"] or h(r.io23.path_for("EAST", sid)) != rec["sha256"]:
            raise ValueError("original data changed before outcome reveal")
        t, p = baseline_for(rank, sid, baseline)
        if len(t) != rec["n"]:
            raise ValueError("original score clock differs after seal")
        flag, down = r.io23.read_outcome("EAST", sid)
        j = slice(o[k], o[k + 1])
        v = opportunity(t, p, flag, down, b[j], ready[j], cfg["threshold"])
        rows.append(dict(shot=sid, rank=rank, **v))
    full = subset_summary(rows)
    if (full["tp"]["n"], full["fp"]["n"], full["reasonable_tp"]["n"]) != (116, 29, 100):
        raise ValueError("fixed static outcomes or timing did not reproduce")
    ans = dict(
        status="RECOUNT_COHERENCE_OPPORTUNITY",
        new_alarm_predictions=0,
        seal_sha256=h(root / "completed.json"),
        full360=full,
        after20=subset_summary(rows[20:]),
        per_shot=rows,
        seconds=time.monotonic() - start,
    )
    h.check()
    r.write_new(output, ans)
    for k in ("fp", "tp", "miss_or_tardy"):
        print(k, full[k], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="mode", required=True)
    s.add_parser("seal").add_argument("output", type=Path)
    q = s.add_parser("inspect")
    q.add_argument("root", type=Path)
    q.add_argument("output", type=Path)
    a = p.parse_args()
    if a.mode == "seal":
        produce(a.output)
    else:
        inspect(a.root, a.output)
