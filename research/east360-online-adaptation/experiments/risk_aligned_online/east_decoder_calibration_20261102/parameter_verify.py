"""Independent CPU scoring/receipt checks; explicit optional four-shot replay."""

import argparse
import math
from pathlib import Path
import time

import numpy as np

from .. import io23
from .audit360 import (
    classify_curve,
    finite,
    paired_flips,
    read,
    recount,
    require,
    sha,
    vector,
)
from .functional_audit import (
    check_trial_steps,
    empty_totals,
    verify_snapshot,
    valid_hash,
)

REFERENCE = io23.ROOT / "probe_outputs/east_functional_protection360_v1"
ARMS = ("static", "real", "reverse", "hold")


def gate(scores):
    base, candidate = scores["static"], scores["real"]
    net = lambda s: s["tp"] - s["fp"]  # noqa: E731
    result = dict(
        operational=candidate["fp"] <= 36
        and candidate["tp"] > base["tp"]
        and net(candidate) > net(base),
        controls={a: net(candidate) > net(scores[a]) for a in ("reverse", "hold")},
    )
    result["passed"] = result["operational"] and all(result["controls"].values())
    return result


def check_arm(events, length, warmup):
    ends = [] if warmup else list(range(159, length, 128))
    require([r["maturity"] for r in events] == ends, "update clock")
    previous = np.zeros(256, np.float32)
    stats = empty_totals()
    stats["events"] = len(events)
    for end, r in zip(ends, events, strict=True):
        require(
            r["origin"] == end - 25
            and r["sealed_through"] == end
            and r["first_affected"] == end + 1,
            "causal clock",
        )
        require(
            all(
                valid_hash(r[k])
                for k in ("forecast_sha256", "mask_sha256", "target_sha256")
            ),
            "receipt hashes",
        )
        gradient, state = vector(r["gradient256"]), vector(r["state256_after"])
        gn, sn = finite(r["gradient_norm"]), finite(r["step_norm"])
        require(gn >= 0 and sn >= 0, "nonnegative norms")
        require(
            math.isclose(
                gn,
                float(np.linalg.norm(gradient.astype(np.float64))),
                rel_tol=1e-9,
                abs_tol=1e-14,
            ),
            "gradient norm",
        )
        require(
            math.isclose(
                sn,
                float(np.linalg.norm((state - previous).astype(np.float64))),
                rel_tol=1e-9,
                abs_tol=1e-14,
            ),
            "step norm",
        )
        n, accepted = r["observed"], r["accepted"]
        require(
            type(n) is int and 0 <= n <= 550 and type(accepted) is bool, "receipt types"
        )
        if n:
            before, proposal = finite(r["loss_before"]), finite(r["loss_proposal"])
            require(
                accepted == (proposal < before)
                and r["reason"] == ("improved" if accepted else "not_improved"),
                "strict acceptance",
            )
            check_trial_steps(r, accepted, before, stats)
        else:
            require(
                not accepted
                and r["reason"] == "no_target"
                and r["loss_before"] is None
                and r["loss_proposal"] is None
                and gn == sn == 0
                and not gradient.any()
                and r["tested_steps"] == []
                and r["chosen_fraction"] is None,
                "no target",
            )
            stats["no_target"] += 1
        require(accepted or np.array_equal(state, previous), "rejected state changed")
        previous = state
        stats["accepted"] += int(accepted)
        stats["nonzero_gradients"] += int(gn > 0)
        stats["observed_scalars"] += n
        stats["max_gradient_norm"] = max(stats["max_gradient_norm"], gn)
        stats["max_step_norm"] = max(stats["max_step_norm"], sn)
        stats["max_state_norm"] = max(
            stats["max_state_norm"], float(np.linalg.norm(state.astype(np.float64)))
        )
    return stats


def verify(root):
    root = Path(root)
    done = read(root / "completed.json")
    require(done["status"] == "COMPLETED", "COMPLETED required before outcome reads")
    p = read(root / "protocol.json")
    require(
        sha(root / "protocol.json") == done["protocol_sha256"]
        and sha(root / "per_shot.json") == done["per_shot_sha256"],
        "completion bindings",
    )
    require(
        sha(REFERENCE / "completed.json") == p["reference_sha256"], "reference binding"
    )
    for name in ("audit_cpu.json", "verification_neural.json"):
        require(
            read(REFERENCE / name)["status"] == "PASS", "verified reference required"
        )
    ref = read(REFERENCE / "protocol.json")
    require(
        read(REFERENCE / "audit_cpu.json")["evidence"]["completed_sha256"]
        == read(REFERENCE / "verification_neural.json")["completed_sha256"]
        == p["reference_sha256"],
        "reference audit bindings",
    )
    ids = [r["shot"] for r in p["order"]]
    require(
        len(ids) == len(set(ids)) == 360
        and p["order"] == ref["order"]
        and p["arms"] == list(ARMS),
        "cohort",
    )
    require(
        p["scope"] == "MATCHED_PARAMETER_BACKTRACK_DIAGNOSTIC"
        and p["warmup20_cost_included"] is True
        and p["head_training_steps"] == p["suffix_supervised_updates"] == 0
        and p["independent_confirmation"] is False,
        "scope",
    )
    require(
        all(
            p[k] == ref[k]
            for k in (
                "threshold",
                "source_sha256",
                "cold_sha256",
                "input_map_sha256",
                "heads_sha256",
            )
        ),
        "fixed parent",
    )
    verify_snapshot(root / "code_snapshot.zip", p["bound_code_sha256"], io23.ROOT)
    require(
        all(sha(Path(path)) == d for path, d in p["bound_code_sha256"].items()),
        "live code",
    )
    expected = {f"{i:04d}_{sid}" for i, sid in enumerate(ids, 1)}
    for suffix in (".npz", "_updates.json", "_seal.json"):
        require(
            {f.name for f in (root / "shots").glob("*" + suffix)}
            == {s + suffix for s in expected},
            "shot inventory",
        )
    rows = {a: [] for a in ARMS}
    totals = {a: empty_totals() for a in ARMS[1:]}
    for i, sid in enumerate(ids, 1):
        stem = f"{i:04d}_{sid}"
        curve, records, sealpath = (
            root / "shots" / (stem + s) for s in (".npz", "_updates.json", "_seal.json")
        )
        seal = read(sealpath)
        require(
            seal["index"] == i
            and seal["shot"] == sid
            and seal["reused_warmup"] == (i <= 20)
            and seal["current_outcome_used_in_prediction"] is False
            and sha(curve) == seal["curve_sha256"]
            and sha(records) == seal["updates_sha256"],
            "seal",
        )
        refcurve = REFERENCE / "shots" / (stem + ".npz")
        refseal = read(refcurve.with_name(stem + "_seal.json"))
        require(sha(refcurve) == refseal["curve_sha256"], "reference curve seal")
        with (
            np.load(curve, allow_pickle=False) as saved,
            np.load(refcurve, allow_pickle=False) as old,
        ):
            require(set(saved.files) == {"times", *ARMS}, "curve keys")
            times = saved["times"]
            curves = {a: saved[a] for a in ARMS}
            require(
                times.ndim == 1
                and np.isfinite(times).all()
                and np.all(np.diff(times) > 0)
                and np.array_equal(times, old["times"])
                and np.array_equal(curves["static"], old["static"])
                and np.array_equal(curves["real"], old["parameter_real"]),
                "reference values",
            )
            for a, v in curves.items():
                require(
                    v.shape == times.shape
                    and np.isfinite(v).all()
                    and ((0 <= v) & (v <= 1)).all(),
                    "curve range",
                )
                require(
                    np.array_equal(v[:160], curves["static"][:160]), "sealed prefix"
                )
                if i <= 20:
                    require(np.array_equal(v, curves["static"]), "startup cost")
        updates = read(records)
        require(set(updates) == set(ARMS[1:]), "update arms")
        old_records = read(REFERENCE / "shots" / (stem + "_updates.json"))
        require(
            updates["real"] == old_records["parameter_real"],
            "real gradient/state/receipt reference",
        )
        for a in ARMS[1:]:
            stats = check_arm(updates[a], len(times), i <= 20)
            for k, v in stats.items():
                if isinstance(v, dict):
                    for q, n in v.items():
                        totals[a][k][q] += n
                elif k.startswith("max_"):
                    totals[a][k] = max(totals[a][k], v)
                else:
                    totals[a][k] += v
        for group in zip(*(updates[a] for a in ARMS[1:]), strict=True):
            require(
                len({r["mask_sha256"] for r in group})
                == len({r["observed"] for r in group})
                == 1,
                "shared mask",
            )
        flag, down = io23.read_outcome("EAST", sid)
        for a in ARMS:
            rows[a].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    **classify_curve(times, curves[a], flag, down, p["threshold"]),
                )
            )
    score = {
        a: {
            part: recount(r)
            for part, r in (("full360", v), ("warmup20", v[:20]), ("suffix340", v[20:]))
        }
        for a, v in rows.items()
    }
    flip = {
        a: paired_flips(rows["static"], v) for a, v in rows.items() if a != "static"
    }
    require(
        rows == read(root / "per_shot.json")
        and score == done["metrics"]
        and flip == done["flips"]
        and gate({a: score[a]["full360"] for a in ARMS}) == done["gate"],
        "scoring report",
    )
    require(
        score["static"]["full360"]["n_disrupt"]
        == score["static"]["full360"]["n_safe"]
        == 180,
        "180D180S",
    )
    result = dict(
        status="PASS",
        scope="RECORDED_DATA_CONSISTENCY_ONLY",
        metrics=score,
        flips=flip,
        update_statistics=totals,
        diagnostic_gate=done["gate"],
        full_neural_replay=False,
        hidden_rms_verified=False,
        raw_signal_reads=0,
        real_gradients_states_receipts_equal_reference=True,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        independent_confirmation=False,
    )
    write_report(root / "audit_cpu.json", result)
    print(dict(status="PASS", gate=done["gate"]), flush=True)
    return result


def write_report(path, result):
    import json

    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def neural(root):
    import torch
    from unittest.mock import patch
    from .. import simple_data
    from ..causal_input import apply_input_contract
    from ..lora_stream_20261102.core import direct
    from ..warmstart_alignment_20261022.alignment import transform
    from .functional_stream import load
    from .neural_replay import IDS, state_at
    from .parameter_controls import predict

    root = Path(root)
    done = read(root / "completed.json")
    require(done["status"] == "COMPLETED", "completed neural replay required")
    model, head, basis, _, previous, mapping, hashes = load()
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    frozen_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    ids = [r["shot"] for r in previous["order"]]
    start = time.monotonic()
    checks = []
    for sid in IDS:
        i = ids.index(sid) + 1
        require(sha(io23.path_for("EAST", sid)) == hashes[str(sid)], "raw fixed replay")
        with patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("outcomes forbidden in replay"),
        ):
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
            curves, records = predict(model, signal, basis, head)
        path = root / "shots" / f"{i:04d}_{sid}.npz"
        require(
            records == read(path.with_name(path.stem + "_updates.json")),
            "neural receipts",
        )
        with np.load(path, allow_pickle=False) as saved:
            for a in ARMS:
                require(np.array_equal(curves[a], saved[a]), "full curve neural replay")
            for q in (0, 160, len(raw["times"]) - 1):
                lo, hi = q // 32 * 32, min(q // 32 * 32 + 32, len(raw["times"]))
                for a in ARMS:
                    state = torch.tensor(
                        state_at([] if a == "static" else records[a], q), device="cuda"
                    )
                    with torch.no_grad():
                        actual = (
                            direct(
                                model,
                                {k: v[:hi] for k, v in signal.items()},
                                np.arange(lo, hi),
                                basis,
                                state,
                            )["logit"]
                            .sigmoid()
                            .cpu()
                            .numpy()
                        )
                    require(
                        np.array_equal(actual, curves[a][lo:hi]), "actual firstFFN hook"
                    )
        checks.append(dict(shot=sid, index=i, events=len(records["real"])))
    require(
        all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in frozen.items())
        and all(
            torch.equal(v, head.state_dict()[k].cpu()) for k, v in frozen_head.items()
        ),
        "frozen replay model",
    )
    result = dict(
        status="PASS",
        checks=checks,
        curves_replayed=16,
        actual_ffn_batches=48,
        outcome_reads=0,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        seconds=time.monotonic() - start,
        scope="same deployed replay, not independent algorithm or training reproduction",
    )
    write_report(root / "verification_neural.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--neural", action="store_true")
    args = parser.parse_args()
    neural(args.root) if args.neural else verify(args.root)
