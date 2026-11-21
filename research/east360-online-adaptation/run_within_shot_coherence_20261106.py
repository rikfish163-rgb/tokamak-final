"""Run and independently recount a fixed, signal-only perturbation benchmark."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import coherence_injection_20261106 as design
from coherence_token_write_20261106 import offsets, scalar_control, shifted_logits
from fit_layer0_arrival_motion_20261106 import past_arrays
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import parents, raw_for
from trajectory_replay_20261105 import baseline_for, reference
from verify_confirmation_raw_replay_20261105 import Hashes
import within_shot_coherence_20261106 as core


PLAN = r.ROOT / "WITHIN_SHOT_DIAGNOSTIC_COHERENCE_PLAN_20261106.md"
ARMS, CASES = design.ARMS, design.CASES
N, BLOCK, ONSET, OFF = design.N, design.BLOCK, design.ONSET, design.OFF
SINGLE = tuple(k for k, _ in design.GROUPS)
STATES = ("single", "coherent", "stale", "wrong_family")
REGIONS = ("mature", "nonanchor", "pre", "recovery")
LIMIT, BATCH = 60, 256


def score_again(times, clean, curves, case):
    """Vectorized independent definitions (test against the separate scorer)."""
    if set(curves) != set(ARMS) or case not in CASES:
        raise ValueError("complete paired curves and a declared case required")
    q = np.stack([curves[k] for k in ARMS], axis=1).astype(np.float64)
    clean = np.asarray(clean)
    if (
        q.shape != (len(times), len(ARMS))
        or clean.shape != (len(times),)
        or not np.isfinite(q).all()
        or not np.isfinite(clean).all()
        or not np.array_equal(times, np.arange(N))
    ):
        raise ValueError("full aligned sealed trace required")
    mask = (times >= ONSET + 2 * BLOCK) & (times < OFF)
    parts = (mask, mask & (times % BLOCK != 0), times < ONSET, times >= OFF + BLOCK)
    static = q[:, 0]
    score = {}
    for name, active in zip(REGIONS, parts, strict=True):
        target = clean if case in SINGLE and name in ("mature", "nonanchor") else static
        score[name] = dict(
            zip(
                ARMS,
                np.abs(q[active] - target[active, None]).mean(0).tolist(),
                strict=True,
            )
        )
    score["counts"] = dict(zip(REGIONS, (int(x.sum()) for x in parts), strict=True))
    return score


def by_case(rows):
    out = {}
    for case in CASES:
        rows_case = [a["cases"][case] for a in rows if a["cases"][case] is not None]
        out[case] = dict(
            supported=len(rows_case),
            excluded=len(rows) - len(rows_case),
            regions={
                k: {arm: float(np.mean([a[k][arm] for a in rows_case])) for arm in ARMS}
                for k in REGIONS
            }
            if rows_case
            else None,
        )
    return out


def single_rows(rows):
    pooled = []
    for row in rows:
        available = [
            row["cases"][k]["nonanchor"] for k in SINGLE if row["cases"][k] is not None
        ]
        if available:
            pooled.append(
                (row["shot"], [float(np.mean([a[k] for a in available])) for k in ARMS])
            )
    return pooled


def gate(rows):
    primary = [r for r in rows if 21 <= r["rank"] <= LIMIT]
    cases, pairs = by_case(primary), single_rows(primary)
    if not primary or not pairs or any(not a["supported"] for a in cases.values()):
        return dict(passed=False, reason="missing primary support")
    x = np.array([v for _, v in pairs])
    m = dict(zip(ARMS, x.mean(0).tolist(), strict=True))
    actual, comparator = ARMS.index("coherent"), ARMS.index("scalar")
    wins = int((x[:, actual] < x[:, comparator]).sum())
    preserve = {}
    for name in ("common", "staggered", "clean"):
        a = cases[name]["regions"]["mature"]
        limit = 0.005 if name == "clean" else max(0.005, a["single"] / 2)
        preserve[name] = dict(
            error=a["coherent"], limit=limit, passed=a["coherent"] <= limit
        )
    conditions = dict(
        at_least_ten_percent=m["static"] > 0 and m["coherent"] <= 0.90 * m["static"],
        better_than_controls=all(
            m["coherent"] < m[k] for k in ("single", "stale", "wrong_family", "scalar")
        ),
        majority_better_than_scalar=2 * wins > len(pairs),
        preserves=all(v["passed"] for v in preserve.values()),
    )
    delta = x[:, actual] - x[:, comparator]
    samples = (
        np.random.default_rng(131)
        .choice(delta, size=(10000, len(delta)), replace=True)
        .mean(1)
    )
    return dict(
        passed=all(conditions.values()),
        conditions=conditions,
        mean_single_nonanchor=m,
        paired_shots=len(pairs),
        better_than_scalar=wins,
        preserve=preserve,
        coherent_minus_scalar_descriptive95=np.quantile(
            samples, [0.025, 0.975]
        ).tolist(),
    )


def normalized(base, v, a):
    center, scale = (
        base.center.detach().cpu().numpy(),
        base.scale.detach().cpu().numpy(),
    )
    z = np.arcsinh((np.where(a, v, center) - center) / scale)
    if z.shape != a.shape or not np.isfinite(z).all():
        raise ValueError("finite source-coordinate signal required")
    return z


@torch.no_grad()
def predict_points(base, v, mask, t, disturbance, action=None, *, oracle=False):
    device = base.center.device
    pieces = []
    for i in range(0, len(t), BATCH):
        ix = t[i : i + BATCH]
        x, a = past_arrays(v, mask, ix)
        d = offsets(ix, disturbance, action, oracle=oracle)
        logits = shifted_logits(
            base,
            torch.as_tensor(x, device=device),
            torch.as_tensor(a, device=device),
            torch.as_tensor(d, device=device),
        )
        pieces.append(logits.cpu().numpy())
    return np.concatenate(pieces) if pieces else np.empty(0, np.float32)


def curves_for(base, v, a, d, states, native):
    if len(v) != len(native) or d.shape != a.shape or len(v) != len(d):
        raise ValueError("unaligned scenario and native trace")
    event = np.flatnonzero(d.any(axis=1))
    if len(event):
        # Outside this support the original causal window contains no disturbance.
        exposed = np.arange(event[0], min(len(d), event[-1] + 128), dtype=np.int64)
        static = native.copy()
        static[exposed] = predict_points(base, v, a, exposed, d)
    else:
        static = native.copy()
    curves = {"static": static}
    for name in STATES:
        q = states[name]
        t = np.flatnonzero(q.any(axis=1))
        p = static.copy()
        p[t] = predict_points(base, v, a, t, d, q)
        curves[name] = p
    curves["scalar"] = scalar_control(static, curves["coherent"])
    t = (
        np.arange(event[0], min(len(d), event[-1] + BLOCK), dtype=np.int64)
        if len(event)
        else np.empty(0, dtype=np.int64)
    )
    p = static.copy()
    p[t] = predict_points(base, v, a, t, d, oracle=True)
    curves["oracle"] = p
    return curves


def prefix_witness(base, v, a, d, states, curves, native, cut=380):
    short = core.observer(normalized(base, v[:cut], a[:cut]) + d[:cut], a[:cut])
    if not all(np.array_equal(states[k][:cut], short[k]) for k in STATES):
        raise ValueError("unarrived suffix changed previous controller state")
    q = curves_for(base, v[:cut], a[:cut], d[:cut], short, native[:cut])
    max_abs = max(float(np.max(np.abs(curves[k][:cut] - q[k]))) for k in ARMS)
    if max_abs > 2e-5:
        raise ValueError("unarrived suffix changed previous model output")
    return max_abs


def produce(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    if core.GROUPS != design.GROUPS or core.BLOCK != BLOCK:
        raise RuntimeError("declared experiment groups differ")
    r.configure()
    order, parent_cfg, _, base, source, mapping, map_path, _ = parents()
    _, _, original_done = reference()
    h = Hashes()
    files = [
        Path(__file__),
        PLAN,
        Path(design.__file__),
        Path(core.__file__),
        Path(shifted_logits.__code__.co_filename),
        Path(r.prepare_actual.__code__.co_filename),
        Path(past_arrays.__code__.co_filename),
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(parents.__code__.co_filename),
        r.DEFAULT_PARENT,
        map_path,
    ]
    binding = {str(p.resolve()): h(p) for p in files}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    r.write_new(
        output / "protocol.json",
        dict(
            scope="DEVELOPMENT_SIGNAL_ONLY_KNOWN_PERTURBATION",
            order=order[:LIMIT],
            arms=ARMS,
            cases=CASES,
            input_sha256=binding,
            source_state_sha256=source,
            threshold=r.THRESHOLD,
            start=ONSET,
            stop=OFF,
            block=BLOCK,
            samples=N,
            amplitude=design.AMP,
            groups=design.GROUPS,
            mapping=str(map_path),
            labels_read=0,
            alarm_scoring=False,
        ),
    )
    inventory, witness = [], []
    t = np.arange(N, dtype=np.int64)
    with patch.object(
        r.io23, "read_outcome", side_effect=RuntimeError("no outcomes in this run")
    ):
        for i, sid in enumerate(order[:LIMIT], start=1):
            raw = raw_for(sid, parent_cfg["raw_sha256"])
            info = dict(
                rank=i,
                shot=sid,
                n=len(raw["values"]),
                raw_sha256=parent_cfg["raw_sha256"][str(sid)],
            )
            if len(raw["values"]) < N:
                inventory.append(dict(info, reason="short", cases=[]))
                continue
            prepared = r.prepare_actual(raw, 21, mapping)
            v, a = prepared["values"][:N], prepared["available"][:N]
            z = normalized(base, v, a)
            zero = np.zeros_like(v, dtype=np.float32)
            native = predict_points(base, v, a, t, zero)
            if i > 20:
                time_ref, prior = baseline_for(i, sid, original_done)
                e = torch.from_numpy(native).sigmoid().numpy()
                if not np.array_equal(
                    raw["times"][:N], time_ref[:N]
                ) or not np.allclose(e, prior[:N], atol=2e-6, rtol=0.0):
                    raise ValueError(
                        "clean forward does not reproduce sealed native baseline"
                    )
            arrays = dict(times=t, native=native)
            names = []
            for case in CASES:
                d = design.injection(a, sid, case)
                if d is None:
                    continue
                states = core.observer(z + d, a)
                curves = curves_for(base, v, a, d, states, native)
                for k, q in curves.items():
                    arrays[f"{case}__{k}"] = q
                arrays[f"{case}__state"] = np.stack(
                    [states[k][::BLOCK] for k in STATES]
                )
                arrays[f"{case}__support"] = states["completed_supported"]
                arrays[f"{case}__candidate"] = states["candidate_flags"]
                names.append(case)
                if case in SINGLE and i in (1, 30, 60):
                    e = prefix_witness(base, v, a, d, states, curves, native)
                    witness.append(dict(shot=sid, case=case, n=380, max_abs=e))
            path = output / "shots" / f"{i:03}_{sid}.npz"
            r.save_arrays(path, **arrays)
            inventory.append(
                dict(
                    info,
                    cases=names,
                    path=str(path.relative_to(output)),
                    sha256=h(path),
                )
            )
            if i % 10 == 0:
                print(f"sealed synthetic diagnostic {i}/{LIMIT}", flush=True)
    if r.state_digest(base) != source or base.embedding._forward_hooks:
        raise ValueError("source checkpoint state or runtime hooks changed")
    h.check()
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_COHERENCE_SYNTHETIC_PREDICTIONS",
            protocol_sha256=h(output / "protocol.json"),
            shots=inventory,
            prefix=witness,
            seconds=time.monotonic() - start,
        ),
    )


def recount(root, output):
    start, h = time.monotonic(), Hashes()
    cfg, done = [r.read(root / n) for n in ("protocol.json", "completed.json")]
    if (
        done["status"] != "COMPLETE_COHERENCE_SYNTHETIC_PREDICTIONS"
        or h(root / "protocol.json") != done["protocol_sha256"]
        or not all(h(p) == d for p, d in cfg["input_sha256"].items())
        or [x["shot"] for x in done["shots"]] != cfg["order"]
        or cfg["labels_read"] != 0
        or cfg["alarm_scoring"] is not False
    ):
        raise ValueError("invalid experiment seal")
    rows = []
    for item in done["shots"]:
        row = dict(
            rank=item["rank"],
            shot=item["shot"],
            cases=dict.fromkeys(CASES),
            activation={},
        )
        rows.append(row)
        if "path" not in item:
            if item["n"] >= N:
                raise ValueError("missing archive for eligible shot")
            continue
        path = root / item["path"]
        if h(path) != item["sha256"]:
            raise ValueError("changed sealed shot")
        with np.load(path, allow_pickle=False) as a:
            for k in item["cases"]:
                curves = {arm: a[f"{k}__{arm}"] for arm in ARMS}
                if not np.array_equal(
                    curves["scalar"][::BLOCK], curves["coherent"][::BLOCK]
                ):
                    raise ValueError("wrong matched-anchor scalar")
                x, y = (
                    f(a["times"], a["native"], curves, k)
                    for f in (score_again, design.score_case)
                )
                if x["counts"] != y["counts"] or any(
                    not np.allclose(
                        list(x[m].values()), list(y[m].values()), atol=1e-12, rtol=1e-12
                    )
                    for m in REGIONS
                ):
                    raise ValueError("independent paired scoring disagreement")
                state = a[f"{k}__state"]
                mature = slice((ONSET + 2 * BLOCK) // BLOCK, OFF // BLOCK)
                norms = np.linalg.norm(state, axis=2)
                if not np.allclose(norms[1], norms[3], rtol=2e-6, atol=1e-8):
                    raise ValueError("wrong-family action is not norm-matched")
                row["cases"][k] = x
                row["activation"][k] = dict(
                    blocks=dict(
                        zip(STATES, (norms[:, mature] > 0).sum(1).tolist(), strict=True)
                    ),
                    supported_blocks=int(a[f"{k}__support"].sum()),
                    different_peer_acceptance=int(
                        ((norms[1, mature] > 0) != (norms[2, mature] > 0)).sum()
                    ),
                )
    ans = dict(
        status="RECOUNT_COHERENCE_SYNTHETIC",
        experiment_sha256=h(root / "completed.json"),
        labels_read=0,
        alarm_scoring=False,
        gate=gate(rows),
        rows=rows,
        groups={
            name: by_case(q)
            for name, q in (("cold20", rows[:20]), ("next40", rows[20:]))
        },
        seconds=time.monotonic() - start,
    )
    h.check()
    r.write_new(output, ans)
    print(ans["gate"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("predict").add_argument("output", type=Path)
    a = sub.add_parser("recount")
    a.add_argument("root", type=Path)
    a.add_argument("output", type=Path)
    args = p.parse_args()
    if args.mode == "predict":
        produce(args.output)
    else:
        recount(args.root, args.output)
