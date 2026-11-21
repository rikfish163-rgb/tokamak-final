"""Independent saved-array recount for the three timed internal-update screens."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROBE = ROOT / "probe_outputs"
TIMED = PROBE / "timed_token_norm_20261105_v1"
REACH = PROBE / "local_affine_reach_20261105_v1"
ARRIVAL = PROBE / "same_site_arrival_20261105_v2"
STATIC = PROBE / "qualified_source_static360_20261104_v1"
FRESH = PROBE / "hdf5_outcome_authority_20261105_v1"
OUT = PROBE / "local_affine_arrival_verification_20261105_v1.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(folder):
    done = read(folder / "completed.json")
    assert all(sha(folder / k) == v for k, v in done["artifact_sha256"].items())
    cfg = read(folder / "protocol.json")
    assert all(sha(Path(k)) == v for k, v in cfg["input_sha256"].items())
    return done, cfg


def inspect_timed():
    done, cfg = verify(TIMED)
    rows, summary = read(TIMED / "rows.json"), read(TIMED / "summary.json")
    assert done["status"] == "COMPLETE_TIMED_TOKEN_NORM_ROUTE_SCREEN"
    assert len(rows) == done["cases"] == 240 and all(r["supported"] for r in rows)
    for r in rows:
        for name, a in r["arms"].items():
            assert (
                abs(np.logaddexp(0, (1 - 2 * r["label"]) * a["logit"]) - a["bce"])
                < 1e-6
            )
            if name != "none":
                assert abs(a["step_norm"] - cfg["step_norm"]) < 1e-7
    for rule, groups in summary.items():
        for group, result in groups.items():
            subset = [r for r in rows if r["group"] == group]
            shots = sorted({r["shot"] for r in subset})
            for mode, value in result["route_vs_none"].items():
                arm = rule + "__" + mode
                d = [
                    np.mean(
                        [
                            r["arms"][arm]["bce"] - r["arms"]["none"]["bce"]
                            for r in subset
                            if r["shot"] == sid
                        ]
                    )
                    for sid in shots
                ]
                assert np.isclose(np.mean(d), value, atol=1e-14, rtol=0)
                flags = [
                    (
                        r["arms"][arm]["p"] >= cfg["threshold"],
                        r["arms"]["none"]["p"] >= cfg["threshold"],
                    )
                    for r in subset
                ]
                assert result["crossings"][mode] == dict(
                    up=sum(x and not y for x, y in flags),
                    down=sum(y and not x for x, y in flags),
                )
    return dict(
        shots=len(cfg["held_shots"]), windows=len(rows), seconds=done["seconds"]
    )


def inspect_reach(east):
    done, cfg = verify(REACH)
    assert done["status"] == "COMPLETE_POSTHOC_LOCAL_AFFINE_REACH"
    rows = read(REACH / "per_shot.json")
    records = {r["shot"]: r for r in rows}
    threshold = cfg["threshold"]
    rescued = {r[0]: [] for r in cfg["routes"]}
    durations = []
    assert len(records) == done["shots"] == 82
    with np.load(REACH / "points.npz", allow_pickle=False) as f:
        points = {name: f[name] for name in f.files}
    assert len(points["shot"]) == done["points"] == 6004
    for index, sid in enumerate(cfg["order"][20:], 21):
        path = STATIC / "shots" / f"{index:04d}_{sid}.npz"
        with np.load(path, allow_pickle=False) as f:
            baseline, t = f["map_after20"], f["times"]
        r = east[sid]
        lead = np.round((r["down"] - t) * 1000, 9) if r["disruptive"] else None
        eligible = lead >= 0 if lead is not None else np.ones(len(t), dtype=bool)
        flags = np.flatnonzero(eligible & (baseline >= threshold))
        if r["disruptive"] and (not len(flags) or lead[flags[0]] <= 30):
            queries = np.flatnonzero(
                (lead > 30) & (lead <= 125) & (np.arange(len(t)) >= 135)
            )
            kind = "tardy" if len(flags) else "miss"
        elif not r["disruptive"] and len(flags) and flags[0] >= 135:
            queries, kind = flags[:1], "fp_first_point"
            durations.append(len(flags))
        else:
            assert sid not in records
            continue
        row = records[sid]
        mask = points["shot"] == sid
        assert row["kind"] == kind and sha(path) == row["curve_sha256"]
        assert np.array_equal(queries, points["query"][mask])
        assert np.array_equal(baseline[queries], points["baseline_probability"][mask])
        assert (
            np.max(
                abs(
                    1 / (1 + np.exp(-points["baseline_logit"][mask].astype(float)))
                    - baseline[queries]
                )
            )
            < 3e-6
        )
        for name, _, _ in cfg["routes"]:
            a = points[name][mask]
            assert np.isfinite(a).all() and (a[:, 1] >= 0).all()
            for i, radius in enumerate(cfg["radii"]):
                p = 1 / (1 + np.exp(-a[:, 2 + 2 * i].astype(float)))
                crossed = p < threshold if kind == "fp_first_point" else p >= threshold
                cell = row["routes"][name][str(radius)]
                assert (int(crossed.sum()), bool(crossed.any())) == (
                    cell["crossings"],
                    cell["crosses"],
                )
                if radius == 0.5 and lead is not None and crossed.any():
                    rescued[name].append(sid)
    return dict(
        shots=len(rows),
        points=done["points"],
        seconds=done["seconds"],
        rescued_inband_by_radius_0_5=rescued,
        eligible_safe_fp_sealed_crossing_samples=dict(
            n=len(durations),
            median=float(np.median(durations)),
            min=min(durations),
            max=max(durations),
            with_multiple_crossings=sum(n > 1 for n in durations),
        ),
    )


def inspect_arrival(east):
    done, cfg = verify(ARRIVAL)
    rows, output = read(ARRIVAL / "rows.json"), read(ARRIVAL / "summary.json")
    losses = read(ARRIVAL / "fit_objective.json")
    assert done["status"] == "COMPLETE_SAME_SITE_ARRIVAL_DEVELOPMENT"
    assert done["source_cases"] == 1410 and done["east_fit_cases"] == 60
    assert len(rows) == done["east_scored_cases"] == 300
    assert set(cfg["first20_fit"]).isdisjoint(r["shot"] for r in rows)
    tau = np.log(cfg["threshold"] / (1 - cfg["threshold"]))
    for r in rows:
        meta = east[r["shot"]]
        assert r["future"] == r["arrival"] + 24 == r["origin"] + 32
        clock = meta["start"] + r["future"] / meta["rate"]
        lead = round((meta["down"] - clock) * 1000, 9) if meta["disruptive"] else None
        assert r["lead_ms"] == lead and r["label"] == int(
            lead is not None and 30 < lead <= 125
        )
        assert all(0 <= a["norm"] < cfg["radius"] + 1e-6 for a in r["arms"].values())
    for sl, groups in output.items():
        for group, cell in groups.items():
            part = [
                r
                for r in rows
                if r["group"] == group
                and (r["stream_index"] <= 100) == (sl == "held80")
            ]
            ids = {r["shot"] for r in part}
            assert len(part) == cell["cases"] and len(ids) == cell["shots"]
            for key, comp in cell["models"].items():
                real = key + "__real"
                for ref in ("none", "hold", "reversed", "negative"):
                    arm = ref if ref == "none" else key + "__" + ref
                    d = [
                        np.mean(
                            [
                                np.logaddexp(
                                    0, (1 - 2 * r["label"]) * r["arms"][real]["z"]
                                )
                                - np.logaddexp(
                                    0, (1 - 2 * r["label"]) * r["arms"][arm]["z"]
                                )
                                for r in part
                                if r["shot"] == sid
                            ]
                        )
                        for sid in ids
                    ]
                    assert np.isclose(
                        np.mean(d), comp[ref]["delta_bce"], atol=1e-14, rtol=0
                    )
                assert comp["crossings"] == dict(
                    up=sum(
                        r["arms"]["none"]["z"] < tau <= r["arms"][real]["z"]
                        for r in part
                    ),
                    down=sum(
                        r["arms"][real]["z"] < tau <= r["arms"]["none"]["z"]
                        for r in part
                    ),
                )
    diagnostics = {}
    with np.load(ARRIVAL / "fit.npz", allow_pickle=False) as f:
        for key, loss in losses.items():
            scope, site, kind = key.split("__")
            x, z, g, y, w = [
                f[scope + "_" + field]
                for field in (
                    kind + "_features",
                    "z",
                    site + "_jacobian",
                    "y",
                    "weight",
                )
            ]
            matrix = f[key + "_matrix"]
            assert np.allclose(
                np.where(w @ x**2 > 1e-24, np.sqrt(w @ x**2), 1.0), f[key + "_scale"]
            )
            u = (x / f[key + "_scale"]) @ matrix
            delta = cfg["radius"] * u / np.sqrt(1 + (u * u).sum(1, keepdims=True))
            offset = z + (g * delta).sum(1)
            actual = (
                w @ np.logaddexp(0, (1 - 2 * y) * offset)
                + cfg["ridge"] / 2 * (matrix**2).sum()
            )
            assert abs(loss["after"] - actual) < 1e-11
            assert abs(loss["before"] - (w @ np.logaddexp(0, (1 - 2 * y) * z))) < 1e-11
        for name, scope, region in (
            ("source_train", "mixed", slice(0, 1410)),
            ("east_fit", "east20", slice(None)),
        ):
            a = f[scope + "_observed_features"][region, :21]
            forecast = a - f[scope + "_innovation_features"][region, :21]
            comparable = (a != 0) & (forecast != 0)
            diagnostics[name] = dict(
                comparable_nonzero_channel_changes=int(comparable.sum()),
                sign_agreement=float(
                    (np.sign(a[comparable]) == np.sign(forecast[comparable])).mean()
                ),
                median_shot_window_actual_change_norm=float(
                    np.median(np.linalg.norm(a, axis=1))
                ),
                median_shot_window_forecast_change_norm=float(
                    np.median(np.linalg.norm(forecast, axis=1))
                ),
            )
    return dict(
        source_train_cases=done["source_cases"],
        east_fit_cases=done["east_fit_cases"],
        held_cases=240,
        later_cases=60,
        seconds=done["seconds"],
        reproduced_fit_objectives=len(losses),
        training_feature_diagnostics=diagnostics,
    )


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    east = {r["shot"]: r for r in read(FRESH / "hdf5_metadata.json")["EAST"]}
    r = dict(
        status="PASS",
        timed=inspect_timed(),
        reach=inspect_reach(east),
        arrival=inspect_arrival(east),
    )
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
    print(r)


if __name__ == "__main__":
    main()
