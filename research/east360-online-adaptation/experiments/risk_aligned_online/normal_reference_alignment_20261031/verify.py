"""Recount all complete trajectories and independently verify calibration moments."""

import argparse
import hashlib
from pathlib import Path
import zipfile

import numpy as np

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..delayed_recurrent_memory_20261025.learning import original_predictions
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..redesign_20261004.prepare_prior import load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .core import transport


def independent_statistics(rows, center, scale):
    z = []
    m = []
    w = []
    observed = np.zeros(23, int)
    for row in rows:
        indices = np.linspace(
            0, len(row["values"]) - 1, min(128, len(row["values"])), dtype=int
        )
        mask = row["available"][indices]
        values = row["values"][indices].astype(np.float64)
        values[~mask] = np.broadcast_to(center, values.shape)[~mask]
        z.append(np.arcsinh((values - center) / scale))
        m.append(mask)
        w.extend([1 / len(indices)] * len(indices))
        observed += mask.any(0)
    z = np.concatenate(z)
    m = np.concatenate(m)
    w = np.asarray(w)
    mu = np.zeros(23)
    sd = np.zeros(23)
    for c in range(23):
        ix = np.flatnonzero(m[:, c])
        if not len(ix):
            continue
        weights = w[ix] / sum(w[ix])
        mu[c] = sum(weights * z[ix, c])
        if np.ptp(z[ix, c]) != 0:
            sd[c] = np.sqrt(sum(weights * (z[ix, c] - mu[c]) ** 2))
    standardized = np.zeros_like(z)
    for c in np.flatnonzero(sd > 0):
        ix = np.flatnonzero(m[:, c])
        standardized[ix, c] = (z[ix, c] - mu[c]) / sd[c]
    covariance = np.einsum("ni,nj,n->ij", standardized, standardized, w) / sum(w)
    corr = np.eye(23)
    active = np.flatnonzero(sd > 0)
    for a in active:
        for b in active:
            corr[a, b] = covariance[a, b] / np.sqrt(covariance[a, a] * covariance[b, b])
    return dict(mean=mu, std=sd, correlation=corr, observed_shots=observed)


def main(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise FileExistsError(output)
    p = read(root / "protocol.json")
    done = read(root / "completed.json")
    reference = Path(p["reference"])
    assert done["status"] in ("COMPLETED", "COMPLETED_WITH_INVALID_CANDIDATES")
    assert sha(reference / "completed.json") == p["reference_completed_sha256"]
    assert sha(Path(p["source_checkpoint"])) == p["source_sha256"]
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        for path, digest in p["code_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    order = p["order"]
    ids = [r["shot"] for r in order]
    assert len(ids) == len(set(ids)) == 360
    assert order == read(reference / "order.json")
    saved = read(root / "per_shot.json")
    failed = done["failed_candidates"]
    assert set(saved) == set(p["arms"]) - set(failed)
    hashes = read(reference / "input_hashes.json")
    rows = {a: [] for a in saved}
    matched = 0
    negative = {a: dict(sxr_var=0, xuv_var=0) for a in p["arms"][2:]}
    target_rows = []
    target_safe = []
    for i, sid in enumerate(ids):
        prefix = f"{i + 1:04d}_{sid}"
        path = root / "shots" / f"{prefix}.npz"
        seal = read(root / "shots" / f"{prefix}_seal.json")
        assert sha(path) == seal["sha256"] and not seal["outcome_available"]
        assert seal["shot"] == sid and seal["index"] == i + 1
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        flag, down = io23.read_outcome("EAST", sid)
        with (
            np.load(path, allow_pickle=False) as a,
            np.load(reference / "shots" / f"{prefix}.npz", allow_pickle=False) as b,
        ):
            expected = set(p["arms"]) - {
                name for name, reason in failed.items() if i + 1 >= reason["index"]
            }
            assert set(a.files) == expected | {"times"}
            assert np.array_equal(a["times"], b["times"])
            assert sorted(expected) == seal["active_arms"]
            for name in expected:
                assert (
                    a[name].shape == a["times"].shape and np.isfinite(a[name]).all()
                ), (sid, name, "invalid curve")
                if i < 20:
                    assert np.array_equal(a[name], a["source"])
                if name in rows:
                    r = classify(a["times"], a[name], flag, down, p["threshold"])
                    assert dict(shot=sid, is_disrupt=flag, **r) == saved[name][i]
                    rows[name].append(dict(shot=sid, disruptive=flag, **r))
            for name, old in (("source", "source"), ("density", "aligned")):
                assert np.array_equal(a[name], b[old])
                matched += 1
        for name, values in seal["negative_variance_outputs"].items():
            for key in negative[name]:
                negative[name][key] += values[key]
        if i < 20:
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            target_rows.append(raw)
            if not flag:
                target_safe.append(raw)
    metrics = {a: counts(rs) for a, rs in rows.items()}
    for name, rs in rows.items():
        for scope, subset in (
            ("full360", rs),
            ("warmup20", rs[:20]),
            ("suffix340", rs[20:]),
        ):
            recounted = counts(subset)
            for key in (
                "tp",
                "fp",
                "n_disrupt",
                "n_safe",
                "tpr",
                "fpr",
                "J",
                "early_over3000",
            ):
                assert recounted[key] == done["metrics"][name][scope][key]
    source_rows = []
    source_safe = []
    source_hashes = read(root / "source_hashes.json")
    assert list(map(int, source_hashes)) == io23.source_roles()["train"]
    for sid in io23.source_roles()["train"]:
        assert sha(io23.path_for("J-TEXT", sid)) == source_hashes[str(sid)]
        raw = simple_data.read_shot("J-TEXT", sid)
        source_rows.append(raw)
        flag, _ = io23.read_outcome("J-TEXT", sid)
        if not flag:
            source_safe.append(raw)
    center, scale = np.asarray(p["center"]), np.asarray(p["scale"])
    stat_errors = {}
    for domain, groups in (
        ("source", dict(pooled=source_rows, safe=source_safe)),
        ("target", dict(pooled=target_rows, safe=target_safe)),
    ):
        recorded = read(root / f"{domain}_statistics.json")
        for group, records in groups.items():
            actual = independent_statistics(records, center, scale)
            expected = recorded[group]
            assert expected["shot_ids"] == [r["shot"] for r in records]
            stat_errors[domain + "_" + group] = {}
            for key, value in actual.items():
                error = float(np.max(abs(value - np.asarray(expected[key]))))
                stat_errors[domain + "_" + group][key] = error
                assert np.allclose(value, expected[key], rtol=1e-10, atol=1e-12), (
                    domain,
                    group,
                    key,
                    error,
                )
    for name in p["arms"][2:]:
        mapping = read(root / "maps" / f"{name}.json")
        group, _ = name.split("_")
        assert mapping["first_affected_rank"] == 21
        assert mapping["source"] == read(root / "source_statistics.json")[group]
        assert mapping["target"] == read(root / "target_statistics.json")[group]
        assert mapping["preceding_seals"] == {
            str(s): read(root / "shots" / f"{i + 1:04d}_{s}_seal.json")["sha256"]
            for i, s in enumerate(ids[:20])
        }
    pairs = {
        name + "__minus__density": flips(rows["density"], values)
        for name, values in rows.items()
        if name not in ("source", "density")
    }
    for mode in ("diagonal", "correlation"):
        a, b = "pooled_" + mode, "safe_" + mode
        if a in rows and b in rows:
            pairs[b + "__minus__" + a] = flips(rows[a], rows[b])
    report = dict(
        status="PASS",
        metrics=metrics,
        flips=pairs,
        complete_curves_recounted=360 * len(rows),
        failed_candidates=failed,
        partial_failed_curves_not_scored=True,
        reference_curves_exact=matched,
        source_files_verified=1353,
        target_files_verified=360,
        statistics_max_errors=stat_errors,
        statistics_independent_recompute=True,
        negative_variance_points_in_completed_prefixes=negative,
        source_safe_ids=[r["shot"] for r in source_safe],
        target_safe_ids=[r["shot"] for r in target_safe],
        completed_sha256=sha(root / "completed.json"),
        independent_future_confirmation=False,
    )
    _durable_json(root / "count_statistics_audit.json", report)
    configure()
    base, _ = load_parent(Path(p["source_checkpoint"]))
    base.cuda().eval()
    selected = [20, 113, 359]
    exact = 0
    for index in selected:
        sid = ids[index]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = {k: raw[k] for k in ("values", "available")}
        with np.load(
            root / "shots" / f"{index + 1:04d}_{sid}.npz", allow_pickle=False
        ) as saved_trace:
            for name in rows:
                if name == "source":
                    mapped = signal
                elif name == "density":
                    mapped = transform(signal, read(root / "maps/density.json"))
                else:
                    mapped = transport(
                        signal, read(root / "maps" / f"{name}.json"), center, scale
                    )
                pred = original_predictions(base, mapped)
                assert np.array_equal(pred, saved_trace[name]), (sid, name, "replay")
                exact += 1
    report.update(
        neural_replay_shots=[ids[i] for i in selected],
        exact_replay_curves=exact,
        input_map_same_implementation_reused_for_neural_replay=True,
        verifier_sha256=sha(Path(__file__)),
    )
    _durable_json(output, report)
    print({k: v for k, v in report.items() if k not in ("source_safe_ids", "flips")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
