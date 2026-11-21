"""Matched missingness-view coverage for the fixed internal-gradient policy."""

from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import source_supported_internal_update_20261104 as p

ROOT = p.ROOT
PLAN = ROOT / "SOURCE_VIEW_COVERAGE_PLAN_20261104.md"
OUT = ROOT / "probe_outputs/source_view_coverage_20261104_v1"


def paired_views(old_x, old_y, new_x, new_y, records):
    if (
        old_x.shape != new_x.shape
        or old_y.shape != new_y.shape
        or len(old_x) != len(records)
        or len(old_y) != len(records)
    ):
        raise ValueError("exact matched source views required")
    rows = [dict(r, view=v) for v in ("no_MNM_CIII", "no_MNM") for r in records]
    return np.r_[old_x, new_x], np.r_[old_y, new_y], rows


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    p.verify_saved(p.OUT)
    p.verify_saved(p.OLD)
    p.verify_saved(p.original.FEATURES)
    original = p.original
    records = p.data.read_json(p.OUT / "source_records.json")
    source_receipts = p.data.read_json(p.OUT / "source_inputs.json")
    raw_hashes = {r["shot"]: r["raw_sha256"] for r in source_receipts}
    protocol = p.data.read_json(p.OUT / "protocol.json")
    if [r["shot"] for r in records] != [
        r["shot"] for r in protocol["selected_records"]
    ]:
        raise ValueError("old source record identities changed")
    bound = {
        str(path): p.data.sha(path)
        for path in (
            Path(__file__),
            Path(p.__file__),
            Path(original.__file__),
            PLAN,
            p.OUT / "completed.json",
            p.OUT / "source_records.json",
            p.OUT / "source_inputs.json",
            p.OUT / "fit.npz",
            p.OLD / "fit.npz",
            p.original.FEATURES / "features_and_logits.npz",
            p.data.DEFAULT_PARENT,
        )
    }
    OUT.mkdir()
    p.data.write_json(
        OUT / "protocol.json",
        dict(
            scope="SOURCE_PAIRED_VIEW_COVERAGE_MATCHED_INTERNAL_UPDATE_DIAGNOSTIC",
            previous_completion_sha256=p.data.sha(p.OUT / "completed.json"),
            fixed_shots=len(raw_hashes),
            fixed_windows=len(records),
            source_views=["no_MNM_CIII", "no_MNM"],
            input_sha256=bound,
            covariance_or_threshold_recalibration=False,
            source_parent_changed=False,
            ridge=0.1,
            step=0.05,
            mixed_source_mass=0.5,
        ),
    )
    base, parent = p.data.load_parent(p.data.DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    source_state = p.data.state_digest(base)
    model = original.make_adapter(base, "norm")
    params = tuple(q for q in model.parameters() if q.requires_grad)
    initial = tuple(q.detach().clone() for q in params)
    advanced = p.io23.advanced_times()
    byshot = defaultdict(list)
    for k, row in enumerate(records):
        byshot[row["shot"]].append((k, row))
    nx, ny = np.empty((len(records), 85)), np.empty((len(records), 256))
    receipts = []
    for n, (sid, rows) in enumerate(sorted(byshot.items()), 1):
        path = p.io23.path_for("J-TEXT", sid)
        if p.data.sha(path) != raw_hashes[sid]:
            raise ValueError("source raw changed")
        raw = p.simple_data.read_shot("J-TEXT", sid)
        disruptive, down = p.io23.read_outcome("J-TEXT", sid)
        horizon = p.io23.effective_horizon_ms(sid, advanced)[0]
        labels, valid = p.io23.labels_at(raw["times"], disruptive, down, horizon)
        signal = dict(values=raw["values"].copy(), available=raw["available"].copy())
        signal["available"][:, 2] = False
        for k, row in rows:
            q = row["query"]
            if (
                not valid[q]
                or float(labels[q]) != row["label"]
                or disruptive != row["actual_hdf_disruptive"]
            ):
                raise ValueError("source label changed between views")
            nx[k] = p.feature_at(base, signal, row["maturity"], sid)
            x, a = original.window_batch(signal, [q], device="cpu")
            z = model(x, a)["logit"]
            loss = F.binary_cross_entropy_with_logits(
                z, torch.full_like(z, float(labels[q]))
            )
            g = torch.autograd.grad(loss, params)
            vector = torch.cat([v.flatten() for v in g]).detach().double()
            norm = vector.norm()
            ny[k] = (vector / norm if norm else vector).numpy()
        if p.data.sha(path) != raw_hashes[sid]:
            raise ValueError("raw source changed while reading")
        receipts.append(
            dict(
                shot=sid,
                raw_sha256=raw_hashes[sid],
                CIII_available_fraction=float(signal["available"][:, 1].mean()),
            )
        )
        if n % 200 == 0:
            print(dict(paired_source_shots=n), flush=True)
    with np.load(p.OUT / "fit.npz", allow_pickle=False) as f:
        old_x, old_y = f["source_features"], f["source_gradients"]
        sx, sy, srows = paired_views(old_x, old_y, nx, ny, records)
    uniform = p.family_weights(srows, use_family=False)
    ew = p.family_weights(srows, use_family=True)
    # Match v1's per-record family permutation, then repeat it in both views;
    # otherwise a different random draw would be confounded with view coverage.
    shuffled = [r.copy() for r in records]
    rng = np.random.default_rng(131)
    for group in ("safe", "early", "danger"):
        idx = [i for i, r in enumerate(shuffled) if r["group"] == group]
        values = rng.permutation([shuffled[i]["family"] for i in idx])
        for i, v in zip(idx, values, strict=True):
            shuffled[i]["family"] = int(v)
    srows_shuffled = [
        dict(r, view=v) for v in ("no_MNM_CIII", "no_MNM") for r in shuffled
    ]
    sw = p.family_weights(srows_shuffled, use_family=True)
    with np.load(p.OLD / "fit.npz", allow_pickle=False) as f:
        eg, weight = f["teachers"], f["weights"]
        old_fit = {k: f["real_" + k] for k in ("mean", "scale", "coef")}
    with np.load(
        original.FEATURES / "features_and_logits.npz", allow_pickle=False
    ) as f:

        def matrix(stage):
            return np.concatenate(
                [
                    f[stage + "_endpoint"],
                    f[stage + "_endpoint_mask"],
                    f[stage + "_common"],
                    f[stage + "_native_logit"][:, None],
                    f[stage + "_long"],
                ],
                axis=1,
            ).astype(float)

        ex, hx = matrix("fit"), matrix("held")
    learners = dict(east20=old_fit, source_only=original.fit_transport(sx, sy, uniform))
    x, y = np.r_[sx, ex], np.r_[sy, eg]
    for name, w in (
        ("mixed_uniform", uniform),
        ("mixed_event", ew),
        ("mixed_shuffled", sw),
    ):
        learners[name] = original.fit_transport(x, y, np.r_[0.5 * w, 0.5 * weight])
    with (OUT / "fit.npz").open("xb") as f:
        np.savez_compressed(
            f,
            new_source_features=nx,
            new_source_gradients=ny,
            **{name + "_" + k: v for name, m in learners.items() for k, v in m.items()},
        )
    print("paired-view rule fitted; fixed held80 evaluation follows", flush=True)
    protocol360 = p.data.read_json(p.data.RUN / "protocol.json")
    if (
        source_state != protocol360["source_state_sha256"]
        or parent["threshold"] != protocol360["threshold"]
    ):
        raise ValueError("source parent or threshold changed")
    ids = [r["shot"] for r in protocol360["order"]]
    mapping, hashes = p.data.read_json(p.data.MAP), p.data.read_json(p.data.RAW_HASHES)
    held_meta = [
        r
        for r in p.data.read_json(original.FEATURES / "cases.json")
        if r["stage"] == "held"
    ]
    results, held_receipts = p.evaluate(
        base, model, params, initial, learners, hx, held_meta, ids, mapping, hashes
    )
    old = p.data.read_json(p.OUT / "results.json")
    if any(
        a["arms"]["none"] != b["arms"]["none"]
        or any(a["arms"]["east20__" + m] != b["arms"]["east20__" + m] for m in p.MODES)
        for a, b in zip(results, old, strict=True)
    ):
        raise ValueError("E20-only reference changed")
    if p.data.state_digest(base) != source_state or any(
        p.data.sha(Path(k)) != v for k, v in bound.items()
    ):
        raise ValueError("source or bound inputs changed")
    summary = p.summarise(results)
    # The reused one-rule summarizer calls its extra comparator "state".
    # Here it is explicitly the frozen E20 rule, not a state-only learner.
    for groups in summary.values():
        for value in groups.values():
            pairs = value["pairs"]
            pairs["east20_vs_none"] = pairs.pop("state_vs_none")
            pairs["real_vs_east20"] = pairs.pop("real_vs_state")
    for filename, value in (
        ("results.json", results),
        ("summary.json", summary),
        ("source_inputs.json", receipts),
        ("held_inputs.json", held_receipts),
    ):
        p.data.write_json(OUT / filename, value)
    p.data.write_json(
        OUT / "completed.json",
        dict(
            status="COMPLETE_PAIRED_SOURCE_VIEW_INTERNAL_DIRECTION_SCREEN",
            source_model_unchanged=True,
            east20_control_exact=True,
            seconds=time.monotonic() - start,
            artifact_sha256={
                path.name: p.data.sha(path) for path in OUT.iterdir() if path.is_file()
            },
        ),
    )
    print(
        {
            name: {g: v["pairs"]["real_vs_none"]["delta_bce"] for g, v in rows.items()}
            for name, rows in summary.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    run()
