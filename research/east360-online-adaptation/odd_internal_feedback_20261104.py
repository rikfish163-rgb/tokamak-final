"""Apply just the measured-motion component of fixed internal update rules."""

from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import source_supported_internal_update_20261104 as p

ROOT = p.ROOT
FIT = ROOT / "probe_outputs/source_view_coverage_20261104_v1"
OUT = ROOT / "probe_outputs/odd_internal_feedback_20261104_v1"
PLAN = ROOT / "ODD_INTERNAL_FEEDBACK_PLAN_20261104.md"
MODES = ("real", "reverse", "permuted")


def odd_direction(fit, motion):
    m = np.asarray(motion, dtype=np.float64)
    if (
        m.ndim != 2
        or m.shape[1] != 21
        or fit["scale"].shape != (85,)
        or fit["coef"].shape[0] != 86
        or not np.isfinite(m).all()
        or not np.isfinite(fit["scale"]).all()
        or not np.isfinite(fit["coef"]).all()
        or not (fit["scale"] > 0).all()
    ):
        raise ValueError("fixed 64-state/21-motion linear rule required")
    return (m / fit["scale"][64:]) @ fit["coef"][65:]


def group_report(rows, threshold):
    result = {}
    for name in p.LEARNERS:
        groups = {}
        for group in ("safe", "earlier", "positive"):
            subset = [r for r in rows if r["group"] == group]
            shots = sorted({r["shot"] for r in subset})
            stat = dict(shots=len(shots), cases=len(subset))

            def compare(a, b):
                by_shot = [
                    np.mean(
                        [
                            r["arms"][a]["bce"] - r["arms"][b]["bce"]
                            for r in subset
                            if r["shot"] == sid
                        ]
                    )
                    for sid in shots
                ]
                v = np.asarray(by_shot)
                return dict(
                    mean=float(v.mean()),
                    improved=int((v < 0).sum()),
                    worse=int((v > 0).sum()),
                    tied=int((v == 0).sum()),
                    shot_se=float(v.std(ddof=1) / np.sqrt(len(v)))
                    if len(v) > 1
                    else None,
                )

            true = name + "__real"
            for ref in (
                "none",
                name + "__full",
                name + "__reverse",
                name + "__permuted",
            ):
                stat["real_vs_" + ref.split("__")[-1]] = compare(true, ref)
            full_even, full_odd = [], []
            for sid in shots:
                current = [r for r in subset if r["shot"] == sid]
                full_even.append(
                    np.mean(
                        [
                            (
                                r["arms"][name + "__full"]["logit"]
                                + r["arms"][name + "__full_reverse"]["logit"]
                            )
                            / 2
                            - r["arms"]["none"]["logit"]
                            for r in current
                        ]
                    )
                )
                full_odd.append(
                    np.mean(
                        [
                            (
                                r["arms"][name + "__full"]["logit"]
                                - r["arms"][name + "__full_reverse"]["logit"]
                            )
                            / 2
                            for r in current
                        ]
                    )
                )
            stat.update(
                previous_full_even_logit_mean=float(np.mean(full_even)),
                previous_full_odd_logit_mean=float(np.mean(full_odd)),
                upward_crossings=sum(
                    r["arms"]["none"]["probability"]
                    < threshold
                    <= r["arms"][true]["probability"]
                    for r in subset
                ),
                downward_crossings=sum(
                    r["arms"][true]["probability"]
                    < threshold
                    <= r["arms"]["none"]["probability"]
                    for r in subset
                ),
            )
            groups[group] = stat
        result[name] = groups
    return result


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    p.verify_saved(FIT)
    p.verify_saved(p.original.FEATURES)
    source_files = (
        Path(__file__),
        PLAN,
        Path(p.__file__),
        Path(p.original.__file__),
        Path(p.data.__file__),
        p.data.DEFAULT_PARENT,
        p.data.MAP,
        p.data.RAW_HASHES,
        p.data.RUN / "protocol.json",
        FIT / "completed.json",
        FIT / "fit.npz",
        FIT / "results.json",
        p.original.FEATURES / "completed.json",
        p.original.FEATURES / "features_and_logits.npz",
    )
    bound = {str(f): p.data.sha(f) for f in source_files}
    protocol = p.data.read_json(p.data.RUN / "protocol.json")
    ids = [r["shot"] for r in protocol["order"]]
    threshold = protocol["threshold"]
    OUT.mkdir()
    p.data.write_json(
        OUT / "protocol.json",
        dict(
            scope="FIXED_ODD_COMPONENT_REAL_INTERNAL_STEP_HELD80",
            held_shots=ids[20:100],
            input_sha256=bound,
            step=0.05,
            labels_only_for_held_scoring=True,
            no_alarm_calibration=True,
            fixed_learners=p.LEARNERS,
            modes=MODES,
        ),
    )
    previous = p.data.read_json(FIT / "results.json")
    with np.load(FIT / "fit.npz", allow_pickle=False) as f:
        fits = {
            name: {k: f[name + "_" + k] for k in ("mean", "scale", "coef")}
            for name in p.LEARNERS
        }
    with np.load(
        p.original.FEATURES / "features_and_logits.npz", allow_pickle=False
    ) as f:
        measured = f["held_long"]
        permuted = f["held_permuted"]
        if measured.shape != permuted.shape or measured.shape != (240, 21):
            raise ValueError("fixed held80 motion views required")
    directions = {}
    for name, fit in fits.items():
        for mode, features in zip(MODES, (measured, -measured, permuted), strict=True):
            directions[name + "__" + mode] = odd_direction(fit, features)
    base, parent = p.data.load_parent(p.data.DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    frozen = p.data.state_digest(base)
    if frozen != protocol["source_state_sha256"] or parent["threshold"] != threshold:
        raise ValueError("frozen parent changed")
    model = p.original.make_adapter(base, "norm")
    parameters = tuple(q for q in model.parameters() if q.requires_grad)
    initial = tuple(q.detach().clone() for q in parameters)
    if sum(q.numel() for q in parameters) != 256:
        raise ValueError("exact internal norm scope required")
    mapping, hashes = p.data.read_json(p.data.MAP), p.data.read_json(p.data.RAW_HASHES)
    rows, receipts, k = [], [], 0
    for i, sid in enumerate(ids[20:100], 21):
        signal, cases, receipt = p.data._read_sealed(i, sid, mapping, hashes)
        receipts.append(receipt)
        for case in cases:
            old = previous[k]
            if (sid, case["origin"], case["label"]) != (
                old["shot"],
                old["origin"],
                old["label"],
            ):
                raise ValueError("fixed held cases changed")
            x, a = p.original.window_batch(signal, [case["future"]], device="cpu")
            arms = {}
            for arm in ("none", *directions):
                p.reset(parameters, initial)
                updated = (
                    0
                    if arm == "none"
                    else p.original.apply_direction(parameters, directions[arm][k])
                )
                with torch.no_grad():
                    z = model(x, a)["logit"]
                    bce = F.binary_cross_entropy_with_logits(
                        z, torch.full_like(z, float(case["label"]))
                    )
                    arms[arm] = dict(
                        logit=float(z), probability=float(z.sigmoid()), bce=float(bce)
                    )
                if arm != "none":
                    arms[arm]["updated_parameters"] = updated
            if arms["none"] != old["arms"]["none"]:
                raise ValueError("frozen baseline must replay exactly")
            for name in p.LEARNERS:
                arms[name + "__full"] = old["arms"][name + "__real"]
                arms[name + "__full_reverse"] = old["arms"][name + "__reverse"]
            rows.append(dict(shot=sid, **case, arms=arms))
            k += 1
    p.reset(parameters, initial)
    if (
        k != 240
        or p.data.state_digest(base) != frozen
        or any(p.data.sha(Path(f)) != h for f, h in bound.items())
    ):
        raise ValueError("case, parent or artifact changed")
    report = group_report(rows, threshold)
    for name, value in (
        ("rows.json", rows),
        ("summary.json", report),
        ("held_inputs.json", receipts),
    ):
        p.data.write_json(OUT / name, value)
    p.data.write_json(
        OUT / "completed.json",
        dict(
            status="COMPLETE_ODD_FEEDBACK_HELD_SCREEN_NOT_ONLINE",
            seconds=time.monotonic() - started,
            cases=k,
            artifact_sha256={
                f.name: p.data.sha(f) for f in OUT.iterdir() if f.is_file()
            },
        ),
    )
    print(
        {
            name: {g: row["real_vs_none"] for g, row in value.items()}
            for name, value in report.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    run()
