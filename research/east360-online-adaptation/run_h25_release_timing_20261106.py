"""TRAIN-only H25 release-timing pilot; output is a compact post-reveal JSON.

The full 128-dimensional ambient nullspace is intentional: this is not the
earlier strict H25-family-span projection. Both clocks share the same target
gradients and algorithm, but use their own already-reached risk Jacobians.
Raw signal access is needed for prediction; outcome metadata is not read until
all six complete, label-blind traces have been assembled in memory.
"""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import h25_arrived_gradient_20261106 as h25
import h25_direction_geometry_20261106 as geo
import run_h25_direction_mechanism_20261106 as mech
import run_output_trust_mix_capacity_20261106 as mix
from inspect_peak_contrast_first_alarm_20261106 import first
from verify_confirmation_raw_replay_20261105 import Hashes


bank, r = mech.bank, mech.r
PRIOR = r.ROOT / "probe_outputs/output_trust_20261106_v1"
INVENTORY = bank.src.PREVIOUS / "train_coverage.json"
PROTOCOL = r.ROOT / "H25_RELEASE_AND_INCREMENT_PROTOCOL_20261106.md"
PROTOCOL_SHA256 = "a6352c9bbbb192e9a2f8151a1eaba95932b6416db9e891ffec65ac9902765cf5"
AUDIT_SHA256 = "06d68ae4bf2c633da827f170fea46f8a85510767e3748346cda78457646ba33b"
LATE_IDS = (1052388, 1053042, 1053940, 1056284, 1056294, 1057866)
RADIUS, LIMIT = 0.35, 0.10
CLOCKS, ARMS = ("early", "late"), ("true", "reverse")


def schedule(length):
    """Each origin is included iff its fixed H25 target has fully arrived."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError("nonnegative integer shot length required")
    return [
        dict(origin=o, maturity=o + 25, early=o + 25, late=o + 32)
        for o in range(31, length - 25, 32)
    ]


def block_indices(length, release):
    """The next 32 raw points, with no endpoint beyond the shot."""
    if length < 0 or release < 0:
        raise ValueError("nonnegative clock values required")
    return np.arange(release + 1, max(release + 1, min(release + 33, length)))


def write_block(curve, written, release, values):
    ix = block_indices(len(curve), release)
    values = np.asarray(values)
    if curve.shape != (len(written), 2) or values.shape != (len(ix), 2):
        raise ValueError("block length must match the two-arm release window")
    if written[ix].any():
        raise ValueError("release windows overlap")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite risk output")
    curve[ix] = values
    written[ix] = True


def target_gradients(base, x, available, raw, entries):
    origins = np.asarray([q["origin"] for q in entries], dtype=np.int64)
    maturity = origins + 25
    if not len(origins):
        return dict(
            true=np.empty((0, 3, 128)),
            reverse=np.empty((0, 3, 128)),
            groups=np.empty((0, 3), dtype=bool),
        )
    target, mask = h25.arrived_targets(
        raw["values"], raw["available"], origins, maturity
    )
    result = h25.gradients(
        base, x[origins].contiguous(), available[origins].contiguous(), target, mask
    )
    return {key: result[key].detach().cpu().numpy() for key in ("true", "reverse", "groups")}


def ambient_directions(family, groups, past):
    """Project signed unit-family mean onto null(past), not span(family)."""
    family = np.asarray(family, dtype=np.float64)
    groups = np.asarray(groups)
    past = np.asarray(past, dtype=np.float64)
    if (
        family.shape != (2, 3, 128)
        or groups.shape != (3,)
        or groups.dtype != np.bool_
        or past.shape != (3, 128)
        or not np.isfinite(family).all()
        or not np.isfinite(past).all()
    ):
        raise ValueError("finite paired family gradients and three reached rows required")
    directions = np.zeros((2, 128), dtype=np.float64)
    if groups.sum() < 2:
        return directions
    active = mix.orthogonal_columns(past)
    for arm in range(2):
        descent = -mech.unit_average(family[arm, groups])
        v = descent - active @ (active.T @ descent)
        size = np.linalg.norm(v)
        if size > 1e-10:
            directions[arm] = RADIUS * v / size
    return directions


def choose_states(base, x, available, native, release, family, groups):
    """One common feasible grid radius across two arms, signs and reached times."""
    zero = np.zeros((2, 128), dtype=np.float32)
    if np.count_nonzero(groups) < 2:
        return zero, 0.0, 0.0
    reached = np.array([release - 16, release - 8, release], dtype=np.int64)
    if reached[0] < 0 or reached[-1] >= len(native):
        raise ValueError("three already-reached risk indices required")
    xp, ap = x[reached].contiguous(), available[reached].contiguous()
    past = mix.gradient(base, xp, ap)
    directions = ambient_directions(family, groups, past)
    if (np.linalg.norm(directions, axis=1) <= 1e-7).any():
        return zero, 0.0, 0.0
    u = torch.as_tensor(directions, dtype=x.dtype, device=x.device)
    scales = u.new_tensor(bank.tr.SCALES)
    signs = u.new_tensor([1.0, -1.0])
    vectors = (
        scales[:, None, None, None, None]
        * u[None, :, None, None, :]
        * signs[None, None, :, None, None]
    ).expand(-1, -1, -1, 3, -1)
    shape = vectors.shape[:-1]
    xx = xp[None, None, None].expand(*shape, *xp.shape[1:]).reshape(-1, *xp.shape[1:])
    aa = ap[None, None, None].expand(*shape, *ap.shape[1:]).reshape(-1, *ap.shape[1:])
    with torch.no_grad():
        z = bank.old.risk(base, xx, aa, vectors.reshape(-1, 128))
    shift = z.reshape(*shape).detach().cpu().numpy() - native[reached]
    choice = mech.shared_grid(shift, LIMIT - 2e-5)
    if choice is None:
        return zero, 0.0, 0.0
    states = (u * scales[choice]).detach().cpu().numpy()
    drift = float(np.abs(shift[choice]).max())
    if drift > LIMIT or np.linalg.norm(states, axis=1).max() > RADIUS + 5e-7:
        raise ValueError("actual direct-forward trust check failed")
    return states, float(scales[choice]), drift


def trace_shot(base, raw):
    """Assemble all four label-blind traces; no outcome enters this function."""
    x, available = bank.prior.views(base, raw)
    native = bank.prior.scan(base, x, available).detach().cpu().numpy()
    if native.shape != (len(raw["times"]),) or not np.isfinite(native).all():
        raise ValueError("finite native logit for every raw time required")
    entries = schedule(len(native))
    targets = target_gradients(base, x, available, raw, entries)
    curves = {clock: np.repeat(native[:, None], 2, axis=1) for clock in CLOCKS}
    written = {clock: np.zeros(len(native), dtype=bool) for clock in CLOCKS}
    supported = {clock: np.zeros(len(native), dtype=bool) for clock in CLOCKS}
    active = {clock: np.zeros(len(native), dtype=bool) for clock in CLOCKS}
    events = []
    for j, entry in enumerate(entries):
        family = np.stack((targets["true"][j], targets["reverse"][j]))
        groups = targets["groups"][j]
        event = {**entry, "groups": int(np.count_nonzero(groups))}
        for clock in CLOCKS:
            release = entry[clock]
            ix = block_indices(len(native), release)
            states, scale, drift = np.zeros((2, 128), dtype=np.float32), 0.0, 0.0
            if len(ix):
                states, scale, drift = choose_states(
                    base, x, available, native, release, family, groups
                )
                supported[clock][ix] = np.count_nonzero(groups) >= 2
                active[clock][ix] = scale > 0
                if scale:
                    v = torch.as_tensor(states, dtype=x.dtype, device=x.device)
                    xx = x[ix].contiguous().repeat_interleave(2, dim=0)
                    aa = available[ix].contiguous().repeat_interleave(2, dim=0)
                    with torch.no_grad():
                        z = bank.old.risk(base, xx, aa, v.repeat(len(ix), 1))
                    block = z.detach().cpu().numpy().reshape(len(ix), 2)
                else:
                    block = np.repeat(native[ix, None], 2, axis=1)
                write_block(curves[clock], written[clock], release, block)
            event[clock + "_scale"] = scale
            event[clock + "_trust_drift"] = drift
        events.append(event)
    return dict(
        native=native, curves=curves, written=written, supported=supported,
        active=active, events=events,
    )


def score_shot(trace, times, down, horizon):
    """Only called after trace assembly and authoritative outcome reveal."""
    times = np.asarray(times)
    lead = np.round((down - times) * 1000, 9)
    positive = (lead > 10) & (lead <= horizon + 1e-6)
    eligible = np.flatnonzero(positive)
    outputs = {"original": trace["native"]}
    outputs.update(
        {f"{clock}_{arm}": trace["curves"][clock][:, k]
         for clock in CLOCKS for k, arm in enumerate(ARMS)}
    )
    metrics = {}
    for name, curve in outputs.items():
        alarm = first(times, curve, True, down, mech.TAU)
        peak_index = int(eligible[np.argmax(curve[eligible])]) if len(eligible) else None
        metrics[name] = dict(
            first_index=alarm["first"], lead_ms=alarm["lead_ms"], timely=alarm["timely"],
            canonical_peak_logit=float(curve[peak_index]) if peak_index is not None else None,
            canonical_peak_index=peak_index,
        )
    coverage = {"canonical_positive_points": int(positive.sum())}
    for clock in CLOCKS:
        index = np.flatnonzero(trace["active"][clock])
        coverage[clock] = dict(
            scheduled_positive_points=int((positive & trace["written"][clock]).sum()),
            supported_positive_points=int((positive & trace["supported"][clock]).sum()),
            action_positive_points=int((positive & trace["active"][clock]).sum()),
            nonzero_actions=sum(e[clock + "_scale"] > 0 for e in trace["events"]),
            first_action_index=int(index[0]) if len(index) else None,
            first_action_time_s=float(times[index[0]]) if len(index) else None,
            max_trust_drift=max((e[clock + "_trust_drift"] for e in trace["events"]), default=0.0),
        )
    per_release = []
    for event in trace["events"]:
        lo, hi = event["early"] + 1, min(event["late"], len(times) - 1)
        ix = np.arange(lo, max(lo, hi + 1))
        early_action = event["early_scale"] > 0
        per_release.append(dict(
            origin=event["origin"], maturity=event["maturity"], early=event["early"],
            late=event["late"], groups=event["groups"],
            early_scale=event["early_scale"], late_scale=event["late_scale"],
            newly_actionable_start=lo if len(ix) else None,
            newly_actionable_end=hi if len(ix) else None,
            newly_actionable_positive_points=int(positive[ix].sum()),
            early_new_gradient_action_positive_points=int(positive[ix].sum()) if early_action else 0,
            previous_late_action_positive_points=int((positive[ix] & trace["active"]["late"][ix]).sum()),
        ))
    return dict(metrics=metrics, coverage=coverage, per_release=per_release)


def locked_inputs(h):
    """Use the sealed TRAIN inventory, not a holdout cohort or event labels."""
    if h(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen release-and-increment protocol changed")
    receipt_path = PRIOR / "fit_completed.json"
    h(receipt_path)
    receipt = r.read(receipt_path)
    audit = PRIOR / "train_crossfit.json"
    if h(audit) != AUDIT_SHA256 or h(audit) != receipt["train_audit_sha256"]:
        raise ValueError("locked original TRAIN audit changed")
    ids = tuple(r.read(audit)["geometry"]["non_timely_shots"])
    table = PRIOR / "completed.json"
    protocol = PRIOR / "protocol.json"
    if (
        ids != LATE_IDS
        or h(table) != receipt["train_table_sha256"]
        or h(protocol) != r.read(table)["protocol_sha256"]
    ):
        raise ValueError("original six TRAIN late cases or table receipt changed")
    config = r.read(protocol)
    if (
        config["status"] != "LOCKED_OUTPUT_TRUST_TABLE"
        or config["threshold"] != r.THRESHOLD
        or config["logit_threshold"] != mech.TAU
        or config["limit"] != LIMIT
        or tuple(config["scales"]) != bank.tr.SCALES
        or config["source_state_sha256"] != receipt["source_state_sha256"]
    ):
        raise ValueError("frozen source and original fixed trust grid required")
    train = config["roles"]["train"]
    if len(train) != 1353 or len(set(train)) != 1353 or not set(ids) <= set(train):
        raise ValueError("original TRAIN split membership required")
    if h(INVENTORY) != config["input_sha256"][str(INVENTORY.resolve())]:
        raise ValueError("TRAIN raw-byte inventory changed")
    inventory = r.read(INVENTORY)["inventory"]
    if [item["shot"] for item in inventory] != train:
        raise ValueError("TRAIN inventory and frozen split differ")
    raw_hashes = {q["shot"]: q["sha256"] for q in inventory if q["shot"] in ids}
    if set(raw_hashes) != set(ids):
        raise ValueError("six original raw hashes required")
    parent = Path(r.DEFAULT_PARENT).resolve()
    if h(parent) != config["input_sha256"][str(parent)]:
        raise ValueError("qualified parent checkpoint changed")
    advanced = r.io23.JTEXT_ADVANCED_TIME_PATH.resolve()
    if h(advanced) != config["input_sha256"][str(advanced)]:
        raise ValueError("canonical J-TEXT warning-time source changed")
    for path in (
        Path(__file__), Path(h25.__file__), Path(geo.__file__), Path(mix.__file__),
        Path(mech.__file__), Path(bank.old.__file__), Path(bank.tr.__file__),
        Path(bank.prior.__file__), Path(bank.src.__file__),
        Path(bank.old.policy.__file__), Path(bank.old.latent.__file__),
        Path(bank.prior.features.__file__), Path(r.load_parent.__code__.co_filename),
        Path(r.state_digest.__code__.co_filename), Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        r.ROOT / "HDF5_OUTCOME_CONTRACT_20261103.md",
    ):
        digest = h(path)
        previous = config["input_sha256"].get(str(path.resolve()))
        if previous is not None and digest != previous:
            raise ValueError(f"frozen source dependency changed: {path}")
    return config, raw_hashes


def run(output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    h = Hashes()
    config, raw_hashes = locked_inputs(h)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    if (
        r.state_digest(base) != config["source_state_sha256"]
        or parent["threshold"] != r.THRESHOLD
        or config["threshold"] != r.THRESHOLD
        or RADIUS != mix.RADIUS
        or LIMIT != bank.tr.LIMIT
    ):
        raise ValueError("unchanged frozen source, threshold and trust limits required")
    sealed = []
    for sid in LATE_IDS:
        if h(r.io23.path_for("J-TEXT", sid)) != raw_hashes[sid]:
            raise ValueError(f"original TRAIN raw bytes changed: {sid}")
        raw = r.simple_data.read_shot("J-TEXT", sid)
        trace = trace_shot(base, raw)
        sealed.append((sid, raw["times"].copy(), trace))
        print(f"fixed label-blind trace {sid}", flush=True)
    # All traces are fixed before *any* HDF5 outcome metadata is opened.
    advance = r.simple_data.advanced_times()
    rows = []
    for sid, times, trace in sealed:
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        if not flag:
            raise ValueError(f"locked late case is not HDF5 disruptive: {sid}")
        horizon = r.io23.effective_horizon_ms(sid, advance)[0]
        scored = score_shot(trace, times, down, horizon)
        if scored["metrics"]["original"]["timely"]:
            raise ValueError(f"locked original late case is now timely: {sid}")
        rows.append(dict(shot=sid, raw_sha256=raw_hashes[sid], horizon_ms=horizon, **scored))
    if r.state_digest(base) != config["source_state_sha256"] or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("frozen model or its gradients changed")
    h.check()
    r.write_new(output, dict(
        status="TRAIN_ONLY_H25_RELEASE_TIMING_PILOT",
        warning="descriptive six-shot pilot; no assumed timing improvement",
        action="ambient 128-D nullspace of reached risk Jacobian; signed family-mean descent",
        clocks="origins 31+32k, H25 maturity o+25, release o+25 versus o+32",
        grid=list(bank.tr.SCALES), radius=RADIUS, trust_limit_logit=LIMIT,
        grid_acceptance_limit_logit=LIMIT - 2e-5,
        probability_threshold=r.THRESHOLD, logit_threshold=mech.TAU,
        input_sha256={str(path): value[1] for path, value in h.cache.items()},
        shots=rows, seconds=time.monotonic() - started,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new compact JSON file; must not exist")
    args = parser.parse_args()
    run(args.output)
