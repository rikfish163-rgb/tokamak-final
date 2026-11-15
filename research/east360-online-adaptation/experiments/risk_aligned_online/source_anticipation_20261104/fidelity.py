"""Sealed source-VAL post-query forecast fidelity, not adaptive risk validation."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..mask_separated_intra_20261103.run import state_digest
from ..native_token_probe_20261104.core import forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.data import load_records
from .data import _validate_records, annotate as annotate_val
from .design import ARMS, STRATA
from .evaluate import HEADS, VIEWS
from .train import load_windows

PLAN = io23.ROOT / "SOURCE_FORECAST_FIDELITY_PLAN_20261104.md"


def verify_manifest(root, manifest):
    """Check every listed top-level file without following an escaping symlink."""
    root = Path(root).resolve()
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("nonempty artifact hash manifest required")
    for name, expected in manifest.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in (".", "..")
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
        ):
            raise ValueError("safe artifact name and SHA-256 required")
        path = root / name
        if path.resolve().parent != root or not path.is_file() or sha(path) != expected:
            raise ValueError(f"artifact hash or path mismatch: {name}")
    return manifest


def verify_fit_design(fit, design):
    """Bind the selected packet, trained heads, source and archived producer."""
    fit, design = Path(fit).resolve(), Path(design).resolve()
    fit_done, design_done = read(fit / "completed.json"), read(design / "completed.json")
    if (
        fit_done.get("status") != "TRAINED_NOT_VALIDATED"
        or design_done.get("status") != "COMPLETED"
        or (design / "superseded.json").exists()
        or fit_done.get("source_and_native_head_unchanged") is not True
        or any(fit_done.get(k) != 0 for k in ("val_signal_reads", "test_signal_reads", "east_signal_reads"))
    ):
        raise ValueError("completed source-only fit and current design required")
    for root, done, required in (
        (fit, fit_done, {"protocol.json", "heads.pt", "code.zip"}),
        (design, design_done, {"protocol.json", "val_records.json", "code.zip"}),
    ):
        manifest = done.get("artifact_sha256")
        if not isinstance(manifest, dict) or not required <= set(manifest):
            raise ValueError("required completed artifacts missing")
        verify_manifest(root, manifest)
    protocol, design_protocol = read(fit / "protocol.json"), read(design / "protocol.json")
    if protocol.get("teaching_objective") not in ("cosine", "risk_utility"):
        raise ValueError("fit must explicitly declare cosine or risk_utility teaching_objective")
    if (
        protocol.get("source_sha256") != PARENT_SHA256
        or design_protocol.get("source_sha256") != PARENT_SHA256
        or protocol.get("source_parameter_updates") != 0
        or protocol.get("arms") != list(ARMS)
        or Path(protocol.get("design", "")).resolve() != design
        or sha(DEFAULT_PARENT) != PARENT_SHA256
    ):
        raise ValueError("qualified source, matched heads and original design required")
    inputs = protocol.get("input_sha256", {})
    def resolve(path):
        path = Path(path)
        return (path if path.is_absolute() else io23.ROOT / path).resolve()
    bound = {resolve(path): digest for path, digest in inputs.items()}
    if bound.get(design / "completed.json") != sha(design / "completed.json") or bound.get(DEFAULT_PARENT.resolve()) != PARENT_SHA256:
        raise ValueError("fit is not bound to this design and qualified source")
    producer = protocol.get("code_sha256")
    if not isinstance(producer, dict) or not producer:
        raise ValueError("archived fit producer hashes required")
    with zipfile.ZipFile(fit / "code.zip") as archive:
        for name, expected in producer.items():
            try:
                member = str(Path(name).relative_to(io23.ROOT))
                contents = archive.read(member)
            except (KeyError, ValueError) as exc:
                raise ValueError("archived fit producer path missing or outside workspace") from exc
            if hashlib.sha256(contents).hexdigest() != expected:
                raise ValueError("archived fit producer hash mismatch")
    return protocol


def validate_val_records(records, archived, allowed_ids):
    """Require exactly the hash-locked source VAL coordinates and HDF annotations."""
    if len(records) != 946 or len(archived) != 946 or len({r["shot"] for r in records}) != 145:
        raise ValueError("fixed 946 source VAL episodes on 145 shots required")
    _validate_records(records, allowed_ids, "val")
    original = {(r["shot"], r["query"]): r for r in archived}
    if len(original) != 946 or set(original) != {(r["shot"], r["query"]) for r in records}:
        raise ValueError("source VAL episode coordinates differ from locked archive")
    for row in records:
        key = row["shot"], row["query"]
        try:
            expected = annotate_val(
                original[key],
                disruptive=row["hdf5_disruptive"],
                down=row["hdf5_down_time"],
                query_time=row["hdf5_query_time"],
                horizon_ms=row["source_horizon_ms"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("complete HDF-derived VAL annotations required") from exc
        if row != expected or row["anticipation_stratum"] not in STRATA or not row["hdf5_source_point_valid"]:
            raise ValueError("VAL source or HDF-derived annotation mismatch")


def missing_future_points(query, samples):
    if type(query) is not int or type(samples) is not int or query < 127 or query >= samples:
        raise ValueError("valid query and recorded shot length required")
    return max(0, query + 26 - samples)


def future_target(signal, query, view):
    """Full q+1..q+25 truth with the native matched availability rule."""
    if view not in VIEWS or set(signal) != {"values", "available"}:
        raise ValueError("declared view and signal-only arrays required")
    values, available = np.asarray(signal["values"]), np.asarray(signal["available"])
    if (
        values.ndim != 2
        or values.shape[1:] != (23,)
        or values.dtype.kind not in "iuf"
        or available.shape != values.shape
        or available.dtype != np.bool_
    ):
        raise ValueError("numeric Nx23 values and matching boolean mask required")
    missing = missing_future_points(query, len(values))
    if missing:
        return dict(eligible=False, missing_future_points=missing, truth=None, mask=None, hold=None)
    observed = available.copy()
    if view == "no_MNM_CIII":
        observed[:, [1, 2]] = False
    result, mask = targets(dict(values=values, available=observed), query, query + 25)
    return dict(
        eligible=True,
        missing_future_points=0,
        truth=result["real"][0],
        mask=mask[0],
        hold=result["hold"][0],
    )


def _coordinates(values, center, scale):
    return np.arcsinh((np.asarray(values, dtype=np.float64) - center) / scale)


def forecast_pack(base, heads, pack, location, records, lengths):
    """Forecast only from query windows; omit records without 25 future indices."""
    if set(heads) != set(HEADS) or heads["native_reference"] is not None:
        raise ValueError("native source head and exactly three frozen heads required")
    if not {"xq", "aq"} <= set(pack):
        raise ValueError("query prefix values and availability required")
    indices = [i for i, r in enumerate(records) if not missing_future_points(r["query"], lengths[r["shot"]])]
    forecasts = np.empty((len(VIEWS), len(HEADS), len(indices), 25, 23), dtype=np.float32)
    holds = np.empty((len(VIEWS), len(indices), 25, 23), dtype=np.float64)
    center = base.center.detach().cpu().numpy().astype(np.float64)
    scale = base.scale.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("finite source center and positive scale required")
    with torch.no_grad():
        for j, index in enumerate(indices):
            x = pack["xq"][[location[index]]]
            a = pack["aq"][[location[index]]]
            for vi, view in enumerate(VIEWS):
                mask = a.clone()
                if view == "no_MNM_CIII":
                    mask[:, :, [1, 2]] = False
                last = x[0, -1].detach().cpu().numpy().astype(np.float64)
                available = mask[0, -1].detach().cpu().numpy()
                last = np.where(available, last, center)
                holds[vi, j] = np.broadcast_to(_coordinates(last, center, scale), (25, 23))
                zero = base.center.new_zeros(1, 128)
                for hi, name in enumerate(HEADS):
                    predicted = forward(base, x, mask, zero, decoder=heads[name])["forecast"]
                    forecasts[vi, hi, j] = predicted[0].detach().cpu().numpy()
    if not np.isfinite(forecasts).all() or not np.isfinite(holds).all():
        raise FloatingPointError("nonfinite sealed forecast or hold baseline")
    return dict(indices=np.asarray(indices, dtype=np.int64), forecasts=forecasts, hold=holds)


def seal_forecasts(output, bundle):
    """Persist forecasts before any post-query truth is consumed."""
    if not isinstance(bundle, dict) or set(bundle) != {"indices", "forecasts", "hold"}:
        raise ValueError("forecast bundle requires indices, forecasts and hold")
    indices, forecasts, holds = (np.asarray(bundle[k]) for k in ("indices", "forecasts", "hold"))
    if (
        indices.ndim != 1
        or indices.dtype != np.int64
        or (indices < 0).any()
        or (np.diff(indices) <= 0).any()
        or forecasts.shape != (len(VIEWS), len(HEADS), len(indices), 25, 23)
        or holds.shape != (len(VIEWS), len(indices), 25, 23)
        or forecasts.dtype.kind != "f"
        or holds.dtype.kind != "f"
        or not np.isfinite(forecasts).all()
        or not np.isfinite(holds).all()
    ):
        raise ValueError("finite forecast bundle with ordered unique indices required")
    output = Path(output)
    path = output / "forecasts.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, **bundle)
    seal = dict(
        forecast_sha256=sha(path),
        complete_coordinates=len(bundle["indices"]),
        heads=list(HEADS),
        views=list(VIEWS),
    )
    write(output / "prediction_seal.json", seal)
    return seal


def score_window(forecasts, hold, truth, mask, center, scale):
    """Mean SmoothL1 over observed scalar positions in source coordinates."""
    forecasts = np.asarray(forecasts, dtype=np.float64)
    hold = np.asarray(hold, dtype=np.float64)
    truth, mask = np.asarray(truth), np.asarray(mask)
    center, scale = np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if (
        forecasts.shape != (len(HEADS), 25, 23)
        or hold.shape != (25, 23)
        or truth.shape != (25, 23)
        or mask.shape != truth.shape
        or mask.dtype != np.bool_
        or center.shape != (23,)
        or scale.shape != (23,)
        or not np.isfinite(forecasts).all()
        or not np.isfinite(hold).all()
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or not (scale > 0).all()
        or not np.isfinite(truth[mask]).all()
    ):
        raise ValueError("finite forecasts, observed targets and source coordinates required")
    count = int(mask.sum())
    if not count:
        return dict(scorable=False, observed_values=0, loss=None, hold_loss=None)
    target = _coordinates(np.where(mask, truth, center[None, :]), center, scale)
    def smooth(error):
        absolute = np.abs(error)
        return np.where(absolute < 1, 0.5 * absolute**2, absolute - 0.5)
    errors = smooth(forecasts[:, mask] - target[mask][None, :]).mean(axis=1)
    hold_loss = float(smooth(hold[mask] - target[mask]).mean())
    if not np.isfinite(errors).all() or not math.isfinite(hold_loss):
        raise FloatingPointError("nonfinite observed forecast loss")
    return dict(scorable=True, observed_values=count, loss=dict(zip(HEADS, map(float, errors))), hold_loss=hold_loss)


def score_sealed(output, records, lengths, center, scale, read_signal):
    """Verify sealed forecasts before calling the supplied post-query reader."""
    output = Path(output)
    path = output / "forecasts.npz"
    seal = read(output / "prediction_seal.json")
    if sha(path) != seal.get("forecast_sha256") or seal.get("heads") != list(HEADS) or seal.get("views") != list(VIEWS):
        raise ValueError("forecast seal mismatch")
    expected = [i for i, r in enumerate(records) if not missing_future_points(r["query"], lengths[r["shot"]])]
    with np.load(path, allow_pickle=False) as sealed:
        indices = sealed["indices"]
        forecasts = sealed["forecasts"]
        holds = sealed["hold"]
    if (
        indices.dtype != np.int64
        or indices.tolist() != expected
        or seal.get("complete_coordinates") != len(expected)
        or forecasts.shape != (len(VIEWS), len(HEADS), len(expected), 25, 23)
        or holds.shape != (len(VIEWS), len(expected), 25, 23)
        or not np.isfinite(forecasts).all()
        or not np.isfinite(holds).all()
    ):
        raise ValueError("sealed forecast support, shape or finite values invalid")
    indexed = {index: j for j, index in enumerate(expected)}
    rows = [dict(index=i, shot=r["shot"], origin=r.get("origin"), query=r["query"],
                 hdf_disruptive=r.get("hdf5_disruptive"), stratum=r["anticipation_stratum"],
                 eligible=i in indexed, missing_future_points=missing_future_points(r["query"], lengths[r["shot"]]),
                 prediction_index=indexed.get(i), views=None) for i, r in enumerate(records)]
    by_shot = defaultdict(list)
    for i in expected:
        by_shot[records[i]["shot"]].append(i)
    center, scale = np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    for sid in sorted(by_shot):
        signal = read_signal(sid)
        if len(signal["values"]) != lengths[sid]:
            raise ValueError("post-seal shot length differs from prediction preparation")
        for i in by_shot[sid]:
            j, query = indexed[i], records[i]["query"]
            views = {}
            for vi, view in enumerate(VIEWS):
                target = future_target(signal, query, view)
                mask = target["mask"]
                hold_target = _coordinates(target["hold"], center, scale)
                if not np.array_equal(holds[vi, j][mask], hold_target[mask]):
                    raise ValueError("sealed hold baseline differs from query observation")
                views[view] = score_window(
                    forecasts[vi, :, j], holds[vi, j], target["truth"], mask, center, scale
                )
            rows[i]["views"] = views
    if sha(path) != seal["forecast_sha256"]:
        raise ValueError("forecast seal changed during scoring")
    return rows


def read_future_shot(sid, allowed_ids, raw_hashes, lengths):
    """Read one already-sealed VAL shot only after checking its archived hash."""
    if sid not in allowed_ids or str(sid) not in raw_hashes or sid not in lengths:
        raise ValueError("only declared source VAL shots may be read")
    path = io23.path_for("J-TEXT", sid)
    stat = path.stat()
    if sha(path) != raw_hashes[str(sid)]:
        raise ValueError("VAL raw file hash changed before future scoring")
    raw = apply_input_contract(simple_data.read_shot("J-TEXT", sid))
    if (
        len(raw["values"]) != lengths[sid]
        or (path.stat().st_size, path.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns)
        or sha(path) != raw_hashes[str(sid)]
    ):
        raise ValueError("VAL raw file hash or length changed during future scoring")
    return {key: raw[key] for key in ("values", "available")}


def summarize(rows):
    """Average observed-scalar loss within windows, then shots, then strata."""
    result = {}
    for view in VIEWS:
        result[view] = {}
        for head in HEADS:
            result[view][head] = {}
            for group in ("overall", *STRATA):
                selected = [r for r in rows if r["eligible"] and (group == "overall" or r["stratum"] == group)
                            and r["views"][view]["loss"] is not None]
                by_shot = defaultdict(list)
                for row in selected:
                    by_shot[row["shot"]].append(row["views"][view])
                def macro(fn):
                    return float(np.mean([np.mean([fn(x) for x in values]) for values in by_shot.values()])) if by_shot else None
                result[view][head][group] = dict(
                    shots=len(by_shot), windows=len(selected),
                    observed_values=sum(r["views"][view]["observed_values"] for r in selected),
                    mean_loss=macro(lambda v: v["loss"][head]),
                    delta_vs_native=macro(lambda v: v["loss"][head] - v["loss"]["native_reference"]),
                    hold_loss=macro(lambda v: v["hold_loss"]),
                    delta_vs_hold=macro(lambda v: v["loss"][head] - v["hold_loss"]),
                )
    return result


def main(fit, design, output):
    """Production entry: fixed source VAL, sealed predictions, then truth."""
    fit, design, output = Path(fit).resolve(), Path(design).resolve(), Path(output).resolve()
    if output.exists() or any(output.is_relative_to(p) for p in (fit, design, io23.DATA_ROOT.resolve())):
        raise ValueError("new output directory outside input artifacts and raw data required")
    fit_protocol = verify_fit_design(fit, design)
    records = read(design / "val_records.json")
    archived, allowed, provenance = load_records("val")
    validate_val_records(records, archived, allowed)
    shot_ids = {r["shot"] for r in records}
    if len(shot_ids) != 145:
        raise ValueError("exactly 145 fixed source VAL shots required")

    configure()
    started = time.monotonic()
    base, _ = load_parent(DEFAULT_PARENT)
    base.cuda()
    saved = torch.load(fit / "heads.pt", map_location="cuda", weights_only=True)
    if saved.get("source_sha256") != PARENT_SHA256 or set(saved.get("heads", {})) != set(ARMS):
        raise ValueError("three hash-bound heads from the qualified source required")
    heads = {"native_reference": None}
    for name in ARMS:
        head = deepcopy(base.forecast_head)
        head.load_state_dict(saved["heads"][name], strict=True)
        head.requires_grad_(False).eval()
        if not all(torch.isfinite(p).all() for p in head.parameters()):
            raise ValueError("all frozen diagnostic head parameters must be finite")
        heads[name] = head
    del saved
    before = {name: state_digest(base if head is None else head) for name, head in heads.items()}
    code = imported_code_hashes()
    own_file = Path(__file__).resolve()
    code[str(own_file)] = sha(own_file)
    bound = {str(path): sha(path) for path in (fit / "completed.json", design / "completed.json", DEFAULT_PARENT, PLAN)}
    phase = {"name": "prepare"}
    calls = {"prepare": [], "score": []}
    lengths = {}
    reader = io23.read_signals

    def checked(device, sid, *args, **kwargs):
        if device != "J-TEXT" or sid not in shot_ids:
            raise ValueError("only the fixed 145 J-TEXT VAL signal shots are allowed")
        if phase["name"] == "score" and not (output / "prediction_seal.json").is_file():
            raise RuntimeError("future scoring is forbidden before the forecast seal")
        raw = reader(device, sid, *args, **kwargs)
        calls[phase["name"]].append(sid)
        length = len(raw["times"])
        if sid in lengths and lengths[sid] != length:
            raise ValueError("source VAL shot length changed")
        lengths[sid] = length
        return raw

    with (
        patch.object(io23, "read_signals", side_effect=checked),
        patch.object(io23, "read_outcome", side_effect=RuntimeError("outcomes are not read in forecast fidelity")),
    ):
        pack, location, raw_hashes = load_windows(
            records, [dict(indices=list(range(len(records))))], base.center.device, role="val"
        )
        if (
            set(lengths) != shot_ids
            or Counter(calls["prepare"]) != Counter({sid: 1 for sid in shot_ids})
            or set(raw_hashes) != {str(sid) for sid in shot_ids}
            or any(raw_hashes[str(sid)] != provenance["input_hashes"][str(sid)] for sid in shot_ids)
        ):
            raise AssertionError("exactly the 145 hash-bound VAL shots are required")
        output.mkdir(parents=True, exist_ok=False)
        protocol = dict(
            scope="SOURCE_VAL_POSTQUERY_FIDELITY_NOT_ADAPTIVE_FORECAST_OR_RISK_GATE",
            fit=str(fit), design=str(design), fit_completion_sha256=bound[str(fit / "completed.json")],
            design_completion_sha256=bound[str(design / "completed.json")],
            source_sha256=PARENT_SHA256, plan_sha256=bound[str(PLAN)],
            source_role="val", records=len(records), source_val_signal_shots=len(shot_ids),
            views=list(VIEWS), heads=list(HEADS), strata=list(STRATA), horizon=25,
            forecast="direct source-coordinate H25 at q with zero native fast offset; no inner update",
            target="q+1..q+25; native targets() availability incl. q and reversed support, mode index10 excluded",
            scoring="float64 source-coordinate asinh SmoothL1; observed scalar mean then per-shot then stratum",
            hold="q last observed value transformed to identical source coordinates; same target mask",
            eligibility="full 25 future sample indices required; incomplete records have no sealed prediction",
            future_truth_access="scoring reader called only after forecast NPZ hash seal",
            reader_scope="whole ended-shot arrays loaded; model receives only <=q prefix, not literal streaming IO",
            current_outcomes_read=False, source_updates=0, head_updates=0, test_signal_reads=0, east_signal_reads=0,
            adaptive_forecast_evaluated=False, risk_gate_changed=False, operational_gain_measured=False,
            fit_teaching_objective=fit_protocol["teaching_objective"],
            raw_sha256=raw_hashes, input_sha256=bound, code_sha256=code,
            torch_version=torch.__version__, numpy_version=np.__version__, device=torch.cuda.get_device_name(),
        )
        write(output / "protocol.json", protocol)
        with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for name in code:
                path = Path(name)
                archive.write(path, str(path.relative_to(io23.ROOT)))
            archive.write(PLAN, PLAN.name)
        bundle = forecast_pack(base, heads, pack, location, records, lengths)
        seal_forecasts(output, bundle)
        del bundle, pack
        phase["name"] = "score"
        rows = score_sealed(
            output, records, lengths, base.center.detach().cpu().numpy(), base.scale.detach().cpu().numpy(),
            lambda sid: read_future_shot(sid, shot_ids, raw_hashes, lengths),
        )

    coverage = {}
    for group in ("overall", *STRATA):
        subset = [r for r in rows if group == "overall" or r["stratum"] == group]
        coverage[group] = dict(
            records=len(subset), shots=len({r["shot"] for r in subset}),
            complete_horizon=sum(r["eligible"] for r in subset),
            incomplete_horizon=sum(not r["eligible"] for r in subset),
            missing_future_points=dict(sorted(Counter(str(r["missing_future_points"]) for r in subset if not r["eligible"]).items())),
            scorable_by_view={view: sum(r["eligible"] and r["views"][view]["scorable"] for r in subset) for view in VIEWS},
        )
    write(output / "rows.json", rows)
    write(output / "summary.json", dict(
        groups=summarize(rows), coverage=coverage, source_forecast_fidelity_only=True,
        adaptive_forecast_evaluated=False, risk_gate_changed=False, operational_online_gain_measured=False,
    ))
    after = {name: state_digest(base if head is None else head) for name, head in heads.items()}
    unchanged = before == after
    if not unchanged or any(p.grad is not None for model in (base, *(h for h in heads.values() if h is not None)) for p in model.parameters()):
        raise AssertionError("frozen source and diagnostic heads must remain unchanged and gradient-free")
    if any(sha(Path(path)) != digest for path, digest in {**bound, **code}.items()):
        raise AssertionError("bound inputs or current producer code changed")
    seal = read(output / "prediction_seal.json")
    if sha(output / "forecasts.npz") != seal["forecast_sha256"]:
        raise AssertionError("forecast seal changed during scoring")
    done = dict(
        status="COMPLETED_SOURCE_VAL_FIDELITY", seconds=time.monotonic() - started,
        source_and_heads_unchanged=True, state_sha256_before=before, state_sha256_after=after,
        source_updates=0, head_updates=0,
        source_val_prepare_signal_shots=len(calls["prepare"]),
        source_val_postseal_signal_shots=len(calls["score"]),
        test_signal_reads=0, east_signal_reads=0, records=len(rows),
        complete_horizon=coverage["overall"]["complete_horizon"],
        incomplete_horizon=coverage["overall"]["incomplete_horizon"],
        adaptive_forecast_evaluated=False, risk_gate_changed=False, operational_online_gain_measured=False,
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "completed.json", done)
    print({k: done[k] for k in ("status", "records", "complete_horizon", "incomplete_horizon")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.fit, args.design, args.output)
