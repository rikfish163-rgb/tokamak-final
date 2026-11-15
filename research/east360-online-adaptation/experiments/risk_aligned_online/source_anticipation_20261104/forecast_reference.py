"""Sealed source-VAL fidelity of the existing forecast-only decoder."""

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

from . import fidelity as f

PRIOR_SHA256 = "cdde96086517cbc468a911bbb5926dde1c783b3711b0a8abc09caf0a6d2e6730"
TRIAL_SHA256 = "5eb353e131f081fb984825dc97fb1956b6d42964252a82ba17750426792a5c33"
TRIAL_NAME = "experiments/risk_aligned_online/direct_decoder_meta_20261102/trial.py"
PLAN = f.io23.ROOT / "FORECAST_ONLY_REFERENCE_PLAN_20261104.md"


def validate_prior_claims(done, protocol):
    if (
        done.get("status") != "COMPLETED"
        or done.get("source_unchanged") is not True
        or any(done.get(k) != 0 for k in ("east_reads", "test_signals_read"))
        or protocol.get("scope") != "SOURCE_ONLY_DIRECT_DIAGNOSTIC_DECODER_NOT_JTEXT_ONLINE"
        or protocol.get("source_sha256") != f.PARENT_SHA256
        or any(protocol.get(k) != 0 for k in ("source_parameters_updated", "east_reads", "test_signals_read"))
        or "forecast_only" not in protocol.get("arms", [])
        or protocol.get("absolute_maintenance_weight") != 0.2
        or protocol.get("initial_parameter_anchor_weight") != 1e-4
    ):
        raise ValueError("completed source-only forecast prior and frozen source required")


def _verify_archive(root, archive_name, hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("archived producer hashes required")
    with zipfile.ZipFile(root / archive_name) as archive:
        for name, digest in hashes.items():
            try:
                member = str(Path(name).relative_to(f.io23.ROOT))
                contents = archive.read(member)
            except (KeyError, ValueError) as exc:
                raise ValueError("archived producer file missing or outside workspace") from exc
            if hashlib.sha256(contents).hexdigest() != digest:
                raise ValueError("archived producer hash mismatch")


def verify_prior(prior):
    prior = Path(prior).resolve()
    done, protocol = f.read(prior / "completed.json"), f.read(prior / "protocol.json")
    validate_prior_claims(done, protocol)
    manifest = done.get("artifact_sha256", {})
    if not {"protocol.json", "decoders.pt", "code_snapshot.zip"} <= set(manifest):
        raise ValueError("complete forecast prior manifest required")
    f.verify_manifest(prior, manifest)
    if manifest["decoders.pt"] != PRIOR_SHA256 or f.sha(prior / "decoders.pt") != PRIOR_SHA256:
        raise ValueError("forecast-only prior checkpoint hash mismatch")
    producer = protocol.get("bound_code_sha256", {})
    trial = str(f.io23.ROOT / TRIAL_NAME)
    if producer.get(trial) != TRIAL_SHA256:
        raise ValueError("forecast-only archived trial identity mismatch")
    # This hash identifies the inspected combine() branch: forecast_only uses
    # 0.20*diagnostic loss plus 1e-4*initial anchor; risk enters only other arms.
    _verify_archive(prior, "code_snapshot.zip", producer)
    return protocol


def validate_reference_claims(done, protocol, design_hash):
    before, after = done.get("state_sha256_before"), done.get("state_sha256_after")
    raw = protocol.get("raw_sha256")
    if (
        done.get("status") != "COMPLETED_SOURCE_VAL_FIDELITY"
        or done.get("source_and_heads_unchanged") is not True
        or any(done.get(k) != 0 for k in ("source_updates", "head_updates", "test_signal_reads", "east_signal_reads"))
        or (done.get("records"), done.get("complete_horizon"), done.get("incomplete_horizon")) != (946, 764, 182)
        or not isinstance(before, dict)
        or set(before) != set(f.HEADS)
        or before != after
        or protocol.get("scope") != "SOURCE_VAL_POSTQUERY_FIDELITY_NOT_ADAPTIVE_FORECAST_OR_RISK_GATE"
        or protocol.get("source_sha256") != f.PARENT_SHA256
        or protocol.get("source_role") != "val"
        or protocol.get("horizon") != 25
        or protocol.get("records") != 946
        or protocol.get("heads") != list(f.HEADS)
        or protocol.get("views") != list(f.VIEWS)
        or protocol.get("design_completion_sha256") != design_hash
        or protocol.get("fit_teaching_objective") != "risk_utility"
        or not isinstance(raw, dict)
        or len(raw) != 145
        or any(protocol.get(k) is not False for k in ("adaptive_forecast_evaluated", "risk_gate_changed", "operational_gain_measured"))
    ):
        raise ValueError("reference source, design, frozen states or VAL coverage mismatch")


def verify_reference(reference, design):
    """Bind prior fidelity artifacts and reuse only its fixed VAL support."""
    reference, design = Path(reference).resolve(), Path(design).resolve()
    done, design_done = f.read(reference / "completed.json"), f.read(design / "completed.json")
    if design_done.get("status") != "COMPLETED" or (design / "superseded.json").exists():
        raise ValueError("current completed source design required")
    for root, manifest, required in (
        (reference, done.get("artifact_sha256", {}), {"protocol.json", "rows.json", "forecasts.npz", "prediction_seal.json", "code.zip"}),
        (design, design_done.get("artifact_sha256", {}), {"protocol.json", "val_records.json", "code.zip"}),
    ):
        if not isinstance(manifest, dict) or not required <= set(manifest):
            raise ValueError("reference/design artifact manifest incomplete")
        f.verify_manifest(root, manifest)
    protocol = f.read(reference / "protocol.json")
    validate_reference_claims(done, protocol, f.sha(design / "completed.json"))
    if f.read(design / "protocol.json").get("source_sha256") != f.PARENT_SHA256:
        raise ValueError("reference design source mismatch")
    _verify_archive(reference, "code.zip", protocol.get("code_sha256"))
    relevant = (
        "source_anticipation_20261104/fidelity.py", "source_anticipation_20261104/train.py",
        "dual_history_ttt_20261101/stream.py", "causal_input.py", "simple_data.py", "io23.py",
        "native_token_probe_20261104/core.py", "simple23.py",
    )
    for name in relevant:
        path = f.io23.ROOT / "experiments/risk_aligned_online" / name
        if protocol["code_sha256"].get(str(path)) != f.sha(path):
            raise ValueError(f"reference scoring or source path changed: {name}")
    seal = f.read(reference / "prediction_seal.json")
    if (
        seal.get("forecast_sha256") != f.sha(reference / "forecasts.npz")
        or seal.get("complete_coordinates") != 764
        or seal.get("heads") != list(f.HEADS)
        or seal.get("views") != list(f.VIEWS)
    ):
        raise ValueError("reference forecast seal mismatch")
    with np.load(reference / "forecasts.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"indices", "forecasts", "hold"}:
            raise ValueError("reference forecast archive schema mismatch")
        old = {name: saved[name] for name in saved.files}
    if (
        old["forecasts"].shape != (2, 4, 764, 25, 23)
        or old["hold"].shape != (2, 764, 25, 23)
        or not np.isfinite(old["forecasts"]).all()
        or not np.isfinite(old["hold"]).all()
    ):
        raise ValueError("reference finite forecast shape mismatch")
    records, old_rows = f.read(design / "val_records.json"), f.read(reference / "rows.json")
    archived, allowed, provenance = f.load_records("val")
    f.validate_val_records(records, archived, allowed)
    validate_reference_rows(records, old_rows, old["indices"])
    if {str(r["shot"]) for r in records} != set(protocol["raw_sha256"]):
        raise ValueError("reference VAL shot hashes do not match fixed records")
    return dict(protocol=protocol, completed=done, records=records, rows=old_rows, old=old,
                provenance=provenance, allowed=allowed)


def validate_reference_rows(records, rows, indices):
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype != np.int64 or len(rows) != len(records):
        raise ValueError("reference coordinate support mismatch")
    complete = [i for i, row in enumerate(rows) if row.get("eligible") is True]
    if indices.tolist() != complete:
        raise ValueError("reference prediction indices mismatch")
    positions = {index: j for j, index in enumerate(complete)}
    for i, (record, row) in enumerate(zip(records, rows)):
        if (
            row.get("index") != i
            or row.get("shot") != record["shot"]
            or row.get("origin") != record["origin"]
            or row.get("query") != record["query"]
            or row.get("stratum") != record["anticipation_stratum"]
            or row.get("hdf_disruptive") != record["hdf5_disruptive"]
            or type(row.get("eligible")) is not bool
            or row.get("prediction_index") != positions.get(i)
            or (row["eligible"] and row.get("missing_future_points") != 0)
            or (not row["eligible"] and (row.get("views") is not None or not 1 <= row.get("missing_future_points", 0) <= 25))
        ):
            raise ValueError("reference coordinate or eligibility mismatch")


def forecast_pack(base, decoder, pack, location, indices, reference_forecasts):
    """Produce only forecast_only output from query prefixes at zero fast offset."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype != np.int64 or not {"xq", "aq"} <= set(pack):
        raise ValueError("fixed indices and query prefix tensors required")
    if reference_forecasts.shape != (len(f.VIEWS), len(f.HEADS), len(indices), 25, 23):
        raise ValueError("matching reference source forecasts required")
    out = np.empty((len(f.VIEWS), len(indices), 25, 23), dtype=np.float32)
    maximum = 0.0
    with torch.no_grad():
        for j, index in enumerate(indices):
            x = pack["xq"][[location[int(index)]]]
            a = pack["aq"][[location[int(index)]]]
            zero = base.center.new_zeros(1, 128)
            for vi, view in enumerate(f.VIEWS):
                mask = a.clone()
                if view == "no_MNM_CIII":
                    mask[:, :, [1, 2]] = False
                native = f.forward(base, x, mask, zero)["forecast"][0].detach().cpu().numpy()
                maximum = max(maximum, float(np.max(np.abs(native.astype(np.float64) - reference_forecasts[vi, 0, j]))))
                out[vi, j] = f.forward(base, x, mask, zero, decoder=decoder)["forecast"][0].detach().cpu().numpy()
    if not np.isfinite(out).all() or not math.isfinite(maximum):
        raise FloatingPointError("finite single-head and source forecasts required")
    if maximum > 1e-6:
        raise ValueError("zero-fast native forecast differs from sealed reference")
    return dict(indices=indices.copy(), forecast_only=out), maximum


def seal_forecasts(output, bundle):
    output = Path(output)
    if not isinstance(bundle, dict) or set(bundle) != {"indices", "forecast_only"}:
        raise ValueError("single forecast_only bundle required")
    indices, forecasts = np.asarray(bundle["indices"]), np.asarray(bundle["forecast_only"])
    if (
        indices.ndim != 1
        or indices.dtype != np.int64
        or (indices < 0).any()
        or (np.diff(indices) <= 0).any()
        or forecasts.shape != (len(f.VIEWS), len(indices), 25, 23)
        or forecasts.dtype.kind != "f"
        or not np.isfinite(forecasts).all()
    ):
        raise ValueError("finite single-head forecast and unique ordered indices required")
    with (output / "forecasts.npz").open("xb") as handle:
        np.savez_compressed(handle, **bundle)
    seal = dict(forecast_sha256=f.sha(output / "forecasts.npz"), complete_coordinates=len(indices),
                heads=["forecast_only"], views=list(f.VIEWS))
    f.write(output / "prediction_seal.json", seal)
    return seal


def single_score(prediction, truth, mask, center, scale):
    prediction, truth, mask = np.asarray(prediction, dtype=np.float64), np.asarray(truth), np.asarray(mask)
    center, scale = np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if (
        prediction.shape != (25, 23)
        or truth.shape != (25, 23)
        or mask.shape != truth.shape
        or mask.dtype != np.bool_
        or center.shape != (23,)
        or scale.shape != (23,)
        or not np.isfinite(prediction).all()
        or not np.isfinite(truth[mask]).all()
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError("finite single forecast and source-coordinate targets required")
    if not mask.any():
        return None
    target = f._coordinates(np.where(mask, truth, center[None, :]), center, scale)
    error = np.abs(prediction[mask] - target[mask])
    loss = float(np.where(error < 1, 0.5 * error**2, error - 0.5).mean())
    if not math.isfinite(loss):
        raise FloatingPointError("nonfinite forecast-only loss")
    return loss


def score_sealed(output, records, old_rows, old, lengths, center, scale, read_signal):
    output = Path(output)
    seal = f.read(output / "prediction_seal.json")
    path = output / "forecasts.npz"
    if f.sha(path) != seal.get("forecast_sha256") or seal.get("heads") != ["forecast_only"] or seal.get("views") != list(f.VIEWS):
        raise ValueError("single forecast seal mismatch")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"indices", "forecast_only"}:
            raise ValueError("only the forecast_only head may be sealed")
        indices, forecasts = saved["indices"], saved["forecast_only"]
    validate_reference_rows(records, old_rows, indices)
    if (
        not np.array_equal(indices, old["indices"])
        or seal.get("complete_coordinates") != len(indices)
        or forecasts.shape != (len(f.VIEWS), len(indices), 25, 23)
        or not np.isfinite(forecasts).all()
        or old["forecasts"].shape != (len(f.VIEWS), len(f.HEADS), len(indices), 25, 23)
        or old["hold"].shape != (len(f.VIEWS), len(indices), 25, 23)
    ):
        raise ValueError("sealed single-head/reference coverage or shape mismatch")
    positions = {int(index): j for j, index in enumerate(indices)}
    rows, by_shot = [], defaultdict(list)
    for i, (record, old_row) in enumerate(zip(records, old_rows)):
        missing = f.missing_future_points(record["query"], lengths[record["shot"]])
        if missing != old_row["missing_future_points"] or (missing == 0) != old_row["eligible"]:
            raise ValueError("reference horizon coverage changed")
        rows.append(dict(index=i, shot=record["shot"], origin=record["origin"], query=record["query"],
                         stratum=record["anticipation_stratum"], eligible=not missing,
                         missing_future_points=missing, prediction_index=positions.get(i), views=None))
        if not missing:
            by_shot[record["shot"]].append(i)
    center, scale = np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    errors = dict(single_vs_existing_score_window=0.0, reference_recalculation=0.0)
    for sid in sorted(by_shot):
        signal = read_signal(sid)
        if len(signal["values"]) != lengths[sid]:
            raise ValueError("reference shot length changed")
        for i in by_shot[sid]:
            j = positions[i]
            views = {}
            for vi, view in enumerate(f.VIEWS):
                target = f.future_target(signal, records[i]["query"], view)
                mask = target["mask"]
                observed = int(mask.sum())
                old_cell = old_rows[i]["views"][view]
                recalculated = f.score_window(old["forecasts"][vi, :, j], old["hold"][vi, j],
                                              target["truth"], mask, center, scale)
                if observed != old_cell["observed_values"] or recalculated["scorable"] != old_cell["scorable"]:
                    raise ValueError("reference observed mask or coverage changed")
                if not observed:
                    views[view] = dict(scorable=False, observed_values=0, mean_loss=None,
                                       delta_vs_native=None, delta_vs_risk_taught=None, delta_vs_hold=None)
                    continue
                held = f._coordinates(target["hold"], center, scale)
                if not np.array_equal(old["hold"][vi, j][mask], held[mask]):
                    raise ValueError("reference hold baseline or mask changed")
                for hi, name in enumerate(f.HEADS):
                    single = single_score(old["forecasts"][vi, hi, j], target["truth"], mask, center, scale)
                    errors["single_vs_existing_score_window"] = max(errors["single_vs_existing_score_window"],
                                                                      abs(single - recalculated["loss"][name]))
                    errors["reference_recalculation"] = max(errors["reference_recalculation"],
                                                              abs(recalculated["loss"][name] - old_cell["loss"][name]))
                errors["reference_recalculation"] = max(errors["reference_recalculation"],
                                                          abs(recalculated["hold_loss"] - old_cell["hold_loss"]))
                if max(errors.values()) > 1e-12:
                    raise ValueError("reference per-window scoring or mask does not reproduce")
                score = single_score(forecasts[vi, j], target["truth"], mask, center, scale)
                views[view] = dict(
                    scorable=True, observed_values=observed, mean_loss=score,
                    delta_vs_native=score - old_cell["loss"]["native_reference"],
                    delta_vs_risk_taught=score - old_cell["loss"]["anticipation125"],
                    delta_vs_hold=score - old_cell["hold_loss"],
                )
            rows[i]["views"] = views
    if f.sha(path) != seal["forecast_sha256"]:
        raise ValueError("single forecast seal changed during scoring")
    return rows, errors


def summarize(rows):
    result = {}
    for view in f.VIEWS:
        result[view] = {}
        for group in ("overall", *f.STRATA):
            selected = [r for r in rows if r["eligible"] and (group == "overall" or r["stratum"] == group)
                        and r["views"][view]["mean_loss"] is not None]
            by_shot = defaultdict(list)
            for row in selected:
                by_shot[row["shot"]].append(row["views"][view])
            metrics = {}
            for key in ("mean_loss", "delta_vs_native", "delta_vs_risk_taught", "delta_vs_hold"):
                metrics[key] = float(np.mean([np.mean([v[key] for v in values]) for values in by_shot.values()])) if by_shot else None
            result[view][group] = dict(shots=len(by_shot), windows=len(selected), **metrics)
    return result


def main(reference, design, prior, output):
    """Produce one sealed forecast-only head on the matched source VAL support."""
    reference, design, prior, output = (Path(p).resolve() for p in (reference, design, prior, output))
    if output.exists() or any(output.is_relative_to(p) for p in (reference, design, prior, f.io23.DATA_ROOT.resolve())):
        raise ValueError("new output directory outside inputs and raw data required")
    if not PLAN.is_file():
        raise FileNotFoundError(f"comparison plan required before running: {PLAN}")
    prior_protocol = verify_prior(prior)
    context = verify_reference(reference, design)
    records, old_rows, old = context["records"], context["rows"], context["old"]
    reference_protocol = context["protocol"]
    if reference_protocol.get("plan_sha256") != f.sha(f.PLAN):
        raise ValueError("reference fidelity plan changed")
    shot_ids = {r["shot"] for r in records}
    if len(shot_ids) != 145 or len(old["indices"]) != 764:
        raise ValueError("fixed 145-shot, 764-complete source VAL support required")

    f.configure()
    started = time.monotonic()
    base, _ = f.load_parent(f.DEFAULT_PARENT)
    base.cuda()
    saved = torch.load(prior / "decoders.pt", map_location="cuda", weights_only=True)
    if saved.get("source_sha256") != f.PARENT_SHA256 or "forecast_only" not in saved.get("heads", {}):
        raise ValueError("hash-locked forecast_only source checkpoint required")
    decoder = deepcopy(base.forecast_head)
    decoder.load_state_dict(saved["heads"]["forecast_only"], strict=True)
    decoder.requires_grad_(False).eval()
    del saved
    if not all(torch.isfinite(p).all() for p in decoder.parameters()):
        raise ValueError("finite forecast_only head required")
    before = {"source": f.state_digest(base), "forecast_only": f.state_digest(decoder)}
    if before["source"] != context["completed"]["state_sha256_before"]["native_reference"]:
        raise ValueError("loaded source differs from frozen fidelity reference")
    code = f.imported_code_hashes()
    own_file = Path(__file__).resolve()
    code[str(own_file)] = f.sha(own_file)
    bound_paths = (reference / "completed.json", design / "completed.json", prior / "completed.json", f.DEFAULT_PARENT, f.PLAN, PLAN)
    bound = {str(path): f.sha(path) for path in bound_paths}
    reader = f.io23.read_signals
    phase = {"name": "prepare"}
    calls, lengths = {"prepare": [], "score": []}, {}

    def checked(device, sid, *args, **kwargs):
        if device != "J-TEXT" or sid not in shot_ids:
            raise ValueError("only the fixed 145 J-TEXT VAL signal shots are allowed")
        if phase["name"] == "score" and not (output / "prediction_seal.json").is_file():
            raise RuntimeError("future scoring is forbidden before the single-head seal")
        raw = reader(device, sid, *args, **kwargs)
        calls[phase["name"]].append(sid)
        length = len(raw["times"])
        if sid in lengths and lengths[sid] != length:
            raise ValueError("source VAL shot length changed")
        lengths[sid] = length
        return raw

    with (
        patch.object(f.io23, "read_signals", side_effect=checked),
        patch.object(f.io23, "read_outcome", side_effect=RuntimeError("no outcome reads in forecast computation")),
    ):
        pack, location, raw_hashes = f.load_windows(
            records, [dict(indices=list(range(len(records))))], base.center.device, role="val"
        )
        if (
            set(lengths) != shot_ids
            or Counter(calls["prepare"]) != Counter({sid: 1 for sid in shot_ids})
            or raw_hashes != reference_protocol["raw_sha256"]
            or any(raw_hashes[str(sid)] != context["provenance"]["input_hashes"][str(sid)] for sid in shot_ids)
        ):
            raise ValueError("source VAL raw hashes differ from the sealed reference")
        for record, previous in zip(records, old_rows):
            missing = f.missing_future_points(record["query"], lengths[record["shot"]])
            if missing != previous["missing_future_points"] or (missing == 0) != previous["eligible"]:
                raise ValueError("source VAL future coverage differs from the sealed reference")
        output.mkdir(parents=True, exist_ok=False)
        protocol = dict(
            scope="SOURCE_VAL_FORECAST_ONLY_POSTQUERY_FIDELITY_NOT_ADAPTIVE_OR_ALARM_GAIN",
            reference=str(reference), design=str(design), prior=str(prior),
            reference_completion_sha256=bound[str(reference / "completed.json")],
            design_completion_sha256=bound[str(design / "completed.json")],
            prior_completion_sha256=bound[str(prior / "completed.json")],
            comparison_plan_sha256=bound[str(PLAN)],
            source_sha256=f.PARENT_SHA256, prior_head_sha256=PRIOR_SHA256,
            prior_scope=prior_protocol["scope"], prior_head="forecast_only",
            prior_objective="absolute diagnostic loss plus initial head anchor; no risk term",
            views=list(f.VIEWS), heads=["forecast_only"], paired_risk_head="anticipation125",
            records=946, complete_horizon=764, incomplete_horizon=182, source_val_shots=145,
            target="reference-matched q+1..q+25; source targets() mask, mode10 excluded",
            scoring="float64 source-coordinate SmoothL1; within-window, then shot, then strata",
            mask_evidence="same raw hashes and frozen target code; count and all four prior losses recomputed; historical bitmask not archived",
            future_truth_access="only after the single-head forecast seal",
            reader_scope="whole ended-shot arrays loaded, model sees <=q prefix; not literal streaming IO",
            no_inner_update=True, source_updates=0, head_updates=0,
            test_signal_reads=0, east_signal_reads=0, operational_online_gain_measured=False,
            raw_sha256=raw_hashes, input_sha256=bound, code_sha256=code,
            torch_version=torch.__version__, numpy_version=np.__version__, device=torch.cuda.get_device_name(),
        )
        f.write(output / "protocol.json", protocol)
        with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for name in code:
                path = Path(name)
                archive.write(path, str(path.relative_to(f.io23.ROOT)))
            archive.write(f.PLAN, f.PLAN.name)
            archive.write(PLAN, PLAN.name)
        bundle, native_parity = forecast_pack(base, decoder, pack, location, old["indices"], old["forecasts"])
        seal_forecasts(output, bundle)
        del bundle, pack
        phase["name"] = "score"
        rows, errors = score_sealed(
            output, records, old_rows, old, lengths, base.center.detach().cpu().numpy(),
            base.scale.detach().cpu().numpy(),
            lambda sid: f.read_future_shot(sid, shot_ids, raw_hashes, lengths),
        )

    scorable = {view: sum(r["eligible"] and r["views"][view]["scorable"] for r in rows) for view in f.VIEWS}
    expected = {view: sum(r["eligible"] and r["views"][view]["scorable"] for r in old_rows) for view in f.VIEWS}
    if scorable != expected:
        raise ValueError("reference scorable coverage changed")
    f.write(output / "rows.json", rows)
    f.write(output / "summary.json", dict(
        groups=summarize(rows), records=len(rows), complete_horizon=764, incomplete_horizon=182,
        scorable_by_view=scorable, risk_taught_reference="anticipation125",
        zero_fast_source_val_forecast_only=True, adaptive_forecast_evaluated=False,
        risk_gate_changed=False, operational_online_gain_measured=False,
    ))
    after = {"source": f.state_digest(base), "forecast_only": f.state_digest(decoder)}
    if before != after or any(p.grad is not None for model in (base, decoder) for p in model.parameters()):
        raise AssertionError("source and forecast_only head must remain frozen")
    if any(f.sha(Path(path)) != digest for path, digest in {**bound, **code}.items()):
        raise AssertionError("bound inputs or producer code changed")
    if f.sha(reference / "forecasts.npz") != f.read(reference / "prediction_seal.json")["forecast_sha256"]:
        raise AssertionError("reference forecast seal changed")
    if f.sha(output / "forecasts.npz") != f.read(output / "prediction_seal.json")["forecast_sha256"]:
        raise AssertionError("forecast_only seal changed")
    done = dict(
        status="COMPLETED_SOURCE_VAL_FORECAST_ONLY_FIDELITY", seconds=time.monotonic()-started,
        source_and_head_unchanged=True, state_sha256_before=before, state_sha256_after=after,
        source_updates=0, head_updates=0, test_signal_reads=0, east_signal_reads=0,
        source_val_prepare_signal_shots=len(calls["prepare"]), source_val_postseal_signal_shots=len(calls["score"]),
        records=len(rows), complete_horizon=764, incomplete_horizon=182, scorable_by_view=scorable,
        native_source_max_abs_forecast_difference=native_parity, score_verification_max_errors=errors,
        adaptive_forecast_evaluated=False, risk_gate_changed=False, operational_online_gain_measured=False,
        artifact_sha256={p.name: f.sha(p) for p in output.iterdir() if p.is_file()},
    )
    f.write(output / "completed.json", done)
    print({k: done[k] for k in ("status", "records", "complete_horizon", "incomplete_horizon")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.reference, args.design, args.prior, args.output)
