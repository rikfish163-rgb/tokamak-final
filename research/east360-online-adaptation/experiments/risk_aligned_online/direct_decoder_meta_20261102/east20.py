"""Frozen source decoders on ended EAST20 local episodes; no EAST fitting."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..critical_shot_meta_20261102.objectives import margin_loss
from ..dual_history_ttt_20261101.run import load_model
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..event_conditional_feedback_20261102.pilot import summarize
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import encode, forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.data import SOURCE_SHA
from ..source_event_tasks_20261102.trial import source_basis
from ..warmstart_alignment_20261022.alignment import transform
from .core import advance, clone_head

PLAN = io23.ROOT / "DIRECT_DECODER_EAST20_PLAN_20261102.md"
SOURCE = io23.ROOT / "probe_outputs/direct_decoder_source_v1"
LOCAL = io23.ROOT / "probe_outputs/difference_feedback_probe20_v1"


def gates(s):
    candidate = s["forecast_only_real"]
    g = {
        name + "_improved": candidate[name]["mean_delta_bce"] < -1e-8
        for name in ("disruptive", "safe", "danger_window")
    }
    for name in ("original_real", "forecast_only_reverse", "forecast_only_hold"):
        g["better_than_" + name] = candidate["balanced"] < s[name]["balanced"] - 1e-8
    g["passed"] = all(g.values())
    return g


def main(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    source_done = read(SOURCE / "completed.json")
    p, local_done = read(LOCAL / "protocol.json"), read(LOCAL / "completed.json")
    if source_done["status"] != "COMPLETED" or not source_done["source_unchanged"]:
        raise ValueError("completed fixed source decoder trial required")
    for name, digest in source_done["artifact_sha256"].items():
        if sha(SOURCE / name) != digest:
            raise ValueError("source candidate artifact changed")
    if sha(LOCAL / "samples.json") != local_done["samples_sha256"]:
        raise ValueError("fixed local sample selection changed")
    model, configuration, cold = load_model()
    if sha(cold) != p["cold_state_sha256"] or p["source_sha256"] != SOURCE_SHA:
        raise ValueError("EAST parent/source changed")
    mapping_path = Path(p["calibration_reference"]) / "input_map.json"
    if sha(mapping_path) != p["input_map_sha256"]:
        raise ValueError("input mapping changed")
    mapping = read(mapping_path)
    ids = p["ids"]
    if mapping["calibration_ids"] != ids or len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError("fixed ended20 calibration roster required")
    all_samples = read(LOCAL / "samples.json")
    keys = ("shot", "query", "origin", "disruptive", "actual_lead_ms", "label")
    samples = [{k: r[k] for k in keys} for r in all_samples]
    if len(samples) != 100 or {r["shot"] for r in samples} != set(ids):
        raise ValueError("complete fixed 100 local queries required")
    basis, threshold = source_basis(model), float(configuration["threshold"])
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    saved = torch.load(SOURCE / "decoders.pt", map_location="cpu", weights_only=True)
    if saved["source_sha256"] != SOURCE_SHA:
        raise ValueError("decoder parent mismatch")
    heads = {"original": clone_head(model).eval().requires_grad_(False)}
    for arm, state in saved["heads"].items():
        head = clone_head(model)
        head.load_state_dict(state, strict=True)
        heads[arm] = head.eval().requires_grad_(False)
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    root.mkdir(parents=True)
    protocol = dict(
        scope="ENDED_EAST20_FROZEN_SOURCE_DECODER_LOCAL_DIAGNOSTIC",
        ids=ids,
        source_sha256=SOURCE_SHA,
        cold_sha256=sha(cold),
        input_map_sha256=sha(mapping_path),
        threshold=threshold,
        source_decoder_completed_sha256=sha(SOURCE / "completed.json"),
        source_decoder_gate=source_done["gate"],
        bound_code_sha256=bound,
        arms=list(heads),
        modes=["real", "reverse", "hold"],
        optimizer_steps=0,
        east_parameter_updates=0,
        cold_parent_previously_saw_all20=True,
        query_selection_uses_previously_revealed_outcomes=True,
        local_reset_every_sample=True,
        label_for_inner_updates=False,
        source_labels_unchanged=True,
        east_point_horizon_ms=125,
        evaluated_task="revealed shot outcome at query lead>30ms; danger30-125 separately",
        online_tpr_fpr_measured=False,
        independent_confirmation=False,
        suffix_raw_reads=0,
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(bound):
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
    readers = (io23.read_signals, io23.read_outcome)
    signals_read, outcomes_read = [], []

    def guarded(reader, log):
        def call(device, sid, *args, **kwargs):
            if device != "EAST" or sid not in ids:
                raise ValueError("outside already-ended EAST20")
            log.append(sid)
            return reader(device, sid, *args, **kwargs)

        return call

    rows = []
    with (
        patch.object(
            io23, "read_signals", side_effect=guarded(readers[0], signals_read)
        ),
        patch.object(
            io23, "read_outcome", side_effect=guarded(readers[1], outcomes_read)
        ),
    ):
        for sid in ids:
            if sha(io23.path_for("EAST", sid)) != p["input_hashes"][str(sid)]:
                raise ValueError("fixed EAST raw changed")
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
            flag, down = io23.read_outcome(
                "EAST", sid
            )  # Already ended; never passed to advance.
            for r in samples:
                if r["shot"] != sid:
                    continue
                if flag != r["disruptive"] or r["origin"] != r["query"] - 26:
                    raise ValueError("HDF authority or local clock mismatch")
                lead = (
                    round((down - raw["times"][r["query"]]) * 1000, 9) if flag else None
                )
                if flag and (lead <= 30 or abs(lead - r["actual_lead_ms"]) > 1e-6):
                    raise ValueError("strict timely query mismatch")
                if r["label"] != int(flag and lead <= 125):
                    raise ValueError("canonical EAST125 point label mismatch")
                origin = encode(
                    model,
                    {k: v[: r["origin"] + 1] for k, v in signal.items()},
                    [r["origin"]],
                    basis,
                )
                later = encode(
                    model,
                    {k: v[: r["query"] + 1] for k, v in signal.items()},
                    [r["query"]],
                    basis,
                )
                truth, mask = targets(
                    {k: v[: r["query"]] for k, v in signal.items()},
                    r["origin"],
                    r["query"] - 1,
                )
                observed = torch.as_tensor(mask, device="cuda")
                with torch.no_grad():
                    z0 = forward(model, later, model.base.center.new_zeros(4, 64))[
                        "logit"
                    ]
                    old_loss = margin_loss(z0, threshold, int(flag))
                for arm, head in heads.items():
                    for mode in ("real", "reverse", "hold"):
                        state, receipt = advance(
                            model,
                            origin,
                            model.base.center.new_zeros(4, 64),
                            torch.as_tensor(truth[mode], device="cuda"),
                            observed,
                            head,
                        )
                        with torch.no_grad():
                            output = forward(model, later, state)
                            loss = margin_loss(output["logit"], threshold, int(flag))
                        rows.append(
                            dict(
                                **r,
                                arm=arm + "_" + mode,
                                before_logit=float(z0),
                                after_logit=float(output["logit"]),
                                before_bce=float(old_loss),
                                after_bce=float(loss),
                                delta_bce=float(loss - old_loss),
                                query_rms=float(output["rms"].max()),
                                accepted=receipt["accepted"],
                                loss_before=receipt.get("loss_before"),
                                loss_proposal=receipt.get("loss_proposal"),
                            )
                        )
    summary = {
        a + "_" + m: summarize([r for r in rows if r["arm"] == a + "_" + m])
        for a in heads
        for m in ("real", "reverse", "hold")
    }
    write(root / "records.json", rows)
    write(root / "summary.json", summary)
    if sha(DEFAULT_PARENT) != SOURCE_SHA or any(
        not torch.equal(v, model.state_dict()[k].detach().cpu())
        for k, v in frozen.items()
    ):
        raise AssertionError("source or cold model changed")
    if any(sha(Path(path)) != digest for path, digest in bound.items()):
        raise AssertionError("frozen local code changed")
    done = dict(
        status="COMPLETED",
        seconds=time.monotonic() - started,
        n_shots=20,
        n_samples=100,
        n_rows=len(rows),
        signals_read=signals_read,
        outcomes_read=outcomes_read,
        suffix_raw_reads=0,
        optimizer_steps=0,
        source_unchanged=True,
        cold_unchanged=True,
        online_gain_measured=False,
        gate=gates(summary),
        method_promoted=False,
        full_stream_not_run=True,
        artifact_sha256={p.name: sha(p) for p in root.iterdir() if p.is_file()},
    )
    write(root / "completed.json", done)
    print(done)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
