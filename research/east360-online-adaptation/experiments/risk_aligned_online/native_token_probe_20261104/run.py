"""Fresh-source, post-seal EAST20 probe of the native token adaptation rule."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..direct_decoder_meta_20261102.core import clone_head
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..mask_separated_intra_20261103.pilot_core import coordinates, score
from ..mask_separated_intra_20261103.run import (
    LOCAL,
    PRIOR,
    PRIOR_SHA,
    state_digest,
    tensor_digest,
)
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent, PARENT_SHA256
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..warmstart_alignment_20261022.alignment import transform
from . import core
from .scoring import CONFIGS, MODES, analyze, check_static_reference

PLAN = io23.ROOT / "NATIVE_TOKEN_PROBE_PLAN_20261104.md"
REFERENCE = io23.ROOT / "probe_outputs/mask_separated_ended20_20261104_v1"
LEARNED_PLAN = io23.ROOT / "EAST_LEARNED_AUXILIARY_PROBE_PLAN_20261104.md"


def decoder_source(fit, key):
    """Use either the original source prior or an explicitly verified new fit."""
    if fit is None:
        if key != "forecast_only":
            raise ValueError("legacy decoder requires forecast_only")
        return PRIOR, PRIOR_SHA, {}
    if key not in ("point_contract", "anticipation125", "time_shuffled125"):
        raise ValueError("explicit fitted decoder identity required")
    fit = Path(fit)
    completion, protocol = read(fit / "completed.json"), read(fit / "protocol.json")
    if (
        completion["status"] != "TRAINED_NOT_VALIDATED"
        or protocol["source_sha256"] != PARENT_SHA256
        or protocol["source_parameter_updates"] != 0
        or protocol["east_signal_reads"] != 0
        or protocol["test_signal_reads"] != 0
    ):
        raise ValueError("fixed source-only fitted decoder required")
    for name, digest in completion["artifact_sha256"].items():
        if sha(fit / name) != digest:
            raise ValueError("fitted decoder artifact changed")
    path = fit / "heads.pt"
    return path, completion["artifact_sha256"]["heads.pt"], {
        str(fit / name): sha(fit / name)
        for name in ("completed.json", "protocol.json")
    }


def serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def main(output, *, fit=None, decoder_key="forecast_only"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    prior_path, prior_sha, fit_bound = decoder_source(fit, decoder_key)
    local = read(LOCAL / "protocol.json")
    points = coordinates(read(LOCAL / "samples.json"), local["ids"])
    map_path = Path(local["calibration_reference"]) / "input_map.json"
    if (
        sha(LOCAL / "samples.json") != read(LOCAL / "completed.json")["samples_sha256"]
        or sha(map_path) != local["input_map_sha256"]
        or sha(prior_path) != prior_sha
    ):
        raise ValueError("fixed coordinate/map/source-decoder provenance changed")
    for name, expected in read(REFERENCE / "completed.json")["artifact_sha256"].items():
        if sha(REFERENCE / name) != expected:
            raise ValueError("previous static reference artifact changed")
    reference_rows = read(REFERENCE / "predictions.json")
    mapping = read(map_path)
    if mapping["calibration_ids"] != local["ids"]:
        raise ValueError("numerical map must use exactly the fixed ended20")
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    # clone_head only requires model.base; do not instantiate an unrelated adapter.
    from types import SimpleNamespace

    decoder = clone_head(SimpleNamespace(base=base))
    saved = torch.load(prior_path, map_location="cuda", weights_only=True)
    if saved["source_sha256"] != PARENT_SHA256:
        raise ValueError("external decoder source changed")
    decoder.load_state_dict(saved["heads"][decoder_key], strict=True)
    decoder.requires_grad_(False).eval()
    del saved
    before_states = (state_digest(base), state_digest(decoder))
    code = imported_code_hashes()
    bound = {
        str(p): sha(p)
        for p in (
            PLAN,
            DEFAULT_PARENT,
            prior_path,
            map_path,
            LOCAL / "samples.json",
            LOCAL / "protocol.json",
            REFERENCE / "completed.json",
        )
    }
    bound.update(fit_bound)
    if fit is not None:
        bound[str(LEARNED_PLAN)] = sha(LEARNED_PLAN)
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        dict(
            scope="RETROSPECTIVE_ENDED20_NATIVE_TOKEN_INTERFACE_NOT_FULL_STREAM_GAIN",
            ids=local["ids"],
            n_episodes=100,
            configurations=list(CONFIGS),
            modes=list(MODES),
            primary_configuration="native_absolute",
            copied_static_records_not_independent=True,
            source_sha256=PARENT_SHA256,
            threshold=float(parent["threshold"]),
            source_head="checkpoint native forecast_head",
            external_head="frozen source-only " + decoder_key,
            learned_fit=None if fit is None else str(Path(fit).resolve()),
            diagnostic_after_source_screen_failed=fit is not None,
            learned_primary_configuration=None if fit is None else "source_decoder_absolute",
            source_updates=0,
            temporary_state="zero-start offset128 at source-native pre-Transformer token gain/bias",
            update="original source normalized_step: FAST_STEP=.05, radius1; no label-based acceptance",
            first_ffn_geometry_matched=False,
            clock="origin=q-26; matured diagnostics q-25..q-1; first affected query q",
            truth_access="all neural records sealed before any current hard outcome read",
            reader_scope="whole ended-shot arrays loaded, then causal model prefixes; not literal streaming I/O",
            retrospective_query_selection=True,
            development_only=True,
            numerical_preparation_budget=20,
            suffix340_reads=0,
            old_cold_state_loaded=False,
            support110_used=False,
            input_sha256=bound,
            imported_code_sha256=code,
            raw_sha256={str(s): local["input_hashes"][str(s)] for s in local["ids"]},
            torch_version=torch.__version__,
            device=torch.cuda.get_device_name(),
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in code:
            path = Path(name)
            archive.write(path, str(path.relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
        if fit is not None:
            archive.write(LEARNED_PLAN, LEARNED_PLAN.name)
    rows, states, signal_calls = [], {}, []
    reader = io23.read_signals

    def checked_signal(device, sid, *args, **kwargs):
        if device != "EAST" or sid not in local["ids"]:
            raise ValueError("fixed previously ended EAST20 only")
        signal_calls.append(sid)
        return reader(device, sid, *args, **kwargs)

    with (
        patch.object(io23, "read_signals", side_effect=checked_signal),
        patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError(
                "hard outcome forbidden until all predictions sealed"
            ),
        ),
    ):
        for sid in local["ids"]:
            if sha(io23.path_for("EAST", sid)) != local["input_hashes"][str(sid)]:
                raise ValueError("ended20 raw file changed")
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
            if not np.array_equal(signal["available"], raw["available"]):
                raise AssertionError("truthful masks must stay unchanged")
            for r in [p for p in points if p["shot"] == sid]:
                q, origin = r["query"], r["origin"]
                if q >= len(raw["times"]):
                    raise ValueError("query outside record")
                prefix0 = {k: v[: origin + 1] for k, v in signal.items()}
                prefix1 = {k: v[: q + 1] for k, v in signal.items()}
                x0, a0 = [
                    torch.as_tensor(v[None], device="cuda")
                    for v in simple_data.window(prefix0, origin, 128)
                ]
                x1, a1 = [
                    torch.as_tensor(v[None], device="cuda")
                    for v in simple_data.window(prefix1, q, 128)
                ]
                truth, observed = targets(
                    {k: v[:q] for k, v in signal.items()}, origin, q - 1
                )
                truth = {k: torch.as_tensor(v, device="cuda") for k, v in truth.items()}
                observed = torch.as_tensor(observed, device="cuda")
                zero = base.center.new_zeros(1, 128)
                with torch.no_grad():
                    before = core.forward(base, x1, a1, zero)["logit"]
                    torch.testing.assert_close(
                        before, base(x1, a1)["logit"], atol=0, rtol=0
                    )
                for cfg in CONFIGS:
                    head = decoder if cfg.startswith("source_decoder") else None
                    loss_kind = cfg.rsplit("_", 1)[1]
                    for mode in MODES:
                        state, receipt = (
                            (zero, None)
                            if mode == "static"
                            else core.advance(
                                base,
                                x0,
                                a0,
                                zero,
                                truth[mode],
                                observed,
                                decoder=head,
                                loss_kind=loss_kind,
                            )
                        )
                        with torch.no_grad():
                            after = core.forward(base, x1, a1, state, decoder=head)
                        if mode == "static":
                            torch.testing.assert_close(
                                after["logit"], before, atol=0, rtol=0
                            )
                        key = f"{cfg}_{mode}_{sid}_{q}"
                        states[key] = state.detach().cpu()
                        rows.append(
                            dict(
                                **r,
                                stem="original",
                                configuration=cfg,
                                mode=mode,
                                query_time=float(raw["times"][q]),
                                before_logit=float(before),
                                after_logit=float(after["logit"]),
                                delta_norm=float(state.double().norm()),
                                observed=int(observed.sum()),
                                mask_sha256=tensor_digest(observed),
                                target_sha256=tensor_digest(
                                    truth[mode if mode != "static" else "real"]
                                ),
                                receipt=serializable(receipt),
                            )
                        )
            print(
                dict(
                    stage="native",
                    shot=sid,
                    predictions=len(rows),
                    seconds=time.monotonic() - started,
                ),
                flush=True,
            )
    if len(rows) != 1600 or len(signal_calls) != 20 or len(states) != 1600:
        raise AssertionError("all4x4x100 fixed local comparisons required")
    parity = check_static_reference(rows, reference_rows)
    torch.save(states, output / "temporary_states.pt")
    write(output / "predictions.json", rows)
    write(
        output / "prediction_seal.json",
        dict(
            current_outcome_used_in_prediction=False,
            predictions_sha256=sha(output / "predictions.json"),
            states_sha256=sha(output / "temporary_states.pt"),
            static_parity=parity,
        ),
    )
    outcomes = {sid: io23.read_outcome("EAST", sid) for sid in local["ids"]}
    scored = score(rows, outcomes, float(parent["threshold"]))
    summary = analyze(scored)
    write(output / "scored.json", scored)
    write(output / "summary.json", summary)
    write(
        output / "hdf_outcomes.json",
        {
            str(sid): dict(disruptive=flag, down=t)
            for sid, (flag, t) in outcomes.items()
        },
    )
    unchanged = before_states == (state_digest(base), state_digest(decoder))
    if not unchanged or any(
        p.grad is not None for m in (base, decoder) for p in m.parameters()
    ):
        raise AssertionError("source and both diagnostic heads must stay frozen")
    if any(sha(Path(p)) != expected for p, expected in {**bound, **code}.items()):
        raise AssertionError("bound source/input/code changed")
    if (
        sha(output / "predictions.json")
        != read(output / "prediction_seal.json")["predictions_sha256"]
    ):
        raise AssertionError("sealed neural predictions changed during scoring")
    result = dict(
        status="COMPLETED",
        seconds=time.monotonic() - started,
        source_and_heads_unchanged=unchanged,
        source_signal_reads=0,
        east_signal_reads=20,
        postseal_outcome_reads=20,
        suffix340_reads=0,
        n_predictions=1600,
        static_parity=parity,
        primary_gate=summary["primary_gate"],
        configuration_gates=summary["configuration_gates"],
        operational_online_gain_measured=False,
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "completed.json", result)
    print(
        {
            k: result[k]
            for k in ("status", "seconds", "primary_gate", "configuration_gates")
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--decoder-key", default="forecast_only")
    args = parser.parse_args()
    main(args.output, fit=args.fit, decoder_key=args.decoder_key)
