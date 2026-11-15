"""Eight-arm retrospective EAST20 screen from a fresh qualified source.

No old cold state, target-trained head, alarm tuning, or suffix340 access.
All neural predictions are sealed before any HDF hard outcome is revealed.
"""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..direct_decoder_meta_20261102.core import clone_head
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..east_decoder_calibration_20261102.functional_update import advance, FRACTIONS
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import encode, forward, direct
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent, PARENT_SHA256
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..warmstart_alignment_20261022.alignment import transform
from .model import MaskSeparatedContext
from .pilot_core import STEMS, MODES, coordinates, score, analyze

LOCAL = io23.ROOT / "probe_outputs/difference_feedback_probe20_v1"
PRIOR = io23.ROOT / "probe_outputs/direct_decoder_source_v1/decoders.pt"
PRIOR_SHA = "cdde96086517cbc468a911bbb5926dde1c783b3711b0a8abc09caf0a6d2e6730"
PLAN = io23.ROOT / "MASK_SEPARATED_INTRA_PLAN_20261104.md"
PARITY_SHOTS = (71304, 69854, 54157, 75657)


def tensor_digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    local = read(LOCAL / "protocol.json")
    if sha(LOCAL / "samples.json") != read(LOCAL / "completed.json")["samples_sha256"]:
        raise ValueError("historical coordinate manifest changed")
    points = coordinates(read(LOCAL / "samples.json"), local["ids"])
    map_path = Path(local["calibration_reference"]) / "input_map.json"
    if sha(map_path) != local["input_map_sha256"] or sha(PRIOR) != PRIOR_SHA:
        raise ValueError("fixed numerical map or source-only head changed")
    mapping = read(map_path)
    if mapping["calibration_ids"] != local["ids"]:
        raise ValueError("numeric preparation must use exactly the same ended20")
    base, parent = load_parent(DEFAULT_PARENT)
    models = {s: MaskSeparatedContext(deepcopy(base), policy=s).cuda() for s in STEMS}
    del base
    if any(m.slow.weight.count_nonzero() for m in models.values()):
        raise ValueError("fresh zero-slow models required; no target state restore")
    basis = source_basis(models["original"])
    torch.testing.assert_close(
        basis, source_basis(models["reference_bt"]), atol=0, rtol=0
    )
    saved = torch.load(PRIOR, map_location="cuda", weights_only=True)
    if saved["source_sha256"] != PARENT_SHA256:
        raise ValueError("decoder source provenance changed")
    decoder = clone_head(models["original"])
    decoder.load_state_dict(saved["heads"]["forecast_only"], strict=True)
    decoder.eval().requires_grad_(False)
    del saved
    snapshots = {s: state_digest(m) for s, m in models.items()}
    decoder_snapshot = state_digest(decoder)
    bound = {
        str(p): sha(p)
        for p in (
            PLAN,
            PRIOR,
            DEFAULT_PARENT,
            map_path,
            LOCAL / "samples.json",
            LOCAL / "protocol.json",
        )
    }
    code = imported_code_hashes()
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        dict(
            scope="RETROSPECTIVE_ENDED20_MASK_SEPARATED_NEURAL_SCREEN_NOT_ONLINE_SCORE",
            ids=local["ids"],
            n_episodes=100,
            stems=list(STEMS),
            modes=list(MODES),
            source_sha256=PARENT_SHA256,
            threshold=float(parent["threshold"]),
            head="source-only external forecast_only; frozen",
            decoder_sha256=PRIOR_SHA,
            cold_state_loaded=False,
            target_fitted_head_loaded=False,
            source_parameter_updates=0,
            slow_parameter_updates=0,
            temporary_state="first-FFN rank4x64, zero per episode; no persistent state",
            update="centered matured25 SmoothL1 + .01 mean(W^2); hidden-metric step",
            fractions=list(FRACTIONS),
            total_hidden_cap_rms=0.25,
            step_hidden_bound_rms=0.05,
            clock="origin=q-26; targets=q-25..q-1; first affected query=q",
            outcome_access="denied for every neural calculation; all800 predictions sealed first",
            outcome_authority="HDF5 meta/IsDisrupt and meta/DownTime only",
            canonical_labels="EAST fixed125ms; operational local loss target is HDF shot flag at lead>30ms",
            coordinate_selection="previously revealed outcomes; retrospective development only",
            numerical_map="known ended20 resource, not zero-shot physical calibration",
            preparation_budget=20,
            suffix340_reads=0,
            support110_used=False,
            input_sha256=bound,
            imported_code_sha256=code,
            raw_sha256={str(s): local["input_hashes"][str(s)] for s in local["ids"]},
            parity_shots=list(PARITY_SHOTS),
            torch_version=torch.__version__,
            device=torch.cuda.get_device_name(),
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in code:
            path = Path(name)
            archive.write(path, str(path.relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    rows, states, parity, signal_calls, clocks = [], {}, [], [], {}
    read_signal = io23.read_signals

    def checked_signal(device, sid, *args, **kwargs):
        if device != "EAST" or sid not in local["ids"]:
            raise ValueError("only fixed previously ended EAST20 signals allowed")
        signal_calls.append(sid)
        return read_signal(device, sid, *args, **kwargs)

    with (
        patch.object(io23, "read_signals", side_effect=checked_signal),
        patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError(
                "hard outcome forbidden before all predictions seal"
            ),
        ),
    ):
        for sid in local["ids"]:
            if sha(io23.path_for("EAST", sid)) != local["input_hashes"][str(sid)]:
                raise ValueError("ended20 raw file hash changed")
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
            if not np.array_equal(signal["available"], raw["available"]):
                raise AssertionError("numerical map must not change availability")
            clocks[sid] = raw["times"]
            local_points = [r for r in points if r["shot"] == sid]
            for point_index, r in enumerate(local_points):
                q, origin = r["query"], r["origin"]
                if q >= len(raw["times"]):
                    raise ValueError("query outside raw signal")
                before_signal = {k: v[: origin + 1] for k, v in signal.items()}
                query_signal = {k: v[: q + 1] for k, v in signal.items()}
                truth, observed = targets(
                    {k: v[:q] for k, v in signal.items()}, origin, q - 1
                )
                truth = {k: torch.as_tensor(v, device="cuda") for k, v in truth.items()}
                mask = torch.as_tensor(observed, device="cuda")
                for stem, model in models.items():
                    origin_pack = encode(model, before_signal, [origin], basis)
                    later_pack = encode(model, query_signal, [q], basis)
                    zero = model.base.center.new_zeros(4, 64)
                    with torch.no_grad():
                        initial = forward(model, later_pack, zero)["logit"]
                    for mode in MODES:
                        state, receipt = (
                            (zero, None)
                            if mode == "static"
                            else advance(
                                model,
                                origin_pack,
                                zero,
                                truth[mode],
                                mask,
                                decoder,
                                kind="parameter",
                            )
                        )
                        with torch.no_grad():
                            after = forward(model, later_pack, state)
                            if sid in PARITY_SHOTS and point_index == 0:
                                actual = direct(model, query_signal, [q], basis, state)
                                torch.testing.assert_close(
                                    actual["logit"], after["logit"], atol=0, rtol=0
                                )
                                parity.append(
                                    dict(
                                        shot=sid,
                                        query=q,
                                        stem=stem,
                                        mode=mode,
                                        direct_logit=float(actual["logit"]),
                                        cached_logit=float(after["logit"]),
                                    )
                                )
                        key = f"{stem}_{mode}_{sid}_{q}"
                        states[key] = state.detach().cpu()
                        rows.append(
                            dict(
                                **r,
                                stem=stem,
                                mode=mode,
                                query_time=float(raw["times"][q]),
                                before_logit=float(initial),
                                after_logit=float(after["logit"]),
                                query_hidden_rms=float(after["rms"].max()),
                                observed=int(mask.sum()),
                                mask_sha256=tensor_digest(mask),
                                target_sha256=tensor_digest(
                                    truth[mode if mode != "static" else "real"]
                                ),
                                receipt=receipt,
                            )
                        )
            print(
                dict(
                    stage="neural",
                    shot=sid,
                    completed_predictions=len(rows),
                    elapsed=time.monotonic() - started,
                ),
                flush=True,
            )
    if len(rows) != 800 or len(parity) != 32 or len(signal_calls) != 20:
        raise AssertionError("full fixed local support and predeclared parity required")
    torch.save(states, output / "temporary_states.pt")
    write(output / "predictions.json", rows)
    write(
        output / "prediction_seal.json",
        dict(
            predictions_sha256=sha(output / "predictions.json"),
            states_sha256=sha(output / "temporary_states.pt"),
            n_predictions=len(rows),
            current_outcome_used_in_prediction=False,
            suffix340_reads=0,
        ),
    )
    # This is a post-run audit reader. None of these labels can affect updates.
    outcomes = {sid: io23.read_outcome("EAST", sid) for sid in local["ids"]}
    scored = score(rows, outcomes, float(parent["threshold"]))
    summary = analyze(scored)
    write(
        output / "hdf_outcomes.json",
        {str(sid): dict(disruptive=f, down=t) for sid, (f, t) in outcomes.items()},
    )
    write(output / "scored.json", scored)
    write(output / "summary.json", summary)
    write(output / "direct_parity.json", dict(status="PASS", witnesses=parity))
    unchanged = (
        all(state_digest(models[s]) == snapshots[s] for s in STEMS)
        and state_digest(decoder) == decoder_snapshot
    )
    if not unchanged or any(
        p.grad is not None for m in (*models.values(), decoder) for p in m.parameters()
    ):
        raise AssertionError(
            "frozen source, slow and decoder states must remain unchanged"
        )
    if any(sha(Path(p)) != digest for p, digest in {**bound, **code}.items()):
        raise AssertionError("bound input/code changed while running")
    if (
        sha(output / "predictions.json")
        != read(output / "prediction_seal.json")["predictions_sha256"]
    ):
        raise AssertionError("sealed predictions changed during scoring")
    completed = dict(
        status="COMPLETED",
        seconds=time.monotonic() - started,
        n_predictions=800,
        source_and_frozen_head_unchanged=unchanged,
        source_file_unchanged=sha(DEFAULT_PARENT) == PARENT_SHA256,
        source_signal_reads=0,
        east_signal_reads=20,
        postseal_east_outcome_reads=20,
        suffix340_reads=0,
        direct_parity_witnesses=32,
        candidate_gate=summary["candidate_gate"],
        operational_online_gain_measured=False,
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "completed.json", completed)
    print(
        {k: completed[k] for k in ("status", "seconds", "candidate_gate")}, flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output)
