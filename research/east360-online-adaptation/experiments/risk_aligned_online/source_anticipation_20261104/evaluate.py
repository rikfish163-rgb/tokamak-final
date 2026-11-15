"""Offline VAL episodes for learned auxiliary tasks, not a J-TEXT online run."""

import argparse
from copy import deepcopy
import math
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..mask_separated_intra_20261103.run import state_digest, tensor_digest
from ..native_token_probe_20261104.core import advance, forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent, PARENT_SHA256
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from .design import ARMS, STRATA
from .train import load_windows, batch_view

VIEWS = ("native", "no_MNM_CIII")
HEADS = ("native_reference", *ARMS)
MODES = ("real", "reverse", "hold")
PLAN = io23.ROOT / "SOURCE_ANTICIPATION_EVAL_PLAN_20261104.md"


def aggregate(rows):
    result = {}
    for group in STRATA:
        selected = [r for r in rows if r["stratum"] == group]
        ids = sorted({r["shot"] for r in selected})
        if not ids:
            raise ValueError("all four declared source VAL groups required")
        result[group] = dict(shots=len(ids), records=len(selected))
        for key in ("future_delta", "point_delta"):
            result[group][key] = float(
                np.mean(
                    [
                        np.mean([r[key] for r in selected if r["shot"] == sid])
                        for sid in ids
                    ]
                )
            )
    result["balanced"] = (
        result["safe"]["future_delta"]
        + result["disruptive_30_to_125ms"]["future_delta"]
    ) / 2
    return result


def gate(table):
    candidate = table["no_MNM_CIII"]["anticipation125"]["real"]
    result = {
        g + "_improved": candidate[g]["future_delta"] < -1e-6
        for g in ("safe", "disruptive_30_to_125ms")
    }
    for mode in ("reverse", "hold"):
        result["better_than_" + mode] = (
            candidate["balanced"]
            < table["no_MNM_CIII"]["anticipation125"][mode]["balanced"] - 1e-6
        )
    for head in ("native_reference", "point_contract", "time_shuffled125"):
        result["better_than_" + head] = (
            candidate["balanced"]
            < table["no_MNM_CIII"][head]["real"]["balanced"] - 1e-6
        )
    for group in ("disruptive_earlier_than_125ms", "disruptive_0_to_30ms"):
        result[group + "_protected"] = candidate[group]["future_delta"] <= 1e-6
    result["native_view_protected"] = all(
        table["native"]["anticipation125"]["real"][g]["future_delta"] <= 1e-6
        for g in ("safe", "disruptive_30_to_125ms")
    )
    result["passed"] = all(result.values())
    return result


def bce(logit, cutoff, label):
    z = logit - cutoff
    return float(np.logaddexp(0, (1 - 2 * label) * z))


def main(fit, design, output):
    fit, design, output = Path(fit), Path(design), Path(output)
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    completion = read(fit / "completed.json")
    fit_protocol = read(fit / "protocol.json")
    if completion["status"] != "TRAINED_NOT_VALIDATED":
        raise ValueError(
            "fixed final trained heads required; no fitting during evaluation"
        )
    for root in (fit, design):
        for name, digest in read(root / "completed.json")["artifact_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("bound fit/design artifact changed")
    # The producer is archived; the shared reader gained an explicit VAL role
    # after fitting. Validate the archived numerical producer, not a false claim
    # that every current source file still has its historical bytes.
    with zipfile.ZipFile(fit / "code.zip") as archive:
        import hashlib

        for path, digest in fit_protocol["code_sha256"].items():
            if (
                hashlib.sha256(
                    archive.read(str(Path(path).relative_to(io23.ROOT)))
                ).hexdigest()
                != digest
            ):
                raise ValueError(
                    "archived fitted producer differs from its recorded hash"
                )
    records = read(design / "val_records.json")
    if len(records) != 946 or any(r["role"] != "val" for r in records):
        raise ValueError("all fixed946 source VAL episodes required")
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    pack, location, raw_hashes = load_windows(
        records,
        [dict(indices=list(range(len(records))))],
        base.center.device,
        role="val",
    )
    saved = torch.load(fit / "heads.pt", map_location="cuda", weights_only=True)
    if saved["source_sha256"] != PARENT_SHA256 or set(saved["heads"]) != set(ARMS):
        raise ValueError("same-source three matched heads required")
    heads = {"native_reference": None}
    for name in ARMS:
        h = deepcopy(base.forecast_head)
        h.load_state_dict(saved["heads"][name], strict=True)
        h.requires_grad_(False).eval()
        heads[name] = h
    before = {name: state_digest(base if h is None else h) for name, h in heads.items()}
    cutoff = math.log(float(parent["threshold"]) / (1 - float(parent["threshold"])))
    code = imported_code_hashes()
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        dict(
            scope="SOURCE_VAL_OFFLINE_AUXILIARY_UPDATE_EPISODES_NOT_JTEXT_ONLINE_OR_EAST_RESULT",
            fit=str(fit.resolve()),
            fit_completion_sha256=sha(fit / "completed.json"),
            design_completion_sha256=sha(design / "completed.json"),
            plan_sha256=sha(PLAN),
            source_sha256=PARENT_SHA256,
            views=list(VIEWS),
            heads=list(HEADS),
            modes=list(MODES),
            source_role="val",
            records=946,
            threshold=float(parent["threshold"]),
            code_sha256=code,
            raw_sha256=raw_hashes,
            source_updates=0,
            head_updates=0,
            test_raw_reads=0,
            east_raw_reads=0,
            supervision="future125 auxiliary loss primary; original source point loss separate",
            future_forecast_fidelity_evaluated=False,
            proxy_loss_scope="matured training-target window only, not post-query heldout forecast accuracy",
            reset_each_episode=True,
            prequential_full_stream=False,
            literal_streaming_io=False,
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    raw_rows = []
    with patch.object(
        io23,
        "read_outcome",
        side_effect=RuntimeError(
            "no metadata access inside neural episode computation"
        ),
    ):
        for view in VIEWS:
            for i, r in enumerate(records):
                batch = batch_view(pack, [location[i]], view)
                zero = base.center.new_zeros(1, 128)
                with torch.no_grad():
                    initial = forward(base, batch["xq"], batch["aq"], zero)["logit"]
                targets = {
                    "real": batch["truth"],
                    "reverse": batch["truth"].flip(1),
                    "hold": batch["x0"][:, -1:].expand(-1, 25, -1),
                }
                for head, decoder in heads.items():
                    for mode in MODES:
                        state, receipt = advance(
                            base,
                            batch["x0"],
                            batch["a0"],
                            zero,
                            targets[mode],
                            batch["mask"],
                            decoder=decoder,
                            loss_kind="absolute",
                        )
                        with torch.no_grad():
                            updated = forward(
                                base, batch["xq"], batch["aq"], state, decoder=decoder
                            )["logit"]
                        raw_rows.append(
                            dict(
                                view=view,
                                head=head,
                                mode=mode,
                                shot=r["shot"],
                                query=r["query"],
                                before_logit=float(initial),
                                after_logit=float(updated),
                                state_sha256=tensor_digest(state),
                                step_norm=receipt["step_norm"],
                                proxy_before=receipt["loss_before"],
                                proxy_after=receipt["loss_after"],
                                mask_sha256=tensor_digest(batch["mask"]),
                            )
                        )
                if (i + 1) % 128 == 0 or i + 1 == len(records):
                    print(
                        dict(
                            stage="source_VAL",
                            view=view,
                            episodes=i + 1,
                            predictions=len(raw_rows),
                            seconds=time.monotonic() - started,
                        ),
                        flush=True,
                    )
    if len(raw_rows) != 946 * 2 * 4 * 3:
        raise AssertionError("complete matched source VAL grid required")
    write(output / "predictions.json", raw_rows)
    write(
        output / "prediction_seal.json",
        dict(
            predictions_sha256=sha(output / "predictions.json"),
            neural_prediction_count=len(raw_rows),
        ),
    )
    # Labels are used here only to score the already fixed neural records.
    lookup = {(r["shot"], r["query"]): r for r in records}
    scored = []
    for row in raw_rows:
        r = lookup[row["shot"], row["query"]]
        item = dict(
            row, stratum=r["anticipation_stratum"], hdf_disruptive=r["hdf5_disruptive"]
        )
        for name, target in (
            ("future", r["future_disruption_125ms"]),
            ("point", r["original_source_point_label"]),
        ):
            item[name + "_delta"] = bce(row["after_logit"], cutoff, target) - bce(
                row["before_logit"], cutoff, target
            )
        scored.append(item)
    table = {
        view: {
            head: {
                mode: aggregate(
                    [
                        r
                        for r in scored
                        if r["view"] == view and r["head"] == head and r["mode"] == mode
                    ]
                )
                for mode in MODES
            }
            for head in HEADS
        }
        for view in VIEWS
    }
    criteria = gate(table)
    write(output / "scored.json", scored)
    write(
        output / "summary.json",
        dict(
            groups=table,
            candidate_gate=criteria,
            source_auxiliary_screen_only=True,
            future_forecast_fidelity_evaluated=False,
            operational_online_gain_measured=False,
        ),
    )
    if any(
        state_digest(base if h is None else h) != before[name]
        for name, h in heads.items()
    ):
        raise AssertionError("evaluation changed frozen source/head states")
    if (
        any(sha(Path(path)) != digest for path, digest in code.items())
        or sha(PLAN) != read(output / "protocol.json")["plan_sha256"]
    ):
        raise AssertionError("evaluation producer changed")
    result = dict(
        status="COMPLETED_SOURCE_AUXILIARY_VALIDATION",
        seconds=time.monotonic() - started,
        source_and_heads_unchanged=True,
        source_val_signal_shots=len(raw_hashes),
        test_raw_reads=0,
        east_raw_reads=0,
        predictions=len(raw_rows),
        candidate_gate=criteria,
        future_forecast_fidelity_evaluated=False,
        operational_online_gain_measured=False,
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "completed.json", result)
    print({k: result[k] for k in ("status", "seconds", "candidate_gate")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.fit, args.design, args.output)
