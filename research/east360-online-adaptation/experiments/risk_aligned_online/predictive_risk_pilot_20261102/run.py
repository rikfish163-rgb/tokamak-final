"""Ended20 direct-measurement forecasting -> frozen future-risk diagnostic."""

import argparse
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..critical_shot_meta_20261102.objectives import margin_loss
from ..direct_decoder_meta_20261102.core import prediction
from ..east_decoder_calibration_20261102.functional_stream import OLD, load
from ..east_decoder_calibration_20261102.functional_update import advance
from ..east_decoder_calibration_20261102.run import LOCAL, collect
from ..dual_history_ttt_20261101.stream import targets
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import encode, forward
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.trial import windows_pack
from .core import decode_forecast, future_windows

PLAN = io23.ROOT / "PREDICTIVE_RISK_PILOT_PLAN_20261102.md"
BRANCHES = ("static", "real", "reverse", "hold")


@torch.no_grad()
def infer(model, prefix, basis, head, state):
    if set(prefix) != {"values", "available"}:
        raise ValueError("signals only; no future target or outcome")
    q = len(prefix["values"]) - 1
    point = encode(model, prefix, [q], basis)
    predicted = prediction(model, point, state, head)
    x, mask = simple_data.window(prefix, q, 128)
    future, assumed_mask = decode_forecast(
        predicted[0].cpu().numpy(),
        model.base.center.cpu().numpy(),
        model.base.scale.cpu().numpy(),
        prefix["values"][-1],
        prefix["available"][-1],
    )
    values, masks = future_windows(x, mask, future, assumed_mask)
    pack = windows_pack(model, values, masks, basis)
    logits = forward(model, pack, state.new_zeros(4, 64))["logit"]
    negative_vars = int(((future[:, [17, 22]] < 0) & assumed_mask[:, [17, 22]]).sum())
    return predicted, logits, assumed_mask, negative_vars


def summarize(rows):
    result = {}
    for group in ("safe", "disruptive"):
        local = [r for r in rows if r["group"] == group]
        ids = sorted({r["shot"] for r in local})

        def macro(key):
            return float(
                np.mean(
                    [
                        np.mean([r[key] for r in local if r["shot"] == sid])
                        for sid in ids
                    ]
                )
            )

        result[group] = dict(
            episodes=len(local),
            shots=len(ids),
            mean_risk_loss=macro("risk_loss"),
            crossed_query_points=sum(r["crossed"] for r in local),
            negative_variance_predictions=sum(
                r.get("negative_variances", 0) for r in local
            ),
        )
        measured = [r for r in local if r.get("forecast_loss") is not None]
        if measured:
            result[group]["mean_forecast_loss"] = float(
                np.mean(
                    [
                        np.mean(
                            [r["forecast_loss"] for r in measured if r["shot"] == sid]
                        )
                        for sid in ids
                        if any(r["shot"] == sid for r in measured)
                    ]
                )
            )
    result["balanced_risk_loss"] = (
        result["safe"]["mean_risk_loss"] + result["disruptive"]["mean_risk_loss"]
    ) / 2
    return result


def main(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    model, base_head, basis, cold, previous, _, _ = load()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    heads = {name: deepcopy(base_head) for name in ("jt_prior", "east_diagnostic")}
    saved = torch.load(OLD / "heads.pt", map_location="cuda", weights_only=True)
    with torch.no_grad():
        heads["jt_prior"].delta.zero_()
        heads["east_diagnostic"].delta.copy_(saved["forecast_only"])
    for h in heads.values():
        h.requires_grad_(False)
    head_initial = {
        name: {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
        for name, h in heads.items()
    }
    local = read(LOCAL / "protocol.json")
    examples, signals = collect(model, basis, local)
    root.mkdir()
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    write(
        root / "protocol.json",
        dict(
            scope="ENDED20_PREDICTIVE_RISK_FEASIBILITY_NOT_ONLINE",
            ids=local["ids"],
            input_hashes=local["input_hashes"],
            source_sha256=previous["source_sha256"],
            cold_sha256=sha(cold),
            heads_sha256=sha(OLD / "heads.pt"),
            bound_code_sha256=bound,
            threshold=previous["threshold"],
            horizon=25,
            non_autoregressive=True,
            future_classifier_state="zero",
            forecast_state="one matured internal update",
            future_availability_assumption="current availability persists; mode10 carry; MNM unavailable",
            raw_variance_projection="none; count negatives; diagnostic only",
            suffix_raw_reads=0,
            optimizer_steps=0,
            independent_confirmation=False,
        ),
    )
    references, rows, excluded = [], [], []
    for e in examples:
        q = e["query"]
        signal = signals[e["shot"]]
        if q + 25 >= len(signal["values"]):
            excluded.append(
                dict(
                    shot=e["shot"],
                    query=q,
                    reason="full_future25_not_recorded",
                    disruptive=e["disruptive"],
                )
            )
            continue
        group = "disruptive" if e["disruptive"] else "safe"
        common = {
            k: e[k] for k in ("shot", "query", "label", "disruptive", "actual_lead_ms")
        }
        zero = model.base.center.new_zeros(4, 64)
        with torch.no_grad():
            current = forward(model, e["later_pack"], zero)["logit"]
            # Evaluation-only future truth, never passed to infer or adaptation.
            future_prefix = {k: v[: q + 26] for k, v in signal.items()}
            actual_pack = encode(model, future_prefix, np.arange(q + 1, q + 26), basis)
            actual_logits = forward(model, actual_pack, zero)["logit"]
            oracle = torch.maximum(current, actual_logits.max().reshape_as(current))
            for name, z in (("current", current), ("future_oracle", oracle)):
                references.append(
                    dict(
                        **common,
                        branch=name,
                        group=group,
                        logit=float(z),
                        risk_loss=float(
                            margin_loss(z, previous["threshold"], int(e["disruptive"]))
                        ),
                        crossed=bool(z.sigmoid() >= previous["threshold"]),
                    )
                )
        future_truth, future_mask = targets(future_prefix, q, q + 25)
        prefix = {k: v[: q + 1] for k, v in signal.items()}
        for name, head in heads.items():
            for mode in BRANCHES:
                if mode == "static":
                    state, receipt = zero, dict(accepted=False, reason="static")
                else:
                    state, receipt = advance(
                        model,
                        e["origin_pack"],
                        zero,
                        e["truth"][mode],
                        e["mask"],
                        head,
                        kind="parameter",
                    )
                with patch.object(
                    io23,
                    "read_outcome",
                    side_effect=RuntimeError("outcome forbidden during infer"),
                ):
                    pred, forecast_logits, assumed_mask, negative = infer(
                        model, prefix, basis, head, state
                    )
                with torch.no_grad():
                    z = torch.maximum(
                        current, forecast_logits.max().reshape_as(current)
                    )
                    mask = torch.as_tensor(future_mask & assumed_mask, device="cuda")
                    forecast_loss = (
                        float(
                            model.base.forecast_loss(
                                pred,
                                torch.as_tensor(future_truth["real"], device="cuda"),
                                mask,
                            )
                        )
                        if mask.any()
                        else None
                    )
                rows.append(
                    dict(
                        **common,
                        head=name,
                        branch=mode,
                        group=group,
                        logit=float(z),
                        risk_loss=float(
                            margin_loss(z, previous["threshold"], int(e["disruptive"]))
                        ),
                        crossed=bool(z.sigmoid() >= previous["threshold"]),
                        negative_variances=negative,
                        forecast_loss=forecast_loss,
                        forecast_observed=int(mask.sum()),
                        update=receipt,
                        future_probabilities=forecast_logits.sigmoid().cpu().tolist(),
                    )
                )
        if len(rows) % 80 == 0:
            print(
                dict(records=len(rows), seconds=time.monotonic() - started), flush=True
            )
    summary = {
        name + "_" + mode: summarize(
            [r for r in rows if r["head"] == name and r["branch"] == mode]
        )
        for name in heads
        for mode in BRANCHES
    }
    reference_summary = {
        name: summarize([r for r in references if r["branch"] == name])
        for name in ("current", "future_oracle")
    }
    candidate, baseline = (
        summary["east_diagnostic_real"],
        summary["east_diagnostic_static"],
    )
    gate = {
        g + "_improved": candidate[g]["mean_risk_loss"]
        < baseline[g]["mean_risk_loss"] - 1e-9
        for g in ("safe", "disruptive")
    }
    gate["no_negative_variances"] = (
        sum(
            candidate[g]["negative_variance_predictions"]
            for g in ("safe", "disruptive")
        )
        == 0
    )
    gate.update(
        {
            "better_than_" + m: candidate["balanced_risk_loss"]
            < summary["east_diagnostic_" + m]["balanced_risk_loss"] - 1e-9
            for m in ("reverse", "hold")
        }
    )
    gate["passed"] = all(gate.values())
    if any(
        not torch.equal(v, model.state_dict()[k].cpu()) for k, v in initial.items()
    ) or any(
        not torch.equal(v, heads[name].state_dict()[k].cpu())
        for name, state in head_initial.items()
        for k, v in state.items()
    ):
        raise AssertionError("frozen model changed")
    if any(sha(Path(path)) != d for path, d in bound.items()):
        raise AssertionError("frozen code changed")
    write(root / "records.json", rows)
    write(root / "references.json", references)
    write(root / "excluded.json", excluded)
    write(root / "summary.json", dict(candidate=summary, references=reference_summary))
    done = dict(
        status="COMPLETED_FEASIBILITY_DIAGNOSTIC",
        episodes=len(references) // 2,
        records=len(rows),
        excluded=len(excluded),
        gate=gate,
        seconds=time.monotonic() - started,
        source_unchanged=True,
        cold_unchanged=True,
        heads_unchanged=True,
        suffix_raw_reads=0,
        optimizer_steps=0,
        online_gain_measured=False,
        method_promoted=False,
        artifact_sha256={f.name: sha(f) for f in root.iterdir() if f.is_file()},
    )
    write(root / "completed.json", done)
    print(done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
