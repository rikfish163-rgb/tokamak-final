"""Ended20 anchored-path feasibility with matched real/reverse/hold controls."""

import argparse
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch

import torch

from .. import io23
from ..critical_shot_meta_20261102.objectives import margin_loss
from ..dual_history_ttt_20261101.stream import targets
from ..east_decoder_calibration_20261102.functional_stream import OLD, load
from ..east_decoder_calibration_20261102.functional_update import advance
from ..east_decoder_calibration_20261102.run import LOCAL, collect
from ..history_gradient_metric_20261102.run import read, sha, write
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from .anchored import AnchoredHead, PersistenceHead
from .run import BRANCHES, infer, summarize

PLAN = io23.ROOT / "ANCHORED_PREDICTIVE_RISK_PLAN_20261102.md"
REFERENCE = io23.ROOT / "probe_outputs/predictive_risk_pilot20_v1"


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
    head_initial = {
        name: {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
        for name, h in heads.items()
    }
    local = read(LOCAL / "protocol.json")
    examples, signals = collect(model, basis, local)
    old_done = read(REFERENCE / "completed.json")
    if old_done["status"] != "COMPLETED_FEASIBILITY_DIAGNOSTIC":
        raise ValueError("completed unanchored reference required")
    for name, expected in old_done["artifact_sha256"].items():
        if sha(REFERENCE / name) != expected:
            raise ValueError("old pilot artifact changed")
    references = read(REFERENCE / "references.json")
    by_query = {}
    for r in references:
        by_query.setdefault((r["shot"], r["query"]), {})[r["branch"]] = r
    if len(by_query) != 88:
        raise ValueError("same88 reference queries required")
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    bound[str(REFERENCE / "completed.json")] = sha(REFERENCE / "completed.json")
    root.mkdir()
    write(
        root / "protocol.json",
        dict(
            scope="ENDED20_ANCHORED_FORECAST_NOT_ONLINE",
            ids=local["ids"],
            input_hashes=local["input_hashes"],
            source_sha256=previous["source_sha256"],
            cold_sha256=sha(cold),
            heads_sha256=sha(OLD / "heads.pt"),
            bound_code_sha256=bound,
            threshold=previous["threshold"],
            horizon=25,
            initial_condition="current_asinh + forecast[h] - forecast[0]",
            initialization_error="first future point equals current; one-sample approximation",
            future_classifier_state="zero",
            non_autoregressive=True,
            update_head="original unanchored head; only matured observed targets",
            references_reused_from=str(REFERENCE),
            references_sha256=sha(REFERENCE / "references.json"),
            future_availability_assumption="current persists; mode10 carry; MNM unavailable",
            raw_variance_projection="none; count negatives",
            suffix_raw_reads=0,
            optimizer_steps=0,
            independent_confirmation=False,
        ),
    )
    rows, excluded = [], []
    for e in examples:
        sid, q = e["shot"], e["query"]
        signal = signals[sid]
        if q + 25 >= len(signal["values"]):
            excluded.append(
                dict(shot=sid, query=q, reason="full_future25_not_recorded")
            )
            continue
        reference = by_query[(sid, q)]["current"]
        if reference["disruptive"] != e["disruptive"]:
            raise ValueError("reference metadata changed")
        prefix = {k: v[: q + 1] for k, v in signal.items()}
        observed = torch.as_tensor(prefix["values"][-1], device="cuda")
        zlast = torch.asinh((observed - model.base.center) / model.base.scale)
        zero = model.base.center.new_zeros(4, 64)
        future_truth, future_mask = targets(
            {k: v[: q + 26] for k, v in signal.items()}, q, q + 25
        )
        current = zero.new_tensor([reference["logit"]])
        for name in (*heads, "persistence"):
            modes = ("static",) if name == "persistence" else BRANCHES
            for mode in modes:
                if mode == "static":
                    state, receipt = zero, dict(accepted=False, reason="static")
                else:
                    # No later query measurement in the earlier origin update.
                    state, receipt = advance(
                        model,
                        e["origin_pack"],
                        zero,
                        e["truth"][mode],
                        e["mask"],
                        heads[name],
                        kind="parameter",
                    )
                decoder = (
                    PersistenceHead(zlast)
                    if name == "persistence"
                    else AnchoredHead(heads[name], zlast)
                )
                with patch.object(
                    io23,
                    "read_outcome",
                    side_effect=RuntimeError("no outcome during current forecast"),
                ):
                    pred, logits, assumed, negative = infer(
                        model, prefix, basis, decoder, state
                    )
                with torch.no_grad():
                    z = torch.maximum(current, logits.max().reshape_as(current))
                    mask = torch.as_tensor(future_mask & assumed, device="cuda")
                    loss = (
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
                        **{
                            k: e[k]
                            for k in (
                                "shot",
                                "query",
                                "label",
                                "disruptive",
                                "actual_lead_ms",
                            )
                        },
                        head=name,
                        branch=mode,
                        group="disruptive" if e["disruptive"] else "safe",
                        logit=float(z),
                        crossed=bool(z.sigmoid() >= previous["threshold"]),
                        risk_loss=float(
                            margin_loss(z, previous["threshold"], int(e["disruptive"]))
                        ),
                        forecast_loss=loss,
                        negative_variances=negative,
                        forecast_observed=int(mask.sum()),
                        update=receipt,
                        future_probabilities=logits.sigmoid().cpu().tolist(),
                    )
                )
        if len(rows) % 90 == 0:
            print(
                dict(records=len(rows), seconds=time.monotonic() - started), flush=True
            )
    if len(rows) != 792 or len(excluded) != 12:
        raise AssertionError("same88 queries and all9 predictions required")
    summaries = {
        name + "_" + mode: summarize(
            [r for r in rows if r["head"] == name and r["branch"] == mode]
        )
        for name in (*heads, "persistence")
        for mode in (("static",) if name == "persistence" else BRANCHES)
    }
    candidate, baseline = (
        summaries["east_diagnostic_real"],
        summaries["east_diagnostic_static"],
    )
    gate = {
        g + "_improved": candidate[g]["mean_risk_loss"]
        < baseline[g]["mean_risk_loss"] - 1e-9
        for g in ("safe", "disruptive")
    }
    gate["no_negative_variances"] = all(
        candidate[g]["negative_variance_predictions"] == 0
        for g in ("safe", "disruptive")
    )
    for label in (
        "east_diagnostic_reverse",
        "east_diagnostic_hold",
        "persistence_static",
    ):
        gate["better_than_" + label] = (
            candidate["balanced_risk_loss"]
            < summaries[label]["balanced_risk_loss"] - 1e-9
        )
    gate["passed"] = all(gate.values())
    if any(not torch.equal(v, model.state_dict()[k].cpu()) for k, v in initial.items()):
        raise AssertionError("source/cold mutated")
    if any(
        not torch.equal(v, heads[n].state_dict()[k].cpu())
        for n, h in head_initial.items()
        for k, v in h.items()
    ):
        raise AssertionError("head mutated")
    if any(sha(Path(p)) != h for p, h in bound.items()):
        raise AssertionError("bound artifact/code changed")
    write(root / "records.json", rows)
    write(root / "excluded.json", excluded)
    write(root / "summary.json", summaries)
    done = dict(
        status="COMPLETED_FEASIBILITY_DIAGNOSTIC",
        episodes=88,
        records=len(rows),
        excluded=len(excluded),
        gate=gate,
        seconds=time.monotonic() - started,
        source_unchanged=True,
        cold_unchanged=True,
        heads_unchanged=True,
        suffix_raw_reads=0,
        optimizer_steps=0,
        references_reused=True,
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
