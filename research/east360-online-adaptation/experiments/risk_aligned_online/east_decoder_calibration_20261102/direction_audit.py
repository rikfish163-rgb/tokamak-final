"""Ended-EAST20 loss-direction diagnostic, never an online label policy."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23
from ..critical_shot_meta_20261102.objectives import margin_loss
from ..direct_decoder_meta_20261102.core import advance, clone_head, prediction
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..innovation_metric_20261102.core import innovation_loss, metric_direction
from ..lora_stream_20261102.core import direct, forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.data import SOURCE_SHA
from ..source_event_tasks_20261102.trial import source_basis
from .core import LowRankDecoder
from .run import LOCAL, SOURCE, collect
from .stream360 import CV

PLAN = io23.ROOT / "EAST_FEEDBACK_DIRECTION_AUDIT_PLAN_20261102.md"
STREAM = io23.ROOT / "probe_outputs/east_decoder_complete360_v1"
FRACTIONS = (1.0, 0.25, 0.0625, 0.015625)


def protection_terms(pack, state):
    valid = pack["valid"][:, :16]
    shift = (pack["phi"].double() @ state.double()).masked_fill(~valid[:, :, None], 0)
    return dict(
        parameter_mean_square=float(state.detach().double().square().mean()),
        hidden_mean_square=float(
            shift.detach().square().sum() / (valid.sum().clamp_min(1) * 64)
        ),
    )


def direction_statistics(pack, diagnostic, risk):
    if diagnostic.shape != (4, 64) or risk.shape != diagnostic.shape:
        raise ValueError("two first-FFN4x64 gradients required")
    if not torch.isfinite(risk).all():
        raise ValueError("finite risk gradient required")
    direction = metric_direction(pack, diagnostic).detach()
    d, gd, gr = direction.double(), diagnostic.detach().double(), risk.detach().double()
    norm = float(d.norm() * gr.norm())
    return direction, dict(
        diagnostic_slope=float((gd * d).sum()),
        risk_slope=float((gr * d).sum()),
        risk_cosine=float((gr * d).sum()) / norm if norm else None,
        direction_norm=float(d.norm()),
        diagnostic_gradient_norm=float(gd.norm()),
        risk_gradient_norm=float(gr.norm()),
        step_hidden_rms=protection_terms(pack, direction)["hidden_mean_square"] ** 0.5,
    )


def macro(rows, value):
    by_shot = defaultdict(list)
    for r in rows:
        by_shot[r["shot"]].append(value(r))
    return float(np.mean([np.mean(v) for v in by_shot.values()]))


def summarize(rows):
    fractions = sorted({s["fraction"] for r in rows for s in r["steps"]}, reverse=True)
    result = dict(
        shots=len({r["shot"] for r in rows}),
        episodes=len(rows),
        beneficial_direction_points=sum(r["risk_slope"] < 0 for r in rows),
        harmful_direction_points=sum(r["risk_slope"] > 0 for r in rows),
        shot_macro_beneficial_direction_fraction=macro(
            rows, lambda r: float(r["risk_slope"] < 0)
        ),
        shot_macro_risk_slope=macro(rows, lambda r: r["risk_slope"]),
        steps={},
    )
    for fraction in fractions:
        selected = [
            dict(
                shot=r["shot"],
                **next(s for s in r["steps"] if s["fraction"] == fraction),
            )
            for r in rows
        ]
        result["steps"][str(fraction)] = dict(
            accepted=sum(s["accepted"] for s in selected),
            diagnostic_improved=sum(s["diagnostic_improved"] for s in selected),
            parameter_anchor_blocks=sum(
                s["diagnostic_improved"] and not s["accepted"] for s in selected
            ),
            shot_macro_delta_risk=macro(selected, lambda s: s["delta_risk"]),
            shot_macro_retained_delta_risk=macro(
                selected, lambda s: s["delta_risk"] if s["accepted"] else 0.0
            ),
        )
    return result


def inspect(model, episode, head, mode, risk_gradient, before_risk, threshold):
    variable = model.base.center.new_zeros(4, 64).requires_grad_(True)
    truth, mask = episode["truth"][mode], episode["mask"]
    predicted = prediction(model, episode["origin_pack"], variable, head)
    diagnostic = innovation_loss(model, predicted, truth, mask)
    (gradient,) = torch.autograd.grad(diagnostic, variable)
    direction, stats = direction_statistics(
        episode["origin_pack"], gradient, risk_gradient
    )
    receipt_state, receipt = advance(
        model, episode["origin_pack"], variable.detach(), truth, mask, head
    )
    torch.testing.assert_close(
        gradient.flatten(), gradient.new_tensor(receipt["gradient256"]), atol=0, rtol=0
    )
    steps = []
    for fraction in FRACTIONS:
        proposal = direction * fraction
        with torch.no_grad():
            predicted = prediction(model, episode["origin_pack"], proposal, head)
            forecast = innovation_loss(model, predicted, truth, mask)
            anchor = 0.01 * proposal.square().mean()
            objective = forecast + anchor
            accepted = bool(objective < diagnostic.detach())
            logit = forward(model, episode["later_pack"], proposal)["logit"]
            risk = margin_loss(
                logit, threshold, int(episode["disruptive"])
            )
        steps.append(
            dict(
                fraction=fraction,
                diagnostic_before=float(diagnostic.detach()),
                diagnostic_after=float(forecast),
                parameter_anchor=float(anchor),
                hidden_anchor=0.01
                * protection_terms(episode["origin_pack"], proposal)[
                    "hidden_mean_square"
                ],
                objective_proposal=float(objective),
                accepted=accepted,
                diagnostic_improved=bool(forecast < diagnostic.detach()),
                risk_before=before_risk,
                risk_after=float(risk),
                delta_risk=float(risk) - before_risk,
            )
        )
        if fraction == 1.0:
            if accepted != receipt["accepted"]:
                raise AssertionError("independent diagnostic acceptance differs")
            expected = proposal if accepted else torch.zeros_like(proposal)
            torch.testing.assert_close(receipt_state, expected, atol=0, rtol=0)
    return dict(
        **{
            k: episode[k]
            for k in (
                "shot",
                "origin",
                "query",
                "label",
                "disruptive",
                "actual_lead_ms",
            )
        },
        group="safe"
        if not episode["disruptive"]
        else "danger30_125"
        if episode["actual_lead_ms"] <= 125
        else "disruptive_early",
        mode=mode,
        **stats,
        gradient256=gradient.detach().flatten().cpu().tolist(),
        risk_gradient256=risk_gradient.detach().flatten().cpu().tolist(),
        steps=steps,
    ), direction


def main(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    for name in ("audit_cpu.json", "verification_neural.json"):
        if read(STREAM / name)["status"] != "PASS":
            raise ValueError("verified frozen complete360 reference required")
    protocol = read(STREAM / "protocol.json")
    model, parent, cold = load_model()
    if sha(DEFAULT_PARENT) != SOURCE_SHA or sha(cold) != protocol["cold_sha256"]:
        raise ValueError("source/cold changed")
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    threshold = float(parent["threshold"])
    local = read(LOCAL / "protocol.json")
    if local["ids"] != protocol["head_fit_ids"]:
        raise ValueError("ended20 roster differs")
    mapping = Path(local["calibration_reference"]) / "input_map.json"
    if sha(mapping) != protocol["input_map_sha256"]:
        raise ValueError("input mapping changed")
    ffn_basis = source_basis(model)
    examples, signals = collect(model, ffn_basis, local)
    prior = clone_head(model)
    prior.load_state_dict(
        torch.load(SOURCE / "decoders.pt", map_location="cpu", weights_only=True)[
            "heads"
        ]["forecast_only"],
        strict=True,
    )
    prior.requires_grad_(False)
    basis = torch.load(
        CV / "diagnostic_basis.pt", map_location="cuda", weights_only=True
    )
    saved = torch.load(STREAM / "heads.pt", map_location="cuda", weights_only=True)
    if sha(STREAM / "heads.pt") != read(STREAM / "calibration.json")["heads_sha256"]:
        raise ValueError("calibrated heads changed")
    heads = {name: LowRankDecoder(prior, basis) for name in saved}
    for name, h in heads.items():
        with torch.no_grad():
            h.delta.copy_(saved[name])
        h.requires_grad_(False)
    heads["prior"] = LowRankDecoder(prior, basis).requires_grad_(False)
    initial_heads = {name: h.delta.detach().clone() for name, h in heads.items()}
    root.mkdir()
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    write(
        root / "protocol.json",
        dict(
            scope="POSTHOC_CURRENT_LABEL_DIRECTION_NOT_DEPLOYABLE",
            source_sha256=SOURCE_SHA,
            cold_sha256=sha(cold),
            calibration_heads_sha256=sha(STREAM / "heads.pt"),
            completed360_sha256=sha(STREAM / "completed.json"),
            ids=local["ids"],
            input_map_sha256=sha(mapping),
            input_hashes=local["input_hashes"],
            bound_code_sha256=bound,
            fractions=FRACTIONS,
            n_samples=len(examples),
            threshold=threshold,
            suffix_signals_read=0,
            optimizer_steps=0,
            same20_previously_used_for_cold_and_C=True,
            independent_validation=False,
        ),
    )
    rows, checks, checked = [], [], set()
    for i, e in enumerate(examples):
        variable = model.base.center.new_zeros(4, 64).requires_grad_(True)
        z0 = forward(model, e["later_pack"], variable)["logit"]
        risk = margin_loss(z0, threshold, int(e["disruptive"]))
        (risk_gradient,) = torch.autograd.grad(risk, variable)
        for name, head in heads.items():
            for mode in ("real", "reverse", "hold"):
                row, proposal = inspect(
                    model, e, head, mode, risk_gradient, float(risk.detach()), threshold
                )
                rows.append(dict(arm=name, **row))
                if (
                    name == "worst_meta"
                    and mode == "real"
                    and e["shot"] not in checked
                    and len(checked) < 4
                ):
                    prefix = {
                        k: v[: e["query"] + 1] for k, v in signals[e["shot"]].items()
                    }
                    with torch.no_grad():
                        actual = direct(
                            model, prefix, [e["query"]], ffn_basis, proposal
                        )["logit"]
                        cached = forward(model, e["later_pack"], proposal)["logit"]
                    torch.testing.assert_close(actual, cached, atol=0, rtol=0)
                    checks.append(
                        dict(
                            shot=e["shot"],
                            query=e["query"],
                            actual_ffn_error=float((actual - cached).abs().max()),
                        )
                    )
                    checked.add(e["shot"])
        if (i + 1) % 10 == 0:
            print(
                dict(
                    episodes=i + 1, records=len(rows), seconds=time.monotonic() - start
                ),
                flush=True,
            )
    summary = {
        name: {
            mode: {
                group: summarize(
                    [
                        r
                        for r in rows
                        if (r["arm"], r["mode"], r["group"]) == (name, mode, group)
                    ]
                )
                for group in ("safe", "danger30_125", "disruptive_early")
            }
            for mode in ("real", "reverse", "hold")
        }
        for name in heads
    }
    if any(
        not torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items()
    ) or any(not torch.equal(v, heads[k].delta) for k, v in initial_heads.items()):
        raise AssertionError("frozen model/head changed")
    if any(sha(Path(path)) != digest for path, digest in bound.items()):
        raise AssertionError("frozen diagnostic code changed")
    write(root / "records.json", rows)
    write(root / "summary.json", summary)
    write(root / "neural_checks.json", checks)
    done = dict(
        status="COMPLETED_POSTHOC_DIAGNOSTIC",
        records=len(rows),
        source_unchanged=True,
        cold_unchanged=True,
        heads_unchanged=True,
        suffix_raw_reads=0,
        optimizer_steps=0,
        online_gain_measured=False,
        seconds=time.monotonic() - start,
        artifact_sha256={f.name: sha(f) for f in root.iterdir() if f.is_file()},
    )
    write(root / "completed.json", done)
    print(done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
