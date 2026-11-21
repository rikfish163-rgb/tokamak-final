"""Ended20 ablation: extend only the risk meta-gradient through the full W path."""

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23
from ..adaptivity_capacity_20261102.model import source_basis
from ..critical_shot_meta_20261102.objectives import last_update, margin_loss, select_critical
from ..critical_shot_meta_20261102.prepare import load_inputs, read_example
from ..critical_shot_meta_20261102.stream import take
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import sha, write
from ..innovation_metric_20261102.core import innovation_loss
from ..innovation_metric_20261102.stream import simulate
from ..lora_stream_20261102.core import forward
from ..risk_bridge_forecaster_20261102.core import auxiliary
from .core import advance

PLAN = io23.ROOT / "FULL_PATH_META_PLAN_20261102.md"
ARM = "full_operational"
ARMS = {ARM: "metric"}


def snapshot(root, parent, cold, scope):
    root.mkdir(parents=True)
    files = {Path(__file__), PLAN}
    files.update(
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.") and getattr(m, "__file__", None)
    )
    write(
        root / "protocol.json",
        dict(
            scope=scope,
            ids=parent["ids"],
            source_sha256=parent["source_sha256"],
            cold_state_sha256=sha(cold),
            input_map_sha256=parent["input_map_sha256"],
            heads=ARMS,
            forecast_parameters_per_head=575,
            steps=128,
            optimizer="Adam",
            learning_rate=0.001,
            gradient_clip=1.0,
            seed=131,
            initial_head="zero",
            temporal_gradient_truncation_updates=None,
            risk_meta_gradient_horizon="all_matured_updates_before_fixed_query",
            diagnostic_incoming_state_detached=True,
            first_update=159,
            first_possible_effect=160,
            update_interval=128,
            metric_learning_rate=1.0,
            relative_metric_damping=0.01,
            absolute_metric_damping=1e-8,
            metric_increment_hidden_rms_cap=0.05,
            total_hidden_rms_cap=0.25,
            reset_each_shot=True,
            warmup_cost_retained=True,
            outcome_authority="HDF5",
            lead_strictly_greater_ms=30,
            suffix_shots_read=0,
            plan_sha256=sha(PLAN),
            independent_confirmation=False,
            code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
        ),
    )
    with zipfile.ZipFile(root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))


def diagnostic(model, example, tape, coefficient):
    observed, available = targets(example["signal"], tape["origin"], tape["maturity"])
    truth, mask = [torch.as_tensor(v, device=model.base.center.device) for v in (observed["real"], available)]
    point = take(example["pack"], tape["origin"], tape["origin"] + 1)
    signal = innovation_loss(model, auxiliary(model, point, tape["before"].detach(), coefficient), truth, mask)
    return point, truth, mask, signal


def replay_full_query(model, example, simulation, arm, coefficient, query):
    assert arm == ARM and 0 <= query < len(example["times"])
    maturity = last_update(query)
    if maturity is None:
        return None
    tapes = [t for t in simulation["tapes"][arm] if t["maturity"] < query]
    assert [t["maturity"] for t in tapes] == list(range(159, maturity + 1, 128)), "replay clock mismatch"
    state, receipts = coefficient.new_zeros(4, 64), []
    for tape in tapes:
        assert tape["origin"] == tape["maturity"] - 25 and tape["first_affected"] == tape["maturity"] + 1
        assert torch.equal(state.detach(), tape["before"]), "incoming state replay mismatch"
        observed, available = targets(example["signal"], tape["origin"], tape["maturity"])
        truth, mask = [torch.as_tensor(v, device=model.base.center.device) for v in (observed["real"], available)]
        point = take(example["pack"], tape["origin"], tape["origin"] + 1)
        state, receipt = advance(model, point, state, truth, mask, coefficient, create_graph=True)
        assert torch.equal(state.detach(), tape["after"]), "state replay mismatch"
        assert receipt == tape["receipt"], "receipt replay mismatch"
        receipts.append(receipt)
    lo = query // 32 * 32
    later = take(example["pack"], lo, min(lo + 32, len(example["times"])))
    after = forward(model, later, state)["logit"][query - lo]
    assert float(after.detach()) == float(simulation["logits"][arm][query]), "risk replay mismatch"
    with torch.no_grad():
        before = forward(model, later, tapes[-1]["before"].detach())["logit"][query - lo]
    _, _, _, signal = diagnostic(model, example, tapes[-1], coefficient)
    return dict(after=after, before=before, signal=signal, state=state, receipts=receipts, maturity=maturity)


def episode(model, example, simulation, arm, coefficient, threshold):
    query = select_critical(example["times"], simulation["logits"][arm], example["disruptive"], example["down"])
    result = replay_full_query(model, example, simulation, arm, coefficient, query)
    if result is not None:
        before, after, signal = result["before"], result["after"], result["signal"]
    else:
        before = after = coefficient.new_tensor(float(simulation["logits"][arm][query]))
        signal = coefficient.sum() * 0
        if simulation["tapes"][arm]:
            _, _, _, signal = diagnostic(model, example, simulation["tapes"][arm][0], coefficient)
    label = int(example["disruptive"])
    risk_before = margin_loss(before, threshold, label).detach()
    risk_after = margin_loss(after, threshold, label)
    objective = (risk_after - risk_before) / 0.1 + 0.20 * signal + 1e-4 * coefficient.square().mean()
    receipts = result["receipts"] if result is not None else []
    return objective, dict(
        shot=example["shot"],
        query=query,
        supported=result is not None,
        maturity=result["maturity"] if result is not None else None,
        label=label,
        risk_before=float(risk_before),
        risk_after=float(risk_after.detach()),
        signal=float(signal.detach()),
        replay_updates=len(receipts),
        accepted_path_updates=sum(r["accepted"] for r in receipts),
    )


def run_preflight(model, basis, mapping, parent):
    example = read_example(model, basis, mapping, parent, 69854, 448)
    table = model.base.center.new_full((25, 23), 0.01)
    heads = {ARM: (table, "real", "metric")}
    cached = simulate(model, example["signal"], basis, heads, cached=example["pack"])
    actual = simulate(model, example["signal"], basis, heads)
    assert np.array_equal(cached["logits"][ARM], actual["logits"][ARM])
    assert np.array_equal(cached["probability"][ARM], actual["probability"][ARM])
    assert len(cached["tapes"][ARM]) == len(actual["tapes"][ARM]) == 3
    for old, new in zip(cached["tapes"][ARM], actual["tapes"][ARM], strict=True):
        assert all(torch.equal(old[k], new[k]) for k in ("before", "after"))
        assert old["receipt"] == new["receipt"]
    checks = []
    for query in (160, 287, 288, 415, 416):
        variable = table.clone().requires_grad_(True)
        replay = replay_full_query(model, example, cached, ARM, variable, query)
        (gradient,) = torch.autograd.grad(replay["after"], variable, allow_unused=True)
        accepted = sum(r["accepted"] for r in replay["receipts"])
        if accepted:
            assert gradient is not None and torch.isfinite(gradient).all()
            norm = float(gradient.double().norm())
        else:
            assert gradient is None
            norm = None
        checks.append(
            dict(
                query=query,
                maturity=replay["maturity"],
                replay_updates=len(replay["receipts"]),
                accepted_path_updates=accepted,
                last_accepted=replay["receipts"][-1]["accepted"],
                head_gradient_norm=norm,
            )
        )
    return dict(
        status="PASS",
        cached_stream_bitwise_equal=True,
        full_path_replay_exact=True,
        checks=checks,
        outcome_reads=1,
        suffix_shots_read=0,
    )


def main(root, *, preflight=False):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    model, configuration, cold, parent, mapping = load_inputs()
    assert len(parent["ids"]) == len(set(parent["ids"])) == 20
    assert all(not p.requires_grad for p in model.parameters())
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    snapshot(root, parent, cold, "FULL_PATH_META_PREFLIGHT" if preflight else "ENDED20_FULL_PATH_RISK_META")
    if preflight:
        result = run_preflight(model, basis, mapping, parent)
    else:
        examples = [read_example(model, basis, mapping, parent, sid) for sid in parent["ids"]]
        groups = {flag: [e for e in examples if e["disruptive"] == flag] for flag in (True, False)}
        assert len(groups[True]) == 8 and len(groups[False]) == 12
        coefficient = torch.nn.Parameter(model.base.center.new_zeros(25, 23))
        optimizer = torch.optim.Adam([coefficient], lr=0.001)
        rng, history = np.random.default_rng(131), []
        for step in range(128):
            selected = [groups[flag][int(rng.integers(len(groups[flag])))] for flag in (True, False)]
            optimizer.zero_grad(set_to_none=True)
            record = dict(step=step + 1, sample_shots=[e["shot"] for e in selected], episodes={ARM: []})
            for example in selected:
                simulation = simulate(
                    model,
                    example["signal"],
                    basis,
                    {ARM: (coefficient.detach(), "real", "metric")},
                    cached=example["pack"],
                )
                objective, receipt = episode(
                    model, example, simulation, ARM, coefficient, float(configuration["threshold"])
                )
                if not torch.isfinite(objective):
                    raise FloatingPointError("nonfinite full-path outer objective")
                (objective / 2).backward()
                record["episodes"][ARM].append(dict(**receipt, objective=float(objective.detach())))
            norm = torch.nn.utils.clip_grad_norm_([coefficient], 1.0, error_if_nonfinite=True)
            record["gradient_norms"] = {ARM: float(norm)}
            optimizer.step()
            assert all(p.grad is None for p in model.parameters())
            history.append(record)
            if (step + 1) % 16 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="train",
                            step=step + 1,
                            seconds=time.monotonic() - started,
                            objective=float(np.mean([r["objective"] for r in record["episodes"][ARM]])),
                        )
                    ),
                    flush=True,
                )
        write(root / "heads.json", {ARM: coefficient.detach().cpu().tolist()})
        write(root / "history.json", history)
        write(
            root / "training_cohort.json",
            [
                dict(shot=e["shot"], disruptive=e["disruptive"], down=e["down"], points=len(e["times"]))
                for e in examples
            ],
        )
        result = dict(
            status="COMPLETED",
            steps=128,
            head_optimizer_steps=128,
            number_heads=1,
            ids=parent["ids"],
            outcome_reads=20,
            suffix_shots_read=0,
            heads_sha256=sha(root / "heads.json"),
            history_sha256=sha(root / "history.json"),
            training_cohort_sha256=sha(root / "training_cohort.json"),
            plan_sha256=sha(PLAN),
            support_counts={ARM: sum(r["episodes"][ARM][i]["supported"] for r in history for i in range(2))},
            online_gain_measured=False,
            independent_future_confirmation=False,
        )
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result.update(source_tensors_unchanged=True, seconds=time.monotonic() - started)
    write(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.output, preflight=args.preflight)
