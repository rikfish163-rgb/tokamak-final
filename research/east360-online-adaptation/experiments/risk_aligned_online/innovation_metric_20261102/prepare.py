"""Equal-budget ended20 training: change innovation and step geometry only."""

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
from ..critical_shot_meta_20261102.objectives import (
    last_update,
    margin_loss,
    select_critical,
)
from ..critical_shot_meta_20261102.prepare import load_inputs, read_example
from ..critical_shot_meta_20261102.stream import take
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import forward
from ..risk_bridge_forecaster_20261102.core import auxiliary
from .core import innovation_loss
from .stream import simulate, update

PLAN = io23.ROOT / "INNOVATION_METRIC_PLAN_20261102.md"
ARMS = dict(
    unit_operational="unit", metric_operational="metric", metric_forecast="metric"
)


def snapshot(root, parent, cold, scope):
    root.mkdir(parents=True)
    files = {Path(__file__), PLAN}
    files.update(
        Path(m.__file__)
        for name, m in list(sys.modules.items())
        if name.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
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
            seed=131,
            gradient_clip=1.0,
            temporal_gradient_truncation_updates=1,
            first_update=159,
            update_interval=128,
            parameter_unit_step=0.1,
            metric_learning_rate=1.0,
            relative_metric_damping=0.01,
            absolute_metric_damping=1e-8,
            metric_increment_hidden_rms_cap=0.05,
            total_hidden_rms_cap=0.25,
            plan_sha256=sha(PLAN),
            code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
            independent_confirmation=False,
            warmup_cost_retained=True,
        ),
    )
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))


def replay_query(model, example, simulation, arm, coefficient, query):
    maturity = last_update(query)
    if maturity is None:
        return None
    tape = next(t for t in simulation["tapes"][arm] if t["maturity"] == maturity)
    assert maturity < query
    observed, available = targets(example["signal"], tape["origin"], maturity)
    real, hold, mask = [
        torch.as_tensor(v, device=model.base.center.device)
        for v in (observed["real"], observed["hold"], available)
    ]
    point = take(example["pack"], tape["origin"], tape["origin"] + 1)
    state, receipt = update(
        model,
        point,
        tape["before"],
        real,
        hold,
        mask,
        coefficient,
        ARMS[arm],
        create_graph=True,
    )
    assert torch.equal(state.detach(), tape["after"]), (
        "update replay mismatch",
        arm,
        query,
    )
    lo = query // 32 * 32
    later = take(example["pack"], lo, min(lo + 32, len(example["times"])))
    after = forward(model, later, state)["logit"][query - lo]
    assert float(after.detach()) == float(simulation["logits"][arm][query]), (
        "risk replay mismatch",
        arm,
        query,
    )
    with torch.no_grad():
        before = forward(model, later, tape["before"])["logit"][query - lo]
    signal = innovation_loss(
        model, auxiliary(model, point, tape["before"], coefficient), real, mask
    )
    return dict(
        before=before, after=after, signal=signal, receipt=receipt, maturity=maturity
    )


def episode(model, example, simulation, arm, coefficient, threshold):
    query = select_critical(
        example["times"],
        simulation["logits"][arm],
        example["disruptive"],
        example["down"],
    )
    result = replay_query(model, example, simulation, arm, coefficient, query)
    supported = result is not None
    if supported:
        before, after, signal = result["before"], result["after"], result["signal"]
    else:
        before = after = coefficient.new_tensor(float(simulation["logits"][arm][query]))
        signal = coefficient.sum() * 0
        if simulation["tapes"][arm]:
            first = simulation["tapes"][arm][0]
            observed, available = targets(
                example["signal"], first["origin"], first["maturity"]
            )
            truth, mask = [
                torch.as_tensor(v, device=model.base.center.device)
                for v in (observed["real"], available)
            ]
            point = take(example["pack"], first["origin"], first["origin"] + 1)
            signal = innovation_loss(
                model,
                auxiliary(model, point, first["before"], coefficient),
                truth,
                mask,
            )
    label = int(example["disruptive"])
    risk_before = margin_loss(before, threshold, label).detach()
    risk_after = margin_loss(after, threshold, label)
    regularizer = 1e-4 * coefficient.square().mean()
    objective = (
        signal + regularizer
        if arm == "metric_forecast"
        else (risk_after - risk_before) / 0.1 + 0.20 * signal + regularizer
    )
    return objective, dict(
        shot=example["shot"],
        query=query,
        supported=supported,
        maturity=result["maturity"] if supported else None,
        label=label,
        risk_before=float(risk_before),
        risk_after=float(risk_after.detach()),
        signal=float(signal.detach()),
    )


def main(root, *, preflight=False):
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    model, configuration, cold, parent, mapping = load_inputs()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    snapshot(
        root,
        parent,
        cold,
        "INNOVATION_METRIC_PREFLIGHT"
        if preflight
        else "ENDED20_INNOVATION_METRIC_EQUAL_BUDGET",
    )
    if preflight:
        example = read_example(model, basis, mapping, parent, 69854, 448)
        table = torch.full((25, 23), 0.01, device=model.base.center.device)
        heads = {a: (table, "real", geometry) for a, geometry in ARMS.items()}
        cached = simulate(
            model, example["signal"], basis, heads, cached=example["pack"]
        )
        actual = simulate(model, example["signal"], basis, heads)
        checks = []
        for a in ARMS:
            assert np.array_equal(cached["logits"][a], actual["logits"][a])
            assert np.array_equal(cached["probability"][a], actual["probability"][a])
            assert len(cached["tapes"][a]) == 3
            for q in (160, 287, 288, 415, 416):
                variable = table.clone().requires_grad_(True)
                replay = replay_query(model, example, cached, a, variable, q)
                if replay["receipt"]["accepted"]:
                    (gradient,) = torch.autograd.grad(replay["after"], variable)
                    assert torch.isfinite(gradient).all()
                    head_gradient_norm = float(gradient.double().norm())
                else:
                    # A rejected proposal retains the detached burn-in state.
                    # There is no B->risk path; do not fabricate a zero gradient
                    # or require an unused tensor to be differentiated.
                    tape = next(
                        t
                        for t in cached["tapes"][a]
                        if t["maturity"] == replay["maturity"]
                    )
                    assert torch.equal(tape["before"], tape["after"])
                    head_gradient_norm = None
                checks.append(
                    dict(
                        arm=a,
                        query=q,
                        accepted=replay["receipt"]["accepted"],
                        head_gradient_norm=head_gradient_norm,
                    )
                )
        assert all(
            torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items()
        )
        result = dict(
            status="PASS",
            cached_stream_bitwise_equal=True,
            actual_clock_replay_exact=True,
            source_tensors_unchanged=True,
            checks=checks,
            outcome_reads=1,
            suffix_shots_read=0,
            seconds=time.monotonic() - started,
        )
        write(root / "completed.json", result)
        print(json.dumps(result), flush=True)
        return
    check = io23.ROOT / "probe_outputs/innovation_metric_preflight_v2"
    assert read(check / "completed.json")["status"] == "PASS"
    assert read(check / "protocol.json")["plan_sha256"] == sha(PLAN)
    examples = []
    for sid in parent["ids"]:
        examples.append(read_example(model, basis, mapping, parent, sid))
        print(
            json.dumps(
                dict(stage="cache", shot=sid, points=len(examples[-1]["times"]))
            ),
            flush=True,
        )
    groups = {
        flag: [e for e in examples if e["disruptive"] == flag] for flag in (True, False)
    }
    assert len(groups[True]) == 8 and len(groups[False]) == 12
    heads = {a: torch.nn.Parameter(model.base.center.new_zeros(25, 23)) for a in ARMS}
    optimizers = {a: torch.optim.Adam([b], lr=0.001) for a, b in heads.items()}
    rng, history = np.random.default_rng(131), []
    for step in range(128):
        selected = [
            groups[flag][int(rng.integers(len(groups[flag])))] for flag in (True, False)
        ]
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        record = dict(
            step=step + 1,
            sample_shots=[e["shot"] for e in selected],
            episodes={a: [] for a in ARMS},
        )
        for example in selected:
            simulation = simulate(
                model,
                example["signal"],
                basis,
                {a: (b.detach(), "real", ARMS[a]) for a, b in heads.items()},
                cached=example["pack"],
            )
            for a, b in heads.items():
                objective, receipt = episode(
                    model, example, simulation, a, b, float(configuration["threshold"])
                )
                if not torch.isfinite(objective):
                    raise FloatingPointError("nonfinite innovation outer objective")
                (objective / 2).backward()
                record["episodes"][a].append(
                    dict(**receipt, objective=float(objective.detach()))
                )
        record["gradient_norms"] = {}
        for a, b in heads.items():
            norm = torch.nn.utils.clip_grad_norm_([b], 1.0, error_if_nonfinite=True)
            record["gradient_norms"][a] = float(norm)
            optimizers[a].step()
        assert all(p.grad is None for p in model.parameters())
        history.append(record)
        if (step + 1) % 16 == 0:
            print(
                json.dumps(
                    dict(
                        stage="train",
                        step=step + 1,
                        seconds=time.monotonic() - started,
                        objective={
                            a: float(
                                np.mean([r["objective"] for r in record["episodes"][a]])
                            )
                            for a in ARMS
                        },
                    )
                ),
                flush=True,
            )
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    write(root / "heads.json", {a: b.detach().cpu().tolist() for a, b in heads.items()})
    write(root / "history.json", history)
    write(
        root / "training_cohort.json",
        [
            dict(
                shot=e["shot"],
                disruptive=e["disruptive"],
                down=e["down"],
                points=len(e["times"]),
            )
            for e in examples
        ],
    )
    result = dict(
        status="COMPLETED",
        steps=128,
        head_optimizer_steps=128,
        source_tensors_unchanged=True,
        ids=parent["ids"],
        outcome_reads=20,
        suffix_shots_read=0,
        heads_sha256=sha(root / "heads.json"),
        history_sha256=sha(root / "history.json"),
        plan_sha256=sha(PLAN),
        support_counts={
            a: sum(e["supported"] for r in history for e in r["episodes"][a])
            for a in ARMS
        },
        online_gain_measured=False,
        independent_future_confirmation=False,
        seconds=time.monotonic() - started,
    )
    write(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.output, preflight=args.preflight)
