"""Train three equal-budget auxiliary heads after all cold20 predictions ended."""

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..warmstart_alignment_20261022.alignment import transform
from .replay import episode, replay_query
from .stream import cache_encoding, simulate

PLAN = io23.ROOT / "CRITICAL_SHOT_META_PLAN_20261102.md"
SOURCE = io23.ROOT / "probe_outputs/difference_feedback_probe20_v1"
SOURCE_SHA = "2af94de43602e66cf7936b2fd9ccf7943759c42d31fb09a109941ad29d79fe5d"
ARMS = ("forecast", "point", "operational")


def load_inputs():
    assert sha(SOURCE / "completed.json") == SOURCE_SHA
    parent = read(SOURCE / "protocol.json")
    model, configuration, cold = load_model()
    assert sha(cold) == parent["cold_state_sha256"]
    path = Path(parent["calibration_reference"]) / "input_map.json"
    assert sha(path) == parent["input_map_sha256"]
    return model, configuration, cold, parent, read(path)


def snapshot(root, parent, cold, mode):
    root.mkdir(parents=True)
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    files.update(
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    )
    protocol = dict(
        scope=mode,
        source_completed_sha256=SOURCE_SHA,
        source_sha256=parent["source_sha256"],
        cold_state_sha256=sha(cold),
        input_map_sha256=parent["input_map_sha256"],
        ids=parent["ids"],
        forecast_parameters_per_head=575,
        heads=ARMS,
        optimizer="Adam",
        learning_rate=0.001,
        steps=128,
        gradient_clip=1.0,
        seed=131,
        temporal_gradient_truncation_updates=1,
        first_update=159,
        update_interval=128,
        warmup_cost_retained=True,
        independent_confirmation=False,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))


def read_example(model, basis, mapping, parent, sid, limit=None):
    assert sha(io23.path_for("EAST", sid)) == parent["input_hashes"][str(sid)]
    raw = apply_input_contract(
        simple_data.read_shot("EAST", sid, rate_contract="nominal")
    )
    if limit is not None:
        raw = {
            **raw,
            "values": raw["values"][:limit],
            "available": raw["available"][:limit],
            "times": raw["times"][:limit],
        }
    signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
    flag, down = io23.read_outcome("EAST", sid)
    return dict(
        shot=sid,
        times=raw["times"],
        signal=signal,
        disruptive=flag,
        down=down,
        pack=cache_encoding(model, signal, basis),
    )


def preflight(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    model, _, cold, parent, mapping = load_inputs()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    snapshot(root, parent, cold, "ENDED_SHOT_ACTUAL_CLOCK_REPLAY_PREFLIGHT")
    example = read_example(model, basis, mapping, parent, 69854, 448)
    head = model.base.center.new_zeros(25, 23)
    other = torch.full_like(head, 0.01)
    heads = {"zero": (head, "real"), "nonzero": (other, "real")}
    cached = simulate(model, example["signal"], basis, heads, cached=example["pack"])
    actual = simulate(model, example["signal"], basis, heads)
    checks = []
    for arm, (b, _) in heads.items():
        assert np.array_equal(cached["logits"][arm], actual["logits"][arm])
        assert np.array_equal(cached["probability"][arm], actual["probability"][arm])
        assert len(cached["tapes"][arm]) == 3
        for q in (160, 287, 288, 415, 416):
            variable = b.clone().requires_grad_(True)
            result = replay_query(model, example, cached, arm, variable, q)
            (gradient,) = torch.autograd.grad(result["after"], variable)
            assert torch.isfinite(gradient).all()
            checks.append(
                dict(
                    arm=arm,
                    query=q,
                    maturity=result["maturity"],
                    head_gradient_norm=float(gradient.double().norm()),
                )
            )
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result = dict(
        status="PASS",
        shot=69854,
        points=448,
        cached_stream_bitwise_equal=True,
        actual_clock_cumulative_state_replay_exact=True,
        checks=checks,
        source_tensors_unchanged=True,
        raw_signal_shots_read=[69854],
        outcome_reads=1,
        suffix_shots_read=0,
    )
    write(root / "completed.json", result)
    print(json.dumps(result), flush=True)


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    check = io23.ROOT / "probe_outputs/critical_shot_preflight_v1"
    assert read(check / "completed.json")["status"] == "PASS"
    assert read(check / "protocol.json")["plan_sha256"] == sha(PLAN)
    model, configuration, cold, parent, mapping = load_inputs()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    snapshot(root, parent, cold, "ENDED20_COMPLETE_SHOT_CRITICAL_REPLAY_HEAD_FIT")
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
        flag: [e for e in examples if e["disruptive"] == flag] for flag in (False, True)
    }
    assert len(groups[True]) == 8 and len(groups[False]) == 12
    heads = {
        arm: torch.nn.Parameter(model.base.center.new_zeros(25, 23)) for arm in ARMS
    }
    optimizers = {a: torch.optim.Adam([b], lr=0.001) for a, b in heads.items()}
    rng = np.random.default_rng(131)
    history = []
    for step in range(128):
        selected = [
            groups[flag][int(rng.integers(len(groups[flag])))] for flag in (True, False)
        ]
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
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
                {a: (b.detach(), "real") for a, b in heads.items()},
                cached=example["pack"],
            )
            for arm, b in heads.items():
                objective, receipt = episode(
                    model,
                    example,
                    simulation,
                    arm,
                    b,
                    float(configuration["threshold"]),
                )
                if not torch.isfinite(objective):
                    raise FloatingPointError(
                        "nonfinite critical endpoint outer objective"
                    )
                (objective / 2).backward()
                record["episodes"][arm].append(
                    dict(**receipt, objective=float(objective.detach()))
                )
        record["gradient_norms"] = {}
        for arm, b in heads.items():
            norm = torch.nn.utils.clip_grad_norm_([b], 1.0, error_if_nonfinite=True)
            record["gradient_norms"][arm] = float(norm)
            optimizers[arm].step()
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
    write(root / "history.json", history)
    write(root / "heads.json", {a: b.detach().cpu().tolist() for a, b in heads.items()})
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
        head_optimizer_steps=128,
        number_heads=3,
        source_tensors_unchanged=True,
        ids=parent["ids"],
        outcome_reads=20,
        suffix_shots_read=0,
        support_counts={
            a: sum(r["supported"] for step in history for r in step["episodes"][a])
            for a in ARMS
        },
        heads_sha256=sha(root / "heads.json"),
        history_sha256=sha(root / "history.json"),
        plan_sha256=sha(PLAN),
        independent_future_confirmation=False,
        online_gain_measured=False,
        seconds=time.monotonic() - started,
    )
    write(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    (preflight if args.preflight else main)(args.output)
