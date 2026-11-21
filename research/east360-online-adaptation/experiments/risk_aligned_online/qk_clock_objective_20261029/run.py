"""Source-only factorial test with the previous run's exact episode manifest."""

import argparse
from pathlib import Path
import sys
import time
import zipfile

import torch

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..context_memory_20261028.checkpoint import read
from ..qk_matched_head_20261029.learning import GROUPS, query_pools, paired_risk
from ..qk_matched_head_20261029.run import load_split, snapshot, group_summary
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import assemble, train_step

PRIOR = io23.ROOT / "probe_outputs/qk_matched_source_v1"
ARMS = {f"{c}_{o}": (c, o) for c in ("blocked", "fresh") for o in ("mean", "worst")}


def source_files(entry):
    """-m executes the entry as __main__, outside the package-name filter."""
    return {Path(entry)} | {
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    }


def evaluate(nets, shots, prior):
    indices = {s["shot"]: i for i, s in enumerate(shots)}
    old = prior["per_endpoint"]["matched_qk"]
    choices = [(indices[r["shot"]], r["query"]) for r in old]
    rows = {}
    receipts = {c: [] for c in ("blocked", "fresh")}
    for net in nets.values():
        net.requires_grad_(False)
    for lo in range(0, len(choices), 12):
        episodes = {
            c: assemble(shots, choices[lo : lo + 12], "cuda", c) for c in receipts
        }
        for c in receipts:
            receipts[c].extend(episodes[c]["receipts"])
        baseline = None
        for name, net in nets.items():
            clocks = (ARMS[name][0],) if name in ARMS else ("blocked", "fresh")
            modes = ("true", "reverse", "hold") if name in ARMS else ("true",)
            for c in clocks:
                for mode in modes:
                    key = f"{name}_{mode}" if name in ARMS else f"{name}_{c}"
                    e = episodes[c]
                    r = paired_risk(net, e, mode=mode)
                    if baseline is None:
                        baseline = r["before"].detach()
                    elif not torch.equal(r["before"].detach(), baseline):
                        raise ValueError("static risk changed")
                    values = {
                        k: v.detach().cpu().numpy()
                        for k, v in r.items()
                        if k != "alignment"
                    }
                    target = rows.setdefault(key, [])
                    for j, receipt in enumerate(e["receipts"]):
                        target.append(
                            dict(
                                shot=receipt["shot"],
                                query=receipt["query"],
                                group=old[lo + j]["group"],
                                **{k: float(v[j]) for k, v in values.items()},
                            )
                        )
    summary = {k: group_summary(v) for k, v in rows.items()}
    screen = {}
    for name, (c, _) in ARMS.items():
        m = summary[f"{name}_true"]
        screen[name] = all(
            m[g]["mean_risk_change"] <= 0 for g in ("safe", "danger")
        ) and all(
            m["macro_change"] < summary[k]["macro_change"] - 1e-6
            for k in (
                f"original_{c}",
                f"forecast_{c}",
                f"{name}_reverse",
                f"{name}_hold",
            )
        )
    return dict(
        summary=summary,
        per_endpoint=rows,
        receipts=receipts,
        source_screen=screen,
        not_east_validation=True,
        not_shot_tpr_fpr=True,
    )


def run(output):
    if output.exists() or output.absolute().parent != io23.ROOT / "probe_outputs":
        raise ValueError("new worktree output required")
    previous = read(PRIOR, "complete.json")
    audit = read(PRIOR, "audit.json")
    if not audit["all_validation_replayed"] or audit["complete_sha256"] != sha(
        PRIOR / "complete.json"
    ):
        raise ValueError("audited prior required")
    training = read(PRIOR, "training.json")
    prior = read(PRIOR, "validation.json")
    if (
        sha(PRIOR / "training.json") != previous["training_sha256"]
        or sha(PRIOR / "validation.json") != previous["validation_sha256"]
    ):
        raise ValueError("prior manifest changed")
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("rl_env CUDA required")
    roles = simple_data.source_roles()
    advanced = simple_data.advanced_times()
    if [len(roles[k]) for k in ("train", "val", "test")] != [1353, 160, 220] or len(
        set(sum(roles.values(), []))
    ) != 1733:
        raise ValueError("canonical disjoint roles required")
    nets = {}
    snapshots = {}
    opts = {}
    for name in list(ARMS) + ["original", "forecast"]:
        base, parent = load_parent(DEFAULT_PARENT)
        base.cuda()
        nets[name] = MatrixAttention(base)
        if name == "forecast":
            if sha(PRIOR / "forecast_head.pt") != previous["heads"]["forecast"]:
                raise ValueError("forecast reference changed")
            base.forecast_head.load_state_dict(
                torch.load(
                    PRIOR / "forecast_head.pt", map_location="cuda", weights_only=True
                )
            )
        snapshots[name] = snapshot(nets[name])
        if name in ARMS:
            base.forecast_head.requires_grad_(True)
            opts[name] = torch.optim.AdamW(
                base.forecast_head.parameters(), lr=1e-4, weight_decay=0
            )
    files = source_files(__file__)
    plan = io23.ROOT / "QK_CLOCK_OBJECTIVE_PLAN.md"
    protocol = dict(
        source_sha256=PARENT_SHA256,
        prior_complete_sha256=sha(PRIOR / "complete.json"),
        plan_sha256=sha(plan),
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in files},
        arms=ARMS,
        steps=512,
        batch_size=12,
        seed=131,
        lr=1e-4,
        weight_decay=0,
        clip=1,
        inner_step=0.05,
        train_ids=roles["train"],
        val_ids=roles["val"],
        test_access=False,
        east_access=False,
        threshold=parent["threshold"],
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        episode_manifest_sha256=previous["training_sha256"],
        forecast_control="previous blocked-clock forecast-only fit",
    )
    output.mkdir()
    _durable_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, str(p.relative_to(io23.ROOT)))
        z.write(plan, plan.name)
    started = time.monotonic()
    train = load_split(
        roles["train"], {k: advanced[k] for k in roles["train"] if k in advanced}
    )
    pools, _ = query_pools(train)
    indices = {s["shot"]: i for i, s in enumerate(train)}
    eligible = {
        g: {train[i]["shot"]: set(map(int, q)) for i, q in entries}
        for g, entries in pools.items()
    }
    history = []
    for step in range(512):
        selected = training["receipts"][12 * step : 12 * (step + 1)]
        if len(selected) != 12:
            raise ValueError("identical episode budget required")
        for j, r in enumerate(selected):
            if r["query"] not in eligible[GROUPS[j // 4]].get(r["shot"], set()):
                raise ValueError("episode group mismatch")
        choices = [(indices[r["shot"]], r["query"]) for r in selected]
        episodes = {
            c: assemble(train, choices, "cuda", c) for c in ("blocked", "fresh")
        }
        if episodes["blocked"]["receipts"] != selected:
            raise ValueError("replayed training episode changed")
        row = dict(step=step + 1)
        for name, (c, o) in ARMS.items():
            row[name] = train_step(nets[name], opts[name], episodes[c], o)
        history.append(row)
        if (step + 1) % 64 == 0:
            print(
                dict(
                    step=step + 1,
                    seconds=time.monotonic() - started,
                    delta={n: row[n]["delta"] for n in ARMS},
                ),
                flush=True,
            )
    fit_seconds = time.monotonic() - started
    for name in ARMS:
        torch.save(
            {
                k: v.detach().cpu()
                for k, v in nets[name].base.forecast_head.state_dict().items()
            },
            output / f"{name}_head.pt",
        )
    _durable_json(
        output / "training.json",
        dict(history=history, prior_receipts_sha256=previous["training_sha256"]),
    )
    del train, pools, eligible
    val = load_split(
        roles["val"], {k: advanced[k] for k in roles["val"] if k in advanced}
    )
    result = evaluate(nets, val, prior)
    _durable_json(output / "validation.json", result)
    for name, net in nets.items():
        if any(
            not torch.equal(v, snapshots[name][k]) for k, v in snapshot(net).items()
        ):
            raise ValueError("classifier modified")
    for path, h in protocol["code_sha256"].items():
        if sha(io23.ROOT / path) != h:
            raise ValueError("implementation changed")
    for sid, h in read(PRIOR, "input_hashes.json").items():
        if sha(io23.path_for("J-TEXT", int(sid))) != h:
            raise ValueError("source input changed")
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise ValueError("source checkpoint changed")
    done = dict(
        complete=True,
        seconds=time.monotonic() - started,
        fit_seconds=fit_seconds,
        source_screen=result["source_screen"],
        target_or_test_signals_read=0,
        source_unchanged=True,
        protocol_sha256=sha(output / "protocol.json"),
        training_sha256=sha(output / "training.json"),
        validation_sha256=sha(output / "validation.json"),
        heads={n: sha(output / f"{n}_head.pt") for n in ARMS},
        scientific_goal_achieved=False,
    )
    _durable_json(output / "complete.json", done)
    print(
        dict(
            source_screen=result["source_screen"],
            changes={
                n: {g: r[g]["mean_risk_change"] for g in GROUPS}
                for n, r in result["summary"].items()
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
