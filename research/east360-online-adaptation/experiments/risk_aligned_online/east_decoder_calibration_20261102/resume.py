"""Continue an interrupted fold-level experiment without replacing old evidence.

Only whole completed folds can be reused. No optimizer state was checkpointed
inside a fold, so unfinished folds start from the declared zero initialization.
This is explicit interruption recovery, not an error/accuracy fallback.
"""

import argparse
from pathlib import Path
import time

import torch

from ..direct_decoder_meta_20261102.core import clone_head
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..source_event_tasks_20261102.data import SOURCE_SHA
from ..source_event_tasks_20261102.trial import source_basis
from .core import LowRankDecoder
from .run import LOCAL, SOURCE, ARMS, collect, fit, evaluate, summarize, gates, folds


def completed_folds(root):
    root = Path(root)
    if (root / "completed.json").exists():
        raise FileExistsError(root / "completed.json")
    found = []
    for index in range(4):
        pair = [root / f"fold{index}_training.json", root / f"fold{index}_heads.pt"]
        if pair[0].exists() != pair[1].exists():
            raise ValueError("incomplete checkpoint pair requires investigation")
        if all(p.exists() for p in pair):
            found.append(index)
    return found


def main(root):
    root = Path(root)
    reused = completed_folds(root)
    if any(
        (root / n).exists() for n in ("records.json", "summary.json", "resumption.json")
    ):
        raise FileExistsError("ambiguous existing resume/output artifact")
    protocol = read(root / "protocol.json")
    for path, digest in protocol["bound_code_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("interrupted run's frozen code changed")
    preserved = {
        f.name: sha(f)
        for i in reused
        for f in (root / f"fold{i}_training.json", root / f"fold{i}_heads.pt")
    }
    configure()
    started = time.monotonic()
    p = read(LOCAL / "protocol.json")
    model, parent, cold = load_model()
    if (
        sha(cold) != protocol["cold_sha256"]
        or sha(SOURCE / "decoders.pt") != protocol["prior_sha256"]
    ):
        raise ValueError("parent/prior changed since interruption")
    if (
        sha(Path(p["calibration_reference"]) / "input_map.json")
        != protocol["input_map_sha256"]
    ):
        raise ValueError("input mapping changed")
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    prior = clone_head(model)
    saved = torch.load(SOURCE / "decoders.pt", map_location="cpu", weights_only=True)
    prior.load_state_dict(saved["heads"]["forecast_only"], strict=True)
    prior.eval().requires_grad_(False)
    basis = torch.load(
        root / "diagnostic_basis.pt", map_location="cuda", weights_only=True
    )
    examples, _ = collect(model, source_basis(model), p)
    if (
        folds(examples, p["ids"]) != protocol["folds"]
        or float(parent["threshold"]) != protocol["threshold"]
    ):
        raise ValueError("fold roster or threshold drift")
    receipt = dict(
        reason="user turn interruption; old session missing, GPU/pgrep confirmed stopped",
        reused_completed_folds=reused,
        preserved_sha256=preserved,
        old_protocol_sha256=sha(root / "protocol.json"),
        resumed_code_sha256=sha(Path(__file__)),
        optimizer_midfold_resume=False,
        unfinished_fold_policy="declared zero initialization and seed131; no completed fold retraining",
    )
    write(root / "resumption.json", receipt)
    rows = []
    for fold in protocol["folds"]:
        index = fold["fold"]
        if index in reused:
            state = torch.load(
                root / f"fold{index}_heads.pt", map_location="cuda", weights_only=True
            )
            history = read(root / f"fold{index}_training.json")
            if len(history) != 128 or set(state) != set(ARMS):
                raise ValueError("completed fold checkpoint schema mismatch")
            heads = {a: LowRankDecoder(prior, basis) for a in ARMS}
            for a, h in heads.items():
                with torch.no_grad():
                    h.delta.copy_(state[a])
                h.delta.requires_grad_(False)
            print(dict(stage="reuse_completed_fold", fold=index), flush=True)
        else:
            training = [e for e in examples if e["shot"] in fold["fit"]]
            heads, history = fit(
                model, training, fold["fit"], prior, basis, protocol["threshold"]
            )
            write(root / f"fold{index}_training.json", history)
            torch.save(
                {a: h.delta.cpu() for a, h in heads.items()},
                root / f"fold{index}_heads.pt",
            )
        heads["prior"] = LowRankDecoder(prior, basis).requires_grad_(False)
        rows.extend(
            evaluate(
                model,
                [e for e in examples if e["shot"] in fold["held"]],
                heads,
                protocol["threshold"],
                index,
            )
        )
    summary = {
        arm: summarize([r for r in rows if r["arm"] == arm])
        for arm in sorted({r["arm"] for r in rows})
    }
    write(root / "records.json", rows)
    write(root / "summary.json", summary)
    if any(sha(root / name) != digest for name, digest in preserved.items()):
        raise AssertionError("completed-fold evidence overwritten")
    if sha(DEFAULT_PARENT) != SOURCE_SHA or any(
        not torch.equal(v, model.state_dict()[k].detach().cpu())
        for k, v in frozen.items()
    ):
        raise AssertionError("source/cold changed")
    if any(
        sha(Path(path)) != digest
        for path, digest in protocol["bound_code_sha256"].items()
    ):
        raise AssertionError("frozen experiment code changed")
    done = dict(
        status="COMPLETED",
        resumed=True,
        seconds=time.monotonic() - started,
        runtime_scope="resumed invocation only; interrupted runtime unavailable",
        n_rows=len(rows),
        n_samples=100,
        gate=gates(summary),
        source_unchanged=True,
        cold_unchanged=True,
        suffix_raw_reads=0,
        online_gain_measured=False,
        method_promoted=False,
        artifact_sha256={f.name: sha(f) for f in root.iterdir() if f.is_file()},
    )
    write(root / "completed.json", done)
    print(done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
