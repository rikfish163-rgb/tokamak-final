"""Source validation and ended-cold20 readout diagnostics, not online scores."""

import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..qk_matched_head_20261029.run import snapshot
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import VIEWS, examples, fit_probes, predictions, scores

PREVIOUS = io23.ROOT / "probe_outputs/qk_forecast_east360_v1"


def collect(net, ids, device, heads, mapping=None, slow=None):
    allowed = set(ids)
    reader = io23.read_signals
    calls, values, receipts, exclusions, hashes = [], [], [], [], {}

    def guarded(d, sid, *args, **kwargs):
        if d != device or sid not in allowed:
            raise ValueError("out-of-scope signals")
        calls.append(sid)
        return reader(d, sid, *args, **kwargs)

    def deny(*args, **kwargs):
        raise AssertionError("no outcome/event access in signal readout diagnostic")

    with (
        patch.object(io23, "read_signals", side_effect=guarded),
        patch.object(io23, "read_outcome", side_effect=deny),
    ):
        for sid in ids:
            raw = apply_input_contract(
                simple_data.read_shot(
                    device,
                    sid,
                    rate_contract="nominal" if device == "EAST" else "strict",
                )
            )
            sig = {k: raw[k] for k in ("values", "available")}
            if mapping is not None:
                sig = transform(sig, mapping)
            clocks = np.arange(256, len(sig["values"]) - 24, 128)
            hashes[str(sid)] = sha(io23.path_for(device, sid))
            if not len(clocks):
                exclusions.append(
                    dict(
                        shot=sid,
                        length=len(sig["values"]),
                        reason="no complete H25 after clock256",
                    )
                )
                continue
            clocks = clocks[
                np.unique(
                    np.linspace(0, len(clocks) - 1, min(16, len(clocks))).astype(int)
                )
            ].tolist()
            v = examples(net, sig, clocks, slow)
            v["ids"] = np.full(len(clocks), sid, dtype=np.int64)
            v["clocks"] = np.array(clocks)
            h = torch.as_tensor(v["all_tokens"], device="cuda")
            with torch.no_grad():
                for name, head in heads.items():
                    v[name] = (
                        F.linear(h, head["weight"], head["bias"])
                        .reshape(-1, 25, 23)
                        .cpu()
                        .numpy()
                    )
            values.append(v)
            receipts.append(
                dict(
                    shot=sid,
                    length=len(sig["values"]),
                    clocks=clocks,
                    observed_channels=v["mask"].sum(0).tolist(),
                )
            )
    if calls != ids or not values:
        raise ValueError("unexpected data access or empty sample set")
    data = {k: np.concatenate([v[k] for v in values]) for k in values[0]}
    return data, dict(
        device=device,
        read_ids=calls,
        receipts=receipts,
        exclusions=exclusions,
        input_hashes=hashes,
        outcome_reads=0,
    )


def baselines(data):
    return dict(
        hold=np.repeat(data["current"][:, None], 25, axis=1),
        trend=data["trend"],
        **{k: v for k, v in data.items() if k.endswith("_forecast")},
    )


def fit_info(fits):
    return {
        view: [
            None
            if f is None
            else {k: v for k, v in f.items() if not isinstance(v, np.ndarray)}
            for f in fits[view]
        ]
        for view in VIEWS
    }


def save_fit(path, fits):
    arrays = {}
    for view in VIEWS:
        for c, fit in enumerate(fits[view]):
            if fit is not None:
                for name in ("mean", "scale", "target_mean", "coef"):
                    arrays[f"{view}_{c}_{name}"] = fit[name]
    with path.open("xb") as file:
        np.savez_compressed(file, **arrays)


def save_predictions(path, data, preds, mask):
    with path.open("xb") as file:
        np.savez_compressed(
            file,
            ids=data["ids"],
            clocks=data["clocks"],
            current=data["current"],
            future=data["future"],
            mask=mask,
            **{"prediction_" + k: v for k, v in preds.items()},
        )


def subset(data, take):
    return {k: v[take] for k, v in data.items()}


def run(output):
    if output.exists() or output.absolute().parent != io23.ROOT / "probe_outputs":
        raise ValueError("new isolated worktree output required")
    prior, audit = read(PREVIOUS, "protocol.json"), read(PREVIOUS, "audit.json")
    if not audit["complete"] or audit["complete_sha256"] != sha(
        PREVIOUS / "complete.json"
    ):
        raise ValueError("audited preceding stream required")
    source = Path(prior["source_root"])
    ref = Path(prior["reference"])
    for name, h in prior["reference_hashes"].items():
        if sha(ref / name) != h:
            raise ValueError("fixed cold reference changed")
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("rl_env CUDA required")
    base, parent = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = snapshot(net)
    heads = {}
    paths = dict(
        source_forecast=source / "forecast_head.pt",
        cold_forecast=PREVIOUS / "cold_head.pt",
    )
    expected = dict(
        source_forecast=prior["source_head_sha256"],
        cold_forecast=read(PREVIOUS, "complete.json")["cold_head_sha256"],
    )
    for name, path in paths.items():
        if sha(path) != expected[name]:
            raise ValueError("auxiliary head changed")
        heads[name] = torch.load(path, map_location="cuda", weights_only=True)
    roles = simple_data.source_roles()
    if [len(roles[k]) for k in ("train", "val", "test")] != [1353, 160, 220]:
        raise ValueError("canonical source roles required")
    plan = io23.ROOT / "READOUT_IDENTIFIABILITY_PLAN.md"
    files = {Path(__file__)} | {
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    }
    protocol = dict(
        scope="FROZEN_READOUT_DIAGNOSTIC_NOT_ONLINE_RESULT",
        source_sha256=PARENT_SHA256,
        previous=str(PREVIOUS),
        previous_complete_sha256=sha(PREVIOUS / "complete.json"),
        train_ids=roles["train"],
        val_ids=roles["val"],
        east_ids=prior["order"][:20],
        test_or_suffix_access=False,
        penalty=0.01,
        max_clocks_per_shot=16,
        first_clock=256,
        stride=128,
        threshold=parent["threshold"],
        heads=expected,
        views=VIEWS,
        reference=str(ref),
        reference_hashes=prior["reference_hashes"],
        fold_note="conditional readout only; fixed operator/mapping already used all20; not clean transfer OOF",
        plan_sha256=sha(plan),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        independent_confirmation=False,
    )
    output.mkdir()
    _durable_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(
        output / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
        archive.write(plan, plan.name)
    start = time.monotonic()
    train, tr = collect(
        net, roles["train"], "J-TEXT", {"source_forecast": heads["source_forecast"]}
    )
    val, vr = collect(
        net, roles["val"], "J-TEXT", {"source_forecast": heads["source_forecast"]}
    )
    _durable_json(output / "source_sampling.json", dict(train=tr, val=vr))
    print(
        dict(
            source_train_windows=len(train["ids"]),
            source_val_windows=len(val["ids"]),
            seconds=time.monotonic() - start,
        ),
        flush=True,
    )
    fitted = fit_probes(train)
    save_fit(output / "source_readouts.npz", fitted)
    _durable_json(output / "source_fits.json", fit_info(fitted))
    support = [f is not None for f in fitted["all_tokens"]]
    train_preds = dict(baselines(train), **predictions(fitted, train))
    val_preds = dict(baselines(val), **predictions(fitted, val))
    results = dict(
        source_train=scores(train_preds, train, support),
        source_val=scores(val_preds, val, support),
    )
    save_predictions(
        output / "source_val_predictions.npz",
        val,
        val_preds,
        val["mask"] & np.array(support)[None],
    )
    del train, train_preds, val, val_preds, fitted
    print(
        dict(source_probes_complete=True, seconds=time.monotonic() - start), flush=True
    )
    slow = torch.load(ref / "models/cold.pt", map_location="cuda", weights_only=True)[
        "slow"
    ]
    cold, cr = collect(
        net, prior["order"][:20], "EAST", heads, read(ref, "input_map.json"), slow
    )
    _durable_json(output / "cold_sampling.json", cr)
    pooled = {k: v.copy().astype(float) for k, v in baselines(cold).items()}
    mask = cold["mask"].copy()
    fold_records = []
    for fold in range(5):
        held_ids = prior["order"][:20][fold::5]
        held = np.isin(cold["ids"], held_ids)
        training = subset(cold, ~held)
        fitted = fit_probes(training)
        out = predictions(fitted, subset(cold, held))
        support = np.array([f is not None for f in fitted["all_tokens"]])
        mask[held] &= support[None]
        for k, v in out.items():
            if k not in pooled:
                pooled[k] = np.full((len(cold["ids"]), *v.shape[1:]), np.nan)
            pooled[k][held] = v
        fold_records.append(
            dict(
                fold=fold,
                fit_ids=[s for s in prior["order"][:20] if s not in held_ids],
                held_ids=held_ids,
                fits=fit_info(fitted),
                unestimable_channels=np.flatnonzero(~support).tolist(),
            )
        )
        print(dict(cold_fold=fold, seconds=time.monotonic() - start), flush=True)
    _durable_json(output / "cold_fold_fits.json", fold_records)
    results["cold_readout_oof"] = scores(
        pooled, dict(cold, mask=mask), np.ones(23, bool)
    )
    save_predictions(output / "cold_oof_predictions.npz", cold, pooled, mask)
    fitted = fit_probes(cold)
    save_fit(output / "cold_readouts.npz", fitted)
    _durable_json(output / "cold_fits.json", fit_info(fitted))
    support = np.array([f is not None for f in fitted["all_tokens"]])
    fitted_preds = dict(baselines(cold), **predictions(fitted, cold))
    results["cold_train"] = scores(fitted_preds, cold, support)
    save_predictions(
        output / "cold_train_predictions.npz",
        cold,
        fitted_preds,
        cold["mask"] & support[None],
    )
    _durable_json(output / "results.json", results)
    if (
        any(not torch.equal(v, frozen[k]) for k, v in snapshot(net).items())
        or sha(DEFAULT_PARENT) != PARENT_SHA256
    ):
        raise ValueError("classifier changed")
    for name, h in protocol["code_sha256"].items():
        if sha(io23.ROOT / name) != h:
            raise ValueError("implementation changed during run")
    final = dict(
        complete=True,
        seconds=time.monotonic() - start,
        source_unchanged=True,
        test_or_suffix_access=False,
        artifact_sha256={f.name: sha(f) for f in output.iterdir() if f.is_file()},
        scientific_goal_achieved=False,
        independent_confirmation=False,
    )
    _durable_json(output / "complete.json", final)
    print(
        {
            stage: {
                k: dict(smooth_l1=v["smooth_l1"], mse=v["mse"], shots=v["shots"])
                for k, v in r.items()
            }
            for stage, r in results.items()
            if stage != "source_train"
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
