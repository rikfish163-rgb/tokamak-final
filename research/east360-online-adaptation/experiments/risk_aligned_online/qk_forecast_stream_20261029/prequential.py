"""Post-hoc next-window forecasting diagnostic; never selects online updates."""

import argparse
from pathlib import Path
from unittest.mock import patch
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import feedback_step


def channel_losses(base, prediction, truth, mask):
    mask = mask & torch.isfinite(truth)
    target = torch.asinh(
        (torch.where(mask, truth, base.center) - base.center) / base.scale
    )
    losses = F.smooth_l1_loss(prediction, target, reduction="none").masked_fill(
        ~mask, 0.0
    )
    return losses.double().sum((0, 1)).cpu().tolist(), mask.sum((0, 1)).cpu().tolist()


def run(root):
    path = root / "prequential.json"
    if path.exists():
        raise FileExistsError(path)
    p = read(root, "protocol.json")
    audit = read(root, "audit.json")
    if not audit["complete"] or audit["complete_sha256"] != sha(root / "complete.json"):
        raise ValueError("audited stream required")
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise ValueError("source changed")
    code_before = sha(Path(__file__))
    plan = io23.ROOT / "QK_FORECAST_PREQUENTIAL_DIAGNOSTIC.md"
    plan_before = sha(plan)
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    nets = {}
    for name in ("original", "source", "cold"):
        base, _ = load_parent(DEFAULT_PARENT)
        nets[name] = MatrixAttention(base.cuda())
        if name != "original":
            weight_path = (
                Path(p["source_root"]) / "forecast_head.pt"
                if name == "source"
                else root / "cold_head.pt"
            )
            expected = (
                p["source_head_sha256"]
                if name == "source"
                else read(root, "complete.json")["cold_head_sha256"]
            )
            if sha(weight_path) != expected:
                raise ValueError("auxiliary head changed")
            base.forecast_head.load_state_dict(
                torch.load(weight_path, map_location="cuda", weights_only=True)
            )
    ref = Path(p["reference"])
    slow = torch.load(ref / "models/cold.pt", map_location="cuda", weights_only=True)[
        "slow"
    ]
    mapping = read(ref, "input_map.json")
    hashes = read(ref, "input_hashes.json")
    rows, exclusions = [], []
    start = time.monotonic()

    def deny(*args, **kwargs):
        raise AssertionError("no outcomes during next-window prediction")

    for sid in p["order"][20:]:
        if sha(io23.path_for("EAST", sid)) != hashes[str(sid)]:
            raise ValueError("raw signal changed")
        with patch.object(io23, "read_outcome", side_effect=deny):
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            sig = transform({k: raw[k] for k in ("values", "available")}, mapping)
            clocks = np.arange(256, len(sig["values"]) - 24, 128)
            if not len(clocks):
                exclusions.append(
                    dict(shot=sid, reason="no complete post-feedback H25")
                )
                continue
            clocks = clocks[
                np.unique(
                    np.linspace(0, len(clocks) - 1, min(4, len(clocks))).astype(int)
                )
            ]
            shot_rows = []
            for c in clocks.tolist():
                prefix = {k: v[:c] for k, v in sig.items()}
                predicted = {}
                for name, net in nets.items():
                    # Only the prefix enters BOTH the adaptation and prediction calls.
                    delta, receipt = feedback_step(net, prefix, slow, "true")
                    x, a = batch(prefix, [c - 1], "cuda")
                    with torch.no_grad():
                        state = torch.stack((torch.zeros_like(delta), delta))
                        predicted[name] = (
                            net(
                                x.repeat(2, 1, 1), a.repeat(2, 1, 1), state, slow, "qk"
                            )["forecast"],
                            receipt,
                        )
                # All forecasts now exist; future truth is read only for this diagnostic.
                truth = torch.as_tensor(sig["values"][c : c + 25], device="cuda")[None]
                mask = torch.as_tensor(sig["available"][c : c + 25], device="cuda")[
                    None
                ]
                mask = (
                    mask & torch.isfinite(truth) & a[:, -1:] & torch.isfinite(x[:, -1:])
                )
                mask[:, :, 10] = False
                hold = torch.asinh((x[:, -1:] - base.center) / base.scale).expand_as(
                    truth
                )
                sums, counts = channel_losses(base, hold, truth, mask)
                item = dict(
                    shot=sid,
                    count=c,
                    feedback_last=c - 1,
                    target_first=c,
                    target_last=c + 24,
                    time=float(raw["times"][c - 1]),
                    target_end_time=float(raw["times"][c + 24]),
                    counts=counts,
                    hold_sums=sums,
                    heads={},
                )
                for name, (values, receipt) in predicted.items():
                    before, _ = channel_losses(base, values[:1], truth, mask)
                    after, _ = channel_losses(base, values[1:], truth, mask)
                    item["heads"][name] = dict(
                        before_sums=before,
                        after_sums=after,
                        past_loss_before=receipt["loss_before"],
                        past_loss_after=receipt["loss_after"],
                    )
                shot_rows.append(item)
        # Class/outcome annotations are retrospective and cannot affect these updates.
        flag, down = io23.read_outcome("EAST", sid)
        for row in shot_rows:
            row.update(
                flag=flag,
                preterminal=bool(
                    not flag
                    or round((down - row["target_end_time"]) * 1000, 9) >= -1e-6
                ),
                lead_ms=float(round((down - row["time"]) * 1000, 9)) if flag else None,
            )
        rows.extend(shot_rows)
    usable = [r for r in rows if r["preterminal"] and sum(r["counts"])]
    summaries = {}
    # Equal-shot summaries prevent long shots or many sample clocks dominating.
    for name in nets:
        per_shot = []
        for sid in p["order"][20:]:
            selected = [r for r in usable if r["shot"] == sid]
            if not selected:
                continue
            before = [
                sum(r["heads"][name]["before_sums"]) / sum(r["counts"])
                for r in selected
            ]
            after = [
                sum(r["heads"][name]["after_sums"]) / sum(r["counts"]) for r in selected
            ]
            hold = [sum(r["hold_sums"]) / sum(r["counts"]) for r in selected]
            per_shot.append(
                dict(
                    shot=sid,
                    flag=selected[0]["flag"],
                    clocks=len(selected),
                    before=float(np.mean(before)),
                    after=float(np.mean(after)),
                    hold=float(np.mean(hold)),
                )
            )
        b = np.array([r["before"] for r in per_shot])
        a = np.array([r["after"] for r in per_shot])
        h = np.array([r["hold"] for r in per_shot])
        summaries[name] = dict(
            shots=len(per_shot),
            before_mean=float(b.mean()),
            after_mean=float(a.mean()),
            hold_mean=float(h.mean()),
            change_mean=float((a - b).mean()),
            change_sd=float((a - b).std(ddof=1)),
            improved_shots=int(np.sum(a < b - 1e-9)),
            harmed_shots=int(np.sum(a > b + 1e-9)),
            beats_hold_shots=int(np.sum(a < h)),
            per_shot=per_shot,
        )
    if (
        sha(Path(__file__)) != code_before
        or sha(plan) != plan_before
        or sha(DEFAULT_PARENT) != PARENT_SHA256
    ):
        raise ValueError("diagnostic inputs changed")
    result = dict(
        complete=True,
        stream_complete_sha256=sha(root / "complete.json"),
        audit_sha256=sha(root / "audit.json"),
        code_sha256=code_before,
        plan_sha256=plan_before,
        source_sha256=PARENT_SHA256,
        sampled_clocks=len(rows),
        preterminal_observed_clocks=len(usable),
        exclusions=exclusions,
        summaries=summaries,
        records=rows,
        seconds=time.monotonic() - start,
        scope="retrospective sampled next-window diagnostic; not online selection or new stream performance",
        independent_confirmation=False,
    )
    _durable_json(path, result)
    print(
        dict(
            clocks=len(rows),
            usable=len(usable),
            exclusions=exclusions,
            summaries={
                k: {j: v for j, v in r.items() if j != "per_shot"}
                for k, r in summaries.items()
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
