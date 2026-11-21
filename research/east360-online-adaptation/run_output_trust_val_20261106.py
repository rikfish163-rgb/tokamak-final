"""Blind causal replay of source VAL with fully frozen source-only selectors."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import fit_output_trust_20261106 as fit
from verify_confirmation_raw_replay_20261105 import Hashes


bank, r, SEEDS, ARMS = fit.bank, fit.r, fit.SEEDS, fit.ARMS


def predict(base, frozen_fits, heads, raw):
    t = raw["times"]
    x, a = bank.prior.views(base, raw)
    native = bank.prior.scan(base, x, a).cpu().numpy()
    z = np.broadcast_to(native[:, None, None], (len(t), len(SEEDS), len(ARMS))).copy()
    rows = bank.rows_for_shot(base, raw, native, t, 0)
    m = rows["maturity"]
    n = len(m)
    shape = (n, len(SEEDS), len(ARMS) - 1)
    if not n:
        return dict(
            times=t,
            logits=z,
            maturity=m,
            action=np.empty(shape, dtype=np.int16),
            norms=np.empty(shape, dtype=np.float32),
            anchor=np.empty(shape, dtype=np.float32),
            past=np.empty(shape, dtype=np.float32),
            stale_supported=np.empty(shape[:2], dtype=bool),
        )
    tape = bank.gather(
        base,
        dict(
            values=np.pad(raw["values"], ((127, 0), (0, 0))),
            available=np.pad(raw["available"], ((127, 0), (0, 0))),
        ),
    )
    d = bank.gather(base, rows)
    p = bank.proposal(base, frozen_fits, tape, d)
    desc = p["descriptor"].cpu().numpy()
    lag, supported = fit.previous_descriptor(desc, np.zeros(n, dtype=np.int64), m)
    picks = fit.choose_all(heads, desc, lag, p["available"].cpu().numpy())
    device = base.center.device
    k = torch.as_tensor(picks.clip(min=0), device=device)
    good = torch.as_tensor(picks >= 0, device=device)
    mode = torch.as_tensor(fit.MODE, device=device)[None, None, :]
    row = torch.arange(n, device=device)[:, None, None]
    seed = torch.arange(len(SEEDS), device=device)[None, :, None]
    used = (row, mode, seed, k)
    if (good & ~p["available"][used]).any():
        raise ValueError("selector used an unavailable causal step")
    u = p["state"][used].masked_fill(~good[..., None], 0.0)
    anchor = p["anchor"][used].masked_fill(~good, 0.0)
    past = torch.stack((p["free_largest"], p["largest"]), 1)[used].masked_fill(
        ~good, 0.0
    )
    real = u[:, :, :-1]
    norms = u.norm(dim=-1)
    j = ARMS.index("guarded") - 1
    c = ARMS.index("reversed_guarded") - 1
    if (
        not torch.equal(u[:, :, j], -u[:, :, c])
        or (past[:, :, 1:] > bank.tr.LIMIT + 1e-7).any()
        or not torch.isfinite(norms).all()
    ):
        raise ValueError("actual matched reverse or forward trust geometry differs")
    row, clock = d["present"].nonzero(as_tuple=True)
    source = bank.src.field(native, device)
    for start in range(0, len(row), 160):
        stop = min(start + 160, len(row))
        i, off = row[start:stop], clock[start:stop]
        point = d["maturity"][i] + 1 + off
        values, avail = bank.src.tape_views(tape, d["future"][i, off])
        w = real[i].reshape(len(i), len(SEEDS) * (len(ARMS) - 2), bank.old.WIDTH)
        out = bank.fwd.batched_risk(base, bank.fwd.encode(base, values, avail), w)
        out = out.reshape(len(i), len(SEEDS), len(ARMS) - 2)
        out = torch.where(good[i, :, :-1], out, source[point, None, None])
        z[point.cpu().numpy(), :, 1:-1] = out.cpu().numpy()
        z[point.cpu().numpy(), :, -1] = (
            (source[point, None] + anchor[i, :, -1]).cpu().numpy()
        )
    if not np.isfinite(z).all():
        raise FloatingPointError("invalid causal source trajectory")
    # No scalar offset is committed in any real internal update arm.
    norm_payload = norms.cpu().numpy().copy()
    norm_payload[..., -1] = 0.0
    return dict(
        times=t,
        logits=z,
        maturity=m,
        action=picks.astype(np.int16),
        norms=norm_payload,
        anchor=anchor.cpu().numpy(),
        past=past.cpu().numpy(),
        stale_supported=supported,
    )


def run(root):
    h, start = Hashes(), time.monotonic()
    fit_receipt = fit.checked_fit(root, h)
    cfg = r.read(root / "protocol.json")
    if (
        fit_receipt["source_state_sha256"] != cfg["source_state_sha256"]
        or cfg["roles"] != r.simple_data.source_roles()
        or len(cfg["roles"]["val"]) != 160
        or set(cfg["roles"]["train"]) & set(cfg["roles"]["val"])
        or set(cfg["roles"]["test"]) & set(cfg["roles"]["val"])
    ):
        raise ValueError("same disjoint source validation cohort required")
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    if (
        r.state_digest(base) != cfg["source_state_sha256"]
        or parent["threshold"] != r.THRESHOLD
    ):
        raise ValueError("same frozen risk and threshold required")
    base.cuda()
    head = fit.load_heads(root / "heads.npz")
    diag = bank.frozen_fits(base)
    out = root / "val"
    out.mkdir(exist_ok=False)
    protocol = out / "protocol.json"
    r.write_new(
        protocol,
        dict(
            status="LOCKED_BLIND_OUTPUT_TRUST_VAL",
            fit_sha256=h(root / "fit_completed.json"),
            source_protocol_sha256=h(root / "protocol.json"),
            val=cfg["roles"]["val"],
            input_sha256={str(Path(__file__).resolve()): h(Path(__file__))},
            arms=ARMS,
            seeds=SEEDS,
            threshold=r.THRESHOLD,
            held_outcome_reads=0,
        ),
    )
    archive, witness = [], []
    with patch.object(
        r.io23, "read_outcome", side_effect=RuntimeError("VAL truth is sealed")
    ):
        for rank, sid in enumerate(cfg["roles"]["val"], 1):
            raw_hash = h(r.io23.path_for("J-TEXT", sid))
            raw = r.simple_data.read_shot("J-TEXT", sid)
            payload = predict(base, diag, head, raw)
            if rank in (3, 56, 119):
                stop = min(103, len(raw["times"]) - 1)
                if stop >= 65:
                    truncated = {
                        k: raw[k][:stop] for k in ("values", "available", "times")
                    }
                    prefix = predict(base, diag, head, truncated)
                    selected = payload["maturity"] < stop - 1
                    err = float(
                        np.abs(prefix["logits"] - payload["logits"][:stop]).max()
                    )
                    if (
                        err > 8e-5
                        or not np.array_equal(
                            prefix["maturity"], payload["maturity"][selected]
                        )
                        or not np.array_equal(
                            prefix["action"], payload["action"][selected]
                        )
                    ):
                        raise ValueError("unarrived VAL signal changed earlier replay")
                    witness.append(
                        dict(
                            shot=sid,
                            points=stop,
                            error=err,
                            actions=int((prefix["action"] >= 0).sum()),
                        )
                    )
            path = out / f"{rank:04d}_{sid}.npz"
            r.save_arrays(path, **payload)
            archive.append(
                dict(
                    shot=sid,
                    rank=rank,
                    file=path.name,
                    raw_sha256=raw_hash,
                    sha256=h(path),
                )
            )
            if rank % 40 == 0:
                print(f"blind VAL replay {rank}/160", flush=True)
    if r.state_digest(base) != cfg["source_state_sha256"]:
        raise ValueError("held replay changed the source classifier")
    h.check()
    r.write_new(
        out / "completed.json",
        dict(
            status="SEALED_OUTPUT_TRUST_VAL",
            protocol_sha256=h(protocol),
            rows=archive,
            held_outcome_reads=0,
            prefix=witness,
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
