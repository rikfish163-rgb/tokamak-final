"""Matched hindsight norm-scope capacity, not a causal deployment candidate."""

from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import joint_lora_hindsight_capacity_20261104 as h

ROOT = h.ROOT
PLAN = ROOT / "JOINT_NORM_HINDSIGHT_PLAN_20261104.md"
OUT = ROOT / "probe_outputs/joint_norm_hindsight_20261104_v1"


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    upstream = h.read(h.OUTPUT / "completed.json")
    if upstream["status"] != "COMPLETED_HINDSIGHT_ORACLE_DIAGNOSTIC":
        raise ValueError("finished Q/V capacity control required")
    for name, digest in upstream["artifact_sha256"].items():
        if h.sha(h.OUTPUT / name) != digest:
            raise ValueError("original capacity experiment changed")
    pre = h.read(h.OUTPUT / "precommit.json")
    protocol, seals, ids, metadata = h.verify_inputs()
    bound = {
        str(p): h.sha(p)
        for p in (
            Path(__file__),
            Path(h.__file__),
            PLAN,
            h.OUTPUT / "precommit.json",
            h.OUTPUT / "completed.json",
            h.DEFAULT_PARENT,
        )
    }
    OUT.mkdir()
    h.write_new(
        OUT / "protocol.json",
        dict(
            scope="MATCHED_SIX_SHOT_NORM_HINDSIGHT_NOT_ONLINE",
            selected_shots=pre["shots"],
            trainable_scope="last-layer norm1/norm2 affine",
            parameters=256,
            comparison_parameters=512,
            steps=60,
            lr=0.01,
            clip=1.0,
            radius=0.5,
            input_sha256=bound,
            parent_frozen=True,
            support_labels_are_hindsight=True,
        ),
    )
    base, parent = h.load_parent(h.DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    state = h.state_digest(base)
    if state != protocol["source_state_sha256"] or parent["threshold"] != h.THRESHOLD:
        raise ValueError("wrong source parent")
    map_path = next(
        Path(k) for k in protocol["input_sha256"] if k.endswith("/input_map.json")
    )
    mapping = h.read(map_path)
    if h.sha(map_path) != protocol["input_sha256"][str(map_path)]:
        raise ValueError("input map changed")
    rows = []
    for case in pre["shots"]:
        sid, index = case["shot"], case["index"]
        if ids[index - 1] != sid:
            raise ValueError("fixed six-shot case changed")
        times, native, seal = h.sealed_curve(index, sid, seals, metadata)
        path = h.io23.path_for("EAST", sid)
        raw_hash = protocol["raw_sha256"][str(sid)]
        if h.sha(path) != raw_hash:
            raise ValueError("case raw data changed")
        raw = h.simple_data.read_shot("EAST", sid, rate_contract="nominal")
        truth = h.io23.read_outcome("EAST", sid)
        if truth != (case["disruptive"], case["down_s"]) or not np.array_equal(
            raw["times"], times
        ):
            raise ValueError("raw labels or clock differ")
        signal = h.prepare_actual(raw, index, mapping)
        support = case["support_indices"]
        if support != h.fixed_support(times, *truth, native, h.THRESHOLD):
            raise ValueError("support changed")
        model = h.make_adapter(base, "norm")
        params = tuple(p for p in model.parameters() if p.requires_grad)
        if len(params) != 4 or sum(p.numel() for p in params) != 256:
            raise ValueError("exact internal norm scope required")
        init = tuple(p.detach().clone() for p in params)
        frozen = h.frozen_digest(model, params)
        zero = h.curve(model, signal)
        gap = float(np.max(np.abs(zero - native)))
        if gap > h.PARITY_ATOL:
            raise ValueError("initial norm model failed native parity")
        labels, _ = h.hindsight_labels(times, *truth)
        x, a = h.windows(signal, support)
        y = torch.from_numpy(labels[support])
        optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=0)
        losses = []
        for step in range(60):
            optimizer.zero_grad(set_to_none=True)
            z = model(x, a)["logit"]
            loss = F.binary_cross_entropy_with_logits(z, y)
            loss.backward()
            grad = float(
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            distance = h.project_joint(params, init, 0.5)
            losses.append(
                dict(
                    step=step + 1,
                    loss=float(loss.detach()),
                    grad_preclip=grad,
                    displacement=distance,
                )
            )
        full = h.curve(model, signal)
        full[:135] = native[:135]
        splice = h.held_splice(native, full, support)
        if (
            h.frozen_digest(model, params) != frozen
            or h.state_digest(base) != state
            or h.sha(path) != raw_hash
        ):
            raise ValueError("frozen parent or raw input changed")
        output = OUT / f"{index:04d}_{sid}.npz"
        with output.open("xb") as f:
            np.savez_compressed(f, adapted_full=full, held_splice=splice)
        row = dict(
            shot=sid,
            kind=case["kind"],
            support=support,
            native=h.timeline(times, native, *truth, h.THRESHOLD),
            full=h.timeline(times, full, *truth, h.THRESHOLD),
            held=h.timeline(times, splice, *truth, h.THRESHOLD),
            collateral=h.collateral(times, native, full, *truth, h.THRESHOLD, support),
            native_parity_gap=gap,
            history=losses,
            raw_sha256=raw_hash,
            output_sha256=h.sha(output),
        )
        if truth[0]:
            row["band"] = h.band_metrics(
                times, native, full, truth[1], support, 30, 125
            )
        rows.append(row)
        print({k: row[k] for k in ("shot", "kind", "full", "held")}, flush=True)
    if any(h.sha(Path(p)) != digest for p, digest in bound.items()):
        raise ValueError("bound input changed")
    h.write_new(OUT / "result.json", rows)
    h.write_new(
        OUT / "completed.json",
        dict(
            status="COMPLETE_MATCHED_INTERNAL_NORM_HINDSIGHT_DIAGNOSTIC",
            seconds=time.monotonic() - started,
            artifact_sha256={p.name: h.sha(p) for p in OUT.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    run()
