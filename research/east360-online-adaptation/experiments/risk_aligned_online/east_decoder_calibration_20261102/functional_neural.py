"""Fixed four-shot neural replay of the protected-update complete360 run."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import direct, encode, forward
from ..warmstart_alignment_20261022.alignment import transform
from .functional_stream import ARMS, load, predict
from .neural_replay import IDS, state_at


def main(root):
    root = Path(root)
    if (root / "verification_neural.json").exists():
        raise FileExistsError(root / "verification_neural.json")
    done, protocol = read(root / "completed.json"), read(root / "protocol.json")
    if done["status"] != "COMPLETED":
        raise ValueError("completed stream required")
    for path, digest in protocol["bound_code_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("bound code changed")
    model, head, basis, _, previous, mapping, hashes = load()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    head_before = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    ids = [r["shot"] for r in previous["order"]]
    checks = []
    actual_batches = 0
    max_step_rms, max_origin_rms = 0.0, 0.0
    start = time.monotonic()
    for sid in IDS:
        index = ids.index(sid) + 1
        if index <= 20 or sha(io23.path_for("EAST", sid)) != hashes[str(sid)]:
            raise ValueError("fixed suffix identity changed")
        path = root / "shots" / f"{index:04d}_{sid}.npz"
        recordpath = path.with_name(path.stem + "_updates.json")
        seal = read(path.with_name(path.stem + "_seal.json"))
        if (
            sha(path) != seal["curve_sha256"]
            or sha(recordpath) != seal["updates_sha256"]
        ):
            raise ValueError("sealed data changed")
        with patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("outcome forbidden in replay"),
        ):
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
            curves, receipts = predict(model, signal, basis, head)
        if receipts != read(recordpath):
            raise AssertionError("gradient/state/backtracking receipts not exact")
        prior = {a: model.base.center.new_zeros(4, 64) for a in ARMS[1:]}
        for event in range(len(receipts["old_real"])):
            end = receipts["old_real"][event]["maturity"]
            origin = end - 25
            lo = end // 32 * 32
            prefix = {k: v[: end + 1] for k, v in signal.items()}
            pack = encode(model, prefix, np.arange(lo, end + 1), basis)
            point = {k: v[origin - lo : origin - lo + 1] for k, v in pack.items()}
            valid = point["valid"][:, :16]
            for a in ARMS[1:]:
                after = torch.tensor(
                    receipts[a][event]["state256_after"], device="cuda"
                ).reshape(4, 64)
                shift = (
                    point["phi"].double() @ (after - prior[a]).double()
                ).masked_fill(~valid[:, :, None], 0)
                step_rms = float(
                    (shift.square().sum() / (valid.sum().clamp_min(1) * 64)).sqrt()
                )
                with torch.no_grad():
                    total_rms = float(forward(model, point, after)["rms"].max())
                if step_rms > 0.05 + 1e-7 or total_rms > 0.25 + 1e-7:
                    raise AssertionError("actual observed hidden RMS bound")
                max_step_rms, max_origin_rms = (
                    max(max_step_rms, step_rms),
                    max(max_origin_rms, total_rms),
                )
                prior[a] = after
        with np.load(path, allow_pickle=False) as saved:
            for a in ARMS:
                if not np.array_equal(curves[a], saved[a]):
                    raise AssertionError((sid, a, "full curve mismatch"))
            for q in (0, 160, len(raw["times"]) - 1):
                lo, hi = q // 32 * 32, min(q // 32 * 32 + 32, len(raw["times"]))
                prefix = {k: v[:hi] for k, v in signal.items()}
                for a in ARMS:
                    state = torch.as_tensor(
                        state_at([] if a == "static" else receipts[a], q), device="cuda"
                    )
                    with torch.no_grad():
                        actual = (
                            direct(model, prefix, np.arange(lo, hi), basis, state)[
                                "logit"
                            ]
                            .sigmoid()
                            .cpu()
                            .numpy()
                        )
                    if not np.array_equal(actual, curves[a][lo:hi]):
                        raise AssertionError(
                            (sid, a, q, "actual first-FFN hook mismatch")
                        )
                    actual_batches += 1
        checks.append(
            dict(
                shot=sid,
                index=index,
                points=len(raw["times"]),
                events=len(receipts["old_real"]),
            )
        )
        print(dict(replayed=sid, complete=len(checks)), flush=True)
    if any(
        not torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items()
    ) or any(
        not torch.equal(v, head.state_dict()[k].cpu()) for k, v in head_before.items()
    ):
        raise AssertionError("frozen model changed during replay")
    result = dict(
        status="PASS",
        checks=checks,
        fixed_preselected_shots=list(IDS),
        curves_replayed=len(IDS) * len(ARMS),
        actual_ffn_hook_batches=actual_batches,
        curve_gradient_state_acceptance_exact=True,
        max_replayed_step_hidden_rms=max_step_rms,
        max_replayed_origin_total_hidden_rms=max_origin_rms,
        source_unchanged=True,
        head_unchanged=True,
        outcome_reads=0,
        seconds=time.monotonic() - start,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        scope="Same deployed algorithm replay plus actual FFN hooks; not independent algorithm or training reproduction.",
    )
    write(root / "verification_neural.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
