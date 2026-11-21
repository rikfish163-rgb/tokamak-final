"""Fixed4-shot complete gradient replay and uncached real-FFN risk replay.

Reuses the model/update implementation: reproducibility evidence, not an
independent mathematical implementation or independent future validation.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.run import digest_array, load_model
from ..east_context_operator_20261031.run import configure, read
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .core import direct, encode
from .run import ARMS, LEARNERS, predict

REPLAY_POSITIONS = (20, 118, 153, 359)


def verify(root):
    output = root / "verification_neural.json"
    if output.exists():
        raise FileExistsError(output)
    p, done = read(root / "protocol.json"), read(root / "completed.json")
    assert done["status"] == "COMPLETED" and p["arms"] == list(ARMS)
    assert len(p["order"]) == 360
    assert all(
        sha(io23.ROOT / name) == digest for name, digest in p["code_sha256"].items()
    )
    configure()
    model, parent, cold = load_model()
    assert sha(cold) == p["cold_state_sha256"]
    assert parent["threshold"] == p["threshold"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert sha(root / "source_basis.pt") == p["basis_sha256"]
    basis = torch.load(root / "source_basis.pt", weights_only=True)["basis"].cuda()
    assert torch.equal(basis, source_basis(model))
    calibration = Path(p["calibration_reference"])
    assert sha(calibration / "input_map.json") == p["input_map_sha256"]
    mapping, raw_hashes = (
        read(calibration / "input_map.json"),
        read(calibration / "input_hashes.json"),
    )
    results = []
    for position in REPLAY_POSITIONS:
        sid = p["order"][position]["shot"]
        prefix = f"{position + 1:04d}_{sid}"
        path, log_path = (
            root / "shots" / f"{prefix}.npz",
            root / "shots" / f"{prefix}_updates.json",
        )
        seal = read(root / "shots" / f"{prefix}_seal.json")
        assert sha(path) == seal["sha256"] and sha(log_path) == seal["updates_sha256"]
        assert sha(io23.path_for("EAST", sid)) == raw_hashes[str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        repeated, receipt = predict(model, signal, basis)
        assert receipt == read(log_path)
        states = {a: torch.zeros(4, 64, device="cuda") for a in ARMS}
        events = {
            a: {r["target_last"]: r for r in receipt["updates"][a]} for a in LEARNERS
        }
        curves = {a: [] for a in ARMS}
        max_rms, seals_verified = 0.0, 0
        with torch.no_grad():
            for lo in range(0, len(signal["values"]), 32):
                ends = np.arange(lo, min(lo + 32, len(signal["values"])))
                end = int(ends[-1])
                has_feedback = end in events["real"]
                original_pack = encode(model, signal, ends, basis)
                for arm in ARMS:
                    actual = direct(
                        model,
                        signal,
                        ends,
                        basis,
                        states[arm],
                        forecast=has_feedback and arm in LEARNERS,
                    )
                    curves[arm].append(actual["logit"].sigmoid().cpu().numpy())
                    valid = original_pack["valid"][:, :16]
                    delta = (
                        (actual["h"][:, :16] - original_pack["h"][:, :16])
                        .double()
                        .masked_fill(~valid[:, :, None], 0)
                    )
                    rms = (
                        torch.linalg.vector_norm(delta.flatten(1), dim=1)
                        / (64 * valid.sum(1).clamp_min(1)).double().sqrt()
                    )
                    max_rms = max(max_rms, float(rms.max()))
                    assert torch.equal(actual["h"][:, -1], original_pack["h"][:, -1])
                    if has_feedback and arm in LEARNERS:
                        record = events[arm][end]
                        assert (
                            states[arm].flatten().cpu().tolist()
                            == record["state256_before"]
                        )
                        origin = record["origin"]
                        forecast = (
                            actual["forecast"][origin - lo : origin - lo + 1]
                            .cpu()
                            .numpy()
                        )
                        assert digest_array(forecast) == record["forecast_sha256"]
                        seals_verified += 1
                # Apply the saved state only after direct risk and issued
                # forecast checks through this endpoint have been completed.
                if has_feedback:
                    for arm in LEARNERS:
                        states[arm] = torch.as_tensor(
                            events[arm][end]["state256_after"], device="cuda"
                        ).reshape(4, 64)
        with np.load(path, allow_pickle=False) as saved:
            assert np.array_equal(saved["times"], raw["times"])
            for arm in ARMS:
                direct_curve = np.concatenate(curves[arm])
                assert np.array_equal(repeated[arm], saved[arm]), (
                    sid,
                    arm,
                    "update replay",
                )
                assert np.array_equal(direct_curve, saved[arm]), (
                    sid,
                    arm,
                    "real FFN replay",
                )
        assert max_rms <= 0.250001
        results.append(
            dict(
                shot=sid,
                curves=len(ARMS),
                all_update_receipts_exact=True,
                direct_ffn_curves_exact=True,
                forecast_seals_checked=seals_verified,
                max_direct_hidden_rms=max_rms,
            )
        )
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result = dict(
        status="PASS",
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        replay_positions_zero_based=REPLAY_POSITIONS,
        replay=results,
        update_replay_curves=4 * len(ARMS),
        direct_ffn_replay_curves=4 * len(ARMS),
        source_tensors_unchanged=True,
        code_files_unchanged=len(p["code_sha256"]),
        forward_gradient_implementation_reused=True,
        independent_future_confirmation=False,
    )
    _durable_json(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)
