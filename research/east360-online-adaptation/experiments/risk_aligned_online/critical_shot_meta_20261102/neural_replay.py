"""Fixed four-shot trajectory replay and actual-layer-hook spot checks."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import direct
from ..warmstart_alignment_20261022.alignment import transform
from .objectives import last_update
from .run import ARMS, export_traces
from .stream import simulate

IDS = (74087, 73403, 63656, 62573)


def main():
    root = io23.ROOT / "probe_outputs/critical_shot_meta360_v1"
    report = root / "verification_neural.json"
    if report.exists():
        raise FileExistsError(report)
    done, protocol = read(root / "completed.json"), read(root / "protocol.json")
    assert done["status"] == "COMPLETED"
    configure()
    start = time.monotonic()
    model, _, cold = load_model()
    assert sha(cold) == protocol["cold_state_sha256"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for filename, expected in protocol["code_sha256"].items():
        assert sha(io23.ROOT / filename) == expected, filename
    calibration = Path(protocol["calibration_reference"])
    assert sha(calibration / "input_map.json") == protocol["input_map_sha256"]
    mapping, hashes = (
        read(calibration / "input_map.json"),
        read(calibration / "input_hashes.json"),
    )
    training = Path(protocol["training"])
    assert sha(training / "heads.json") == protocol["heads_sha256"]
    tables = {
        k: torch.tensor(v, dtype=torch.float32, device="cuda")
        for k, v in read(training / "heads.json").items()
    }
    heads = dict(
        cold=(None, "real"),
        original_difference=(torch.zeros_like(tables["operational"]), "real"),
        forecast=(tables["forecast"], "real"),
        point=(tables["point"], "real"),
        operational=(tables["operational"], "real"),
        reverse=(tables["operational"], "reverse"),
        hold=(tables["operational"], "hold"),
    )
    basis = source_basis(model)
    order = [r["shot"] for r in protocol["order"]]
    checks = []
    direct_batches = 0
    for sid in IDS:
        index = order.index(sid) + 1
        assert index > 20
        stem = f"{index:04d}_{sid}"
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        result = simulate(model, signal, basis, heads)
        arrays, info = export_traces(result["tapes"], result["actions"])
        seal = read(root / "shots" / (stem + "_seal.json"))
        path = root / "shots" / (stem + ".npz")
        assert sha(path) == seal["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            for arm in ARMS:
                assert np.array_equal(result["probability"][arm], saved[arm]), (
                    sid,
                    arm,
                    "trajectory",
                )
            for q in (0, 160, len(raw["times"]) - 1):
                lo, hi = q // 32 * 32, min(q // 32 * 32 + 32, len(raw["times"]))
                for arm in ARMS:
                    maturity = last_update(q)
                    if maturity is None or arm == "cold":
                        state = torch.zeros(4, 64, device="cuda")
                    else:
                        tape = next(
                            t for t in result["tapes"][arm] if t["maturity"] == maturity
                        )
                        state = tape["after"]
                    with torch.no_grad():
                        actual = (
                            direct(model, signal, np.arange(lo, hi), basis, state)[
                                "logit"
                            ]
                            .sigmoid()
                            .cpu()
                            .numpy()
                        )
                    assert np.array_equal(actual, saved[arm][lo:hi]), (
                        sid,
                        arm,
                        q,
                        "actual layer hook",
                    )
                    direct_batches += 1
        trace = root / "shots" / (stem + "_states.npz")
        assert sha(trace) == seal["states_sha256"]
        with np.load(trace, allow_pickle=False) as saved:
            for key, value in arrays.items():
                assert np.array_equal(value, saved[key]), (sid, key, "state/gradient")
        assert info == read(root / "shots" / (stem + "_updates.json")), (sid, "receipt")
        checks.append(
            dict(
                shot=sid,
                index=index,
                curves=len(ARMS),
                points=len(raw["times"]),
                real_updates=len(result["tapes"]["operational"]),
            )
        )
        print(json.dumps(dict(checked_shot=sid, checks=len(checks))), flush=True)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result = dict(
        status="PASS",
        fixed_preselected_shots=list(IDS),
        checks=checks,
        curves_replayed=4 * len(ARMS),
        actual_layer_hook_batches=direct_batches,
        full_trajectory_state_gradient_receipts_exact=True,
        source_tensors_unchanged=True,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        outcome_reads=0,
        raw_signal_shots_read=list(IDS),
        seconds=time.monotonic() - start,
        scope="Replay uses the deployed simulation implementation; fixed endpoint batches additionally use the uncached actual first-FFN hook. Not an independent reimplementation or complete training rerun.",
    )
    write(report, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
