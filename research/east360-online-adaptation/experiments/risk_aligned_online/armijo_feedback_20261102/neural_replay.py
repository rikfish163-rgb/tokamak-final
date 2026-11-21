"""Fixed four-shot replay of actual Armijo neural states and layer hooks."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..critical_shot_meta_20261102.objectives import last_update
from ..critical_shot_meta_20261102.run import export_traces
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import direct
from ..warmstart_alignment_20261022.alignment import transform
from .run import ARMS, FRESH, REFERENCE, REUSED, build_heads
from .stream import simulate

IDS = (74087, 73403, 63656, 62573)


def reference_state(arm, query, archive, device):
    if arm not in REUSED:
        raise ValueError("known reused arm required")
    maturity = last_update(query)
    if arm == "cold" or maturity is None:
        return torch.zeros(4, 64, device=device)
    values = archive["full_real_after"]
    index = (maturity - 159) // 128
    if values.ndim != 3 or values.shape[1:] != (4, 64) or index >= len(values):
        raise ValueError("fixed reference state unavailable")
    if values.dtype != np.float32 or not np.isfinite(values[index]).all():
        raise ValueError("finite float32 archived state required")
    return torch.as_tensor(values[index], device=device)


def main(root):
    root = Path(root)
    report = root / "verification_neural.json"
    if report.exists():
        raise FileExistsError(report)
    done, protocol = read(root / "completed.json"), read(root / "protocol.json")
    assert done["status"] == "COMPLETED"
    configure()
    started = time.monotonic()
    for path, digest in protocol["bound_files_sha256"].items():
        assert sha(Path(path)) == digest, path
    for path, digest in protocol["code_sha256"].items():
        assert sha(io23.ROOT / path) == digest, path
    model, _, cold = load_model()
    assert sha(cold) == protocol["cold_state_sha256"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    calibration = Path(protocol["calibration_reference"])
    mapping, hashes = [
        read(calibration / n) for n in ("input_map.json", "input_hashes.json")
    ]
    tables = {
        a: torch.tensor(v, dtype=torch.float32, device="cuda")
        for a, v in read(Path(protocol["training"]) / "heads.json").items()
    }
    heads, basis = build_heads(tables), source_basis(model)
    order = [r["shot"] for r in protocol["order"]]
    reference_records = read(REFERENCE / "verification_records.json")["hashes"][
        "shot_files"
    ]
    checks, hooks = [], 0
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
        tapes = {a: result["tapes"][a] if a in FRESH else [] for a in ARMS}
        actions = {
            a: result["actions"][a]
            if a in FRESH
            else dict(endpoints=0, capped_endpoints=0, max_rms=0.0)
            for a in ARMS
        }
        arrays, info = export_traces(tapes, actions)
        ref_path = REFERENCE / "shots" / (stem + ".npz")
        ref_seal_path = REFERENCE / "shots" / (stem + "_seal.json")
        ref_seal, record = read(ref_seal_path), reference_records[index - 1]
        assert (record["shot"], record["index"]) == (sid, index)
        assert sha(ref_seal_path) == record["seal_sha256"]
        assert sha(ref_path) == ref_seal["sha256"] == record["curve_sha256"]
        ref_trace = REFERENCE / "shots" / (stem + "_states.npz")
        assert sha(ref_trace) == ref_seal["states_sha256"]
        info["statistics_scope"] = "fresh_recomputed_only_not_historical_activity"
        info["execution"] = {
            a: dict(
                kind="reused_reference" if a in REUSED else "fresh_uncached_gpu",
                reference=str(ref_path) if a in REUSED else None,
                reference_arm=REUSED.get(a),
            )
            for a in ARMS
        }
        info["reference_curve_sha256"] = record["curve_sha256"]
        path = root / "shots" / (stem + ".npz")
        seal = read(root / "shots" / (stem + "_seal.json"))
        assert sha(path) == seal["sha256"]
        with (
            np.load(path, allow_pickle=False) as saved,
            np.load(ref_path, allow_pickle=False) as reference,
            np.load(ref_trace, allow_pickle=False) as states,
        ):
            np.testing.assert_array_equal(saved["times"], raw["times"])
            for arm in FRESH:
                np.testing.assert_array_equal(
                    saved[arm], result["probability"][arm], err_msg=f"{sid}:{arm}:fresh"
                )
            for arm, previous in REUSED.items():
                np.testing.assert_array_equal(
                    saved[arm], reference[previous], err_msg=f"{sid}:{arm}:reference"
                )
            for query in (0, 160, len(raw["times"]) - 1):
                lo, hi = query // 32 * 32, min(query // 32 * 32 + 32, len(raw["times"]))
                for arm in ARMS:
                    if arm in REUSED:
                        state = reference_state(arm, query, states, "cuda")
                    elif last_update(query) is None:
                        state = torch.zeros(4, 64, device="cuda")
                    else:
                        state = next(
                            t for t in tapes[arm] if t["maturity"] == last_update(query)
                        )["after"]
                    with torch.no_grad():
                        actual = (
                            direct(model, signal, np.arange(lo, hi), basis, state)[
                                "logit"
                            ]
                            .sigmoid()
                            .cpu()
                            .numpy()
                        )
                    np.testing.assert_array_equal(
                        actual,
                        saved[arm][lo:hi],
                        err_msg=f"{sid}:{arm}:{query}:actual layer hook",
                    )
                    hooks += 1
        trace = root / "shots" / (stem + "_states.npz")
        assert sha(trace) == seal["states_sha256"]
        with np.load(trace, allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key, value in arrays.items():
                np.testing.assert_array_equal(
                    saved[key], value, err_msg=f"{sid}:{key}:state/gradient"
                )
        metadata = root / "shots" / (stem + "_updates.json")
        assert sha(metadata) == seal["updates_sha256"]
        assert info == read(metadata), (sid, "receipts/reuse metadata")
        checks.append(
            dict(
                shot=sid,
                index=index,
                points=len(raw["times"]),
                fresh_curves=len(FRESH),
                referenced_curves=len(REUSED),
            )
        )
        print(json.dumps(dict(checked_shot=sid, checks=len(checks))), flush=True)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result = dict(
        status="PASS",
        fixed_preselected_shots=list(IDS),
        checks=checks,
        fresh_curves_replayed=4 * len(FRESH),
        referenced_curves_checked=4 * len(REUSED),
        actual_layer_hook_batches=hooks,
        fresh_state_gradient_receipts_exact=True,
        referenced_fixed_states_borrowed_not_recomputed=True,
        source_tensors_unchanged=True,
        outcome_reads=0,
        raw_signal_shots_read=list(IDS),
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        seconds=time.monotonic() - started,
        scope="Six deployed neural trajectories replayed; two archived references checked with borrowed fixed states. Not an independent algorithm implementation, all-shot neural replay or training rerun.",
    )
    write(report, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
