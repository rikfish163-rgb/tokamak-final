"""Fixed-ended-shot local update replay and uncached query-layer check."""

import argparse
from pathlib import Path

import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure, read
from ..lora_stream_20261102.core import direct, encode
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import ARMS, sample


def verify(root, *, report_name="verification_neural.json"):
    destination = root / report_name
    if destination.exists():
        raise FileExistsError(destination)
    p, done = read(root / "protocol.json"), read(root / "completed.json")
    assert done["status"] == "COMPLETED_ENDED20_LOCAL_PROBE"
    assert sha(root / "samples.json") == done["samples_sha256"]
    assert all(
        sha(io23.ROOT / name) == value for name, value in p["code_sha256"].items()
    )
    configure()
    model, _, cold = load_model()
    assert sha(cold) == p["cold_state_sha256"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert sha(root / "source_basis.pt") == p["basis_sha256"]
    basis = torch.load(root / "source_basis.pt", weights_only=True)["basis"].cuda()
    assert torch.equal(basis, source_basis(model))
    calibration = Path(p["calibration_reference"])
    assert sha(calibration / "input_map.json") == p["input_map_sha256"]
    mapping = read(calibration / "input_map.json")
    selected = [p["ids"][i] for i in (0, 4, 8, 19)]
    rows = read(root / "samples.json")
    checked, direct_checked, max_rms = 0, 0, 0.0
    for sid in selected:
        assert sha(io23.path_for("EAST", sid)) == p["input_hashes"][str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        for row in rows:
            if row["shot"] != sid:
                continue
            repeated = sample(model, signal, basis, row["query"], row["label"])
            assert all(value == row[key] for key, value in repeated.items()), (
                sid,
                row["query"],
            )
            with torch.no_grad():
                pack = encode(model, signal, [row["query"]], basis)
                for arm in ARMS:
                    state = torch.as_tensor(
                        row["results"][arm]["receipt"]["state256_after"], device="cuda"
                    ).reshape(4, 64)
                    actual = direct(model, signal, [row["query"]], basis, state)
                    assert float(actual["logit"]) == row["results"][arm]["after_logit"]
                    valid = pack["valid"][:, :16]
                    delta = (
                        (actual["h"][:, :16] - pack["h"][:, :16])
                        .double()
                        .masked_fill(~valid[:, :, None], 0)
                    )
                    rms = (
                        torch.linalg.vector_norm(delta.flatten(1), dim=1)
                        / (64 * valid.sum(1).clamp_min(1)).double().sqrt()
                    )
                    max_rms = max(max_rms, float(rms.max()))
                    assert torch.equal(actual["h"][:, -1], pack["h"][:, -1])
                    direct_checked += 1
            checked += 1
    assert checked > 0 and max_rms <= 0.250001
    assert all(
        torch.equal(value, model.state_dict()[key].cpu())
        for key, value in original.items()
    )
    result = dict(
        status="PASS",
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        selected_shots=selected,
        local_samples_replayed=checked,
        direct_query_states_checked=direct_checked,
        all_update_gradients_and_receipts_exact=True,
        direct_query_logits_exact=True,
        max_direct_query_hidden_rms=max_rms,
        source_tensors_unchanged=True,
        forward_gradient_implementation_reused=True,
        online_gain_measured=False,
        independent_future_confirmation=False,
    )
    _durable_json(destination, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-name", default="verification_neural.json")
    args = parser.parse_args()
    verify(args.output, report_name=args.report_name)
