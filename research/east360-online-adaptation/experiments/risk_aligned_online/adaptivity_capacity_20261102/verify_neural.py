"""Direct, uncached replay of fixed cases and first finite witnesses.

This reuses the tested neural intervention implementation, not its feature
cache. It verifies reproducibility, not an independent proof of that model or
causal validity of the deliberately labelled post-hoc parameter search.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.model import latent_forward
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure, read
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .model import direct_lora, encode, source_basis


def replay_ids(outcomes):
    """Fixed ends of each class, plus first witness per class and family."""
    if not outcomes:
        raise ValueError("completed outcomes required")
    selected = set()
    for disruptive in (True, False):
        rows = [r for r in outcomes if r["disruptive"] == disruptive]
        if not rows:
            continue  # A one-case preflight has only one outcome class.
        selected.update((rows[0]["shot"], rows[-1]["shot"]))
        for family in ("gates", "ffn_lora"):
            matches = [
                r for r in rows if r["family"] == family and r["score"]["witness"]
            ]
            if matches:
                selected.add(matches[0]["shot"])
    return sorted(selected)


@torch.no_grad()
def replay(root):
    destination = root / "verification_neural.json"
    if destination.exists():
        raise FileExistsError(destination)
    protocol, done = read(root / "protocol.json"), read(root / "completed.json")
    assert done["status"] == "COMPLETED_POSTHOC_CAPACITY"
    assert protocol["scope"] == "POSTHOC_LABEL_ORACLE_FINITE_WITNESSES_NOT_ONLINE"
    configure()
    model, parent, cold = load_model()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert sha(cold) == protocol["cold_sha256"]
    assert float(parent["threshold"]) == protocol["threshold"]
    assert sha(root / "source_basis.pt") == protocol["basis_sha256"]
    basis = torch.load(root / "source_basis.pt", weights_only=True)["basis"].cuda()
    assert torch.equal(basis, source_basis(model))
    reference = Path(protocol["reference"])
    reference_protocol = read(reference / "protocol.json")
    old = Path(reference_protocol["reference"])
    mapping = read(old / "input_map.json")
    input_hashes = read(old / "input_hashes.json")
    assert sha(old / "input_map.json") == reference_protocol["input_map_sha256"]
    selected = set(replay_ids(done["outcomes"]))
    records = []
    for rank, sid in protocol["order"]:
        if sid not in selected:
            continue
        folder = root / "cases" / f"{rank:04d}_{sid}"
        assert sha(io23.path_for("EAST", sid)) == input_hashes[str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        for family in protocol["families"]:
            receipt = read(folder / f"{family}_receipt.json")
            state_path = folder / f"{family}_state.pt"
            curves_path = folder / f"{family}_curves.npz"
            assert sha(state_path) == receipt["state_sha256"]
            assert sha(curves_path) == receipt["curves_sha256"]
            state = torch.load(state_path, weights_only=True)["state"].cuda()
            parts = []
            for lo in range(0, len(signal["values"]), 32):
                indices = np.arange(lo, min(lo + 32, len(signal["values"])))
                enabled = torch.as_tensor(indices >= 160, device="cuda")
                if family == "ffn_lora":
                    z = direct_lora(model, signal, indices, basis, state, enabled)[
                        "logit"
                    ]
                else:
                    assert bool(enabled.all()) or not bool(enabled.any())
                    pack = encode(model, signal, indices, basis)
                    native = {k: pack[k] for k in ("h", "valid", "long", "snapshot")}
                    gates = state if bool(enabled.all()) else torch.zeros_like(state)
                    z = latent_forward(model, native, gates)["logit"]
                parts.append(z.sigmoid().cpu().numpy())
            probabilities = np.concatenate(parts)
            with np.load(curves_path, allow_pickle=False) as saved:
                assert np.array_equal(raw["times"], saved["times"])
                assert np.array_equal(probabilities, saved["probability"]), (
                    sid,
                    family,
                    float(np.max(np.abs(probabilities - saved["probability"]))),
                )
                assert np.array_equal(probabilities[:160], saved["cold"][:160])
            records.append(dict(shot=sid, family=family, complete_curve_exact=True))
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result = dict(
        status="PASS_DIRECT_NEURAL_REPLAY",
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        selected_shots=sorted(selected),
        replay=records,
        replay_curves=len(records),
        direct_layer_path=True,
        cached_features_reused=False,
        intervention_implementation_reused=True,
        source_tensors_unchanged=True,
        demonstrates_online_gain=False,
        independent_future_confirmation=False,
    )
    _durable_json(destination, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    replay(parser.parse_args().output)
