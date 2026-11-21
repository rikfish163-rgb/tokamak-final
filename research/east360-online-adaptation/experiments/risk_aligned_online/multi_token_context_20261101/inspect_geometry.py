"""Post-run fixed-index128 description of token-specific versus uniform shifts."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..delayed_recurrent_memory_20261025.model import patch_signal
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .model import action, context_at, context_trace, token_basis


@torch.no_grad()
def main(root):
    output = root / "fixed128_geometry.json"
    if output.exists():
        raise FileExistsError(output)
    done, p = read(root / "completed.json"), read(root / "protocol.json")
    assert done["status"] == "COMPLETED"
    configure()
    base, _ = load_parent(Path(p["source_checkpoint"]))
    base.cuda().eval()
    model = build_adapter(base, readout="full", feedback="progressive", context=True)
    assert (
        load_context_memory(model, Path(p["context_source"]["path"]))
        == p["context_source"]
    )
    model.restore(torch.load(p["cold_state"], map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    weight = torch.load(root / "preparation/readouts.pt", weights_only=True)[
        "per_token"
    ].cuda()
    mapping = read(Path(p["reference"]) / "input_map.json")
    cold = read(root / "per_shot.json")["cold"]
    rows = []
    for index, ordered in enumerate(p["order"]):
        if index < 20:
            continue
        sid = ordered["shot"]
        s = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        if len(s["values"]) <= 128:
            rows.append(dict(shot=sid, status="SHORT_RECORD", end=128))
            continue
        # All inspected features stop at128, even though the file is complete.
        signal = transform({k: s[k][:129] for k in ("values", "available")}, mapping)
        means, complete = patch_signal(signal, base.center, base.scale)
        c = context_at(context_trace(model.memory, means, complete), [128])
        x, a = simple_data.window(signal, 128, 128)
        x, a = (
            torch.as_tensor(x, device="cuda")[None],
            torch.as_tensor(a, device="cuda")[None],
        )
        b, supported = token_basis(model.memory, x, a, base.center, base.scale)
        valid_data = a.reshape(1, 16, 8 * 23).any(-1)
        valid = torch.cat(
            (valid_data, torch.ones(1, 1, dtype=torch.bool, device="cuda")), 1
        )
        full, _ = action(weight, b, c, valid, "per_token")
        tile, _ = action(weight, b, c, valid, "tiled_norm")
        n = valid_data.sum().clamp_min(1)
        mean_shift = full.sum(1, keepdim=True) / n
        shape = (full - mean_shift) * valid_data[:, :, None]
        energy = full.square().sum()
        denominator = full.norm() * tile.norm()
        level = torch.asinh(
            (torch.where(a, x, base.center) - base.center) / base.scale
        ).reshape(1, 16, 8, 23)
        channel_complete = a.reshape(1, 16, 8, 23).all(2)
        patch_mean = torch.where(channel_complete, level.mean(2), 0.0)
        key = model.memory.key(
            model.memory.key_features(patch_mean, channel_complete)
        ).tanh()
        chosen_keys = key[supported]
        flag, _ = io23.read_outcome("EAST", sid)
        r = cold[index]
        label = (
            ("TP" if r["timely"] else "NOT_TIMELY")
            if flag
            else ("FP" if r["alarm"] else "TN")
        )
        rows.append(
            dict(
                shot=sid,
                status="COMPUTED",
                end=128,
                outcome_group=label,
                context_norm=float(c.norm()),
                shift_norm=float(full.norm()),
                shape_energy_fraction=float(shape.square().sum() / energy)
                if energy > 0
                else None,
                cosine_to_tiled=float((full * tile).sum() / denominator)
                if denominator > 0
                else None,
                supported_tokens=int(supported.sum()),
                key_saturation_over095=float((chosen_keys.abs() > 0.95).float().mean())
                if chosen_keys.numel()
                else None,
            )
        )
    summaries = {}
    fields = (
        "shape_energy_fraction",
        "cosine_to_tiled",
        "key_saturation_over095",
        "context_norm",
        "shift_norm",
    )
    for group in ("ALL", "TP", "NOT_TIMELY", "FP", "TN"):
        selected = [
            r
            for r in rows
            if r["status"] == "COMPUTED"
            and (group == "ALL" or r["outcome_group"] == group)
        ]
        summaries[group] = dict(n_shots=len(selected))
        for field in fields:
            values = [r[field] for r in selected if r[field] is not None]
            summaries[group][field] = (
                dict(
                    n=len(values),
                    q10=float(np.quantile(values, 0.1)),
                    median=float(np.median(values)),
                    q90=float(np.quantile(values, 0.9)),
                )
                if values
                else None
            )
    result = dict(
        status="COMPLETED_DESCRIPTIVE",
        scope="fixed index128 from saved-record start, suffix340 only",
        not_actual_alarm_time_explanation=True,
        outcome_used_only_after_features=True,
        rows=rows,
        summaries=summaries,
        completed_sha256=sha(root / "completed.json"),
        code_sha256=sha(Path(__file__)),
        independent_future_confirmation=False,
    )
    _durable_json(output, result)
    print(summaries, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
