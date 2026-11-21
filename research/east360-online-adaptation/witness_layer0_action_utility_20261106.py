"""Rebuild the original fit and witness exact arm splicing via real mixed writes."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from run_layer0_action_credit_history_20261106 import load_history, refit
from run_layer0_signed_projection_20261106 import (
    action_targets,
    as_case,
    causal_phase_support,
    layer,
    load_bank,
    parents,
    project_layer0,
    r,
    raw_for,
)


RUNS = {"fixed360": (21, 181, 359), "conditional835": (1, 418, 835)}
MENU = ("static", "intra", "signed", "blind_signed")
BANK = r.ROOT / "probe_outputs/layer0_signed_bank_20261106_v1"


def witness(output):
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, src, _, base, source, mapping, _, fit_path = parents()
    rule = r.load_rule(fit_path, base.center.device)
    bank, _ = load_bank(BANK, base.center.device)
    records, _, _ = load_history(order, src["raw_sha256"], source, fit_path)
    rows, inputs = [], {}
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("witness cannot open a current outcome"),
    ):
        raw = {x["shot"]: raw_for(x["shot"], src["raw_sha256"]) for x in records[:20]}
        actual, trace = refit(base, records[:20], raw, mapping)
        fit_max = max(
            float((p - actual.state_dict()[k]).abs().max().item())
            for k, p in rule.state_dict().items()
        )
        for key, ranks in RUNS.items():
            path = r.ROOT / "probe_outputs" / f"layer0_signed_{key}_20261106_v1"
            cfg, done = [r.read(path / n) for n in ("protocol.json", "completed.json")]
            inputs[key] = r.sha(path / "completed.json")
            if (
                done["protocol_sha256"] != r.sha(path / "protocol.json")
                or cfg["source_state_sha256"] != source
                or cfg["threshold"] != r.THRESHOLD
                or cfg["bank_receipt_sha256"] != r.sha(BANK / "completed.json")
            ):
                raise ValueError("mixed-write witness requires same frozen inputs")
            for rank in ranks:
                sid = cfg["order"][rank - 1]
                data = raw_for(sid, cfg["raw_sha256"])
                ctx, _, keys = as_case(
                    base, data, rank if key == "fixed360" else 21, mapping, rule
                )
                orig, old = (
                    x[:, :128].contiguous() for x in r.actions(rule, ctx, stale=True)
                )
                bank_flags = causal_phase_support(keys, bank)["allow"]
                values = action_targets(
                    orig, old, bank_flags, project_layer0(base, ctx, orig)
                )
                values["static"] = torch.zeros_like(values["intra"])
                arms = torch.stack([values[k] for k in MENU], dim=1)
                idx = torch.arange(len(arms), device=arms.device)
                selectors = (idx % 4, (3 * idx + 1) % 4)
                mixtures = {
                    f"mix{j}": arms[idx, sel] for j, sel in enumerate(selectors)
                }
                live, _ = r.forecast(base, ctx, mixtures)
                f = path / "shots" / f"{rank:04d}_{sid}.npz"
                if r.sha(f) != done["sealed"][rank - 1]["curve_sha256"]:
                    raise ValueError("mixed-write reference curve changed")
                with np.load(f, allow_pickle=False) as a:
                    if not np.array_equal(a["times"], data["times"]):
                        raise ValueError("mixed-write clock changed")
                    start, stride = r.policy.FIRST, r.policy.STRIDE
                    coords = np.arange(start, len(data["times"]))
                    p = np.stack([a[k] for k in MENU])
                    for j, selector in enumerate(selectors):
                        expected = p[0].copy()
                        expected[coords] = p[
                            selector.cpu().numpy()[(coords - start) // stride], coords
                        ]
                        err = layer.require_curve_parity(
                            live[f"mix{j}"], expected, r.THRESHOLD
                        )
                        rows.append(
                            dict(
                                mode=key, rank=rank, shot=sid, mixture=j, max_error=err
                            )
                        )
    if r.state_digest(base) != source or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("mixed-write witness modified source")
    r.write_new(
        output,
        dict(
            status="PASS_REAL_MIXED_WRITES",
            rows=rows,
            max_curve_error=max(x["max_error"] for x in rows),
            original_20_refit_max_parameter_error=fit_max,
            refit_steps=trace["initial"]["optimizer_steps"]
            + sum(x["steps"] for x in trace["rounds"]),
            input_completion_sha256=inputs,
        ),
    )
    print(
        f"verified {len(rows)} mixed-write curves; full fit parameter max difference {fit_max:g}",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    witness(p.parse_args().output)
