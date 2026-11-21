"""Reconstruct the cold bank and replay actual truncated signal prefixes."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from run_layer0_signed_projection_20261106 import (
    ARMS,
    as_case,
    forecast_signed,
    layer,
    load_bank,
    parents,
    r,
    raw_for,
)


RANKS = {"fixed": (20, 21, 181, 359), "conditional": (1, 210, 418, 835)}


def witness(fixed, conditional, bank_dir, output):
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, fixed_cfg, _, base, source, mapping, _, fit = parents()
    bank, receipt = load_bank(bank_dir, base.center.device)
    rule = r.load_rule(fit, base.center.device)
    paths = {"fixed": fixed, "conditional": conditional}
    parents_sha = {k: r.sha(path / "completed.json") for k, path in paths.items()}
    rows = []
    bank_max = 0.0
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("prefix witness cannot access outcomes"),
    ):
        for row in receipt["historical_disruptive_rows"]:
            sid, slot = row["shot"], row["slot"]
            if order[row["rank"] - 1] != sid or row["rank"] > 20:
                raise ValueError("bank may only use released cold roles")
            raw = raw_for(sid, fixed_cfg["raw_sha256"])
            _, _, keys = as_case(base, raw, 21, mapping, rule)
            for stored, name in zip(bank[slot], ("precursor", "early"), strict=True):
                j = (np.array(row[name]) - r.policy.FIRST) // r.policy.STRIDE
                err = float((keys[j] - stored).abs().max().item())
                if err > 4e-6:
                    raise ValueError(f"cold bank key cannot be reconstructed for {sid}")
                bank_max = max(bank_max, err)
        for mode, path in paths.items():
            cfg, done = [r.read(path / p) for p in ("protocol.json", "completed.json")]
            if (
                done["status"] != "COMPLETE_LAYER0_SIGNED_PROJECTION"
                or cfg["mode"] != mode
                or cfg["threshold"] != r.THRESHOLD
                or cfg["source_state_sha256"] != source
                or cfg["bank_receipt_sha256"] != r.sha(bank_dir / "completed.json")
                or done["protocol_sha256"] != r.sha(path / "protocol.json")
            ):
                raise ValueError("unbound witness input")
            for rank in RANKS[mode]:
                sid = cfg["order"][rank - 1]
                raw = raw_for(sid, cfg["raw_sha256"])
                start, stride = r.policy.FIRST, r.policy.STRIDE
                full_len = len(raw["times"])
                cuts = sorted({start, start + 1, start + stride, full_len // 2})
                f = path / "shots" / f"{rank:04d}_{sid}.npz"
                if r.sha(f) != done["sealed"][rank - 1]["curve_sha256"]:
                    raise ValueError("sealed prefix reference changed")
                active = rank >= cfg["bank_active_rank"]
                map_rank = rank if mode == "fixed" else 21
                with np.load(f, allow_pickle=False) as saved:
                    for n in cuts:
                        if not 0 < n < full_len:
                            raise ValueError("witness cut must be a strict prefix")
                        data = {
                            **raw,
                            **{
                                k: raw[k][:n].copy()
                                for k in ("times", "values", "available")
                            },
                        }
                        ctx, _, keys = as_case(
                            base, data, map_rank, mapping, rule if active else None
                        )
                        if active:
                            p, details, _ = forecast_signed(base, ctx, rule, keys, bank)
                            for name in (
                                "anchor_supported",
                                "anchor_positive",
                                "anchor_zero",
                            ):
                                d = details[name]
                                if not np.array_equal(d, saved[name][: len(d)]):
                                    raise ValueError(f"prefix changed {sid}/{n}/{name}")
                        else:
                            p = dict.fromkeys(ARMS, ctx[2])
                        if not np.array_equal(data["times"], saved["times"][:n]):
                            raise ValueError("prefix changed physical clock")
                        diffs = {
                            k: layer.require_curve_parity(
                                p[k], saved[k][:n], r.THRESHOLD
                            )
                            for k in ARMS
                        }
                        rows.append(
                            dict(
                                mode=mode, rank=rank, shot=sid, n=n, max_abs_error=diffs
                            )
                        )
    if r.state_digest(base) != source or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("prefix run altered frozen source")
    r.write_new(
        output,
        dict(
            status="PASS_SIGNED_TRUNCATED_PREFIXES",
            rows=rows,
            n_curves=len(rows) * len(ARMS),
            n_cold_banks=len(bank),
            cold_key_max_abs_error=bank_max,
            max_probability_error=max(
                v for row in rows for v in row["max_abs_error"].values()
            ),
            parent_completion_sha256=parents_sha,
            scope="Causal prefixes, exact threshold crossings and no outcome access; not a timing or deployment test.",
        ),
    )
    print(
        f"verified {len(rows)} truncated prefixes / {len(rows) * len(ARMS)} curves",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "conditional", "bank", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    witness(a.fixed, a.conditional, a.bank, a.output)
