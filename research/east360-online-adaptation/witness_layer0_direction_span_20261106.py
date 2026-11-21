"""Re-run the real direction candidates on truncated raw signal, with no labels."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from run_layer0_direction_span_20261106 import (
    ARMS,
    FIRST,
    STRIDE,
    load_basis,
    layer,
    parents,
    predict,
    r,
    raw_for,
)


RANKS = {"fixed": (20, 21, 181, 359), "conditional": (1, 210, 418, 835)}


def witness(fixed, conditional, output):
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    _, _, _, base, digest, mapping, map_path, fit = parents()
    rule = r.load_rule(fit, base.center.device)
    basis, singular = load_basis(fit, base.center.device)
    rows, inputs = [], {}
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("no current outcome in prefix witness"),
    ):
        for mode, path in (("fixed", fixed), ("conditional", conditional)):
            cfg, done = [r.read(path / k) for k in ("protocol.json", "completed.json")]
            inputs[mode] = r.sha(path / "completed.json")
            if (
                cfg["mode"] != mode
                or done["mode"] != mode
                or done["status"] != "COMPLETE_LAYER0_DIRECTION_SPAN"
                or done["protocol_sha256"] != r.sha(path / "protocol.json")
                or cfg["source_state_sha256"] != digest
                or cfg["source_map_sha256"] != r.sha(map_path)
                or cfg["source_fit_sha256"] != r.sha(fit)
                or cfg["threshold"] != r.THRESHOLD
                or cfg["basis_sha256"] != r.sha(path / "basis.npz")
            ):
                raise ValueError("witness requires the exact frozen replays")
            with np.load(path / "basis.npz", allow_pickle=False) as q:
                if not np.array_equal(
                    q["basis"], basis.cpu().numpy()
                ) or not np.array_equal(q["singular_values"], singular):
                    raise ValueError("basis did not reproduce exactly")
            for rank in RANKS[mode]:
                sid = cfg["order"][rank - 1]
                data = raw_for(sid, cfg["raw_sha256"])
                active = rank >= cfg["active_rank"]
                path_shot = path / "shots" / f"{rank:04d}_{sid}.npz"
                if r.sha(path_shot) != done["sealed"][rank - 1]["curve_sha256"]:
                    raise ValueError("witness target seal mismatch")
                cuts = sorted(
                    {FIRST, FIRST + 1, FIRST + STRIDE, len(data["times"]) // 2}
                )
                with np.load(path_shot, allow_pickle=False) as saved:
                    for n in cuts:
                        if not 0 < n < len(data["times"]):
                            raise ValueError("witness requires a strict prefix")
                        cut = {
                            k: data[k][:n].copy()
                            for k in ("times", "available", "values")
                        }
                        cut.update(shot=sid, device=data["device"])
                        if not np.array_equal(cut["times"], saved["times"][:n]):
                            raise ValueError("witness clock mismatch")
                        p, receipt = predict(
                            base,
                            cut,
                            rank if mode == "fixed" else 21,
                            active,
                            mapping,
                            rule,
                            basis,
                        )
                        errs = {
                            k: layer.require_curve_parity(
                                p[k], saved[k][:n], r.THRESHOLD
                            )
                            for k in ARMS
                        }
                        for key, v in receipt.items():
                            ref = saved[key][: len(v)]
                            if key == "anchor_valid":
                                same = np.array_equal(v, ref)
                            else:
                                same = np.allclose(v, ref, rtol=1e-5, atol=1e-6)
                            if not same:
                                raise ValueError(
                                    f"future length changed {sid}/{n}/{key}"
                                )
                        rows.append(
                            dict(
                                mode=mode,
                                rank=rank,
                                shot=sid,
                                cut=n,
                                arm_abs_errors=errs,
                            )
                        )
    if r.state_digest(base) != digest or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("witness modified frozen source")
    r.write_new(
        output,
        dict(
            status="PASS_DIRECTION_SPAN_RAW_PREFIXES",
            rows=rows,
            completion_sha256=inputs,
            n_curves=len(rows) * len(ARMS),
            max_probability_error=max(
                e for row in rows for e in row["arm_abs_errors"].values()
            ),
        ),
    )
    print(
        f"span witness: {len(rows)} strict prefixes and {len(rows) * len(ARMS)} exact threshold comparisons",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "conditional", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    witness(a.fixed, a.conditional, a.output)
