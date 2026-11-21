"""Target sensitivity at the same zero state, from first-update saved vectors.

This does not rerun neural VJPs. Gradient differences are paired target
sensitivities, not an additive decomposition of SmoothL1 loss values, and do
not establish alignment with the disruption task.
"""

import argparse
from pathlib import Path

import numpy as np

from ..east_context_operator_20261031.run import read
from ..simple_online import _durable_json
from ..simple_train import sha


def paired(reference, other):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(other, dtype=np.float64)
    if (
        a.ndim != 1
        or a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("matching finite gradient vectors required")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dict(
        cosine=float(np.clip(a @ b / (na * nb), -1, 1)) if na and nb else None,
        relative_difference=float(np.linalg.norm(a - b) / na) if na else None,
    )


def audit(root):
    destination = root / "first_feedback_sensitivity.json"
    if destination.exists():
        raise FileExistsError(destination)
    assert read(root / "completed.json")["status"] == "COMPLETED"
    p = read(root / "protocol.json")
    rows = []
    for index, record in enumerate(p["order"][20:], 21):
        sid = record["shot"]
        prefix = f"{index:04d}_{sid}"
        path = root / "shots" / f"{prefix}_updates.json"
        seal = read(root / "shots" / f"{prefix}_seal.json")
        assert sha(path) == seal["updates_sha256"]
        info = read(path)
        first = {a: info["updates"][a][0] for a in ("real", "reverse", "hold")}
        assert all(
            r["origin"] == 134 and r["target_last"] == 159 for r in first.values()
        )
        assert all(
            len(r["state256_before"]) == 256 and not np.any(r["state256_before"])
            for r in first.values()
        )
        assert len({r["mask_sha256"] for r in first.values()}) == 1
        assert len({r["observed"] for r in first.values()}) == 1
        g = {
            a: np.asarray(r["gradient256"], dtype=np.float64) for a, r in first.items()
        }
        assert all(v.shape == (256,) for v in g.values())
        assert all(
            np.isclose(np.linalg.norm(g[a]), first[a]["gradient_norm"], rtol=1e-12)
            for a in first
        )
        original = paired(g["real"], g["reverse"])
        hold = paired(g["real"], g["hold"])
        differences = paired(g["real"] - g["hold"], g["reverse"] - g["hold"])
        rows.append(
            dict(
                shot=sid,
                real_reverse_cosine=original["cosine"],
                real_hold_cosine=hold["cosine"],
                real_hold_relative_difference=hold["relative_difference"],
                difference_reverse_cosine=differences["cosine"],
                first_accepted={a: r["accepted"] for a, r in first.items()},
                updates_sha256=sha(path),
            )
        )
    assert len(rows) == 340
    summaries = {}
    for key in (
        "real_reverse_cosine",
        "real_hold_cosine",
        "real_hold_relative_difference",
        "difference_reverse_cosine",
    ):
        values = np.asarray([r[key] for r in rows if r[key] is not None])
        summaries[key] = dict(
            n=len(values),
            undefined=len(rows) - len(values),
            quantiles=dict(
                zip(
                    ("min", "p10", "median", "p90", "max"),
                    np.quantile(values, (0, 0.1, 0.5, 0.9, 1)).tolist(),
                    strict=True,
                )
            )
            if len(values)
            else None,
        )
    result = dict(
        status="COMPLETED_MATCHED_ZERO_STATE_TARGET_SENSITIVITY",
        first_matched_states=len(rows),
        summaries=summaries,
        rows=rows,
        first_accepted={a: sum(r["first_accepted"][a] for r in rows) for a in first},
        completed_sha256=sha(root / "completed.json"),
        auditor_sha256=sha(Path(__file__)),
        gradient_vectors_recomputed=False,
        source_regularizer_gradient_at_zero=0,
        only_first_update_not_all_danger_stages=True,
        risk_gradient_alignment_measured=False,
        differences_are_not_additive_loss_values=True,
    )
    _durable_json(destination, result)
    print(
        {
            "n": len(rows),
            "summaries": summaries,
            "first_accepted": result["first_accepted"],
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
