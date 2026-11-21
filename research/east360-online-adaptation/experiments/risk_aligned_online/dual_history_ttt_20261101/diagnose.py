"""Post-run description of gate movement and order-control sensitivity.

No raw signal/outcome access, training, parameter selection, or GPU work.
Statistics are descriptive and do not certify physical mechanisms.
"""

import argparse
from pathlib import Path

import numpy as np

from ..east_context_operator_20261031.run import read
from ..simple_online import _durable_json
from ..simple_train import sha


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        key: float(np.quantile(values, q))
        for key, q in (("q10", 0.1), ("median", 0.5), ("q90", 0.9), ("max", 1.0))
    }


def main(root):
    out = root / "gate_diagnostics.json"
    if out.exists():
        raise FileExistsError(out)
    p = read(root / "protocol.json")
    gradients, beta_differences = [], []
    summaries = {
        a: {
            k: []
            for k in (
                "last_long",
                "last_snapshot",
                "objective_decrease",
                "step_norm",
                "risk_rms",
                "risk_max",
                "moving_events",
            )
        }
        for a in ("real", "reverse", "hold")
    }
    exact_real_reverse, risk_difference = 0, []
    for i, record in enumerate(p["order"][20:], 21):
        prefix = f"{i:04d}_{record['shot']}"
        log = read(root / "shots" / f"{prefix}_updates.json")["updates"]
        for real, reverse in zip(log["real"], log["reverse"], strict=True):
            g, h = np.asarray(real["gradient32"]), np.asarray(reverse["gradient32"])
            denom = np.linalg.norm(g) * np.linalg.norm(h)
            if denom:
                gradients.append(float(g @ h / denom))
            beta_differences.append(
                float(
                    np.linalg.norm(
                        np.asarray(real["beta32_after"]) - reverse["beta32_after"]
                    )
                )
            )
        with np.load(root / "shots" / f"{prefix}.npz", allow_pickle=False) as curves:
            exact_real_reverse += np.array_equal(curves["real"], curves["reverse"])
            risk_difference.append(
                float(
                    np.max(
                        np.abs(curves["real"].astype(np.float64) - curves["reverse"])
                    )
                )
            )
            for arm, summary in summaries.items():
                history = log[arm]
                beta = (
                    np.asarray(history[-1]["beta32_after"]).reshape(16, 2)
                    if history
                    else np.zeros((16, 2))
                )
                summary["last_long"].append(float(beta[:, 0].mean()))
                summary["last_snapshot"].append(float(beta[:, 1].mean()))
                summary["moving_events"].append(
                    sum(r["step_norm"] > 0 for r in history)
                )
                for r in history:
                    if r["observed"]:
                        summary["objective_decrease"].append(
                            r["loss_before"] - r["loss_retained"]
                        )
                        summary["step_norm"].append(r["step_norm"])
                difference = curves[arm].astype(np.float64) - curves["cold"]
                summary["risk_rms"].append(float(np.sqrt(np.mean(difference**2))))
                summary["risk_max"].append(float(np.max(np.abs(difference))))
    result = dict(
        scope="POSTHOC_DESCRIPTIVE_SAVED_RECEIPTS_AND_CURVES_ONLY",
        shots=340,
        summary={
            a: {k: quantiles(v) for k, v in d.items()} for a, d in summaries.items()
        },
        real_reverse_gradient_cosine=quantiles(gradients),
        real_reverse_gate_l2=quantiles(beta_differences),
        real_reverse_risk_max=quantiles(risk_difference),
        real_reverse_identical_complete_curves=exact_real_reverse,
        completed_sha256=sha(root / "completed.json"),
        script_sha256=sha(Path(__file__)),
        no_training=True,
        no_gpu=True,
        raw_signals_or_outcomes_read=0,
        new_physics_established=False,
    )
    _durable_json(out, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
