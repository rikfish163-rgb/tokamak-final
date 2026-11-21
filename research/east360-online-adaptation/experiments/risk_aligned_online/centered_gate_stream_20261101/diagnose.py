"""Saved-receipt description of signal versus known quadratic-anchor gradient.

This is a post-run numerical description, not a second training/evaluation
policy. Subtracting the analytic anchor from stored float32 total gradients is
subject to float rounding; it is not an independently replayed signal VJP.
"""

import argparse
from pathlib import Path

import numpy as np

from ..east_context_operator_20261031.run import read
from ..simple_online import _durable_json
from ..simple_train import sha


def distribution(values):
    x = np.asarray([v for v in values if v is not None], np.float64)
    if not len(x):
        return dict(n=0, median=None, q10=None, q90=None)
    assert np.isfinite(x).all()
    return dict(
        n=len(x),
        median=float(np.median(x)),
        q10=float(np.quantile(x, 0.1)),
        q90=float(np.quantile(x, 0.9)),
    )


def separate(receipt, anchor):
    g = np.asarray(receipt["gradient32"], np.float64)
    before = np.asarray(receipt["beta32_before"], np.float64)
    regularizer = 2 * anchor / 32 * before
    signal = g - regularizer
    total_norm, anchor_norm, signal_norm = map(
        float, (np.linalg.norm(g), np.linalg.norm(regularizer), np.linalg.norm(signal))
    )
    return signal, dict(
        total_norm=total_norm,
        anchor_norm=anchor_norm,
        signal_norm=signal_norm,
        anchor_to_signal=anchor_norm / signal_norm if signal_norm else None,
        anchor_larger=anchor_norm > signal_norm,
    )


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else None


def main(root):
    output = root / "gradient_diagnostics.json"
    if output.exists():
        raise FileExistsError(output)
    p = read(root / "protocol.json")
    done = read(root / "completed.json")
    assert done["status"] == "COMPLETED"
    old = Path(p["old"])
    records = {
        a: []
        for a in ("absolute_real", "centered_real", "centered_reverse", "centered_hold")
    }
    last_gates = {a: [] for a in records}
    total_cosines, signal_cosines, risk_rms, risk_max = [], [], [], []
    for i, record in enumerate(p["order"][20:], 21):
        prefix = f"{i:04d}_{record['shot']}"
        logs = read(root / "shots" / f"{prefix}_updates.json")["updates"]
        logs["absolute_real"] = read(old / "shots" / f"{prefix}_updates.json")[
            "updates"
        ]["real"]
        for arm in records:
            for event in logs[arm]:
                _, summary = separate(event, p["anchor"])
                records[arm].append(summary)
            if logs[arm]:
                b = np.asarray(logs[arm][-1]["beta32_after"]).reshape(16, 2)
                last_gates[arm].append(
                    dict(long=float(b[:, 0].mean()), snapshot=float(b[:, 1].mean()))
                )
        for real, reverse in zip(
            logs["centered_real"], logs["centered_reverse"], strict=True
        ):
            a, _ = separate(real, p["anchor"])
            b, _ = separate(reverse, p["anchor"])
            total_cosines.append(
                cosine(
                    np.asarray(real["gradient32"]), np.asarray(reverse["gradient32"])
                )
            )
            signal_cosines.append(cosine(a, b))
        with np.load(root / "shots" / f"{prefix}.npz", allow_pickle=False) as curves:
            delta = curves["centered_real"].astype(np.float64) - curves["absolute_real"]
            risk_rms.append(float(np.sqrt(np.mean(delta**2))))
            risk_max.append(float(np.abs(delta).max()))
    result = dict(
        scope="POSTHOC_SAVED_RECEIPT_DESCRIPTION_NO_TRAINING",
        analytical_anchor_formula="2 * .01 /32 * beta_before",
        rounding_limit="signal = stored float32 total minus analytical float64 anchor, not a replayed VJP",
        per_arm={
            a: dict(
                events=len(v),
                anchor_larger_events=sum(r["anchor_larger"] for r in v),
                anchor_to_signal=distribution([r["anchor_to_signal"] for r in v]),
                signal_norm=distribution([r["signal_norm"] for r in v]),
                anchor_norm=distribution([r["anchor_norm"] for r in v]),
                final_mean_long=distribution([r["long"] for r in last_gates[a]]),
                final_mean_snapshot=distribution(
                    [r["snapshot"] for r in last_gates[a]]
                ),
            )
            for a, v in records.items()
        },
        centered_real_reverse_total_cosine=distribution(total_cosines),
        centered_real_reverse_approx_signal_cosine=distribution(signal_cosines),
        centered_vs_absolute_per_shot_risk_rms=distribution(risk_rms),
        centered_vs_absolute_per_shot_risk_max=distribution(risk_max),
        cosine_compares_separate_trajectories=True,
        no_raw_signals_or_outcomes=True,
        no_training=True,
        no_gpu=True,
        completed_sha256=sha(root / "completed.json"),
        script_sha256=sha(Path(__file__)),
        independent_confirmation=False,
    )
    _durable_json(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
