"""Post-hoc labelled cone-capacity diagnostic, never a deployable policy."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from ..history_gradient_metric_20261102.run import ROOT, read, sha, write


def cone_projection(gradients, risk):
    g, r = np.asarray(gradients, np.float64), np.asarray(risk, np.float64)
    if (
        g.ndim != 2
        or r.shape != g.shape[1:]
        or not np.isfinite(g).all()
        or not np.isfinite(r).all()
    ):
        raise ValueError("finite channel gradients and risk vector required")
    nr = np.linalg.norm(r)
    if nr == 0:
        raise ValueError("defined risk direction required")
    r = r / nr
    norms = np.linalg.norm(g, axis=1)
    active = np.flatnonzero(norms > 0)
    if len(active):
        # Positive column normalization changes neither the generated cone nor
        # its projection. It prevents arbitrary diagnostic units conditioning
        # the NNLS solve; it is not an input calibration or online update.
        a = (g[active] / norms[active, None]).T
        coefficients, _ = nnls(a, r, maxiter=10000)
        projected = a @ coefficients
        derivative = a.T @ (projected - r)
        free = coefficients > 1e-10
        kkt = max(
            float(np.max(np.abs(derivative[free]))) if free.any() else 0.0,
            float(np.max(np.maximum(-derivative[~free], 0))) if (~free).any() else 0.0,
        )
        if kkt > 1e-6:
            raise FloatingPointError("NNLS cone projection failed KKT")
        original_weights = np.zeros(len(g))
        original_weights[active] = coefficients / norms[active]
    else:
        projected = np.zeros_like(r)
        original_weights = np.zeros(len(g))
        kkt = 0.0
    magnitude = float(np.linalg.norm(projected))
    if magnitude > 1 + 1e-6 or abs(float(projected @ r) - magnitude**2) > 1e-6:
        raise FloatingPointError("invalid projection geometry")
    # The cone is intentionally larger than the allowed box/sum-constrained
    # weights. A large value is only a loose labelled capacity reference.
    rescaled = (
        (original_weights * len(g) / original_weights.sum())
        if original_weights.sum() > 0
        else original_weights
    )
    return dict(
        cosine=magnitude,
        direction_defined=magnitude > 0,
        normalized_projection=projected.tolist(),
        nonzero_channels=len(active),
        kkt_residual=kkt,
        normalized_weight_max=float(rescaled.max()),
        rescaled_ray_box_feasible=bool(magnitude > 0 and rescaled.max() <= 4),
    )


def main():
    root = ROOT / "probe_outputs/risk_conditioned_feedback20_v1"
    protocol = read(root / "protocol.json")
    source = Path(protocol["source_collection"])
    assert sha(source / "gradients.npz") == protocol["source_gradients_sha256"]
    rows = read(source / "records.json")
    with np.load(source / "gradients.npz", allow_pickle=False) as stored:
        arrays = {
            name: stored[name].astype(np.float64)
            for name in ("real", "reverse", "task")
        }
        observations = stored["observations"]
    records = []
    for i, row in enumerate(rows):
        active = observations[i] > 0
        record = {key: row[key] for key in ("shot", "query", "label", "disruptive")}
        for mode in ("real", "reverse"):
            record[mode] = cone_projection(arrays[mode][i, active], arrays["task"][i])
        records.append(record)
    summaries = {}
    for mode in ("real", "reverse"):
        summaries[mode] = {}
        for name in ("class0", "class1", "safe"):
            group = [
                r
                for r in records
                if (
                    not r["disruptive"]
                    if name == "safe"
                    else r["label"] == int(name[-1])
                )
            ]
            ids = {r["shot"] for r in group}
            means = [
                np.mean([r[mode]["cosine"] for r in group if r["shot"] == sid])
                for sid in ids
            ]
            summaries[mode][name] = dict(
                n=len(group),
                n_shots=len(ids),
                shot_balanced_mean=float(np.mean(means)),
                point_quantiles=np.quantile(
                    [r[mode]["cosine"] for r in group], [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
                rescaled_ray_box_feasible=sum(
                    r[mode]["rescaled_ray_box_feasible"] for r in group
                ),
            )
    result = dict(
        status="COMPLETED_POSTHOC_LABELLED_CONE_REFERENCE",
        summary=summaries,
        records=records,
        diagnostic_only=True,
        current_query_label_used=True,
        deployable=False,
        online_gain_measured=False,
        warning="per-sample known-risk-gradient projection; ignores shared fitting, coefficient box when unsupported and finite neural-step dynamics",
        box_certificate_scope="Only tests the zero-unused-coefficient ray rescaled to mean one. A false value is not proof that all bounded representations are impossible.",
        supersedes="labelled_cone_reference.json: same numeric projection, ambiguous box_representable field replaced by limited feasibility certificate",
        source_completed_sha256=sha(root / "completed.json"),
        source_gradients_sha256=sha(source / "gradients.npz"),
        code_sha256=sha(Path(__file__)),
    )
    write(root / "labelled_cone_reference_v2.json", result)
    print(json.dumps(dict(status=result["status"], summary=summaries)), flush=True)


if __name__ == "__main__":
    main()
