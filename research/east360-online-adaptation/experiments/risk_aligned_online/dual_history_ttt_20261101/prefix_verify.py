"""Additional CUDA future-perturbation check, specified before result inspection.

Only the first ended preparation shot is read, without its outcome. A320-point
prefix includes risk after the first possible learned long-view effect at288.
This script is a separate audit, not part of the running experiment snapshot.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..east_context_operator_20261031.run import configure, read
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE
from ..warmstart_alignment_20261022.alignment import transform
from .model import encode_views, latent_forward
from .run import LEARNERS, load_model, predict


def main(root):
    output = root / "cuda_prefix_verification.json"
    if output.exists():
        raise FileExistsError(output)
    configure()
    model, _, _ = load_model()
    sid = read(REFERENCE / "order.json")[0]["shot"]
    raw = apply_input_contract(
        simple_data.read_shot("EAST", sid, rate_contract="nominal")
    )
    signal = transform(
        {k: raw[k][:416] for k in ("values", "available")},
        read(REFERENCE / "input_map.json"),
    )
    assert len(signal["values"]) == 416
    curves, receipt = predict(model, signal)
    prefix = {k: v[:320].copy() for k, v in signal.items()}
    truncated, truncated_receipt = predict(model, prefix)
    changed = {k: v.copy() for k, v in signal.items()}
    changed["values"][320:] += 1e5
    changed["available"][320:] = ~changed["available"][320:]
    perturbed, perturbed_receipt = predict(model, changed)
    assert all(
        np.array_equal(curves[a][:320], truncated[a])
        and np.array_equal(curves[a][:320], perturbed[a][:320])
        for a in curves
    )
    for a in LEARNERS:
        earlier = [r for r in receipt["updates"][a] if r["target_last"] < 320]
        altered = [r for r in perturbed_receipt["updates"][a] if r["target_last"] < 320]
        assert earlier == truncated_receipt["updates"][a] == altered
        assert [r["target_last"] for r in earlier] == [159, 287]
    # Same-shaped batches isolate a forecast's prefix, including later rows
    # inside that very batch. Nonzero gates exercise the two alternate views.
    origin_checks = []
    beta = torch.full((16, 2), 0.2, device="cuda")
    for origin in (134, 262):
        lo = origin // 32 * 32
        ends = np.arange(lo, lo + 32)
        altered = {k: v.copy() for k, v in signal.items()}
        altered["values"][origin + 1 :] += 1e5
        altered["available"][origin + 1 :] = ~altered["available"][origin + 1 :]
        original_pack = encode_views(model, signal, ends)
        changed_pack = encode_views(model, altered, ends)
        point = {k: v[origin - lo : origin - lo + 1] for k, v in original_pack.items()}
        other = {k: v[origin - lo : origin - lo + 1] for k, v in changed_pack.items()}
        assert all(torch.equal(point[k], other[k]) for k in point)
        with torch.no_grad():
            forecast = latent_forward(model, point, beta, forecast=True)["forecast"]
            changed_forecast = latent_forward(model, other, beta, forecast=True)[
                "forecast"
            ]
        assert torch.equal(forecast, changed_forecast)
        origin_checks.append(origin)
    cutoff_checks = []
    for cutoff in (159, 160, 287, 288):
        altered = {k: v.copy() for k, v in signal.items()}
        altered["values"][cutoff + 1 :] += 1e5
        altered["available"][cutoff + 1 :] = ~altered["available"][cutoff + 1 :]
        changed_curves, changed_info = predict(model, altered)
        assert all(
            np.array_equal(curves[a][: cutoff + 1], changed_curves[a][: cutoff + 1])
            for a in curves
        )
        for a in LEARNERS:
            original_matured = [
                r for r in receipt["updates"][a] if r["target_last"] <= cutoff
            ]
            changed_matured = [
                r for r in changed_info["updates"][a] if r["target_last"] <= cutoff
            ]
            assert original_matured == changed_matured
        cutoff_checks.append(cutoff)
    result = dict(
        status="PASS",
        shot=sid,
        input_points=416,
        preserved_prefix_points=320,
        curves_exact=len(curves),
        receipts_exact=2 * len(LEARNERS),
        target_suffix_shots_read=0,
        outcomes_read=0,
        device="CUDA",
        same_batch_origin_checks=origin_checks,
        same_length_risk_cutoff_checks=cutoff_checks,
        includes_risk_after_long_gate_first_possible_effect=True,
        script_sha256=sha(Path(__file__)),
        scope="SINGLE_ENDED20_SHOT_FUTURE_PERTURBATION_NOT_GENERAL_PROOF",
    )
    _durable_json(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
