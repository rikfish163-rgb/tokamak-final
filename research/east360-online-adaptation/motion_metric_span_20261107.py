"""Retrospective gradient-span diagnostic; its weights are not deployable."""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import moving_peak_replay_20261105 as r
import motion_metric_model_20261107 as model
from motion_metric_kernel_20261107 import descent, potential
from run_motion_metric_20261107 import GROUPS, STEP, batch, prepare


def coefficients(matrix, target):
    """Signed least squares in the numerically retained column span."""
    matrix, target = matrix.double(), target.double()
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    keep = singular > 1e-4 * singular[:, :1]
    component = u.transpose(1, 2) @ target[:, :, None]
    inverse = torch.where(keep, 1 / singular.clamp_min(1e-300), 0)
    weight = (vh.transpose(1, 2) @ (component * inverse[:, :, None])).squeeze(-1)
    factor = (2 / weight.abs().amax(1, keepdim=True).clamp_min(1e-300)).clamp_max(1)
    weight = weight * factor
    projected = (component.squeeze(-1).square() * keep).sum(1).sqrt()
    fraction = projected / target.norm(dim=1).clamp_min(1e-300)
    return weight, keep.sum(1), fraction


def run(out):
    start = time.monotonic()
    r.configure()
    old = r.read(out / "val_result.json")
    meta = r.read(out / "val_sealed.json")["rows"]
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    data, rows, skipped = prepare(base, "val")
    if rows != meta:
        raise ValueError("the exact same already scored VAL episodes required")
    r.write_new(out / "span_protocol.json", dict(scope="RETROSPECTIVE_NOT_DEPLOYABLE", rows=len(rows),
        relative_svd_cutoff=1e-4, coefficient_bound=2, step=STEP, source_unchanged=True,
        source_files={str(p): p.read_text() for p in (Path(__file__), r.ROOT / "MOTION_METRIC_PROTOCOL_20261107.md")}))
    tau = math.log(parent["threshold"]/(1-parent["threshold"]))
    columns = [c for c in range(23) if c not in (2, 10)]
    result = []
    for lo in range(0, len(rows), 32):
        d = batch(data, np.arange(lo, min(lo+32, len(rows))))
        n = len(d["context"])
        variable = torch.zeros(n, 4, 64, device="cuda", requires_grad=True)
        prediction = model.forward_state(base, d["packs"]["0"], variable)["forecast"]
        channel_potential = -(prediction * d["motion"] * d["mask"]).sum(1) / d["mask"].sum((1, 2)).clamp_min(1)[:, None]
        gradients = [torch.autograd.grad(channel_potential[:, c].sum(), variable, retain_graph=True)[0]
                     .flatten(1).detach() for c in columns]
        matrix = torch.stack(gradients, dim=-1)
        q_variable = torch.zeros_like(variable, requires_grad=True)
        z = model.forward_state(base, d["packs"]["q"], q_variable)["logit"]
        risk = F.binary_cross_entropy_with_logits(z-tau, d["labels"].float(), reduction="sum")
        target, = torch.autograd.grad(risk, q_variable)
        weight, rank, fraction = coefficients(matrix, target.flatten(1))
        weights = torch.zeros(n, 23, device="cuda")
        weights[:, columns] = weight.float()
        actual_loss = potential(prediction, d["motion"], d["mask"], weights)
        gradient, = torch.autograd.grad(actual_loss.sum(), variable)
        update = descent(gradient, step=STEP)
        with torch.no_grad():
            after = model.forward_state(base, d["packs"]["q"], update)["logit"]
            anchor = model.forward_state(base, d["packs"]["r"], update)["logit"]
            scalar = d["packs"]["q"]["native"] + anchor - d["packs"]["r"]["native"]
            reconstructed = (matrix.double() @ weight[:, :, None]).squeeze(-1)
            gap = (gradient.flatten(1).double() - reconstructed).norm(dim=1)
            relative = gap / reconstructed.norm(dim=1).clamp_min(1e-300)
            cosine = F.cosine_similarity(gradient.flatten(1), target.flatten(1), dim=1)
            before = d["packs"]["q"]["native"]
            bce0 = F.binary_cross_entropy_with_logits(before.double()-tau, d["labels"].double(), reduction="none")
            bce1 = F.binary_cross_entropy_with_logits(after.double()-tau, d["labels"].double(), reduction="none")
            bces = F.binary_cross_entropy_with_logits(scalar.double()-tau, d["labels"].double(), reduction="none")
        for j in range(n):
            source = old["rows"][lo+j]
            if abs(float(before[j]) - source["logits"]["static"]) > 5e-6:
                raise ValueError("original source prediction changed")
            result.append(dict(rows[lo+j], rank=int(rank[j]), projection=float(fraction[j]),
                actual_cosine=float(cosine[j]), relative_reconstruction_error=float(relative[j]),
                coefficient_max=float(weights[j].abs().max()), state_norm=float(update[j].norm()),
                before_bce=float(bce0[j]), oracle_delta=float(bce1[j]-bce0[j]),
                scalar_delta=float(bces[j]-bce0[j]), learned_delta=source["after_bce"]["conditional"]-source["before_bce"]))
    table = {}
    for group in GROUPS:
        members = [q for q in result if q["group"] == group]
        table[group] = dict(n=len(members), oracle_delta=float(np.mean([q["oracle_delta"] for q in members])),
            scalar_delta=float(np.mean([q["scalar_delta"] for q in members])),
            learned_delta=float(np.mean([q["learned_delta"] for q in members])),
            projection_median=float(np.median([q["projection"] for q in members])),
            rank_median=float(np.median([q["rank"] for q in members])),
            actual_cosine_median=float(np.median([q["actual_cosine"] for q in members])),
            improved=sum(q["oracle_delta"] < 0 for q in members),
            improved_over_scalar=sum(q["oracle_delta"] < q["scalar_delta"] for q in members))
    if any(not torch.equal(frozen[k], v) for k, v in base.state_dict().items()):
        raise ValueError("source was modified")
    r.write_new(out / "span_result.json", dict(status="RETROSPECTIVE_NOT_ONLINE", table=table, rows=result,
        max_relative_reconstruction_error=max(q["relative_reconstruction_error"] for q in result),
        seconds=time.monotonic()-start))
    print(table, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
