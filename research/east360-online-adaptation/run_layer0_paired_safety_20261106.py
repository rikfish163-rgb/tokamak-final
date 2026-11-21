"""Train matched safety directions; score real frozen-model writes on held shots."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import adjacent_patch_feedback_20261106 as feedback
import layer0_arrival_motion_20261106 as arrival
import paired_diagnostic_step_20261106 as layer
import run_patch_sustained_margin_20261106 as previous
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import deterministic_folds, first_alarm


b, m = previous.b, previous.m
PLAN = b.r.ROOT / "LAYER0_PAIRED_SAFETY_PROTOCOL_20261106.md"
DELAY, BLOCK, RADIUS = 8, 32, .35
SCALES = previous.SCALES
BOUND = previous.BOUND
ARMS = ("static", "matched", "shuffled", "reverse", "scalar")


@dataclass(frozen=True)
class Release:
    view: str
    shot: int
    clock: int
    context: tuple
    families: int
    points: tuple

    @property
    def key(self):
        return self.view, self.shot, self.clock


def shuffled_partners(view):
    out = np.arange(len(view))
    for v in np.unique(view):
        x = np.flatnonzero(view == v)
        if len(x) < 2:
            raise ValueError("at least two pairs per view for a derangement")
        out[x] = np.roll(x, 1)
    return out


def contrast(danger_shift, safe_shift):
    return F.relu(safe_shift - danger_shift + .05).square().mean()


def direction(coeff, gradient, margin):
    d = torch.einsum("bk,bkd->bd", coeff, gradient)
    return F.normalize(d, dim=1, eps=1e-9) * (RADIUS / (1. + margin.abs()))[:, None]


def write_grid(clock, active, length):
    ix = clock[:, None] + np.arange(DELAY + 1, BLOCK + 1)
    row, col = np.nonzero(active[:, None] & (ix < length))
    return row, ix[row, col]


def context_at(native, times, clock):
    end = clock + DELAY
    z = native[end]
    return np.column_stack((z - b.TAU, (times[end] - times[0]), z - native[clock - 16]))


def teacher_pool(view, tape):
    positives, negatives = [], []
    for q in tape.shots:
        c = tape.data["clock"][q.blocks]
        recorded = tape.data["ordered"][q.blocks, 0, -9:] > 0
        groups = recorded.reshape(-1, 3, 3).any(axis=2)
        enough = (c + BLOCK < len(q.times)) & (groups.sum(1) >= 2)
        c, groups = c[enough], groups[enough]
        ctx = context_at(q.native, q.times, c)
        pattern = groups @ np.array([1, 2, 4])
        rows = []
        for k, clock in enumerate(c):
            ix = np.arange(clock + DELAY + 1, clock + BLOCK + 1)
            if q.disruptive:
                lead = np.round((q.down - q.times[ix]) * 1000, 9)
                ix = ix[(lead > 10.) & (lead <= q.horizon + 1e-6)]
            if len(ix):
                peak = ix[np.argsort(-q.native[ix], kind="stable")[:4]]
                rows.append(Release(view, q.sid, int(clock), tuple(ctx[k]), int(pattern[k]),
                                    tuple(np.pad(peak, (0, 4-len(peak)), mode="edge"))))
        if q.disruptive and rows:
            positives.append(min(rows, key=lambda x: (abs(x.context[0]), x.clock)))
        elif not q.disruptive:
            negatives.extend(rows)
    return positives, negatives


def match_pairs(positives, negatives):
    """Causal variables determine proximity; only the ended-shot pool uses labels."""
    nc = np.asarray([n.context for n in negatives])
    family = np.asarray([n.families for n in negatives])
    shot = np.asarray([n.shot for n in negatives])
    used, result = set(), []
    for p in sorted(positives, key=lambda q: (abs(q.context[0]), q.shot)):
        delta = np.abs(nc - p.context)
        fit = (delta[:, 0] <= 1.) & (delta[:, 1] <= .5) & (family == p.families)
        ix = np.flatnonzero(fit)
        cost = delta[ix] @ np.array([1., 2., .25])
        for k in ix[np.argsort(cost, kind="stable")]:
            if shot[k] not in used:
                used.add(shot[k])
                result.append((p, negatives[k]))
                break
    return result


def raw_view(view, sid, times):
    raw = b.r.simple_data.read_shot("J-TEXT", sid)
    if not np.array_equal(raw["times"], times):
        raise ValueError(f"changed source clock: {sid}")
    return raw if view == "native" else previous.mask(raw, view)


def reached(base, raw, x, a, clock, native):
    """The actual target indices are m-24,...,m, never the score suffix."""
    origin = clock - arrival.H
    ix = origin[:, None] + np.arange(1, arrival.H + 1)
    y = x.new_tensor(raw["values"][ix])
    mask = torch.as_tensor(raw["available"][ix], device=x.device)
    observed = mask & torch.isfinite(y)
    keep = torch.zeros(23, device=x.device, dtype=torch.bool)
    for _, ids in feedback.GROUPS:
        keep[list(ids)] = True
    observed &= ((observed.sum(1) >= feedback.MIN_OBSERVED) & keep)[:, None]
    reverse = feedback._reverse_observed(y, observed)
    pairs = observed[:, 8:] & observed[:, :-8]
    with torch.enable_grad():
        u = x.new_zeros((len(clock), layer.WIDTH), requires_grad=True)
        _, forecast = arrival.features(base, x[origin].contiguous(), a[origin].contiguous(), u)
        g1 = feedback._losses(forecast, y, observed, pairs, base.center, base.scale)
        g2 = feedback._losses(forecast, reverse, observed, pairs, base.center, base.scale)
        odd = (g1 - g2).reshape(len(clock), 3, 3).mean(2) * .5
        grads = [torch.autograd.grad(odd[:, j].sum(), u, retain_graph=j < 2)[0]
                 for j in range(3)]
    gradient = -torch.stack(grads, dim=1).detach()
    strength = gradient.norm(dim=2)
    support = (strength > 1e-9).sum(1) >= 2
    feat = torch.cat((x.new_tensor(context_at(native, raw["times"], clock)), strength.log1p()), dim=1)
    return feat, F.normalize(gradient, dim=2, eps=1e-9), support


def responses(base, x, a, at, native):
    index = at.ravel()
    u = x.new_zeros((len(index), layer.WIDTH), requires_grad=True)
    z = layer.risk(base, x[index].contiguous(), a[index].contiguous(), u)
    if np.max(np.abs(z.detach().cpu().numpy() - native[index])) > 8e-5:
        raise ValueError("frozen first-layer zero-state mismatch")
    (g,) = torch.autograd.grad(z.sum(), u)
    return g.reshape(*at.shape, layer.WIDTH).detach()


def collect(base, episodes, shots):
    by_shot = defaultdict(dict)
    for p in episodes:
        by_shot[p.view, p.shot][p.clock] = p
    cache = {}
    for i, ((view, sid), ep) in enumerate(by_shot.items(), 1):
        q = shots[view, sid]
        raw = raw_view(view, sid, q.times)
        x, a = b.bank.prior.views(base, raw)
        c = np.array(list(ep), dtype=np.int64)
        f, g, ok = reached(base, raw, x, a, c, q.native)
        targets = np.asarray([ep[k].points for k in c])
        jac = responses(base, x, a, targets, q.native)
        ref = x.new_tensor(q.native[targets])
        for j in torch.nonzero(ok, as_tuple=True)[0].tolist():
            cache[ep[c[j]].key] = tuple(v[j].cpu() for v in (f, g, jac, ref))
        if i % 150 == 0 or i == len(by_shot):
            print(f"source teacher extraction {i}/{len(by_shot)} shot-views", flush=True)
    return cache


def fit(base, pairs, cache):
    pairs = [(p, n) for p, n in pairs if p.key in cache and n.key in cache]
    if min(len({p.shot for p, _ in pairs}), len({n.shot for _, n in pairs})) < 20:
        raise ValueError("insufficient independent matched TRAIN shots")
    data = [tuple(torch.stack([cache[row[k].key][j] for row in pairs]).to(base.center.device)
                  for j in range(4)) for k in range(2)]
    x = torch.cat((data[0][0], data[1][0]))
    mean, std = x.mean(0), x.std(0, correction=0).clamp_min(1e-5)
    identity = np.arange(len(pairs))
    shuffle = shuffled_partners(np.array([p.view for p, _ in pairs]))
    heads, history = {}, {}
    for name, order in (("matched", identity), ("shuffled", shuffle)):
        net = nn.Linear(x.shape[1], 3, device=x.device)
        nn.init.zeros_(net.weight)
        nn.init.constant_(net.bias, .2)
        optim = torch.optim.AdamW(net.parameters(), lr=.025, weight_decay=.005)
        f, g, j, native = data[0]
        sf, sg, sj, sn = (d[order] for d in data[1])
        for _ in range(160):
            u = direction(net((f - mean) / std).tanh(), g, f[:, 0])
            v = direction(net((sf - mean) / std).tanh(), sg, sf[:, 0])
            danger = (native + (j @ u[..., None]).squeeze(-1)).amax(1)
            safe = (sn + (sj @ v[..., None]).squeeze(-1)).amax(1)
            sep = contrast(danger - native.amax(1), safe - sn.amax(1))
            loss = F.softplus(b.TAU - danger).mean() + F.softplus(safe - b.TAU).mean() + sep
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite safety-direction objective")
        heads[name] = net.eval().requires_grad_(False)
        history[name] = dict(loss=float(loss.detach()), contrast=float(sep.detach()))
    stat = dict(pairs=len(pairs), distinct_danger=len({p.shot for p, _ in pairs}),
                distinct_safe=len({s.shot for _, s in pairs}), fit=history,
                identities=[[p.view, p.shot, p.clock, n.shot, n.clock] for p, n in pairs])
    return (heads, mean, std), stat


@torch.no_grad()
def exact_risk(base, x, a, frames, states):
    z = []
    for k in range(0, len(frames), 1024):
        sl = slice(k, k+1024)
        z.append(layer.risk(base, x[frames[sl]].contiguous(), a[frames[sl]].contiguous(), states[sl]))
    return torch.cat(z).cpu().numpy()


def predict(base, raw, native, clock, fitted):
    clock = clock[clock + DELAY + 1 < len(native)]
    trace = np.broadcast_to(native, (len(ARMS), len(native))).copy()
    count, max_guard, max_norm_error = 0, 0., 0.
    if len(clock):
        x, a = b.bank.prior.views(base, raw)
        f, g, ok = reached(base, raw, x, a, clock, native)
        heads, mean, std = fitted
        with torch.no_grad():
            initial = torch.stack([direction(heads[k]((f - mean) / std).tanh(), g, f[:, 0])
                                   for k in ("matched", "shuffled")], dim=1)
        ok &= (initial.norm(dim=2) > 1e-8).all(1)
        use = torch.nonzero(ok, as_tuple=True)[0].cpu().numpy()
        c = clock[use]
        if len(c):
            v = initial[use]
            anchors = np.column_stack((c, c + DELAY))
            grid = np.broadcast_to(anchors[:, None, None, :], (len(c), 4, len(SCALES), 2))
            signed = torch.cat((v, -v), dim=1)
            edits = (signed[:, :, None, None] * x.new_tensor(SCALES)[None, None, :, None, None])
            edits = edits.expand(-1, -1, -1, 2, -1).reshape(-1, layer.WIDTH)
            shift = exact_risk(base, x, a, grid.ravel(), edits).reshape(grid.shape) - native[grid]
            acceptable = (np.abs(shift) <= BOUND).all(axis=(1, 3))
            chosen = acceptable.argmax(1)
            active = acceptable.any(1)
            chosen_shift = shift[np.arange(len(c)), :, chosen]
            steps = np.where(active, SCALES[chosen], 0.)
            post = signed[:, :3] * x.new_tensor(steps)[:, None, None]
            if active.any():
                count = int(active.sum())
                max_guard = float(np.abs(chosen_shift[active]).max())
                max_norm_error = float((post.norm(dim=2).amax(1) - post.norm(dim=2).amin(1)).max().cpu())
                row, at = write_grid(c, active, len(native))
                edited = exact_risk(base, x, a, np.repeat(at, 3), post[row].reshape(-1, layer.WIDTH))
                trace[1:4, at] = edited.reshape(-1, 3).T
                trace[4, at] = native[at] + chosen_shift[row, 0, 1]
    if max_norm_error > 2e-6 or not np.isfinite(trace).all():
        raise ValueError("held writes or equal-norm controls are invalid")
    return trace, dict(active=count, max_guard=max_guard, max_norm_error=max_norm_error)


def evaluate(views, tau):
    records = {}
    for view, shots in views.items():
        counts = defaultdict(dict)
        for arm in ARMS:
            for metric, relevant in (("timely", True), ("false_alarm", False)):
                hits, added, lost = [], [], []
                for row in shots:
                    if row["disruptive"] != relevant:
                        continue
                    pred, static = (row["alarms"][k][metric] for k in (arm, "static"))
                    sid = row["shot"]
                    if pred:
                        hits.append(sid)
                    if pred and not static:
                        added.append(sid)
                    if static and not pred:
                        lost.append(sid)
                counts[arm][metric] = dict(hits=hits, added=added, lost=lost)
        records[view] = dict(arms=dict(counts), n={role: sum(q["role"] == role for q in shots)
                                                  for role in previous.STRATA})
    return records


def decision(report):
    def union(arm, metric, kind):
        return {sid for row in report.values() for sid in row["arms"][arm][metric][kind]}

    other = set().union(*(union(a, "timely", "added") for a in ARMS[2:]))
    novel = union("matched", "timely", "added") - other
    lost = union("matched", "timely", "lost")
    fp = {v: len(a["arms"]["matched"]["false_alarm"]["hits"])
          - len(a["arms"]["static"]["false_alarm"]["hits"]) for v, a in report.items()}
    return dict(matched_unique_timely=sorted(novel), lost_timely=sorted(lost),
                fp_delta=fp, pass_to_full_source=len(novel) >= 2 and not lost and max(fp.values()) <= 0)


def run(native, masked, baseline, output):
    if output.exists():
        raise FileExistsError(output)
    h, start = Hashes(), time.monotonic()
    tapes, ids, _ = m.load(native, masked, baseline, h)
    by_id = dict(zip(ids, deterministic_folds(ids), strict=True))
    b.r.configure()
    base, parent = b.r.load_parent(b.r.DEFAULT_PARENT)
    initial = b.r.state_digest(base)
    if (initial != b.r.read(native / "protocol.json")["source_state_sha256"]
            or parent["threshold"] != b.r.THRESHOLD):
        raise ValueError("qualified source changed")
    base.cuda()
    shots = {(view, q.sid): q for view, tape in tapes.items() for q in tape.shots}
    pools = {v: teacher_pool(v, t) for v, t in tapes.items()}
    pairs = [[row for p, n in pools.values()
              for row in match_pairs([q for q in p if by_id[q.shot] != k],
                                     [q for q in n if by_id[q.shot] != k])]
             for k in range(5)]
    print("matching before gradient support", [len(p) for p in pairs], flush=True)
    cached = collect(base, (q for p in pairs for pair in p for q in pair), shots)
    models, fits = {}, {}
    for k, rows in enumerate(pairs):
        models[k], fits[k] = fit(base, rows, cached)
        print("fit", k, fits[k]["pairs"], fits[k]["fit"], flush=True)
    output.mkdir(parents=True, exist_ok=False)
    b.r.write_new(output/"protocol.json", dict(status="SOURCE_TRAIN_HELD_ONLY",
        source_state=initial, tau=b.TAU, arms=ARMS, scales=SCALES.tolist(), cap=RADIUS,
        bound=BOUND, folds={int(k): int(v) for k,v in by_id.items()},
        locked_plan=h(PLAN), source_seal=h(native/"seal.json"), fits=fits))
    coeff = {}
    for k, (heads, mean, std) in models.items():
        coeff.update({f"{k}_mean": mean.cpu().numpy(), f"{k}_std": std.cpu().numpy()})
        for arm, net in heads.items():
            for name, v in net.state_dict().items():
                coeff[f"{k}_{arm}_{name}"] = v.cpu().numpy()
    b.r.save_arrays(output/"coefficients.npz", **coeff)
    scored, witness = {}, {}
    for view, tape in tapes.items():
        (output/view).mkdir()
        scored[view] = []
        for role, q in previous.cohort(tape.shots, b.TAU):
            raw = raw_view(view, q.sid, q.times)
            c = tape.data["clock"][q.blocks]
            head = models[by_id[q.sid]]
            trace, info = predict(base, raw, q.native, c, head)
            if view not in witness and info["active"] >= 3:
                cut = int(c[len(c)//2] + 16)
                if cut < len(q.native):
                    p = {key: raw[key][:cut] for key in ("times","values","available")}
                    thin, _ = predict(base, p, q.native[:cut], c, head)
                    err = float(np.abs(thin - trace[:, :cut]).max())
                    if err > 8e-5 or not np.array_equal(thin.astype(float) >= b.TAU,
                                                       trace[:, :cut].astype(float) >= b.TAU):
                        raise ValueError("held prefix changed when later observations were removed")
                    witness[view] = dict(shot=q.sid, cut=cut, max_logit_error=err)
            b.r.save_arrays(output/view/f"{q.sid}.npz", time=q.times, logits=trace, clock=c)
            alarms = {arm: first_alarm(q.times, z, q.disruptive, q.down, b.TAU)
                      for arm, z in zip(ARMS, trace, strict=True)}
            if alarms["static"] != q.teacher["baseline"]:
                raise ValueError("sealed static first alarm changed")
            scored[view].append(dict(shot=q.sid, fold=int(by_id[q.sid]), role=role,
                                     disruptive=q.disruptive, down=q.down, alarms=alarms, **info))
        print(view, "held", len(scored[view]), "prefix", witness.get(view), flush=True)
    if b.r.state_digest(base) != initial or any(p.grad is not None for p in base.parameters()):
        raise ValueError("frozen J-TEXT parent was changed")
    report = evaluate(scored, b.TAU)
    b.r.write_new(output/"result.json", dict(status="HELD_TRAIN_DIAGNOSTIC_NOT_EAST",
        views=report, rows=scored, prefix=witness, decision=decision(report),
        duration=time.monotonic()-start))
    print(decision(report), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("native", "masked", "baseline", "output"):
        cli.add_argument(key, type=Path)
    args = cli.parse_args()
    run(args.native, args.masked, args.baseline, args.output)
