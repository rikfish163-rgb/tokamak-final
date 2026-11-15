"""Fit causal selectors by held source pair; score already-sealed real interventions."""

import argparse
from pathlib import Path
import time

import numpy as np

import peak_contrast_policy_20261106 as policy
import verify_peak_contrast_capacity_20261106 as verify
from verify_confirmation_raw_replay_20261105 import Hashes


r = verify.run.r
SEEDS = verify.run.SEEDS
ARMS = ("static", "raw", "context", "fixed", "shape", "reverse", "score")
TRAINED = ("raw", "context", "fixed", "shape")
PLAN = r.ROOT / "PEAK_CONTRAST_CROSSFIT_PROTOCOL_20261106.md"


def select_fold(
    z, offset, native, descriptors, available, positive, protect, tau, tr, val
):
    raw, shape = policy.utility(
        native[tr], z[tr], offset[tr], positive[tr], protect[tr], tau
    )
    shape_ctx = descriptors.copy()
    shape_ctx[..., :22] = 0.0
    available = np.repeat(available[:, None], 2, axis=1)
    a = available.shape[-1]
    values = {
        name: np.empty((len(val), 2, len(SEEDS)), dtype=np.int64) for name in TRAINED
    }
    for j in range(len(SEEDS)):
        tr_mask = available[tr, :, j].reshape(-1, a)
        va_mask = available[val, :, j].reshape(-1, a)
        for name, target, obs in (
            ("raw", raw, descriptors),
            ("context", shape, shape_ctx),
            ("shape", shape, descriptors),
        ):
            tr_x = obs[tr, :, j].reshape(-1, obs.shape[-1])
            va_x = obs[val, :, j].reshape(-1, obs.shape[-1])
            y = target[:, :, j].reshape(-1, a)
            fit = policy.fit_values(tr_x, y, tr_mask)
            values[name][:, :, j] = policy.choose(fit, va_x, va_mask).reshape(-1, 2)
        # Identical rewards, no inference features, one global training-fold action.
        count = tr_mask.sum(0)
        target = shape[:, :, j].reshape(-1, a)
        mean = (target * tr_mask).sum(0) / count.clip(min=1)
        best = int(policy.choose_scores(mean[None], count[None] >= policy.MIN_ROWS)[0])
        values["fixed"][:, :, j] = (
            np.where(va_mask[:, best], best, -1).reshape(-1, 2) if best >= 0 else -1
        )
    return values


def chosen_payload(z, shift, norms, native, present, choices):
    # No predicted proxy score enters the risk trace. Take the archived real
    # forward of the chosen parameter state, or the exact frozen trajectory.
    i, side, seed = np.indices(z.shape[:3])
    ref = native[:, :, None, :]
    names = {
        **choices,
        "reverse": np.where(choices["shape"] >= 0, choices["shape"] ^ 1, -1),
    }
    actual, offsets, magnitudes = [], [], []
    for name in ARMS:
        if name == "static":
            actual.append(np.broadcast_to(ref, z.shape[:3] + (32,)))
            offsets.append(np.zeros(z.shape[:3]))
            magnitudes.append(np.zeros(z.shape[:3]))
            continue
        pick = names["shape" if name == "score" else name]
        on = pick >= 0
        j = pick.clip(min=0)
        o = np.where(on, shift[i, side, seed, j], 0.0).astype(np.float64)
        updated = (
            np.where(present[:, :, None], ref.astype(np.float64) + o[..., None], ref)
            if name == "score"
            else np.where(on[..., None], z[i, side, seed, j], ref)
        )
        actual.append(updated)
        offsets.append(o)
        magnitudes.append(
            np.zeros_like(o)
            if name == "score"
            else np.where(on, norms[i, side, seed, j], 0.0)
        )
    return tuple(np.stack(q, 3) for q in (actual, offsets, magnitudes))


def compact(summary, picks, timely):
    index = {a: i for i, a in enumerate(ARMS)}
    state = {a: dict(shape=[], raw=[], net=[], losses=[]) for a in ARMS}
    rows = summary["pairs"]
    for name, j in index.items():
        entry = state[name]
        entry["shape"] = summary["arms"][j]["shape_contrast"]
        for k in range(len(SEEDS)):
            cases = [q["seed_actions"][k][j] for q in rows]
            entry["raw"].append(sum(q["new_danger_crossing_raw"] for q in cases))
            entry["net"].append(
                sum(q["safe_induced_raw"] - q["safe_suppressed_raw"] for q in cases)
            )
            entry["losses"].append(
                sum(q["lost_protected_crossing"] is True for q in cases)
            )
        entry["changed_windows"] = (
            (picks[name] >= 0).sum((0, 1)).tolist() if name in TRAINED else None
        )
    diffs = {}
    base = index["shape"]
    for name in ("raw", "context", "fixed", "reverse", "score"):
        j = index[name]
        d = np.array(
            [
                [
                    q[base]["shape_contrast"] - q[j]["shape_contrast"]
                    for q in row["seed_actions"]
                ]
                for row in rows
            ],
            dtype=np.float64,
        ).mean(1)
        rng = np.random.default_rng(61106)
        ci = np.percentile(
            d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1), [2.5, 97.5]
        )
        diffs[name] = dict(mean=float(d.mean()), percentile=ci.tolist())
    s = state["shape"]
    criteria = dict(
        shape_margin=s["shape"]["three_seed_mean"] >= 0.02,
        two_seed_positive=sum(v > 0 for v in s["shape"]["per_seed_mean"]) >= 2,
        above_raw_context_fixed=all(
            diffs[a]["mean"] >= 0.01 for a in ("raw", "context", "fixed")
        ),
        no_raw_net_new_safe_cross=all(v <= 0 for v in s["net"]),
        keeps_protected_first=not any(s["losses"]),
    )
    return dict(
        n_pairs=len(rows),
        n_native_not_timely=int((~timely).sum()),
        by_arm=state,
        differences=diffs,
        criteria=criteria,
        passed=all(criteria.values()),
    )


def replay(root, out):
    h, start = Hashes(), time.monotonic()
    if out.exists():
        raise FileExistsError(out)
    audit_path = root / "audit.json"
    hashes = {
        str(p.resolve()): h(p)
        for p in (
            PLAN,
            Path(__file__),
            Path(policy.__file__),
            Path(verify.__file__),
            Path(verify.scores.__file__),
            audit_path,
        )
    }
    cfg, parts, raw = verify.preflight(root, h)
    audit = r.read(audit_path)
    if (
        audit["status"] != "INDEPENDENT_TRAIN_ONLY_CAPACITY_RECOUNT"
        or h(root / "completed.json") != audit["input_seal_sha256"]
        or not any(audit["gate"].values())
    ):
        raise ValueError("verified, sufficient action capacity required before fitting")
    m, path = parts["all"]
    ids, positive, protect, timely = verify.as_arrays(m, raw)
    folds = policy.grouped_folds(ids)
    present = np.array(
        [[q["danger"]["present"], q["safe"]["present"]] for q in m["pairs"]]
    )
    result, selections = {}, {}
    with np.load(path, allow_pickle=False) as a:
        native, z, shifts, norms, common = verify.check_payload(a, len(ids), m)
        descriptors = a["descriptor"]
        for k, site in enumerate(cfg["sites"]):
            if not audit["gate"][site]:
                continue
            picks = {
                name: np.full((len(ids), 2, len(SEEDS)), -2, dtype=np.int64)
                for name in TRAINED
            }
            for train, hold in folds:
                selected = select_fold(
                    z[:, :, k],
                    shifts[:, :, k],
                    native,
                    descriptors,
                    common,
                    positive,
                    protect,
                    cfg["logit_threshold"],
                    train,
                    hold,
                )
                for name in TRAINED:
                    if (picks[name][hold] != -2).any():
                        raise ValueError("held pair selected twice")
                    picks[name][hold] = selected[name]
            for q in picks.values():
                j = np.maximum(q, 0)
                active = np.take_along_axis(common[:, None], j[..., None], axis=-1)[
                    ..., 0
                ]
                if (q < -1).any() or ((q >= 0) & ~active).any():
                    raise ValueError(
                        "held pair missing or selected an unavailable action"
                    )
            xyz = chosen_payload(
                z[:, :, k], shifts[:, :, k], norms[:, :, k], native, present, picks
            )
            # The choices are fully determined for every held pair before a
            # scoring pass can inspect that pair's actual outcomes or effects.
            report = verify.scores.score_pairs(
                native,
                *xyz,
                positive,
                protect,
                timely,
                cfg["logit_threshold"],
                ARMS,
                ids,
            )
            result[site] = compact(report, picks, timely)
            selections[site] = {name: q.tolist() for name, q in picks.items()}
            print(site, result[site]["criteria"], result[site]["passed"], flush=True)
    h.check()
    r.write_new(
        out,
        dict(
            status="CROSSFITTED_SOURCE_PAIR_REPLAY",
            input_sha256=hashes,
            capacity_audit_sha256=h(audit_path),
            sealed_effects_sha256=h(path),
            pair_ids=ids.tolist(),
            folds=[dict(train=a.tolist(), held=b.tolist()) for a, b in folds],
            selections=selections,
            by_site=result,
            held_val_east_reads=0,
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    replay(a.root, a.output)
