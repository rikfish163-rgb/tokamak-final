"""Fixed action menu: rare hard events and a causal first-write selector."""

import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

import run_patch_utility_selection_20261106 as main
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = main.bank.r.ROOT / "PATCH_UTILITY_SPARSE_PROTOCOL_20261106.md"
QUANTILES = (0.50, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995)


def event_weights(hard, allowed, owner):
    """One third per outcome class; normalize again by shot, block, action."""
    w = np.zeros(hard.shape, dtype=np.float64)
    for y in (-1, 0, 1):
        group = allowed & (hard == y)
        group[:, 0] = False
        counts = group.sum(axis=1)
        blocks = np.flatnonzero(counts)
        if not len(blocks):
            raise ValueError(f"training has no eligible hard class {y}")
        _, group_id, shot_counts = np.unique(
            owner[blocks], return_inverse=True, return_counts=True
        )
        scale = 3 * len(shot_counts) * shot_counts[group_id] * counts[blocks]
        w[blocks] += group[blocks] / scale[:, None]
    return w


def eligible_menu(data, tau):
    allow = data["allowed"].copy()
    # Already-sealed alarms can never be withdrawn by a later action.
    allow[:, 1:] &= data["risk"][:, 8:9].astype(np.float64) < tau
    return allow


def top(score, allowed):
    score = np.where(allowed, score, -np.inf)
    score[:, 0] = -np.inf
    action = score.argmax(axis=1)
    return action, score[np.arange(len(score)), action]


def choose_once(score, allowed, owner, margin):
    if (owner[1:] < owner[:-1]).any():
        raise ValueError("rows must be chronological within contiguous shots")
    action, best = top(score, allowed)
    qualified = np.flatnonzero(best > margin)
    _, first = np.unique(owner[qualified], return_index=True)
    chosen = np.zeros(len(owner), dtype=np.int64)
    chosen[qualified[first]] = action[qualified[first]]
    return chosen


def fit_events(x, hard, allowed, owner):
    w = event_weights(hard, allowed, owner)
    rows = w > 0
    x, y, weight = x[rows].astype(np.float64), hard[rows], w[rows]
    center = np.average(x, axis=0, weights=weight)
    var = np.average((x - center) ** 2, axis=0, weights=weight)
    scale = np.where(var > 1e-12, np.sqrt(var), 1.0)
    # With normalized weights, C = 1/alpha gives the stated L2 coefficient.
    model = LogisticRegression(
        C=1.0 / main.ALPHA, solver="lbfgs", max_iter=300, tol=1e-9
    ).fit((x - center) / scale, y, sample_weight=weight)
    if not np.array_equal(model.classes_, [-1, 0, 1]) or model.n_iter_.max() >= 300:
        raise RuntimeError("incomplete balanced multiclass fit")
    return dict(
        center=center.tolist(), scale=scale.tolist(),
        coefficient=model.coef_.tolist(), intercept=model.intercept_.tolist(),
        steps=int(model.n_iter_.max()),
        support={
            str(k): {
                "shots": int(np.unique(owner[(rows & (hard == k)).any(axis=1)]).size),
                "actions": int(np.count_nonzero(y == k)),
            }
            for k in (-1, 0, 1)
        },
    )


def event_scores(model, x):
    z = (
        (x.astype(np.float64) - model["center"]) / model["scale"]
    ) @ np.asarray(model["coefficient"]).T + model["intercept"]
    return z[..., 2] - np.logaddexp(z[..., 1], z[..., 0] + np.log(2.0))


def rank_support(score, allowed, hard, owner, ids):
    a, best = top(score, allowed)
    credit = hard[np.arange(len(a)), a]
    return dict(
        oracle_shots=ids[np.unique(owner[((hard > 0) & allowed).any(axis=1)])].tolist(),
        top_action_shots=ids[np.unique(owner[(credit > 0) & np.isfinite(best)])].tolist(),
        top_action_positive_blocks=int(np.count_nonzero((credit > 0) & np.isfinite(best))),
        top_action_negative_blocks=int(np.count_nonzero((credit < 0) & np.isfinite(best))),
    )


def hard_credit(hard, choice, owner, ids, disruptive):
    credit = hard[np.arange(len(choice)), choice]
    s, d = ids[owner], disruptive[owner]
    return dict(
        recovered=s[(credit > 0) & d].tolist(),
        lost=s[(credit < 0) & d].tolist(),
        fp_added=s[(credit < 0) & ~d].tolist(),
        fp_removed=s[(credit > 0) & ~d].tolist(),
        writes=int(np.count_nonzero(choice)),
    )


def run(root, original, out):
    if out.exists():
        raise FileExistsError(out)
    h, start = Hashes(), time.monotonic()
    path = main.bank.r.ROOT
    for p in (Path(__file__), Path(main.__file__), Path(main.fit.__file__),
              Path(main.utility.__file__), PLAN, main.bank.PLAN):
        h(p)
    parent, pred, fit_path = (
        original / key for key in
        ("result.json", "selected_actions.npz", "fit_and_calibration.json")
    )
    reference = main.bank.r.read(parent)
    fits = main.bank.r.read(fit_path)["evidence"]
    if (
        reference["promoted"]
        or reference["choices_sha256"] != h(pred)
        or reference["fit_sha256"] != h(fit_path)
        or reference["bank_seal_sha256"] != h(root / "seal.json")
    ):
        raise ValueError("exploratory protocol is bound to the failed original run")
    h(parent)
    shots, data, hard, _, owner = main.load_bank(root, h)
    ids = np.array([q.sid for q in shots])
    disruptive = np.array([q.disruptive for q in shots])
    folds = main.fit.shot_folds(ids, main.N_FOLDS)
    with np.load(pred, allow_pickle=False) as f:
        if not all(np.array_equal(f[k], v) for k, v in
                   (("hard", hard), ("owner", owner), ("folds", folds))):
            raise ValueError("hard-label and fold inputs differ from parent receipt")
    allowed = eligible_menu(data, main.bank.TAU)
    arms, chosen_by_arm, receipts, scores_by_arm, rank = {}, {}, {}, {}, {}
    n_blocks = len(hard)
    for view in main.VIEWS:
        x = main.features(data, view)
        for design in ("ridge", "events"):
            name = f"{design}_{view}"
            score = np.full(hard.shape, -np.inf, dtype=np.float64)
            selected = np.full((n_blocks, 2), -1, dtype=np.int64)
            logs = []
            for k in range(main.N_FOLDS):
                fold = folds[owner]
                held = fold == k
                cal = fold == (k + 1) % main.N_FOLDS
                tr = ~(held | cal)
                if design == "ridge":
                    saved = fits[view][k]
                    if (saved["held_shots"] != ids[folds == k].tolist()
                            or saved["calibrate_shots"] != ids[folds == (k + 1) % main.N_FOLDS].tolist()):
                        raise ValueError("ridge parameters do not match the active split")
                    model = main.fit.Model(**saved["model"])
                    sc = model.predict(x[cal], allowed[cal])
                    sh = model.predict(x[held], allowed[held])
                    info = {"reference_fold": k}
                else:
                    info = fit_events(x[tr], hard[tr], allowed[tr], owner[tr])
                    sc, sh = event_scores(info, x[cal]), event_scores(info, x[held])
                score[held] = sh
                _, best = top(sc, allowed[cal])
                attained = best[np.isfinite(best)]
                if not len(attained):
                    raise ValueError(f"calibration fold {k} contains no writable blocks")
                candidates = np.unique(np.r_[-np.inf, np.quantile(attained, QUANTILES), np.inf])

                def evaluate(m, cal=cal, sc=sc):
                    c = choose_once(sc, allowed[cal], owner[cal], m)
                    return hard_credit(hard[cal], c, owner[cal], ids, disruptive)

                margin, grid = main.calibrate(evaluate, candidates)
                for row in grid:
                    if row["margin"] == -np.inf:
                        row["margin"] = "all"
                selected[held, 0] = choose_once(sh, allowed[held], owner[held], -np.inf)
                selected[held, 1] = choose_once(sh, allowed[held], owner[held], margin)
                logs.append(dict(fold=k, model=info, calibration=grid,
                                 margin="all" if margin == -np.inf else main.safe_margin(margin)))
            if (selected < 0).any():
                raise ValueError("incomplete held write policy")
            rank[name] = rank_support(score, allowed, hard, owner, ids)
            receipts[name], scores_by_arm[name] = logs, score
            for j, mode in enumerate(("open", "guarded")):
                c = selected[:, j]
                summary = main.summarize(main.score_policy(shots, data, c))
                credit = hard_credit(hard, c, owner, ids, disruptive)
                if any(summary[p] != credit[p] for p in
                       ("recovered", "lost", "fp_added", "fp_removed")):
                    raise ValueError("single-edit training credit disagrees with full replay")
                key = f"{name}_{mode}"
                arms[key] = summary | {"writes": credit["writes"]}
                chosen_by_arm[key] = c
                if view == "true":
                    arms[f"{key}_scalar"] = main.summarize(
                        main.score_policy(shots, data, c, scalar=True)
                    ) | {"writes": credit["writes"]}
            print(f"Completed single-write selector: {name}", flush=True)
    h.check()
    out.mkdir(parents=True, exist_ok=False)
    npz, calfile = out / "choices.npz", out / "calibration.json"
    main.bank.r.save_arrays(
        npz, owner=owner, folds=folds,
        **chosen_by_arm, **{f"score_{k}": v for k, v in scores_by_arm.items()}
    )
    main.bank.r.write_new(calfile, receipts)
    winner = arms["events_true_guarded"]
    comparators = ("events_risk_guarded", "events_reverse_guarded",
                   "events_true_guarded_scalar")
    distinct = sorted(set(winner["recovered"] + winner["fp_removed"]) -
                      {i for c in comparators for p in ("recovered", "fp_removed")
                       for i in arms[c][p]})
    result = dict(
        status="SCORED_POSTHOC_SAME_TRAIN_EXPLORATION",
        baseline=reference["baseline"], arms=arms, top_rank_support=rank,
        matches_source_gate=bool(distinct and not winner["lost"] and not winner["fp_added"]
                                 and (winner["recovered"] or len(winner["fp_removed"]) >= 3)),
        distinct=distinct, choices_sha256=h(npz), calibration_sha256=h(calfile),
        parent_sha256=h(parent), bank_seal_sha256=h(root / "seal.json"),
        code_sha256=h(Path(__file__)), protocol_sha256=h(PLAN),
        worktree=str(path), elapsed_seconds=time.monotonic() - start,
        new_val_test_east_reads=0,
    )
    main.bank.r.write_new(out / "result.json", result)
    print({k: (v["timely"], v["fp"], v["writes"]) for k, v in arms.items()}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for p in ("bank", "original", "output"):
        cli.add_argument(p, type=Path)
    args = cli.parse_args()
    run(args.bank, args.original, args.output)
