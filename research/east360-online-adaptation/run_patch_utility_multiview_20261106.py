"""Grouped, chronological single-write comparison on three sealed TRAIN views."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

import run_patch_utility_selection_20261106 as sel
import run_patch_utility_sparse_20261106 as sparse
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = sel.bank.r.ROOT / "PATCH_UTILITY_MULTIVIEW_POLICY_PROTOCOL_20261106.md"
VIEWS = ("native", "no_MNM", "no_MNM_CIII")
REGIMES = ("native_only", "view_only", "pooled")
BASELINE_KEYS = ("recovered", "lost", "fp_added", "fp_removed", "writes")


@dataclass
class Tape:
    shots: list
    data: dict
    hard: np.ndarray
    owner: np.ndarray
    allowed: np.ndarray


def paired_weights(hard, allowed, owner, view):
    """Balance by class, physical shot, view, block, then action."""
    w = np.zeros(hard.shape, dtype=np.float64)
    for y in (-1, 0, 1):
        group = allowed & (hard == y)
        group[:, 0] = False
        counts = group.sum(axis=1)
        blocks = np.flatnonzero(counts)
        if not len(blocks):
            raise ValueError(f"no usable action for hard class {y}")
        keys, key_id, blocks_per_view = np.unique(
            np.column_stack((owner[blocks], view[blocks])),
            axis=0, return_inverse=True, return_counts=True,
        )
        ids, shot_id, views_per_shot = np.unique(
            keys[:, 0], return_inverse=True, return_counts=True,
        )
        denominator = (
            3 * len(ids) * views_per_shot[shot_id[key_id]]
            * blocks_per_view[key_id] * counts[blocks]
        )
        w[blocks] += group[blocks] / denominator[:, None]
    return w


def fit(x, hard, allowed, owner, view):
    w = paired_weights(hard, allowed, owner, view)
    rows = w > 0
    values, y, weight = x[rows].astype(np.float64), hard[rows], w[rows]
    center = np.average(values, axis=0, weights=weight)
    var = np.average((values - center) ** 2, axis=0, weights=weight)
    scale = np.where(var > 1e-12, np.sqrt(var), 1.0)
    model = sparse.LogisticRegression(
        C=1 / sel.ALPHA, solver="lbfgs", max_iter=300, tol=1e-9,
    ).fit((values - center) / scale, y, sample_weight=weight)
    if not np.array_equal(model.classes_, [-1, 0, 1]) or model.n_iter_.max() >= 300:
        raise RuntimeError("incomplete balanced multiview fit")
    return dict(
        center=center.tolist(), scale=scale.tolist(),
        coefficient=model.coef_.tolist(), intercept=model.intercept_.tolist(),
        steps=int(model.n_iter_.max()),
        support={
            str(k): dict(
                shots=int(np.unique(owner[(rows & (hard == k)).any(axis=1)]).size),
                actions=int(np.count_nonzero(y == k)),
            ) for k in (-1, 0, 1)
        },
    )


def rows_for_fold(folds, owner, k):
    held, cal = folds[owner] == k, folds[owner] == (k + 1) % sel.N_FOLDS
    return ~(held | cal), cal, held


def union(records, key, views=VIEWS):
    return sorted({s for v in views for s in records[v][key]})


def safe_margin(margin):
    return "all" if margin == -np.inf else sel.safe_margin(margin)


def calibrate(evaluate, margins):
    best, choice, rows = None, None, []
    for margin in margins:
        effects = evaluate(margin)
        eligible = not (union(effects, "fp_added") or union(effects, "lost"))
        recovered, removed = union(effects, "recovered"), union(effects, "fp_removed")
        key = (len(recovered), len(removed), margin)
        rows.append(dict(margin=safe_margin(margin), eligible=eligible,
                         recovered=recovered, fp_removed=removed, views=effects))
        if eligible and (best is None or key > best):
            best, choice = key, margin
    if choice is None:
        raise ValueError("the calibration must permit no write")
    return choice, rows


def source_gate(scores):
    target = VIEWS[1:]
    controls = set().union(*(union(scores[k], "recovered", target)
                             for k in ("risk", "reverse", "scalar")))
    distinct = sorted(set(union(scores["true"], "recovered", target)) - controls)
    protected = not (union(scores["true"], "fp_added") or union(scores["true"], "lost"))
    return bool(distinct and protected), distinct


def load(native, masked, baseline, h):
    read = sel.bank.r.read
    complete = read(masked / "completed.json")
    cap = read(masked / "capacity.json")
    witness = read(masked / "verification.json")
    prior = read(baseline)
    if (
        not cap["support_gate"] or witness["source_result_sha256"] != h(masked / "capacity.json")
        or complete["native_seal_sha256"] != prior["bank_seal_sha256"]
        or h(native / "seal.json") != prior["bank_seal_sha256"] or complete["outcome_reads"]
        or set(complete["seals"]) != set(VIEWS[1:])
        or any(h(masked / v / "seal.json") != sha for v, sha in complete["seals"].items())
    ):
        raise ValueError("run requires the independently verified sealed three-view capacity")
    for p in (masked / "completed.json", masked / "verification.json", baseline):
        h(p)
    banks = {}
    for name in VIEWS:
        shots, data, hard, _, owner = sel.load_bank(native if name == VIEWS[0] else masked / name, h)
        summary = sel.summarize([(q.sid, q.teacher["baseline"], q.teacher["baseline"]) for q in shots])
        expected = prior["baseline"] if name == VIEWS[0] else cap["views"][name]["baseline"]
        if summary != expected:
            raise ValueError(f"{name} zero-write baseline does not match sealed evidence")
        banks[name] = Tape(shots, data, hard, owner, sparse.eligible_menu(data, sel.bank.TAU))
    ref = banks[VIEWS[0]]
    ids, flags = (np.array([getattr(q, key) for q in ref.shots]) for key in ("sid", "disruptive"))
    for k in VIEWS[1:]:
        tape = banks[k]
        if (
            not np.array_equal(tape.owner, ref.owner)
            or not np.array_equal(tape.data["clock"], ref.data["clock"])
            or any((a.sid, a.disruptive, a.down) != (b.sid, b.disruptive, b.down)
                   for a, b in zip(tape.shots, ref.shots, strict=True))
        ):
            raise ValueError("all views must share the same HDF5 shot and release schedule")
    return banks, ids, flags


def train_model(tapes, features, rows, training_views):
    a = [(features[v][rows], tapes[v].hard[rows], tapes[v].allowed[rows],
          tapes[v].owner[rows], np.full(int(rows.sum()), VIEWS.index(v), dtype=np.int8))
         for v in training_views]
    return fit(*(np.concatenate(col) for col in zip(*a, strict=True)))


def replay(tape, choices, ids, flags, *, scalar=False):
    result = sel.summarize(sel.score_policy(tape.shots, tape.data, choices, scalar=scalar))
    result["writes"] = int(np.count_nonzero(choices))
    if not scalar:
        expected = sparse.hard_credit(tape.hard, choices, tape.owner, ids, flags)
        if any(result[k] != expected[k] for k in BASELINE_KEYS):
            raise ValueError("single-write hard-credit and actual first alarm disagree")
    return result


def crossfit(tapes, ids, flags, folds):
    owner = tapes[VIEWS[0]].owner
    n = len(owner)
    archive, reports, receipts = {}, {}, {}
    for kind in sel.VIEWS:
        x = {v: sel.features(tapes[v].data, kind) for v in VIEWS}
        for regime in REGIMES:
            prefix = f"{regime}__{kind}"
            scores = {v: np.empty(tapes[v].hard.shape, dtype=np.float64) for v in VIEWS}
            choices = {v: np.full((n, 2), -1, dtype=np.int64) for v in VIEWS}
            logs = []
            for k in range(sel.N_FOLDS):
                tr, cal, held = rows_for_fold(folds, owner, k)
                if regime == "view_only":
                    train = {v: (v,) for v in VIEWS}
                else:
                    common = (VIEWS[0],) if regime == "native_only" else VIEWS
                    train = {v: common for v in VIEWS}
                fits = {key: train_model(tapes, x, tr, key) for key in dict.fromkeys(train.values())}
                p_cal = {v: sparse.event_scores(fits[train[v]], x[v][cal]) for v in VIEWS}
                q = np.concatenate([sparse.top(p_cal[v], tapes[v].allowed[cal])[1] for v in VIEWS])
                finite = q[np.isfinite(q)]
                if not len(finite):
                    raise ValueError("joint calibration requires writable blocks")
                margins = np.unique(np.r_[-np.inf, np.quantile(finite, sparse.QUANTILES), np.inf])

                def evaluate(margin):
                    return {
                        v: sparse.hard_credit(
                            tapes[v].hard[cal],
                            sparse.choose_once(p_cal[v], tapes[v].allowed[cal], owner[cal], margin),
                            owner[cal], ids, flags,
                        ) for v in VIEWS
                    }

                margin, cal_rows = calibrate(evaluate, margins)
                for v in VIEWS:
                    sh = sparse.event_scores(fits[train[v]], x[v][held])
                    scores[v][held] = sh
                    for mode, gate in enumerate((-np.inf, margin)):
                        choices[v][held, mode] = sparse.choose_once(sh, tapes[v].allowed[held], owner[held], gate)
                logs.append(dict(fold=k, margin=safe_margin(margin), calibration=cal_rows,
                                 training_views={v: list(train[v]) for v in VIEWS},
                                 models={"+".join(key): m for key, m in fits.items()}))
            row = dict(open={}, guarded={}, rank={})
            if kind == "true":
                row.update(scalar_open={}, scalar_guarded={})
            for v in VIEWS:
                if (choices[v] < 0).any():
                    raise ValueError("unassigned outer-fold rows")
                tape = tapes[v]
                archive[f"{prefix}__{v}"] = choices[v]
                row["rank"][v] = sparse.rank_support(scores[v], tape.allowed, tape.hard, owner, ids)
                for j, mode in enumerate(("open", "guarded")):
                    row[mode][v] = replay(tape, choices[v][:, j], ids, flags)
                    if kind == "true":
                        row[f"scalar_{mode}"][v] = replay(tape, choices[v][:, j], ids, flags, scalar=True)
            reports[prefix], receipts[prefix] = row, logs
            print(f"Finished {prefix}: " +
                  ", ".join(f"{v} {row['guarded'][v]['timely']}/{row['guarded'][v]['fp']}"
                            for v in VIEWS), flush=True)
    return archive, reports, receipts


def run(native, masked, baseline, out):
    if out.exists():
        raise FileExistsError(out)
    started, h = time.monotonic(), Hashes()
    for p in (Path(__file__), Path(sel.__file__), Path(sparse.__file__),
              Path(sel.fit.__file__), Path(sel.utility.__file__), PLAN):
        h(p)
    tapes, ids, flags = load(native, masked, baseline, h)
    folds = sel.fit.shot_folds(ids, sel.N_FOLDS)
    archive, reports, models = crossfit(tapes, ids, flags, folds)
    gate, unique = source_gate(
        {k: reports[f"pooled__{k}"]["guarded"] for k in sel.VIEWS}
        | {"scalar": reports["pooled__true"]["scalar_guarded"]}
    )
    h.check()
    out.mkdir(parents=True, exist_ok=False)
    choice, receipt = out / "choices.npz", out / "models_and_calibration.json"
    sel.bank.r.save_arrays(choice, ids=ids, folds=folds, owner=tapes[VIEWS[0]].owner, **archive)
    sel.bank.r.write_new(receipt, dict(
        splits=[dict(held=ids[folds == k].tolist(),
                     calibrate=ids[folds == (k+1) % sel.N_FOLDS].tolist(),
                     fit=ids[(folds != k) & (folds != (k+1) % sel.N_FOLDS)].tolist())
                for k in range(sel.N_FOLDS)],
        models=models,
    ))
    result = dict(
        status="SCORED_GROUPED_TRAIN_MULTIVIEW_NOT_ONLINE",
        arms=reports, source_gate=gate, distinct_recovered=unique,
        fixed_tau=sel.bank.TAU, source_shots=len(ids),
        archive_sha256=h(choice), models_sha256=h(receipt),
        input_hashes={str(k): v[1] for k, v in h.cache.items()},
        elapsed_seconds=time.monotonic()-started, new_val_test_east_reads=0,
    )
    sel.bank.r.write_new(out / "result.json", result)
    print({"source_gate": gate, "distinct_recovered": unique}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for p in ("native", "masked", "native_result", "output"):
        cli.add_argument(p, type=Path)
    args = cli.parse_args()
    run(args.native, args.masked, args.native_result, args.output)
