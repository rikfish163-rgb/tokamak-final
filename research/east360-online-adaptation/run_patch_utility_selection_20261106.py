"""Train shot-disjoint selectors on actual first-alarm intervention utility."""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np

import patch_utility_fit_20261106 as fit
import patch_utility_targets_20261106 as utility
import run_patch_utility_bank_20261106 as bank
from verify_adjacent_patch_intra_20261106 import outcome_and_clock
from verify_confirmation_raw_replay_20261105 import Hashes


VIEWS = ("risk", "true", "reverse")
MARGINS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, np.inf)
N_FOLDS = 5
ALPHA = 0.03
T = 32
BASELINE = bank.r.ROOT / "probe_outputs/first_passage_target_replay_20261106_v1.json"


@dataclass
class Shot:
    sid: int
    disruptive: bool
    down: float | None
    horizon: float
    times: np.ndarray
    native: np.ndarray
    blocks: slice
    teacher: dict


def features(data, view):
    e = data["effect"]
    n, a, _ = e.shape
    history = data["risk"]
    anchor = e[..., 2:3]
    # An additive shot context alone cannot change which action wins. Encode
    # shared interactions with action identity and attained historical change.
    action = np.broadcast_to(np.eye(a, dtype=np.float32)[None, :, 1:], (n, a, a - 1))
    risk = np.broadcast_to(history[:, None], (n, a, history.shape[1]))
    margin = history[:, 7] - bank.TAU
    interactions = np.concatenate(
        (
            action * margin[:, None, None],
            anchor * risk[..., [0, 7, 8, 9]],
        ),
        axis=2,
    )
    if view == "risk":
        motion = np.zeros((n, a, 45), dtype=np.float32)
    elif view in ("true", "reverse"):
        k = 0 if view == "true" else 1
        window = data["ordered"][:, k]
        g = data["g"][:, k]
        scale = np.linalg.norm(g, axis=-1, keepdims=True)
        g = np.divide(g, scale, out=np.zeros_like(g), where=scale > 1e-8)
        agreement = np.einsum("baf,btf->bat", data["states"], g, optimize=True)
        motion = np.concatenate(
            (
                np.broadcast_to(window[:, None], (n, a, 27)),
                agreement,
                anchor * window[:, None, 9:18],
            ),
            axis=2,
        )
    else:
        raise ValueError("unknown prefix-information control")
    return np.concatenate((risk, action, e, interactions, motion), axis=2).astype(
        np.float32
    )


def scalar_curves(curves, effect, clock, points):
    ix = clock[:, None] + 1 + np.arange(T)[None]
    return np.where(ix[:, None] < points, curves[:, :1] + effect[..., 2, None], 0.0)


def safe_margin(m):
    return None if np.isposinf(m) else float(m)


def calibrate(evaluate, margins=MARGINS):
    records, key, selected = [], None, None
    for m in margins:
        result = evaluate(m)
        ok = not result["fp_added"] and not result["lost"]
        criterion = (len(result["recovered"]), len(result["fp_removed"]), m)
        records.append(dict(margin=safe_margin(m), eligible=ok, **result))
        if ok and (key is None or criterion > key):
            key, selected = criterion, m
    if selected is None:
        raise ValueError("calibration must contain a feasible no-write rule")
    return selected, records


def load_bank(root, h):
    pp, sp = root / "protocol.json", root / "seal.json"
    cfg, seal = bank.r.read(pp), bank.r.read(sp)
    expected, inv = bank.bank.src.source_inputs(h)
    inventory = {q["shot"]: q["sha256"] for q in inv}
    if (
        cfg["status"] != "WAVEFORM_ONLY_TRAIN_ACTION_BANK"
        or seal["status"] != "SEALED_BEFORE_OUTCOME_REVEAL"
        or cfg["outcome_reads"]
        or seal["outcome_reads"]
        or cfg["shots"] != expected["roles"]["train"]
        or tuple(cfg["actions"]) != bank.ORDER
        or cfg["tau"] != bank.TAU
        or cfg["source_state_sha256"] != expected["source_state_sha256"]
        or h(pp) != seal["protocol_sha256"]
        or not all(h(p) == q for p, q in cfg["inputs"].items())
        or not all(h(root / s["file"]) == s["sha256"] for s in seal["shards"])
    ):
        raise ValueError(
            "fixed source data, raw no-outcome seal or action-bank contract changed"
        )
    if [v["shot"] for s in seal["shards"] for v in s["shots"]] != cfg["shots"]:
        raise ValueError("sealed data omitted or reordered source TRAIN shots")
    h(sp)
    h(bank.r.io23.JTEXT_ADVANCED_TIME_PATH)
    advance = bank.r.simple_data.advanced_times()
    tape, shots, hard, soft, owner = defaultdict(list), [], [], [], []
    count, total = 0, 0
    for shard in seal["shards"]:
        with np.load(root / shard["file"], allow_pickle=False) as f:
            block = {k: f[k] for k in (*bank.FIELDS, "times", "native")}
        t, b = 0, 0
        for row in shard["shots"]:
            sid, n, m = (row[k] for k in ("shot", "points", "blocks"))
            ix, sl = slice(t, t + n), slice(b, b + m)
            times = block["times"][ix]
            native = block["native"][ix]
            path = bank.r.io23.path_for("J-TEXT", sid)
            if h(path) != row["raw_sha256"] or row["raw_sha256"] != inventory[sid]:
                raise ValueError("TRAIN raw hash changed after sealed inference")
            flag, down, hdf_times = outcome_and_clock(path, n)
            clock, allowed = block["clock"][sl], block["allowed"][sl]
            e = block["effect"][sl]
            if (
                not np.array_equal(hdf_times, times)
                or not np.array_equal(clock, np.arange(63, n - 1, 32))
                or not allowed[:, 0].all()
                or not np.array_equal(allowed[:, 1::2], allowed[:, 2::2])
                or np.abs(e[..., :3]).max(initial=0.0) > bank.pair.LIMIT
                or e[..., 3].max(initial=0.0) > bank.pair.RADIUS + 5e-7
                or not np.array_equal(e[~allowed], np.zeros_like(e[~allowed]))
            ):
                raise ValueError("clock or candidate protection did not replay")
            horizon = bank.r.io23.effective_horizon_ms(sid, advance)[0]
            y = utility.outcomes(
                native,
                times,
                clock,
                block["curves"][sl],
                allowed,
                tau=bank.TAU,
                disruptive=flag,
                down=down,
                horizon=horizon,
            )
            hard.append(y.pop("hard"))
            soft.append(y.pop("soft"))
            y.pop("target")
            owner.append(np.full(m, len(shots), dtype=np.int64))
            shots.append(
                Shot(
                    sid, flag, down, horizon, times, native, slice(count, count + m), y
                )
            )
            count += m
            total += n
            t, b = ix.stop, sl.stop
        if t != len(block["times"]) or b != len(block["clock"]):
            raise ValueError("sealed shard contains extra times or block records")
        for key in bank.FIELDS:
            tape[key].append(block[key])
    if count != seal["blocks"] or total != seal["points"] or len(shots) != 1353:
        raise ValueError("incomplete full TRAIN cohort")
    data = {k: np.concatenate(v) for k, v in tape.items()}
    return (
        shots,
        data,
        np.concatenate(hard),
        np.concatenate(soft),
        np.concatenate(owner),
    )


def score_policy(shots, data, choice, selected=None, *, scalar=False):
    use = range(len(shots)) if selected is None else np.flatnonzero(selected)
    records = []
    for k in use:
        q = shots[k]
        sl = q.blocks
        curves, clock, allowed = (data[p][sl] for p in ("curves", "clock", "allowed"))
        if scalar:
            curves = scalar_curves(curves, data["effect"][sl], clock, len(q.native))
        z = utility.replay(q.native, clock, curves, allowed, choice[sl])
        records.append(
            (
                q.sid,
                q.teacher["baseline"],
                utility.score(q.times, z, q.disruptive, q.down, bank.TAU),
            )
        )
    return records


def summarize(records):
    return dict(
        timely=sum(c["timely"] for _, _, c in records),
        fp=sum(c["false_alarm"] for _, _, c in records),
        recovered=[s for s, b, c in records if not b["timely"] and c["timely"]],
        lost=[s for s, b, c in records if b["timely"] and not c["timely"]],
        fp_added=[
            s for s, b, c in records if not b["false_alarm"] and c["false_alarm"]
        ],
        fp_removed=[
            s for s, b, c in records if b["false_alarm"] and not c["false_alarm"]
        ],
        first_changed=sum(b["first"] != c["first"] for _, b, c in records),
    )


def crossfit(shots, data, y, block_shot):
    flags = np.asarray([q.disruptive for q in shots])
    ids = np.asarray([q.sid for q in shots], dtype=np.int64)
    folds = fit.shot_folds(ids, N_FOLDS)
    result = {}
    receipts = {}
    allowed = data["allowed"]
    for name in VIEWS:
        x = features(data, name)
        choices = np.full((len(x), 2), -1, dtype=np.int64)
        pred = np.full(allowed.shape, -np.inf, dtype=np.float32)
        fits = []
        for k in range(N_FOLDS):
            outer, cal = folds == k, folds == (k + 1) % N_FOLDS
            train = ~(outer | cal)
            train_rows, cal_rows, held_rows = (
                v[block_shot] for v in (train, cal, outer)
            )
            # Physically omit held and calibration feature/target rows from fit.
            model = fit.fit_model(
                x[train_rows],
                y[train_rows],
                allowed[train_rows],
                block_shot[train_rows],
                flags & train,
                train,
                alpha=ALPHA,
            )
            cal_p = model.predict(x[cal_rows], allowed[cal_rows])

            def replay_margin(m):
                chosen = np.zeros(len(x), dtype=np.int64)
                chosen[cal_rows] = fit.choices(cal_p, allowed[cal_rows], m)
                return summarize(score_policy(shots, data, chosen, cal))

            margin, candidates = calibrate(replay_margin)
            held = model.predict(x[held_rows], allowed[held_rows])
            pred[held_rows] = held
            for j, m in enumerate((0.0, margin)):
                choices[held_rows, j] = fit.choices(held, allowed[held_rows], m)
            fits.append(
                dict(
                    fit_fold=k,
                    calibrate_fold=(k + 1) % N_FOLDS,
                    held_shots=ids[outer].tolist(),
                    fit_shots=ids[train].tolist(),
                    calibrate_shots=ids[cal].tolist(),
                    margin=safe_margin(margin),
                    calibration=candidates,
                    model=asdict(model),
                )
            )
        if (choices < 0).any() or not np.isfinite(pred[allowed]).all():
            raise ValueError("held-fold evaluation left source blocks unassigned")
        receipts[name] = fits
        result[name] = dict(open=choices[:, 0], guarded=choices[:, 1], utility=pred)
        print(f"Full shot-disjoint selection finished: {name}", flush=True)
    return result, receipts, folds


def support(shots, hard, allowed):
    result = dict(
        available_actions=int(allowed[:, 1:].sum()),
        positive_actions=int((hard > 0).sum()),
        negative_actions=int((hard < 0).sum()),
        single_edit_timely=[],
        single_edit_fp_removed=[],
        oracle_timely=[],
        oracle_fp_removed=[],
    )
    for q in shots:
        b = q.teacher["baseline"]
        if q.disruptive and not b["timely"]:
            if (hard[q.blocks] > 0).any():
                result["single_edit_timely"].append(q.sid)
            if q.teacher["oracle_timely"]:
                result["oracle_timely"].append(q.sid)
        if not q.disruptive and b["false_alarm"]:
            if (hard[q.blocks] > 0).any():
                result["single_edit_fp_removed"].append(q.sid)
            if q.teacher["oracle_no_false_alarm"]:
                result["oracle_fp_removed"].append(q.sid)
    return result


def run(root, output):
    if output.exists():
        raise FileExistsError(output)
    h, started = Hashes(), time.monotonic()
    for p in (
        Path(__file__),
        Path(fit.__file__),
        Path(utility.__file__),
        bank.PLAN,
        BASELINE,
    ):
        h(p)
    shots, data, hard, soft, owner = load_bank(root, h)
    base = summarize(
        [(q.sid, q.teacher["baseline"], q.teacher["baseline"]) for q in shots]
    )
    reference = bank.r.read(BASELINE)["summary"]
    if (
        base["timely"] != reference["static_timely"]
        or base["fp"] != reference["static_fp"]
    ):
        raise ValueError("raw frozen first alarms did not match original TRAIN record")
    y = 0.75 * hard + 0.25 * soft
    selected, folds_receipt, folds = crossfit(shots, data, y, owner)
    h.check()
    output.mkdir(parents=True, exist_ok=False)
    choice_path = output / "selected_actions.npz"
    fold_path = output / "fit_and_calibration.json"
    order = [(n, m) for n in VIEWS for m in ("open", "guarded")]
    bank.r.save_arrays(
        choice_path,
        choices=np.stack([selected[n][m] for n, m in order], axis=1),
        utility=np.stack([selected[n]["utility"] for n in VIEWS], axis=0),
        hard=hard,
        soft=soft,
        folds=folds,
        owner=owner,
    )
    bank.r.write_new(
        fold_path,
        dict(
            order=order, evidence=folds_receipt, bank_seal_sha256=h(root / "seal.json")
        ),
    )
    scores, per_shot = {}, {}
    for name, mode in order:
        idx = selected[name][mode]
        records = score_policy(shots, data, idx)
        label = f"{name}_{mode}"
        scores[label] = summarize(records) | {"writes": int(np.count_nonzero(idx))}
        per_shot[label] = records
    for m in ("open", "guarded"):
        idx = selected["true"][m]
        key = f"true_{m}_scalar"
        records = score_policy(shots, data, idx, scalar=True)
        scores[key] = summarize(records) | {"writes": int(np.count_nonzero(idx))}
        per_shot[key] = records

    def benefit(x):
        return set(x["recovered"] + x["fp_removed"])

    main = scores["true_guarded"]
    unique = sorted(
        benefit(main)
        - set().union(
            *(
                benefit(scores[k])
                for k in ("risk_guarded", "reverse_guarded", "true_guarded_scalar")
            )
        )
    )
    promoted = bool(
        unique
        and not main["lost"]
        and not main["fp_added"]
        and (main["recovered"] or len(main["fp_removed"]) >= 3)
    )
    result = dict(
        status="SCORED_TRAIN_SHOT_DISJOINT_DIAGNOSTIC",
        baseline=base,
        scores=scores,
        capacity=support(shots, hard, data["allowed"]),
        distinct_true_benefit=unique,
        promoted=promoted,
        n_shots=len(shots),
        disruptive=sum(q.disruptive for q in shots),
        blocks=len(owner),
        per_shot=per_shot,
        bank_seal_sha256=h(root / "seal.json"),
        choices_sha256=h(choice_path),
        fit_sha256=h(fold_path),
        input_sha256={str(k): v[1] for k, v in h.cache.items()},
        elapsed_seconds=time.monotonic() - started,
        new_val_test_east_reads=0,
    )
    bank.r.write_new(output / "result.json", result)
    print(
        {
            k: result[k]
            for k in (
                "baseline",
                "scores",
                "capacity",
                "distinct_true_benefit",
                "promoted",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("bank", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run(args.bank, args.output)
