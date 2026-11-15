"""Replace only the selected matched TRAIN block and recount full first alarms."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import run_peak_contrast_crossfit_20261106 as cv
import verify_peak_contrast_capacity_20261106 as verify
from verify_confirmation_raw_replay_20261105 import Hashes


def first(t, logit, flag, down, tau):
    z = np.asarray(logit, dtype=np.float64)
    admissible = z >= tau
    if flag:
        admissible &= (down - t) * 1000 >= -1e-6
    hits = np.flatnonzero(admissible)
    k = int(hits[0]) if len(hits) else None
    lead = round(float((down - t[k]) * 1000), 9) if flag and k is not None else None
    return dict(
        first=k,
        lead_ms=lead,
        timely=lead is not None and lead > 10,
        false_alarm=not flag and k is not None,
    )


def effect_decomposition(native, shape_z, anchor, positive):
    """Check whether positive mean shape contrast also raises actual danger."""
    eligible = positive[:, None, None, :]
    native_peaks = np.where(eligible, native[:, :, None, :], -np.inf).max(-1)
    raw = np.where(eligible, shape_z, -np.inf).max(-1) - native_peaks
    adjusted = raw - anchor
    by_pair = (adjusted[:, 0] - adjusted[:, 1]).mean(1)
    return dict(
        raw_danger_safe_mean=raw.mean((0, 2)).tolist(),
        shape_danger_safe_mean=adjusted.mean((0, 2)).tolist(),
        median_paired_shape_contrast=float(np.median(by_pair)),
        positive_paired_contrast=int((by_pair > 0).sum()),
        n_pairs=len(by_pair),
        max_abs_anchor=float(np.abs(anchor).max()),
        seed_window_exposures_abs_anchor_over_half=int((np.abs(anchor) > 0.5).sum()),
    )


def run(root, crossfit, out):
    h, start = Hashes(), time.monotonic()
    if out.exists():
        raise FileExistsError(out)
    cfg, parts, raw_truth = verify.preflight(root, h)
    done = cv.r.read(crossfit)
    manifest, path = parts["all"]
    ids, positive, _, _ = verify.as_arrays(manifest, raw_truth)
    if (
        done["status"] != "CROSSFITTED_SOURCE_PAIR_REPLAY"
        or done["pair_ids"] != ids.tolist()
        or done["sealed_effects_sha256"] != h(path)
        or done["held_val_east_reads"] != 0
        or not all(h(p) == v for p, v in done["input_sha256"].items())
    ):
        raise ValueError(
            "same sealed matched interventions and held selections required"
        )
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = cv.r.load_parent(cv.r.DEFAULT_PARENT)
    if cv.r.state_digest(base) != cfg["state_sha256"]:
        raise ValueError("unmatched frozen model")
    advance, cache = cv.r.simple_data.advanced_times(), {}
    present = np.array(
        [[q["danger"]["present"], q["safe"]["present"]] for q in manifest["pairs"]]
    )
    names = (*cv.ARMS, "shape_no_anchor_offset")
    report, cases, decomposition = {}, {}, {}
    with np.load(path, allow_pickle=False) as a:
        native, logits, shift, norm, _ = verify.check_payload(a, len(ids), manifest)
        for p, q in enumerate(manifest["pairs"]):
            for side, tag in enumerate(("danger", "safe")):
                verify.verify_native(
                    base, q[tag], native[p, side], side == 0, cache, advance
                )
        for i, site in enumerate(cfg["sites"]):
            if site not in done["selections"]:
                continue
            picks = {k: np.asarray(v) for k, v in done["selections"][site].items()}
            z, offsets, _ = cv.chosen_payload(
                logits[:, :, i], shift[:, :, i], norm[:, :, i], native, present, picks
            )
            shape_col = cv.ARMS.index("shape")
            decomposition[site] = effect_decomposition(
                native, z[..., shape_col, :], offsets[..., shape_col], positive
            )
            totals = {name: [[0, 0] for _ in cv.SEEDS] for name in names}
            protected, changed = [0] * len(cv.SEEDS), []
            for p, pair in enumerate(manifest["pairs"]):
                for side, tag in enumerate(("danger", "safe")):
                    meta = pair[tag]
                    sid, m = meta["shot"], meta["maturity"]
                    raw, flag, down, baseline = cache[sid]
                    time_clock = raw["times"]
                    ix = np.arange(m + 1, min(m + 33, len(baseline)))
                    before = first(
                        time_clock, baseline, flag, down, cfg["logit_threshold"]
                    )
                    for j, seed in enumerate(cv.SEEDS):
                        outcomes = {}
                        for k, name in enumerate(names):
                            use = baseline.astype(np.float64).copy()
                            if name != "static":
                                payload = (
                                    z[p, side, j, shape_col, : len(ix)]
                                    - offsets[p, side, j, shape_col]
                                    if name == "shape_no_anchor_offset"
                                    else z[p, side, j, k, : len(ix)]
                                )
                                use[ix] = payload
                            alarm = first(
                                time_clock, use, flag, down, cfg["logit_threshold"]
                            )
                            outcomes[name] = alarm
                            totals[name][j][0] += int(alarm["timely"])
                            totals[name][j][1] += int(alarm["false_alarm"])
                        new = outcomes["shape"]
                        delayed = bool(
                            flag
                            and any(meta["protect"])
                            and (new["first"] is None or new["first"] > before["first"])
                        )
                        protected[j] += delayed
                        if delayed or any(
                            (outcomes[k]["timely"], outcomes[k]["false_alarm"])
                            != (before["timely"], before["false_alarm"])
                            for k in names[1:]
                        ):
                            changed.append(
                                dict(
                                    shot=sid,
                                    seed=seed,
                                    side=tag,
                                    maturity=m,
                                    delayed_protected_first=delayed,
                                    alarms=outcomes,
                                    shape_anchor_offset=float(
                                        offsets[p, side, j, shape_col]
                                    ),
                                )
                            )
            if protected != done["by_site"][site]["by_arm"]["shape"]["losses"]:
                raise ValueError(
                    "protected first-alarm loss differs from matched audit"
                )
            report[site], cases[site] = totals, changed
            print(site, totals, protected, flush=True)
    h.check()
    cv.r.write_new(
        out,
        dict(
            status="MATCHED_ONE_BLOCK_COUNTERFACTUAL_NOT_A_STREAM",
            crossfit_sha256=h(crossfit),
            implementation_sha256=h(Path(__file__)),
            totals=report,
            cases=cases,
            effect_decomposition=decomposition,
            test_val_east_reads=0,
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "crossfit", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.root, a.crossfit, a.out)
