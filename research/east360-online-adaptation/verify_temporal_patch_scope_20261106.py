"""Recount actual HDF5-clock alarms; witness token support in the native model."""

import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

import run_temporal_patch_scope_20261106 as trial
from verify_adjacent_patch_intra_20261106 import direct, first_alarm, outcome_and_clock


def recount(native, f, row, mode, d, s, tau):
    dn, sn, di, si = (native[k] for k in ("danger_native", "safe_native", "di", "si"))
    for ix, clock, length in ((di, row["clock"], len(dn)), (si, row["safe_clock"], len(sn))):
        np.testing.assert_array_equal(ix[4:], np.arange(clock + 9, min(clock + 33, length)))
    if first_alarm(d[2], dn, True, d[1], tau)["timely"] or first_alarm(s[2], sn, False, None, tau)["false_alarm"]:
        raise ValueError("wrong archived static comparator")
    z, y, zm, ym = (f[f"{role}_{mode}{sign}"]
                    for sign in ("", "_minus") for role in ("danger", "safe"))
    tally = dict.fromkeys(trial.KEYS, 0)
    for ix in np.ndindex(z.shape[:2]):
        dg = np.abs(np.stack([z[ix][:4], zm[ix]]) - dn[di[:4]]).max()
        if dg > trial.prior.BOUND:
            continue
        sg = np.abs(np.stack([y[ix][:4], ym[ix]]) - sn[si[:4]]).max()
        da, sa, scalar = dn.copy(), sn.copy(), dn.copy()
        da[di[4:]], sa[si[4:]] = z[ix][4:], y[ix][4:]
        scalar[di[4:]] += z[ix][3] - dn[di[3]]
        dor = first_alarm(d[2], da, True, d[1], tau)
        sos = first_alarm(s[2], sa, False, None, tau)
        sco = first_alarm(d[2], scalar, True, d[1], tau)
        safe = sg <= trial.prior.BOUND and not sos["false_alarm"]
        timely = dor["timely"] and dor["lead_ms"] <= row["horizon"] + 1e-6
        tally["scalar_any"] += int(safe and sco["timely"])
        tally["danger"] += int(timely)
        tally["paired"] += int(timely and safe)
        tally["neural_only"] += int(timely and safe and not sco["timely"])
        tally["safe_reject"] += int(timely and not safe)
    if tally != {k: row["exact"][mode][k] for k in tally}:
        raise ValueError(f"full HDF5 recount mismatch {row['shot']}/{mode}")
    return tally, z.shape[0] * z.shape[1]


def mask_from_blocks(indices, release, scope):
    if scope == "full":
        return np.ones((len(indices), 16))
    a = np.zeros((len(indices), 16))
    for j, end in enumerate(indices):
        r = min(end, release) if scope == "fade" else end
        for k in range(12, 16):
            begin = end - 127 + k * 8
            a[j, k] = len(set(range(begin, begin + 8)) & set(range(r - 31, r + 1))) / 8.
    return a


@torch.no_grad()
def raw_witness(base, pack, ix, state, factor, release, mode, scores, tau):
    x, a, p = pack
    scope = mode.removesuffix("_energy")
    w = p.tokens.new_tensor(mask_from_blocks(ix, release, scope))
    local = replace(p, features=p.features * w[:, None, :, None])
    u = x.new_tensor(state).expand(len(x), -1)
    if factor is None:
        p = local
    else:
        localized = trial.b.op.delta(local, u, 0)
        u = u * x.new_tensor(factor)[:, None]
        spread = trial.b.op.delta(p, u, 0)
        torch.testing.assert_close(localized.norm(dim=(1, 2)), spread.norm(dim=(1, 2)),
                                   rtol=4e-6, atol=2e-7)
    z = direct(base, x, a, p, u, 0)
    minus = direct(base, x[:4], a[:4], p.take(slice(0, 4)), -u[:4], 0)
    errors = []
    for replayed, saved in zip((z, minus), scores, strict=True):
        error = float(np.abs(replayed - saved).max())
        if error > 5e-5 or not np.array_equal(replayed.astype(float) >= tau, saved.astype(float) >= tau):
            raise ValueError("native model hook disagrees with scored temporal write")
        errors.append(error)
    return max(errors)


def run(out):
    start = time.monotonic()
    b = trial.b
    cfg, result = (b.r.read(out/f"{name}.json") for name in ("protocol", "result"))
    old = Path(cfg["previous"])
    gains = b.r.read(Path(cfg["stress"])/"donor_waveform_receipt.json")["profiles"]
    advance = b.r.simple_data.advanced_times()
    b.r.configure()
    base, _ = b.r.load_parent(b.r.DEFAULT_PARENT)
    if b.r.state_digest(base) != cfg["source_state"]:
        raise ValueError("wrong frozen model for raw witnesses")
    base.cuda()
    basis = b.op.basis(base)
    pairs = candidates = witnesses = energy_witnesses = 0
    error = 0.
    totals = {}
    for view, rows in result["rows"].items():
        gain = gains[view]["gain"] if view in gains else None
        sums = totals[view] = {m: dict.fromkeys(trial.KEYS, 0) for m in trial.MODES}
        observed = {m: {k: set() for k in trial.KEYS} for m in trial.MODES}
        for row in rows:
            name = f"{row['shot']}_{row['clock']}.npz"
            with np.load(old/view/name, allow_pickle=False) as native, np.load(out/view/name, allow_pickle=False) as f:
                roles = (("danger", "shot", "clock", "di"), ("safe", "safe", "safe_clock", "si"))
                d, s = (outcome_and_clock(b.r.io23.path_for("J-TEXT", row[key]), len(native[f"{role}_time"]))
                        for role, key, _, _ in roles)
                if not d[0] or s[0] or row["horizon"] != b.r.io23.effective_horizon_ms(row["shot"], advance)[0]:
                    raise ValueError("wrong HDF5 class, disruption time, or supervision window")
                np.testing.assert_array_equal(d[2], native["danger_time"])
                np.testing.assert_array_equal(s[2], native["safe_time"])
                grid = f["grid"]
                norms = np.linalg.norm(grid, axis=-1)
                if np.ptp(norms, axis=1).max() > 1e-8 or norms.max() > cfg["radius"] + 1e-7:
                    raise ValueError("scope search did not use matched state norms")
                packs = {}
                for mode in trial.MODES:
                    n, k = recount(native, f, row, mode, d, s, cfg["tau"])
                    candidates += k
                    for key, count in n.items():
                        sums[mode][key] += count
                        if count:
                            observed[mode][key].add(row["shot"])
                    witness = row["exact"][mode]["witness"]
                    if witness is None and not row["fixed"]:
                        continue
                    ix = tuple(witness["idx"]) if witness is not None else (0, 0)
                    u = grid[ix[0], trial.SCOPES.index(mode.removesuffix("_energy")), ix[1]]
                    for role, key, clock, idx in roles:
                        if role not in packs:
                            packs[role] = trial.packed(base, basis, view, row[key], gain, native[idx])
                        factor = f[f"{role}_{mode}_factor"][ix] if mode.endswith("_energy") else None
                        error = max(error, raw_witness(base, packs[role], native[idx], u, factor,
                            row[clock] + 8, mode, (f[f"{role}_{mode}"][ix], f[f"{role}_{mode}_minus"][ix]), cfg["tau"]))
                        witnesses += 1
                        energy_witnesses += int(factor is not None)
                pairs += 1
        if {m: {k: sorted(v) for k, v in x.items()} for m, x in observed.items()} != result["counts"][view]:
            raise ValueError("deduplicated shot counts do not match full HDF5 recount")
    if pairs != 48 or not energy_witnesses:
        raise ValueError("incomplete recount or no actual equal-energy witnesses")
    b.r.write_new(out/"verification.json", dict(status="HDF5_FULL_TRACE_AND_TEMPORAL_HOOK_MATCH",
        pairs=pairs, candidates=candidates, raw_role_witnesses=witnesses,
        matched_energy_witnesses=energy_witnesses, max_logit_error=error,
        totals_by_view=totals, duration=time.monotonic()-start))
    print(pairs, candidates, witnesses, energy_witnesses, error, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
