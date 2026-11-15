"""Extract the actual EAST 70320 first-alarm activations from the frozen replay."""

import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.risk_aligned_online import commissioning_write_delay_20261112 as delay
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import state_digest


RUN = ROOT / "probe_outputs/commissioning_write_delay_east360_20261112_v2"
AUDIT = ROOT / "probe_outputs/commissioning_first_alarm_witness_east360_20261113_v1"
ARMS = ("bias_only", "earlier_5", "current_0", "current_5")
group, lag = delay.group, delay.lag


def archive(sid, protocol, seal, ref):
    rank = protocol["order"].index(sid) + 1
    name = f"{rank:04d}_{sid}.npz"
    a, b = RUN / "shots" / name, group.REFERENCE / "shots" / name
    for path, lock in ((a, seal), (b, ref)):
        if sha(path) != lock["curve_sha256"][name]:
            raise ValueError(f"changed archived curve for shot {sid}")
    with np.load(a, allow_pickle=False) as data, np.load(b, allow_pickle=False) as clock:
        np.testing.assert_array_equal(data["release"], clock["release"])
        return rank, dict(data), clock["times"]


def extract(raw, rank, args, decoder, path, weights, archived, n, mapping):
    base = args[0]
    # Restrict the actual forward graph and observations to the first alarm.
    signal = lag.prepare_actual(raw, rank, mapping)
    stop = n + 1
    prefix = {k: signal[k][:stop] for k in ("values", "available")}
    q, h, z, beta, prior, phase, correction = group.parent_track(prefix, args, decoder, path)
    eligible, fields, wave = group.feedback_waveform(prefix, q, args[2], z.device)
    states, common = group._estimate_innovation(
        base, prefix, q, eligible, phase, correction, prior, fields, wave, decoder, weights,
    )
    release = q + lag.H + 1
    np.testing.assert_array_equal(release, archived["release"][archived["release"] < stop])
    np.testing.assert_array_equal(common, archived["eligible"][:len(q)])
    cur, earlier, dose, _, equal_error, parent = delay.render(
        base, prefix, q, h, z, beta, phase, correction, states, common, archived["bias_only"][:stop],
    )
    errors = {}
    for arm, p in zip(ARMS, (parent, earlier[-1], cur[0], cur[-1]), strict=True):
        np.testing.assert_allclose(p, archived[arm][:stop], atol=3e-6, rtol=0)
        errors[arm] = float(np.max(np.abs(p - archived[arm][:stop])))
    np.testing.assert_allclose(dose, archived["dose"][:, :stop], atol=3e-6, rtol=1e-6)

    wait, patch_width = delay.WAITS[-1], base.cfg.patch
    slot = int(np.searchsorted(release + wait, n, side="right"))
    original_slot = int(np.searchsorted(release, n, side="right"))
    fraction = group.handoff_fraction(release[slot - 1] + wait, np.array([n]), patch_width)
    j = slot - 1
    i = np.flatnonzero(eligible)
    issued, current, state, forecast, center, spread = group.signed_window(
        base, prefix, q, i, phase, correction, prior, fields, decoder,
    )
    _, _, sp, sf = group.contrast_forces(forecast, wave[i], center, spread)
    force = sf.new_zeros((len(q), *sf.shape[1:]))
    force[i] = sf
    expected, fresh, previous, common_check = group.past_force_parts(q, force, eligible)
    np.testing.assert_array_equal(common_check, common)
    history = (q >= q[j] - group.bias.HISTORY * lag.STRIDE) & (release <= q[j]) & eligible
    history_count = int(history.sum())
    np.testing.assert_allclose(expected[j].detach().cpu(), force[history].sum(0).detach().cpu() / (history_count + 1),
                               atol=3e-7, rtol=0)
    history_counts = ((q[None, :] >= q[:, None] - group.bias.HISTORY * lag.STRIDE)
                      & (release[None, :] <= q[:, None]) & eligible[None, :]).sum(1)
    previous_index = int(np.flatnonzero(history & (history_counts >= 2))[-1])
    torch.testing.assert_close(previous[j], fresh[previous_index])
    gate = torch.as_tensor(common[i], device=z.device)[:, None, None]
    shaped = [group.shape_force(f[i], weights) * gate for f in (fresh, previous)]
    directions = group.signed_directions(issued, current, state, [(sp, f) for f in shaped])
    torch.testing.assert_close(directions, states[-2:, i+1], atol=2e-5, rtol=2e-5)

    before, after = [], []
    add_to_patches = group.layer.add_to_patches

    def capture(tokens, change):
        result = add_to_patches(tokens, change)
        before.append(tokens)
        after.append(result)
        return result

    x, mask = lag._windows(base, prefix, [n])
    with torch.no_grad(), patch.object(group.layer, "add_to_patches", capture):
        view, actual_dose, _ = group.forward_hidden(
            base, x, mask, group.layer.patcher(phase[original_slot:original_slot+1],
                                               correction[original_slot:original_slot+1]),
            writes=states[:, slot:slot+1],
            handoff=(states[:, slot-1:slot] * bool(common[j]),
                     torch.as_tensor(fraction, device=z.device, dtype=z.dtype)),
        )
        logit = z[n] + ((view[:, 0] - h[n]) * beta).sum(-1)
        scores = logit[[0, -2, -1]].sigmoid().cpu().numpy()
        np.testing.assert_allclose(scores, [archived[k][n] for k in ("bias_only", "current_5", "earlier_5")],
                                   atol=3e-6, rtol=0)
        delta = (after[0] - before[0])[-2:]
        torch.testing.assert_close(delta[:, -1], torch.zeros_like(delta[:, -1]), atol=0, rtol=0)
        torch.testing.assert_close(delta[:, :-1].sum(1), torch.zeros_like(delta[:, -1]), atol=6e-6, rtol=0)
        torch.testing.assert_close(delta.flatten(1).norm(dim=-1), actual_dose.expand(2), atol=2e-6, rtol=0)
        hidden = view[[0, -2, -1], 0]
        logit_terms = (hidden[1:] - hidden[:1]) * beta
        torch.testing.assert_close(logit_terms.sum(-1), logit[[-2, -1]] - logit[0], atol=3e-6, rtol=0)

    def to_np(a):
        return a.detach().cpu().numpy()

    k = int(np.flatnonzero(i == j)[0])
    choose = np.asarray(group.USE)
    names = [io23.RAW23[c] for c in lag.CHANNELS]
    trace = dict(
        target_index=q[j] + np.arange(1, lag.H + 1), channels=np.array(names)[choose],
        channel_groups=np.array(group.SIGNED_GROUPS), forecast=to_np(forecast[k])[:, choose],
        observation=to_np(wave[j])[:, choose], reference_center=to_np(center[k])[choose],
        reference_spread=to_np(spread[k])[choose], signed_forecast=to_np(sp[k]),
        signed_observation=to_np(group.signed_levels(wave[j:j+1], center[k:k+1], spread[k:k+1])[0]),
        clipped_error=to_np(force[j]), historical_error=to_np(expected[j]), new_error=to_np(fresh[j]),
        preceding_error=to_np(previous[j]), weighted_error=to_np(shaped[0][k]),
        local_coefficients=to_np(states[-2:, slot]), tokens_before=to_np(before[0][0, :-1]),
        token_delta=to_np(delta[:, :-1]), hidden=to_np(hidden), beta=to_np(beta),
        logit_terms=to_np(logit_terms), logits_at_alarm=to_np(logit[[0, -2, -1]]),
        scores_at_alarm=scores,
    )
    meta = dict(
        issue_index=int(q[j]), release_index=int(release[j]), previous_issue_index=int(q[previous_index]),
        wait_samples=wait, patch_width=patch_width, feature_window=base.cfg.window,
        history_issue_indices=q[history].tolist(), prefix_points=stop,
        newest_slot_if_no_wait=int(q[original_slot-1]), fraction=float(fraction[0]),
        max_prefix_score_difference=max(errors.values()), max_equal_dose_difference=float(equal_error),
        token_l2=to_np(delta.flatten(1).norm(dim=-1)).tolist(),
        largest_abs_token_sum=float(delta[:, :-1].sum(1).abs().max()),
        logit_increments=to_np(logit_terms.sum(-1)).tolist(),
    )
    return trace, meta


def main():
    protocol, seal = (read(RUN / f"{k}.json") for k in ("protocol", "seal"))
    reference = read(group.REFERENCE / "seal.json")
    audit = read(AUDIT / "independent_scores.json")
    witness = read(AUDIT / "prefix_at_first_alarm.json")
    threshold = protocol["calibration"]["0.2"]
    if (audit["references"]["delay"] != sha(RUN / "seal.json")
            or witness["delay_seal"] != audit["references"]["delay"]
            or protocol["history_seal"] != sha(delay.HISTORY / "seal.json")
            or protocol["reference_seal"] != sha(group.REFERENCE / "seal.json")
            or protocol["group_sha256"] != sha(Path(group.__file__))):
        raise ValueError("original experiment and frozen model trace are required")
    rank, a, t = archive(70320, protocol, seal, reference)
    safe_rank, b, u = archive(73743, protocol, seal, reference)
    n = int(np.flatnonzero(a["current_5"] >= threshold)[0])
    if witness["shots"]["70320"]["first_index"] != n:
        raise ValueError("first alarm differs from prefix witness")

    lag.configure()
    lock = lag.load_inputs()
    p, _, args, decoder, path = group.load_reference(lock)
    if p["order"] != protocol["order"] or p["calibration"]["0.2"] != threshold:
        raise ValueError("replay order or operating threshold changed")
    teach = delay.HISTORY / "teaching.json"
    if sha(teach) != read(delay.HISTORY / "protocol.json")["teaching_sha256"]:
        raise ValueError("frozen force weights changed")
    weights = torch.tensor(read(teach)["weights"], device=args[0].center.device, dtype=torch.float32)
    before = [state_digest(m) for m in args[:2]]
    for sid in (70320, 73743):
        if sha(io23.path_for("EAST", sid)) != lock["hashes"][str(sid)]:
            raise ValueError(f"original signal for {sid} changed")
    with (patch.object(io23, "read_outcome", side_effect=RuntimeError("outcomes unavailable during replay")),
          patch.object(group.hdf, "_raw_truth", side_effect=RuntimeError("outcomes unavailable during replay"))):
        raw = simple_data.read_shot("EAST", 70320, rate_contract="nominal")
        np.testing.assert_array_equal(raw["times"], t)
        trace, detail = extract(raw, rank, args, decoder, path, weights, a, n, lock["mapping"])
    if before != [state_digest(m) for m in args[:2]]:
        raise ValueError("model weights changed during extraction")

    per = read(RUN / "per_shot.json")["0.2"]
    cases = {}
    for sid, r, times, pack in ((70320, rank, t, a), (73743, safe_rank, u, b)):
        disrupted, down, raw_times = group.hdf._raw_truth(sid)
        np.testing.assert_array_equal(times, raw_times)
        outcome = {}
        for k in ARMS:
            s = score_curve(times, pack[k], disrupted, down, threshold)
            if per[k][r-1] != dict(shot=sid, disruptive=disrupted, down_s=down, **s):
                raise ValueError(f"sealed outcome disagrees for {sid} / {k}")
            hits = np.flatnonzero(pack[k] >= threshold)
            adjacent = np.flatnonzero((pack[k][:-1] >= threshold) & (pack[k][1:] >= threshold)) + 1
            first = int(hits[0]) if s["alarm"] else None
            outcome[k] = dict(
                **s, first_index=first, first_s=None if first is None else float(times[first]),
                first_score=None if first is None else float(pack[k][first]), max_score=float(pack[k].max()),
                first_two_point_confirmation=None if not adjacent.size else int(adjacent[0]),
            )
        cases[str(sid)] = dict(rank=r, disruptive=disrupted, down_s=down, scores=outcome)

    metrics = audit["bands"]["0.2"]["metrics"]
    timing = read(RUN / "postseal_timing.json")
    metadata = dict(
        scope="POSTSEAL_EXPLANATION_OF_FIXED_EAST360", threshold=threshold, cases=cases,
        trace=detail, timely_strictly_over_ms=30, n_disruptive=180, n_safe=180,
        counts={k: [metrics[k]["by_deadline"]["30"]["tp"], metrics[k]["fp"]] for k in ARMS},
        two_point_counts={k: timing["confirmation"]["0.2"]["2"][k] for k in ARMS},
    )
    np.savez_compressed(HERE / "figure_data.npz", **trace,
                        t70320=t, t73743=u, release70320=a["release"], release73743=b["release"],
                        **{f"s{sid}_{k}": pack[k] for sid, pack in ((70320, a), (73743, b)) for k in ARMS})
    (HERE / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    with (HERE / "score_traces.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("shot", "sample_index_0_based", "time_s", "row4_no_new_update",
                         "row6_earlier_error_wait5", "row7_latest_error_wait0", "row5_latest_error_wait5"))
        for sid, times, data in ((70320, t, a), (73743, u, b)):
            writer.writerows((sid, j, round(float(ts), 9), *(float(data[k][j]) for k in ARMS))
                             for j, ts in enumerate(times))
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
