"""Attach diagnostic feedback to all 360 sealed EAST replay curves."""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from extract_trace import ARMS, HERE, RUN, archive, delay, group, io23, lag, read, sha, simple_data, state_digest
from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve


def focus_sample(sid, a, threshold):
    if sid in (70320, 73405, 73743):
        k = "current_0" if sid == 73743 else "current_5"
        return int(np.flatnonzero(a[k] >= threshold)[0]), "alarm"
    if np.any(a["current_5"] != a["bias_only"]):
        return int(np.argmax(abs(a["current_5"] - a["bias_only"]))), "largest_change"
    return int(np.argmax(a["bias_only"])), "parent_peak"


def feedback(prefix, args, decoder, path, weights, a, focus):
    """Collect the arrays returned by the production forward and error code."""
    base = args[0]
    q, h, z, beta, prior, phase, correction = group.parent_track(prefix, args, decoder, path)
    available, fields, wave = group.feedback_waveform(prefix, q, args[2], z.device)
    targets = (len(q), lag.H, len(group.USE))
    pred = np.full(targets, np.nan, dtype=np.float32)
    estimates = []
    window, past = group.signed_window, group.past_force_parts

    def capture_window(base, signal, q, i, phase, correction, prior, fields, decoder):
        result = window(base, signal, q, i, phase, correction, prior, fields, decoder)
        pred[i] = result[3].detach().cpu().numpy()[..., group.USE]
        return result

    def capture_error(q, force, eligible):
        result = past(q, force, eligible)
        estimates[:] = [force, *result[:3]]
        return result

    with (patch.object(group, "signed_window", side_effect=capture_window),
          patch.object(group, "past_force_parts", side_effect=capture_error)):
        states, common = group._estimate_innovation(
            base, prefix, q, available, phase, correction, prior, fields, wave, decoder, weights,
        )
    release = q + lag.H + 1
    np.testing.assert_array_equal(release, a["release"])
    np.testing.assert_array_equal(common, a["eligible"])
    no_write = a["dose"][-1] == 0.
    np.testing.assert_array_equal(a["current_5"][no_write], a["bias_only"][no_write])
    np.testing.assert_array_equal(a["earlier_5"][no_write], a["bias_only"][no_write])

    # Check one actual scoring endpoint per shot without running the whole alarm archive again.
    wait, width = delay.WAITS[-1], base.cfg.patch
    native_slot = int(np.searchsorted(release, focus, side="right"))
    slot = int(np.searchsorted(release + wait, focus, side="right"))
    err = 0.
    if a["dose"][-1, focus] > 0.:
        x, m = lag._windows(base, prefix, [focus])
        frac = group.handoff_fraction(release[slot-1] + wait, np.array([focus]), width)
        with torch.no_grad():
            view, dose, dose_error = group.forward_hidden(
                base, x, m, group.layer.patcher(phase[native_slot:native_slot+1],
                                               correction[native_slot:native_slot+1]),
                writes=states[:, slot:slot+1],
                handoff=(states[:, slot-1:slot] * bool(common[slot-1]),
                         torch.as_tensor(frac, device=z.device, dtype=z.dtype)),
            )
            score = (z[focus] + ((view[:, 0] - h[focus]) * beta).sum(-1)).sigmoid().cpu().numpy()
        got = score[[0, -2, -1]]
        ref = np.array([a[k][focus] for k in ("bias_only", "current_5", "earlier_5")])
        np.testing.assert_allclose(got, ref, atol=3e-6, rtol=0)
        np.testing.assert_allclose(float(dose[0]), a["dose"][-1, focus], atol=3e-6, rtol=1e-6)
        if float(dose_error) > 1e-5:
            raise ValueError("unmatched perturbation magnitude")
        err = float(np.max(np.abs(got-ref)))

    def cpu(a):
        return a.detach().cpu().numpy()

    old, expected, new, prior_new = map(cpu, estimates)
    observed = cpu(wave)[..., group.USE]
    return dict(
        issue=q, forecast=pred, observation=observed, clipped_error=old, historical_error=expected,
        new_error=new, preceding_error=prior_new, coefficients=cpu(states[-2:, 1:]),
        channels=np.array([io23.RAW23[lag.CHANNELS[c]] for c in group.USE]),
    ), err


def main():
    dest = HERE / "atlas" / "data"
    dest.mkdir(parents=True, exist_ok=True)
    protocol, seal = (read(RUN / f"{k}.json") for k in ("protocol", "seal"))
    if (protocol["history_seal"] != sha(delay.HISTORY / "seal.json")
            or protocol["reference_seal"] != sha(group.REFERENCE / "seal.json")
            or protocol["group_sha256"] != sha(Path(group.__file__))):
        raise ValueError("the frozen group replay and its inputs are required")
    per = read(RUN / "per_shot.json")["0.2"]
    lag.configure()
    lock = lag.load_inputs()
    p, ref, args, decoder, path = group.load_reference(lock)
    if p["order"] != protocol["order"] or p["calibration"] != protocol["calibration"]:
        raise ValueError("replay order or operating threshold changed")
    teach = delay.HISTORY / "teaching.json"
    if sha(teach) != read(delay.HISTORY / "protocol.json")["teaching_sha256"]:
        raise ValueError("frozen force weights changed")
    weights = torch.tensor(read(teach)["weights"], device=args[0].center.device, dtype=torch.float32)
    before = [state_digest(m) for m in args[:2]]
    index = []
    for rank, sid in enumerate(p["order"], 1):
        _, a, times = archive(sid, protocol, seal, ref)
        threshold = p["original"] if rank <= p["startup"] else p["calibration"]["0.2"]
        focus, why = focus_sample(sid, a, threshold)
        detail, error = {}, 0.
        if sha(io23.path_for("EAST", sid)) != lock["hashes"][str(sid)]:
            raise ValueError(f"original signal for {sid} changed")
        with (patch.object(io23, "read_outcome", side_effect=RuntimeError("no outcomes during replay")),
              patch.object(group.hdf, "_raw_truth", side_effect=RuntimeError("no outcomes during replay"))):
            if rank > p["startup"]:
                raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
                np.testing.assert_array_equal(raw["times"], times)
                signal = lag.prepare_actual(raw, rank, lock["mapping"])
                detail, error = feedback(signal, args, decoder, path, weights, a, focus)
            elif a["eligible"].size or np.any(a["dose"]):
                raise ValueError("unexpected write during cold preparation")
        d, down, raw_time = group.hdf._raw_truth(sid)
        np.testing.assert_array_equal(times, raw_time)
        alarms = {}
        for k in ARMS:
            s = score_curve(times, a[k], d, down, threshold)
            if per[k][rank-1] != dict(shot=sid, disruptive=d, down_s=down, **s):
                raise ValueError(f"sealed outcome disagrees for {sid} / {k}")
            hits = np.flatnonzero(a[k] >= threshold)
            alarms[k] = dict(**s, index=int(hits[0]) if s["alarm"] else None,
                             max_score=float(a[k].max()))
        file = f"{rank:03d}_{sid}"
        np.savez_compressed(dest / f"{file}.npz", times=times, threshold=threshold, **detail,
                            **{k: a[k] for k in (*ARMS, "release", "eligible", "dose")})
        index.append(dict(file=file, rank=rank, shot=sid, disruptive=d, down_s=down, focus=focus,
                          focus_reason=why, threshold=threshold, n=len(times),
                          eligible_windows=int(a["eligible"].sum()), changed_points=int((a["dose"][-1] > 0).sum()),
                          max_abs_score_change=float(abs(a["current_5"]-a["bias_only"]).max()),
                          focus_score_error=error, alarms=alarms))
        if rank % 30 == 0 or rank == 1:
            print(f"extracted {rank}/360, shot {sid}, focus error {error:.2g}", flush=True)
    if before != [state_digest(m) for m in args[:2]]:
        raise ValueError("model weights changed during extraction")
    assert len(index) == 360 and sum(r["disruptive"] for r in index) == 180
    (dest.parent / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    with (dest.parent / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rank", "shot", "disruptive", "down_s", "threshold", "samples", "legal_feedback_windows",
                         "points_written", "max_abs_score_change", "focus_index_0_based",
                         *(f"{k}_{field}" for k in ARMS for field in ("alarm", "lead_ms", "timely"))))
        for r in index:
            writer.writerow((r["rank"], r["shot"], int(r["disruptive"]), r["down_s"], r["threshold"], r["n"],
                             r["eligible_windows"], r["changed_points"], r["max_abs_score_change"], r["focus"],
                             *(r["alarms"][k][f] for k in ARMS for f in ("alarm", "lead_ms", "timely"))))
    print(f"360 traces verified. Largest recomputed focus-score difference: "
          f"{max(r['focus_score_error'] for r in index):.3g}", flush=True)


if __name__ == "__main__":
    main()
