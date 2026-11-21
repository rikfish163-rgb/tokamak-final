"""Full scoring audit and selected no-hook neural replay of normalized inputs."""

import argparse
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..delayed_recurrent_memory_20261025.learning import batch, original_predictions
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..redesign_20261004.prepare_prior import load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .normalized_forward import coordinate_signal


@torch.no_grad()
def independent_forward(base, x, mask, mapped, active):
    """Explicit existing source layers; do not register the runner's input hook."""
    assert base.cfg.input_view == "source" and base.cfg.fast_site == "tokens"
    valid = mask & torch.isfinite(x)
    clean = torch.where(valid, x, base.center)
    level = torch.asinh((clean - base.center) / base.scale)
    level = torch.where(valid & active, mapped, level)
    features = torch.cat((level, valid.to(level.dtype)), -1)
    h = base.embedding(features.reshape(len(x), base.cfg.window // base.cfg.patch, -1))
    fast = base.initial_fast(len(x))
    dg, db = (fast - base.initial_fast(len(x))).chunk(2, -1)
    h = h * (1 + 0.1 * dg[:, None].tanh()) + 0.1 * db[:, None].tanh()
    token_valid = valid.reshape(len(x), base.cfg.window // base.cfg.patch, -1).any(-1)
    h = torch.cat((h, base.cls.expand(len(x), -1, -1)), 1) + base.position
    token_valid = torch.cat(
        (token_valid, torch.ones(len(x), 1, dtype=torch.bool, device=x.device)), 1
    )
    with sdpa_kernel(SDPBackend.MATH):
        h = base.encoder(h, src_key_padding_mask=~token_valid)
    gain, bias = base.initial_fast(len(x)).chunk(2, -1)
    h = base.norm(h) * (1 + 0.1 * gain[:, None].tanh()) + 0.1 * bias[:, None].tanh()
    h = h.masked_fill(~token_valid[:, :, None], 0.0)
    return base.risk_head(h[:, -1]).squeeze(-1).sigmoid()


def main(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise FileExistsError(output)
    p = read(root / "protocol.json")
    done = read(root / "completed.json")
    old = Path(p["original_map_run"])
    assert (
        done["status"] == "COMPLETED"
        and sha(old / "completed.json") == p["original_completed_sha256"]
    )
    assert sha(old / "independent_audit.json") == p["original_audit_sha256"]
    old_p = read(old / "protocol.json")
    reference = Path(old_p["reference"])
    raw_hashes = read(reference / "input_hashes.json")
    assert sha(Path(p["source_checkpoint"])) == p["source_sha256"]
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        for path, digest in p["code_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    ids = [r["shot"] for r in p["order"]]
    arms = p["arms"]
    assert len(ids) == len(set(ids)) == 360
    assert p["order"] == old_p["order"] and p["threshold"] == old_p["threshold"]
    access = read(root / "map_access.json")
    assert access["loaded_after_rank"] == 20 and access["ids"] == ids[:20]
    for name, digest in access["map_sha256"].items():
        assert sha(old / "maps" / f"{name}.json") == digest
    source_statistics = read(old / "source_statistics.json")
    target_statistics = read(old / "target_statistics.json")
    safe_ids = [sid for sid in ids[:20] if not io23.read_outcome("EAST", sid)[0]]
    assert target_statistics["pooled"]["shot_ids"] == ids[:20]
    assert target_statistics["safe"]["shot_ids"] == safe_ids
    assert source_statistics["pooled"]["shot_ids"] == io23.source_roles()["train"]
    assert (
        source_statistics["safe"]["shot_ids"]
        == read(old / "independent_audit.json")["source_safe_ids"]
    )
    for name in arms[2:]:
        fit = read(old / "maps" / f"{name}.json")
        group, mode = name.split("_")
        assert (
            fit["source"] == source_statistics[group]
            and fit["target"] == target_statistics[group]
        )
        assert (
            fit["mode"] == mode
            and fit["ridge"] == 1.0
            and fit["direction"] == "target_to_frozen_source"
        )
        assert (
            fit["first_affected_rank"] == 21
            and not fit["physical_calibration_certified"]
        )
        for c, feature in enumerate(io23.RAW23):
            if feature in ("MNM", "mode_number_n"):
                status = "excluded_semantic_uncertainty"
            elif (
                min(
                    fit["source"]["observed_shots"][c],
                    fit["target"]["observed_shots"][c],
                )
                < 3
            ):
                status = "fewer_than_three_observed_shots"
            elif min(fit["source"]["std"][c], fit["target"]["std"][c]) <= 0:
                status = "unidentifiable_zero_variation"
            else:
                status = "mapped"
            assert fit["status"][c] == status and fit["active"][c] == (
                status == "mapped"
            )
        assert fit["preceding_seals"] == {
            str(s): read(old / "shots" / f"{i + 1:04d}_{s}_seal.json")["sha256"]
            for i, s in enumerate(ids[:20])
        }
    recorded = read(root / "per_shot.json")
    rows = {name: [] for name in arms}
    roundtrip = []
    roundtrip_decision_changes = []
    negative = {a: dict(sxr_var=0, xuv_var=0) for a in arms[2:]}
    maximum = {a: 0.0 for a in arms[2:]}
    for i, sid in enumerate(ids):
        prefix = f"{i + 1:04d}_{sid}"
        path = root / "shots" / f"{prefix}.npz"
        seal = read(root / "shots" / f"{prefix}_seal.json")
        assert (
            sha(path) == seal["sha256"]
            and not seal["outcome_available"]
            and seal["shot"] == sid
            and seal["index"] == i + 1
        )
        assert sha(io23.path_for("EAST", sid)) == raw_hashes[str(sid)]
        if i < 20:
            assert access["seals"][str(sid)] == seal["sha256"]
        flag, down = io23.read_outcome("EAST", sid)
        with (
            np.load(path, allow_pickle=False) as data,
            np.load(old / "shots" / f"{prefix}.npz", allow_pickle=False) as old_data,
        ):
            assert set(data.files) == set(arms) | {"times"} and np.array_equal(
                data["times"], old_data["times"]
            )
            for name in arms:
                assert (
                    data[name].shape == data["times"].shape
                    and np.isfinite(data[name]).all()
                )
                r = classify(data["times"], data[name], flag, down, p["threshold"])
                assert dict(shot=sid, is_disrupt=flag, **r) == recorded[name][i]
                rows[name].append(dict(shot=sid, disruptive=flag, **r))
                if i < 20:
                    assert np.array_equal(data[name], data["source"])
            for name in ("source", "density"):
                assert np.array_equal(data[name], old_data[name])
            for name in arms[2:]:
                if name in old_data:
                    previous = classify(
                        old_data["times"], old_data[name], flag, down, p["threshold"]
                    )
                    current = classify(
                        data["times"], data[name], flag, down, p["threshold"]
                    )
                    if i >= 20 and previous != current:
                        roundtrip_decision_changes.append(
                            dict(shot=sid, arm=name, old=previous, new=current)
                        )
                    roundtrip.append(
                        dict(
                            shot=sid,
                            index=i + 1,
                            arm=name,
                            max_probability_error=float(
                                abs(old_data[name] - data[name]).max()
                            ),
                            threshold_point_differences=int(
                                (
                                    (old_data[name] >= p["threshold"])
                                    != (data[name] >= p["threshold"])
                                ).sum()
                            ),
                        )
                    )
        for name, info in seal["input_diagnostics"].items():
            maximum[name] = max(maximum[name], info["max_abs_mapped_coordinate"])
            for key, value in info["negative_variance"].items():
                negative[name][key] += value
    assert roundtrip == read(root / "roundtrip_comparison.json")
    mapped_roundtrip = [r for r in roundtrip if r["index"] > 20]
    mapped_errors = [r["max_probability_error"] for r in mapped_roundtrip]
    numeric_tolerance = 2e-6
    assert max(mapped_errors) <= numeric_tolerance, (
        "roundtrip numerical equivalence failed",
        max(mapped_errors),
    )
    metrics = {name: counts(rs) for name, rs in rows.items()}
    for name, rs in rows.items():
        for group, subset in (
            ("full360", rs),
            ("warmup20", rs[:20]),
            ("suffix340", rs[20:]),
        ):
            actual = counts(subset)
            for k in (
                "tp",
                "fp",
                "n_disrupt",
                "n_safe",
                "tpr",
                "fpr",
                "J",
                "early_over3000",
            ):
                assert actual[k] == done["metrics"][name][group][k]
    paired = {
        name + "__minus__density": flips(rows["density"], rows[name])
        for name in arms[2:]
    }
    for mode in ("diagonal", "correlation"):
        paired["safe_" + mode + "__minus__pooled_" + mode] = flips(
            rows["pooled_" + mode], rows["safe_" + mode]
        )
    result = dict(
        status="PASS",
        complete360_curves=2160,
        reference_curves_exact=720,
        metrics=metrics,
        flips=paired,
        original_prefix_comparisons=len(roundtrip),
        original_prefix_max_probability_error=max(
            r["max_probability_error"] for r in roundtrip
        ),
        original_prefix_changed_threshold_points=sum(
            r["threshold_point_differences"] for r in roundtrip
        ),
        mapped_prefix_only_comparisons=len(mapped_roundtrip),
        mapped_prefix_probability_error_quantiles=np.quantile(
            mapped_errors, [0, 0.5, 0.95, 1]
        ).tolist(),
        mapped_prefix_numeric_tolerance=numeric_tolerance,
        mapped_prefix_numeric_equivalence=True,
        mapped_prefix_first_alarm_decision_changes=roundtrip_decision_changes,
        maps_exact_membership_and_fields_verified=True,
        max_abs_coordinates=maximum,
        negative_variance_coordinate_points=negative,
        source_and_target_statistics_verified_by_parent_audit=True,
        parent_audit_sha256=sha(old / "independent_audit.json"),
        completed_sha256=sha(root / "completed.json"),
        independent_future_confirmation=False,
    )
    _durable_json(root / "count_audit.json", result)
    configure()
    base, _ = load_parent(Path(p["source_checkpoint"]))
    base.cuda().eval()
    # Fixed ranks include both previously failing shots, beginning and end.
    selected = [20, 118, 153, 359]
    exact = 0
    for index in selected:
        sid = ids[index]
        signal0 = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = {k: signal0[k] for k in ("values", "available")}
        with np.load(
            root / "shots" / f"{index + 1:04d}_{sid}.npz", allow_pickle=False
        ) as data:
            for name in arms:
                if name in ("source", "density"):
                    mapped = (
                        signal
                        if name == "source"
                        else transform(signal, read(old / "maps/density.json"))
                    )
                    curve = original_predictions(base, mapped)
                else:
                    fit = read(old / "maps" / f"{name}.json")
                    coordinate = coordinate_signal(
                        signal, fit, base.center.cpu().numpy(), base.scale.cpu().numpy()
                    )
                    active = torch.as_tensor(fit["active"], device="cuda")
                    pieces = []
                    for lo in range(0, len(signal["values"]), 32):
                        ends = range(lo, min(lo + 32, len(signal["values"])))
                        x, m = batch(signal, ends, "cuda")
                        z, _ = batch(coordinate, ends, "cuda")
                        pieces.append(
                            independent_forward(base, x, m, z, active).cpu().numpy()
                        )
                    curve = np.concatenate(pieces)
                assert np.array_equal(curve, data[name]), (
                    sid,
                    name,
                    "no-hook replay mismatch",
                )
                exact += 1
    result.update(
        no_hook_neural_replay_shots=[ids[i] for i in selected],
        exact_no_hook_neural_curves=exact,
        numerical_coordinate_map_implementation_shared=True,
        verifier_sha256=sha(Path(__file__)),
    )
    _durable_json(output, result)
    print({k: v for k, v in result.items() if k != "flips"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
