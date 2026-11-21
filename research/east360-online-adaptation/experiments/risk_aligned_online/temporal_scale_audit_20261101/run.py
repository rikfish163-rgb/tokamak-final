"""TRAIN1353 versus exactly ended cold20: no target-suffix or holdout signals."""

import argparse
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..east_context_operator_20261031.run import read
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE, REFERENCE_SHA
from ..warmstart_alignment_20261022.alignment import transform
from .core import LAGS, variogram, patch_retention

PLAN = io23.ROOT / "TEMPORAL_SCALE_AUDIT_PLAN_20261101.md"
MODES = ("fixed128", "record_windows", "timely_positive")
FIELDS = (
    "raw_variance",
    "asinh_std",
    "patch_mean_retention",
    "observed",
    "complete_patches",
    *("D" + str(lag) for lag in LAGS),
)


def endpoints(shot):
    valid = np.flatnonzero(shot["valid"] & (np.arange(len(shot["values"])) >= 127))
    views = dict(
        fixed128=[127] if 127 in valid else [], record_windows=[], timely_positive=[]
    )
    if len(valid):
        views["record_windows"] = valid[
            np.linspace(0, len(valid) - 1, min(16, len(valid)), dtype=int)
        ].tolist()
    if shot["is_disrupt"]:
        lead = np.round((shot["down"] - shot["times"]) * 1000, 9)
        positive = np.flatnonzero(
            shot["valid"]
            & (shot["y"] == 1)
            & (lead > (10 if shot["device"] == "J-TEXT" else 30))
            & (np.arange(len(lead)) >= 127)
        )
        if len(positive):
            views["timely_positive"] = positive[
                np.linspace(0, len(positive) - 1, min(8, len(positive)), dtype=int)
            ].tolist()
    return views


def window_features(raw, mapped, available, center, scale):
    matrix = np.full((23, len(FIELDS)), np.nan)
    pairs = np.zeros((23, len(LAGS)), np.uint16)
    for c in range(23):
        m = available[:, c] & np.isfinite(raw[:, c])
        v = raw[m, c].astype(np.float64)
        matrix[c, FIELDS.index("observed")] = len(v)
        if len(v):
            matrix[c, 0] = np.square(v - v.mean()).mean()
            coordinates = np.arcsinh(
                (mapped[m, c].astype(np.float64) - center[c]) / scale[c]
            )
            matrix[c, 1] = coordinates.std()
        clean = np.where(m, mapped[:, c], center[c]).astype(np.float64)
        coordinates = np.arcsinh((clean - center[c]) / scale[c])
        retention, complete = patch_retention(coordinates, m)
        matrix[c, 2] = retention if retention is not None else np.nan
        matrix[c, 4] = complete
        matrix[c, 5:], pairs[c] = variogram(raw[:, c], m)
    return matrix, pairs


def summarize(arrays):
    result = {}
    for mode_id, name in enumerate(MODES):
        result[name] = {}
        for group in ("ALL", "SAFE", "DISRUPT"):
            selected = arrays["mode"] == mode_id
            if group != "ALL":
                selected &= arrays["disruptive"] == (group == "DISRUPT")
            ids = np.unique(arrays["shot"][selected])
            by_shot = []
            for sid in ids:
                values = arrays["features"][selected & (arrays["shot"] == sid)]
                valid = np.isfinite(values)
                count = valid.sum(0)
                mean = np.full((23, len(FIELDS)), np.nan)
                np.divide(
                    np.where(valid, values, 0).sum(0), count, out=mean, where=count > 0
                )
                by_shot.append(mean)
            channels = {}
            for c, channel in enumerate(io23.RAW23):
                metrics = {}
                for k, field in enumerate(FIELDS):
                    values = np.array(
                        [v[c, k] for v in by_shot if np.isfinite(v[c, k])]
                    )
                    metrics[field] = (
                        dict(
                            n_shots=len(values),
                            q10=float(np.quantile(values, 0.1)),
                            median=float(np.median(values)),
                            q90=float(np.quantile(values, 0.9)),
                        )
                        if len(values)
                        else dict(n_shots=0, q10=None, median=None, q90=None)
                    )
                channels[channel] = metrics
            result[name][group] = dict(
                n_shots=len(ids), n_windows=int(selected.sum()), channels=channels
            )
    return result


def main(root):
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    source_ids = simple_data.source_roles()["train"]
    target_ids = [r["shot"] for r in read(REFERENCE / "order.json")[:20]]
    mapping = read(REFERENCE / "input_map.json")
    assert mapping["calibration_ids"] == target_ids
    target_hashes = read(REFERENCE / "input_hashes.json")
    for index, sid in enumerate(target_ids):
        prefix = REFERENCE / "shots" / f"{index + 1:04d}_{sid}"
        seal = read(Path(str(prefix) + "_seal.json"))
        assert (
            sha(prefix.with_suffix(".npz")) == seal["sha256"]
            and seal["outcome_available"] is False
        )
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = (
        base.center.numpy().astype(np.float64),
        base.scale.numpy().astype(np.float64),
    )
    advanced = simple_data.advanced_times()
    root.mkdir(parents=True)
    files = {Path(__file__), Path(__file__).with_name("core.py"), PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="SOURCE_TRAIN_AND_ENDED_TARGET20_DESCRIPTIVE_TEMPORAL_AUDIT",
        source_ids=source_ids,
        target_ids=target_ids,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        input_map_sha256=sha(REFERENCE / "input_map.json"),
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=sha(DEFAULT_PARENT),
        center=center.tolist(),
        scale=scale.tolist(),
        modes=MODES,
        fields=FIELDS,
        lags=LAGS,
        target_suffix_signals_read=0,
        target_suffix_outcomes_read=0,
        source_holdout_signals_read=0,
        no_training=True,
        no_gpu=True,
        label_authority="HDF5",
        time_axis_changed=False,
        physical_time_scale_certified=False,
        summary="window mean within each shot, shot-equal median and10/90percentile; separate outcome groups",
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    _durable_json(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    inventory, summaries = [], {}
    for device, ids in (("J-TEXT", source_ids), ("EAST", target_ids)):
        features, pairs, metadata = [], [], []
        for i, sid in enumerate(ids):
            path = io23.path_for(device, sid)
            digest = sha(path)
            if device == "EAST":
                assert digest == target_hashes[str(sid)]
            s = simple_data.read_shot(
                device, sid, rate_contract="nominal" if device == "EAST" else "strict"
            )
            if device == "EAST":
                s = apply_input_contract(s)
            s = simple_data.attach_outcome(
                s, device, advanced if device == "J-TEXT" else {}
            )
            signal = {k: s[k] for k in ("values", "available")}
            mapped = transform(signal, mapping) if device == "EAST" else signal
            views = endpoints(s)
            inventory.append(
                dict(
                    device=device,
                    shot=sid,
                    sha256=digest,
                    disruptive=s["is_disrupt"],
                    n_points=len(s["values"]),
                    eligible={k: len(v) for k, v in views.items()},
                    dt_ms=float(np.median(np.diff(s["times"])) * 1000),
                )
            )
            for mode_id, name in enumerate(MODES):
                for end in views[name]:
                    span = slice(end - 127, end + 1)
                    value, pair = window_features(
                        s["values"][span],
                        mapped["values"][span],
                        s["available"][span],
                        center,
                        scale,
                    )
                    features.append(value)
                    pairs.append(pair)
                    metadata.append((sid, mode_id, end, s["is_disrupt"]))
            if (i + 1) % 100 == 0 or i + 1 == len(ids):
                progress = dict(
                    device=device,
                    shots=i + 1,
                    total=len(ids),
                    windows=len(features),
                    seconds=time.monotonic() - started,
                )
                _durable_json(root / "progress.json", progress)
                print(progress, flush=True)
        meta = np.asarray(metadata, dtype=np.int64)
        arrays = dict(
            features=np.stack(features),
            pair_counts=np.stack(pairs),
            shot=meta[:, 0],
            mode=meta[:, 1],
            end=meta[:, 2],
            disruptive=meta[:, 3].astype(bool),
        )
        path = root / (
            "source_windows.npz" if device == "J-TEXT" else "target_windows.npz"
        )
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        summaries[device] = summarize(arrays)
    _durable_json(root / "inventory.json", inventory)
    _durable_json(root / "summary.json", summaries)
    result = dict(
        status="COMPLETED_DESCRIPTIVE",
        source_shots=len(source_ids),
        target_shots=len(target_ids),
        source_class_counts={
            str(d): sum(
                r["device"] == "J-TEXT" and r["disruptive"] == d for r in inventory
            )
            for d in (False, True)
        },
        target_class_counts={
            str(d): sum(
                r["device"] == "EAST" and r["disruptive"] == d for r in inventory
            )
            for d in (False, True)
        },
        source_windows_sha256=sha(root / "source_windows.npz"),
        target_windows_sha256=sha(root / "target_windows.npz"),
        inventory_sha256=sha(root / "inventory.json"),
        summary_sha256=sha(root / "summary.json"),
        seconds=time.monotonic() - started,
        no_target_suffix_access=True,
        scientific_goal_achieved=False,
    )
    _durable_json(root / "completed.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
