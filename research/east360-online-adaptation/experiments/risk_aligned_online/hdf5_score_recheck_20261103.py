"""Re-score sealed curves using the freshly reconciled HDF5 metadata only."""

import argparse
from pathlib import Path

import numpy as np

from . import io23
from .history_gradient_metric_20261102.run import read, sha, write

INVENTORY = io23.ROOT / "probe_outputs/hdf5_outcome_authority_20261103_v1"
RUNS = (
    "east_decoder_complete360_v1",
    "east_functional_protection360_v1",
    "east_parameter_feedback_controls360_v1",
)


def score_curve(times, scores, disruptive, down, threshold):
    times, scores = np.asarray(times), np.asarray(scores)
    if times.ndim != 1 or scores.shape != times.shape or not len(times):
        raise ValueError("matching nonempty one-dimensional curve required")
    if not np.isfinite(times).all() or not np.isfinite(scores).all():
        raise ValueError("finite curve required")
    valid = (down - times) * 1000 >= -1e-6 if disruptive else np.ones(len(times), bool)
    hits = np.flatnonzero(valid & (scores >= threshold))
    lead = (
        round(float((down - times[hits[0]]) * 1000), 9)
        if disruptive and len(hits)
        else None
    )
    return dict(
        alarm=bool(len(hits)), lead_ms=lead, timely=bool(lead is not None and lead > 30)
    )


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    done = read(INVENTORY / "completed.json")
    if done["status"] != "PASS":
        raise ValueError("fresh metadata reconciliation must pass")
    for name, value in done["artifact_sha256"].items():
        if sha(INVENTORY / name) != value:
            raise ValueError("metadata audit artifact changed")
    metadata = read(INVENTORY / "hdf5_metadata.json")
    east = {r["shot"]: r for r in metadata["EAST"]}
    results, hashes = (
        {},
        {str(INVENTORY / "completed.json"): sha(INVENTORY / "completed.json")},
    )
    for name in RUNS:
        run = io23.ROOT / "probe_outputs" / name
        protocol, per, completed = (
            read(run / f) for f in ("protocol.json", "per_shot.json", "completed.json")
        )
        ids = [r["shot"] for r in protocol["order"]]
        rows, curves = {arm: [] for arm in per}, 0
        for index, sid in enumerate(ids, 1):
            truth = east[sid]
            path = io23.path_for("EAST", sid)
            stat = path.stat()
            if stat.st_size != truth["size"] or stat.st_mtime_ns != truth["mtime_ns"]:
                raise ValueError("HDF file stat changed since fresh inventory")
            if io23.read_outcome("EAST", sid) != (truth["disruptive"], truth["down"]):
                raise ValueError("HDF hard outcome changed since inventory")
            prefix = run / "shots" / f"{index:04d}_{sid}"
            sealed = read(prefix.with_name(prefix.name + "_seal.json"))
            curve = prefix.with_suffix(".npz")
            if sha(curve) != sealed["curve_sha256"]:
                raise ValueError("sealed curve hash changed")
            if sealed["current_outcome_used_in_prediction"] is not False:
                raise ValueError("current outcome was not excluded from prediction")
            hashes[str(curve)] = sealed["curve_sha256"]
            with np.load(curve) as values:
                expected_times = (
                    truth["start"] + np.arange(truth["samples"]) / truth["rate"]
                )
                if not np.array_equal(values["times"], expected_times):
                    raise ValueError("stored curve clock differs from HDF metadata")
                for arm in per:
                    scored = dict(
                        shot=sid,
                        disruptive=truth["disruptive"],
                        **score_curve(
                            values["times"],
                            values[arm],
                            truth["disruptive"],
                            truth["down"],
                            protocol["threshold"],
                        ),
                    )
                    if scored != per[arm][index - 1]:
                        raise AssertionError((name, sid, arm, "saved score differs"))
                    rows[arm].append(scored)
                    curves += 1
        metrics = {}
        for arm, arm_rows in rows.items():
            nd = sum(r["disruptive"] for r in arm_rows)
            ns = len(arm_rows) - nd
            tp = sum(r["timely"] for r in arm_rows)
            fp = sum(r["alarm"] and not r["disruptive"] for r in arm_rows)
            metrics[arm] = dict(
                tp=tp,
                fp=fp,
                n_disrupt=nd,
                n_safe=ns,
                tpr=tp / nd,
                fpr=fp / ns,
                J=tp / nd - fp / ns,
                early_over3000=sum(
                    r["timely"] and r["lead_ms"] > 3000 for r in arm_rows
                ),
            )
            reported = completed["metrics"][arm]["full360"]
            if any(reported[k] != v for k, v in metrics[arm].items()):
                raise AssertionError((name, arm, "published summary differs"))
        results[name] = dict(
            shots=len(ids),
            curves=curves,
            sealed_files=len(ids),
            metadata_and_clock_exact=True,
            saved_per_shot_exact=True,
            metrics=metrics,
        )
    output.mkdir()
    # Readable corrected membership views. Missing raw files stay UNAVAILABLE,
    # rather than becoming safe labels. These do not change any fixed split.
    corrected = {}
    for device, folder, suffix in (
        ("J-TEXT", "J-TEXT", "J-TEXT"),
        ("EAST", "EAST", "EAST_TV"),
        ("EAST", "TEST", "EAST_Test"),
    ):
        rs = {r["shot"]: r for r in metadata[device]}
        ids = set()
        for stem in ("Disruption", "Non-disruption"):
            ids.update(read(io23.ROOT / "shot_list" / folder / f"{stem}_{suffix}.json"))
        corrected[folder] = dict(
            device=device,
            disruption_shots=sorted(i for i in ids if i in rs and rs[i]["disruptive"]),
            non_disruption_shots=sorted(
                i for i in ids if i in rs and not rs[i]["disruptive"]
            ),
            unavailable_shots=sorted(ids - rs.keys()),
        )
    write(output / "resolved_shotlists.json", corrected)
    write(
        output / "curve_recheck.json",
        dict(
            status="PASS",
            scope="POSTRUN_HDF_OUTCOME_AND_SEALED_CURVE_RECHECK",
            runs=results,
            input_sha256=hashes,
            code_sha256=sha(Path(__file__)),
            original_shotlists_modified=False,
            fixed_cohorts_changed=False,
            raw_files_modified=False,
            model_training=False,
        ),
    )
    write(
        output / "completed.json",
        dict(
            status="PASS",
            artifact_sha256={f.name: sha(f) for f in output.iterdir() if f.is_file()},
        ),
    )
    print(
        dict(
            status="PASS",
            runs={
                k: {n: v[n] for n in ("shots", "curves", "sealed_files")}
                for k, v in results.items()
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output)
