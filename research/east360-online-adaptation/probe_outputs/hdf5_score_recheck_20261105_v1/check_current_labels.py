"""Reconcile TEST220 alarm records, the latest endpoint diagnostic, and prior receipts."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.risk_aligned_online import io23  # noqa: E402
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import (  # noqa: E402
    read,
    sha,
    write,
)
from experiments.risk_aligned_online.source_gate import qualify_source  # noqa: E402


def main():
    probe = ROOT / "probe_outputs"
    fresh = probe / "hdf5_outcome_authority_20261105_v1"
    inventory_file = fresh / "hdf5_metadata.json"
    inventory = read(inventory_file)
    digest = sha(inventory_file)
    assert read(fresh / "completed.json")["artifact_sha256"][inventory_file.name] == digest
    assert read(fresh / "audit.json")["status"] == "PASS"
    jt = {r["shot"]: r for r in inventory["J-TEXT"]}
    east = {r["shot"]: r for r in inventory["EAST"]}
    source = ROOT / "experiments/risk_aligned_online/runs/gradient_aligned512_20260929"
    qualification = qualify_source(source / "source.pt")
    assert qualification["qualified"]
    paired = source / "test_intra.json"
    report = read(source / "test220.json")
    assert sha(paired) == report["provenance"]["paired_evaluation_sha256"]
    m = report["jtext"]
    assert read(paired)["arms"]["static"]["metrics"] == m
    rows = m["predictions"]
    ids = [r["shot"] for r in rows]
    assert len(ids) == len(set(ids)) == 220
    assert set(ids) == set(io23.source_roles()["test"])
    tp = fp = nd = alarms_with_lead = 0
    for r in rows:
        truth = jt[r["shot"]]
        flag, down = truth["disruptive"], truth["down"]
        assert r["is_disrupt"] == flag
        lead = r["lead_ms"]
        if flag and r["alarm"]:
            # The archive has leads, not full curves. Check that the implied
            # first crossing lands on the raw HDF clock; do not claim replay.
            index = round((down - lead / 1000 - truth["start"]) * truth["rate"])
            assert 0 <= index < truth["samples"]
            time = truth["start"] + index / truth["rate"]
            assert round((down - time) * 1000, 9) == lead
            alarms_with_lead += 1
        else:
            assert lead is None
        timely = bool(flag and lead is not None and lead > 10)
        assert timely == r["timely"]
        nd += flag
        tp += timely
        fp += not flag and r["alarm"]
    ns = len(ids) - nd
    actual = dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns, tpr=tp / nd, fpr=fp / ns,
                  J=tp / nd - fp / ns)
    assert all(m[k] == v for k, v in actual.items())

    timed = probe / "timed_token_norm_20261105_v1"
    done = read(timed / "completed.json")
    assert done["status"] == "COMPLETE_TIMED_TOKEN_NORM_ROUTE_SCREEN"
    assert all(sha(timed / f) == h for f, h in done["artifact_sha256"].items())
    rows = read(timed / "rows.json")
    assert len(rows) == done["cases"] == 240
    assert len({(r["shot"], r["future"]) for r in rows}) == 240
    cohort = read(probe / "joint_lora_neutral_stream360_20261104_v1/protocol.json")["order"]
    assert {r["shot"] for r in rows} == {r["shot"] for r in cohort[20:100]}
    for r in rows:
        truth = east[r["shot"]]
        assert 0 <= r["origin"] < r["arrival"] < r["future"] < truth["samples"]
        t = np.float64(truth["start"]) + r["future"] / np.float64(truth["rate"])
        flag, down = truth["disruptive"], truth["down"]
        y, valid = io23.labels_at(np.array([t]), flag, down, 125.0)
        expected_lead = round(float((down - t) * 1000), 9) if flag else None
        assert valid[0] and r["label"] == int(y[0]) and r["lead_ms"] == expected_lead
        assert r["group"] == ("safe" if not flag else ("positive" if y[0] else "earlier"))
        if y[0]:
            assert expected_lead > 30

    prior = read(probe / "hdf5_current_results_recheck_20261104_v1/result.json")
    assert prior["status"] == "PASS" and prior["inventory_metadata_sha256"] == digest
    assert all(sha(Path(f)) == h for f, h in prior["input_sha256"].items())
    result = dict(
        status="PASS",
        hdf5_metadata_sha256=digest,
        jtext_test220=dict(
            **actual, saved_leads_matching_hdf_clock=alarms_with_lead,
            threshold=m["threshold"], source_sha256=qualification["checkpoint_sha256"],
            scope="saved alarm records; not new model inference",
        ),
        east_latest_endpoint_diagnostic=dict(
            cases=len(rows), shots=len({r["shot"] for r in rows}),
            groups=dict(Counter(r["group"] for r in rows)), labels_and_leads_match=True,
        ),
        previous_five_runs=dict(
            hashes_rechecked=len(prior["input_sha256"]),
            curves_previously_rescored=sum(r["method_shot_curves"] for r in prior["runs"].values()),
            curves_rescored_again_here=0,
        ),
        script_sha256=sha(Path(__file__)),
    )
    destination = Path(__file__).with_name("current_label_recheck.json")
    if destination.exists():
        raise FileExistsError(destination)
    write(destination, result)
    print(result)


if __name__ == "__main__":
    main()
