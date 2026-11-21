"""Post-hoc operational ranking diagnostic, never a deployed threshold policy."""

import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..east_context_operator_20261031.run import read
from ..simple_online import _durable_json
from ..simple_train import sha


def main(root):
    output = root / "posthoc_ranking.json"
    if output.exists():
        raise FileExistsError(output)
    p = read(root / "protocol.json")
    done = read(root / "completed.json")
    assert done["status"] == "COMPLETED"
    ids = [r["shot"] for r in p["order"]]
    scores = {a: [] for a in p["arms"]}
    labels = []
    for i, sid in enumerate(ids):
        flag, down = io23.read_outcome("EAST", sid)
        labels.append(flag)
        path = root / "shots" / f"{i + 1:04d}_{sid}.npz"
        assert (
            sha(path) == read(root / "shots" / f"{i + 1:04d}_{sid}_seal.json")["sha256"]
        )
        with np.load(path, allow_pickle=False) as data:
            eligible = (
                np.round((down - data["times"]) * 1000, 9) > 30
                if flag
                else np.ones(len(data["times"]), bool)
            )
            for arm in scores:
                scores[arm].append(
                    float(data[arm][eligible].max()) if eligible.any() else -np.inf
                )
    y = np.asarray(labels)
    result = {}
    for arm, values in scores.items():
        values = np.asarray(values)
        positive = values[y]
        negative = values[~y]
        auc = float(
            (
                (positive[:, None] > negative).astype(float)
                + 0.5 * (positive[:, None] == negative)
            ).mean()
        )
        fixed = values >= p["threshold"]
        assert int(fixed[y].sum()) == done["metrics"][arm]["full360"]["tp"]
        assert int(fixed[~y].sum()) == done["metrics"][arm]["full360"]["fp"]
        thresholds = np.unique(
            np.r_[
                values[np.isfinite(values)],
                np.nextafter(values[np.isfinite(values)], np.inf),
                np.nextafter(1.0, 2.0),
            ]
        )
        caps = []
        for cap in (0.1, 0.15, 0.2):
            choices = []
            for threshold in thresholds:
                pred = values >= threshold
                tp = int(pred[y].sum())
                fp = int(pred[~y].sum())
                if fp / (~y).sum() <= cap:
                    choices.append((tp, -fp, float(threshold)))
            tp, negfp, threshold = max(choices)
            caps.append(
                dict(
                    cap=cap,
                    tp=tp,
                    fp=-negfp,
                    tpr=tp / int(y.sum()),
                    fpr=-negfp / int((~y).sum()),
                    threshold=threshold,
                )
            )
        result[arm] = dict(
            operational_auc=auc,
            caps=caps,
            no_timely_observation_shots=int((~np.isfinite(values[y])).sum()),
        )
    out = dict(
        scope="POSTHOC_DEVELOPMENT_DIAGNOSTIC_NOT_DEPLOYED",
        definition="D score=max probability at strict lead>30ms; S=max over entire saved record; AUC=P(Dscore>Sscore)+0.5 ties",
        all360_labels_used_for_caps=True,
        thresholds_must_not_be_reported_as_online_policy=True,
        n_disrupt=int(y.sum()),
        n_safe=int((~y).sum()),
        metrics=result,
        completed_sha256=sha(root / "completed.json"),
        code_sha256=sha(Path(__file__)),
    )
    _durable_json(output, out)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
