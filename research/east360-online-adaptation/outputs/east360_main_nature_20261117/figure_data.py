"""Frozen source data for the EAST main-result figure."""

import csv
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXPECTED = {
    ("EAST360", 0): (136, 32),
    ("EAST360", 1): (138, 31),
    ("EAST360", 2): (141, 34),
    ("EAST360", 3): (143, 34),
    ("EAST360", 4): (144, 34),
    ("EAST360", 5): (146, 34),
    ("EAST2593", 0): (695, 367),
    ("EAST2593", 4): (697, 420),
    ("EAST2593", 5): (706, 434),
}


@lru_cache(maxsize=1)
def metrics():
    with (HERE / "source_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(EXPECTED)
    for row in rows:
        for key in ("step", "tp", "fp", "n_disruptive", "n_safe", "late", "miss"):
            row[key] = int(row[key])
        for key in ("tpr", "fpr"):
            row[key] = float(row[key])
        assert (row["tp"], row["fp"]) == EXPECTED[row["cohort"], row["step"]]
        assert row["tp"] + row["late"] + row["miss"] == row["n_disruptive"]
        assert row["tpr"] == row["tp"] / row["n_disruptive"]
        assert row["fpr"] == row["fp"] / row["n_safe"]
        assert row["source"] and row["owning_section"]
    return rows


@lru_cache(maxsize=1)
def metadata():
    return json.loads((HERE / "source_metadata.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def curves(shot):
    with (HERE / "source_curves.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["shot"] == str(shot)]
    arrays = {
        key: np.array([float(row[key]) for row in rows])
        for key in ("time_s", "before_step5", "with_step5")
    }
    assert len(rows) and np.all(np.diff(arrays["time_s"]) > 0)
    assert all(float(row["threshold"]) == metadata()["threshold"] for row in rows)
    assert all(np.isfinite(values).all() for values in arrays.values())
    return arrays
